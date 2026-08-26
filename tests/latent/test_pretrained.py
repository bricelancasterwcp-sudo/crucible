"""Tests for `crucible.latent.pretrained` -- the ONLY module allowed to
import `transformers`.

NO downloads, NO network anywhere in this file: `transformers.AutoModel`,
`transformers.AutoTokenizer`, and `huggingface_hub.model_info` are all
monkeypatched with tiny local stubs before any wrapper under test is
called. Every stub tensor here is small (hidden dims of 6-8, batches of
2-3) -- these tests exist to pin the CONTRACT (freezing, no-grad, output
shape/dtype/device, the exact `control.py` model-call contract), never to
exercise the real pretrained weights.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import torch

from crucible.latent import pretrained
from crucible.latent.control import _extract_logits, _tokenize_truncated

# -- module hygiene: the lazy-import boundary itself -------------------------


def test_module_top_level_never_imports_transformers_or_hub():
    """Only FUNCTION-BODY imports of `transformers`/`huggingface_hub` are
    allowed -- importing `crucible.latent.pretrained` itself must never
    touch either package. Checked via `ast` over the file's TOP-LEVEL
    statements only (`tree.body`, not a recursive walk), so a lazy import
    nested inside a function is correctly invisible to this check."""
    source = Path(pretrained.__file__).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("transformers"), f"top-level import of {name!r}"
            assert not name.startswith("huggingface_hub"), f"top-level import of {name!r}"


def test_revisions_are_pinned_hex_shas_not_a_branch_name():
    """A regex pin, not just `!= "main"`: an accidental branch name like
    `"main"`, `"master"`, or a stray version tag must fail this, not merely
    the single literal string `"main"`."""
    full_sha = re.compile(r"^[0-9a-f]{40}$")
    assert full_sha.fullmatch(pretrained.JINA_REVISION)
    assert full_sha.fullmatch(pretrained.CODEEXECUTOR_REVISION)


def test_d_model_of_jina_returns_768():
    assert pretrained.d_model_of_jina() == 768
    assert pretrained.JINA_HIDDEN_SIZE == 768


# -- jina_code_embedder --------------------------------------------------------


class _FakeJinaHiddenModel(torch.nn.Module):
    """Tiny stand-in for the real jina encoder: an `nn.Embedding` gives it
    real, gradient-bearing parameters (so freezing is a meaningful thing to
    assert), and its forward returns a plain tuple whose first element is
    the last-hidden-state tensor -- matching a real HF `ModelOutput`'s
    `output[0]` indexing."""

    def __init__(self, hidden: int = 8, vocab: int = 50):
        super().__init__()
        self.hidden = hidden
        self.embed = torch.nn.Embedding(vocab, hidden)

    def forward(self, input_ids, attention_mask=None):
        return (self.embed(input_ids),)


class _FakeJinaTokenizer:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id, revision=None, trust_remote_code=None):
        cls.calls.append(
            {"model_id": model_id, "revision": revision, "trust_remote_code": trust_remote_code}
        )
        return cls()

    def __call__(self, sources, padding=True, truncation=True, max_length=1024, return_tensors="pt"):
        # deterministic tiny "tokenization": token ids that always fit the fake vocab (< 50)
        id_lists = [[2] + [3] * min(len(s), max_length - 1) for s in sources]
        max_len = max((len(ids) for ids in id_lists), default=1)
        input_ids = torch.zeros((len(sources), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(sources), max_len), dtype=torch.long)
        for i, ids in enumerate(id_lists):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeJinaAutoModel:
    calls: list[dict] = []
    instances: list[_FakeJinaHiddenModel] = []

    @classmethod
    def from_pretrained(cls, model_id, revision=None, trust_remote_code=None, torch_dtype=None):
        cls.calls.append(
            {
                "model_id": model_id,
                "revision": revision,
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch_dtype,
            }
        )
        instance = _FakeJinaHiddenModel()
        cls.instances.append(instance)
        return instance


def test_jina_code_embedder_loads_pinned_revision_with_trust_remote_code(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "AutoModel", _FakeJinaAutoModel)
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeJinaTokenizer)

    pretrained.jina_code_embedder("cpu")

    tok_call = _FakeJinaTokenizer.calls[-1]
    model_call = _FakeJinaAutoModel.calls[-1]
    assert tok_call == {
        "model_id": pretrained.JINA_MODEL_ID,
        "revision": pretrained.JINA_REVISION,
        "trust_remote_code": True,
    }
    assert model_call["model_id"] == pretrained.JINA_MODEL_ID
    assert model_call["revision"] == pretrained.JINA_REVISION
    assert model_call["trust_remote_code"] is True
    assert model_call["torch_dtype"] == torch.float32  # bf16 only kicks in for a cuda device


def test_jina_code_embedder_freezes_the_model(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "AutoModel", _FakeJinaAutoModel)
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeJinaTokenizer)

    pretrained.jina_code_embedder("cpu")
    model = _FakeJinaAutoModel.instances[-1]

    assert model.training is False  # .eval() was called
    assert all(not p.requires_grad for p in model.parameters())  # requires_grad_(False) was called


def test_jina_code_embedder_output_shape_dtype_device_and_no_grad(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "AutoModel", _FakeJinaAutoModel)
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeJinaTokenizer)

    embed = pretrained.jina_code_embedder("cpu")
    out = embed(["def f(): pass", "x = 1"])

    assert out.shape == (2, 8)  # (B, d) -- d matches the fake encoder's hidden size
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
    assert out.requires_grad is False  # torch.no_grad() path


# -- codeexecutor_factory + codeexecutor_tokenizer -----------------------------


class _FakeCodeExecutorBase(torch.nn.Module):
    """Tiny RoBERTa-arch stand-in: `.config.hidden_size` (read by
    `_CodeExecutorClassifier`) plus a forward returning an object exposing
    `.last_hidden_state` -- matching a real HF `BaseModelOutput`."""

    def __init__(self, hidden: int = 6, vocab: int = 50):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = torch.nn.Embedding(vocab, hidden)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _FakeCodeExecutorAutoModel:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id, revision=None):
        cls.calls.append({"model_id": model_id, "revision": revision})
        return _FakeCodeExecutorBase()


class _FakeCodeExecutorTokenizer:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id, revision=None):
        cls.calls.append({"model_id": model_id, "revision": revision})
        return cls()

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is True
        return [1] + [(ord(c) % 40) + 2 for c in text] + [2]  # stays < the fake vocab (50)


def test_codeexecutor_factory_loads_pinned_revision(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "AutoModel", _FakeCodeExecutorAutoModel)

    pretrained.codeexecutor_factory("cpu")()

    call = _FakeCodeExecutorAutoModel.calls[-1]
    assert call == {"model_id": pretrained.CODEEXECUTOR_MODEL_ID, "revision": pretrained.CODEEXECUTOR_REVISION}


def test_codeexecutor_factory_matches_control_forward_contract_exactly(monkeypatch):
    """The decisive contract test: builds the model via the SAME factory
    ops wires into `train_control`, calls it POSITIONALLY exactly the way
    `control._forward_batch` does (`model(input_ids, attention_mask)`),
    and feeds the raw output through `control._extract_logits` itself
    (imported, not reimplemented) -- proving this module's output shape is
    accepted by `control.py`'s real extractor, not a hand-rolled
    lookalike."""
    import transformers

    monkeypatch.setattr(transformers, "AutoModel", _FakeCodeExecutorAutoModel)

    model = pretrained.codeexecutor_factory("cpu")()
    assert isinstance(model, torch.nn.Module)

    batch, seq_len = 3, 5
    input_ids = torch.randint(0, 50, (batch, seq_len))
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long)

    output = model(input_ids, attention_mask)  # positional, per control._forward_batch
    logits = _extract_logits(output)

    assert logits.shape == (batch,)


def test_codeexecutor_tokenizer_loads_pinned_revision_and_matches_control_truncation(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeCodeExecutorTokenizer)

    tokenize = pretrained.codeexecutor_tokenizer()
    call = _FakeCodeExecutorTokenizer.calls[-1]
    assert call == {"model_id": pretrained.CODEEXECUTOR_MODEL_ID, "revision": pretrained.CODEEXECUTOR_REVISION}

    full_ids = tokenize("abcdef")
    assert isinstance(full_ids, list) and all(isinstance(i, int) for i in full_ids)

    # control._tokenize_truncated's contract: list(tokenizer(text))[:max_len]
    truncated = _tokenize_truncated(tokenize, "abcdef", max_len=3)
    assert truncated == full_ids[:3]
    assert len(truncated) == 3


# -- resolve_current_revision (ops re-verification utility, never called internally) -----


def test_resolve_current_revision_calls_model_info(monkeypatch):
    import huggingface_hub

    fake_sha = "a" * 40

    def fake_model_info(repo_id):
        assert repo_id == "org/repo"
        return SimpleNamespace(sha=fake_sha)

    monkeypatch.setattr(huggingface_hub, "model_info", fake_model_info)

    assert pretrained.resolve_current_revision("org/repo") == fake_sha
