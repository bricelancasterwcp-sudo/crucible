"""Ops-facing pretrained-model wrappers (prereg §3/§5.2) -- the ONLY module
in this package allowed to import ``transformers``.

Every other `crucible.latent` module (`model.py`, `train.py`, `control.py`,
...) is deliberately kept free of `transformers` so it stays loadable and
mutation-testable with nothing heavier than `torch` on the box, and so a
change in one of those files can never accidentally start a weight
download. This module is the one place that boundary is crossed, and it
does so as narrowly as possible:

* The module's own top-level import is `torch` only. `transformers` and
  `huggingface_hub` are imported LAZILY, inside each factory function's
  body -- importing `crucible.latent.pretrained` itself never touches the
  network and never requires `transformers` to even be installed unless
  one of the factory functions below is actually called.
* Both HF repos this module wires are REVISION-PINNED to a specific commit
  sha (never `"main"`) -- resolved once, at implementation/lock time, via
  `huggingface_hub.model_info(repo_id).sha`, and hardcoded below. Pinning
  protects the pre-registered experiment from a silent upstream weight
  change mid-run; `resolve_current_revision` (bottom of this file) is an
  ops re-verification utility, never called by the wrappers themselves.

`jina_code_embedder` backs `train.py`'s injected `code_embedder`
(`(list[str]) -> torch.Tensor (B, d_model)`, frozen, prereg §5.2).
`codeexecutor_factory` + `codeexecutor_tokenizer` back `control.py`'s
injected `model_factory` / `tokenizer` (prereg §3) -- see
`crucible.latent.control.train_control`'s own docstring for the exact
model-call contract (`forward(input_ids, attention_mask) -> output` with a
`.logits`-bearing output, `(B,)` or `(B, 1)`) this module's classifier
matches.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import torch

# jinaai/jina-embeddings-v2-base-code -- resolved via
# `huggingface_hub.model_info("jinaai/jina-embeddings-v2-base-code").sha`
# at implementation time (2026-08-26). LOCK: pinned to this commit for the
# life of the pre-registered B-lite experiment.
JINA_MODEL_ID = "jinaai/jina-embeddings-v2-base-code"
JINA_REVISION = "516f4baf13dec4ddddda8631e019b5737c8bc250"

# microsoft/codeexecutor -- resolved via
# `huggingface_hub.model_info("microsoft/codeexecutor").sha` at
# implementation time (2026-08-26). LOCK: pinned to this commit for the
# life of the pre-registered B-lite experiment.
CODEEXECUTOR_MODEL_ID = "microsoft/codeexecutor"
CODEEXECUTOR_REVISION = "fcaa2615bd918a68e8c0a478934cfacfe423028e"

# jina-embeddings-v2-base-code's hidden size (a JinaBert encoder). Matches
# `crucible.latent.config.D_MODEL` -- the shared hidden width the frozen
# code encoder, StateEncoder's projection, and LatentPredictor all operate
# in (prereg §5.2). Not imported from config.py here to avoid a
# transformers-free-module <-> ops-module coupling; the two are asserted
# equal by `tests/latent/test_pretrained.py`.
JINA_HIDDEN_SIZE = 768


def d_model_of_jina() -> int:
    """The frozen code encoder's output width -- `JINA_HIDDEN_SIZE` (768),
    a plain constant, not a live model query (no network, no import)."""
    return JINA_HIDDEN_SIZE


def jina_code_embedder(device: str, revision: str = JINA_REVISION) -> Callable[[list[str]], torch.Tensor]:
    """Loads `jinaai/jina-embeddings-v2-base-code` (revision-pinned,
    `trust_remote_code=True` -- the model ships custom modeling code on the
    Hub), freezes it (`.eval()` + `requires_grad_(False)`), and returns
    `embed(sources: list[str]) -> torch.Tensor (B, d_model)`.

    `torch_dtype` is bf16 when `device` is a CUDA device, else fp32 --
    matching this project's other pretrained-load sites
    (`crucible/sleep/train.py`). `embed` tokenizes with `max_length=1024`
    (truncating), runs the encoder under `torch.no_grad()`, mean-pools the
    last hidden state over real (non-pad) token positions using the
    attention mask, and returns the result as float32 ON `device` --
    `train.py`'s `code_embedder` contract (`(list[str]) -> Tensor (B,
    d_model)`) consumes this directly; it also `.detach()`s and `.to
    (device)`s defensively on its own, but this function already satisfies
    both, structurally, before returning.
    """
    from transformers import AutoModel, AutoTokenizer

    torch_device = torch.device(device)
    load_dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL_ID, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        JINA_MODEL_ID, revision=revision, trust_remote_code=True, torch_dtype=load_dtype,
    )
    model.to(torch_device)
    model.eval()
    model.requires_grad_(False)

    def embed(sources: list[str]) -> torch.Tensor:
        encoded = tokenizer(
            sources, padding=True, truncation=True, max_length=1024, return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(torch_device)
        attention_mask = encoded["attention_mask"].to(torch_device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = output[0]
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts

        return pooled.to(device=torch_device, dtype=torch.float32)

    return embed


class _ClassifierOutput(NamedTuple):
    """`.logits`-bearing output, matching `crucible.latent.control
    ._extract_logits`'s accepted shape: it reads `output.logits` when
    present (this satisfies `hasattr(output, "logits")`) and would fall
    back to `output[0]` otherwise -- a `NamedTuple` satisfies both branches
    at once, so this is compatible however `control.py` chooses to read
    it."""

    logits: torch.Tensor


class _CodeExecutorClassifier(torch.nn.Module):
    """`microsoft/codeexecutor` (RoBERTa-arch, loaded via `AutoModel`) plus
    a fresh `Linear(hidden_size, 1)` head over the CLS/first-token hidden
    state. `forward(input_ids, attention_mask) -> _ClassifierOutput` with
    `.logits` of shape `(B, 1)` -- matches `crucible.latent.control
    .train_control`'s `model_factory` model contract exactly (see that
    function's own docstring): `_extract_logits` squeezes a trailing
    `(B, 1)` to `(B,)` for both this module's real wrapper and the test
    suite's tiny stubs.
    """

    def __init__(self, base_model: torch.nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.classifier = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _ClassifierOutput:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_hidden.to(self.classifier.weight.dtype))
        return _ClassifierOutput(logits=logits)


def codeexecutor_factory(device: str, revision: str = CODEEXECUTOR_REVISION) -> Callable[[], torch.nn.Module]:
    """Returns a zero-arg factory for `crucible.latent.control
    .train_control`'s `model_factory: Callable[[], torch.nn.Module]`
    parameter (see that function's docstring for the full model-call
    contract). Each call to the returned factory loads a FRESH
    `microsoft/codeexecutor` base (revision-pinned) plus a newly
    initialized `Linear` head -- `train_control` calls `model_factory()`
    exactly once per training run and then `.to(device)`s the result
    itself; this factory ALSO places the freshly loaded base model on
    `device` before wrapping it (mirroring `jina_code_embedder`'s `device`
    parameter), so `train_control`'s own `.to(device)` is a harmless no-op
    rather than the only placement.

    Unlike the frozen jina embedder, `microsoft/codeexecutor` is FINE-TUNED
    here (that is the whole point of the control arm), so it is loaded in
    plain fp32 -- no bf16 branching -- to avoid mixed-dtype gradient
    surprises with a plain `AdamW` optimizer.
    """
    def factory() -> torch.nn.Module:
        from transformers import AutoModel

        torch_device = torch.device(device)
        base = AutoModel.from_pretrained(CODEEXECUTOR_MODEL_ID, revision=revision)
        base.to(torch_device)
        hidden_size = base.config.hidden_size
        classifier = _CodeExecutorClassifier(base, hidden_size)
        classifier.to(torch_device)
        return classifier

    return factory


def codeexecutor_tokenizer(revision: str = CODEEXECUTOR_REVISION) -> Callable[[str], list[int]]:
    """Returns `tokenize(text: str) -> list[int]` matching
    `crucible.latent.control`'s tokenizer contract (its `_tokenize_truncated`
    usage): the FULL, UNTRUNCATED token-id sequence for `text` over
    `microsoft/codeexecutor`'s own vocabulary, special tokens included.
    Truncation to `config.CTRL_MAXLEN` happens in `control.py`'s own
    batching code, never here.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CODEEXECUTOR_MODEL_ID, revision=revision)

    def tokenize(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=True)

    return tokenize


def resolve_current_revision(repo_id: str) -> str:
    """Ops re-verification utility: queries the HF Hub for `repo_id`'s
    CURRENT commit sha, for confirming `JINA_REVISION`/`CODEEXECUTOR_REVISION`
    above have not silently drifted from what a fresh `model_info` call
    would return. NEVER called by any wrapper in this module -- the
    wrappers always pin to the hardcoded revision constants, never to this
    function's live result -- so importing this module or calling any
    factory above never reaches the network via this path.
    """
    from huggingface_hub import model_info

    return model_info(repo_id).sha
