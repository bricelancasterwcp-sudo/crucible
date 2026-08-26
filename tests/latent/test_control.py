"""RED/GREEN tests for the token-space CONTROL arm's fine-tune harness
(prereg §3).

No real corpus, no real microsoft/codeexecutor anywhere in this file: a
20-sample synthetic corpus is written directly in Task 2/3's on-disk jsonl
format (16 functions hashed into "train", 4 into "val" -- never "test"),
`tokenizer` is a deterministic per-word sha256-hash tokenizer stub (never
the real HuggingFace tokenizer), and `model_factory` builds a tiny 2-layer
(embedding + linear) stub model standing in for
microsoft/codeexecutor-plus-head. Every synthetic sample's `args` field
carries a literal "POS"/"NEG" marker word tied 1:1 to its binary outcome --
a task a 2-layer bag-of-tokens model can separate almost immediately, which
is what makes "val AUROC > 0.9 within 2 epochs" a meaningful proof that the
loop actually trains and evaluates, not a coincidence of random init.

Everything here runs on CPU with tiny dims, per the 4G-scope test command.

`train_control` has no `config_overrides` parameter (see its own
docstring for why -- it never constructs the model, so there is no
model-dimension config to override); tests that need non-default
hyperparameters use `monkeypatch.setattr(control_module.config, ...)`
instead, reading live off the config module exactly as production code
does.

Mutation pins:

* `test_load_raises_on_test_split` -- the literal guard itself, called
  directly. `control_module._load` is `crucible.latent.train._load`,
  imported (not reimplemented), so this also pins that the control arm
  reuses the exact same guard function object as the treatment arm.
* `test_train_control_raises_on_nan_loss` -- a stub model whose first
  forward call returns a NaN logit (via a tuple-shaped output, exercising
  `_extract_logits`'s non-`.logits`-attribute branch); a mutant that
  swallows/skips the finiteness check would let training continue silently
  instead of raising.
* `test_train_control_early_stops_before_max_epochs` -- pins the `>` (not
  `>=`) early-stop comparison, exactly as train.py's own early-stop test
  does: a flat val AUROC must still exhaust PATIENCE and stop, well before
  CTRL_MAX_EPOCHS.
* `test_tokenize_truncated_pins_ctrl_maxlen` -- a tokenizer stub that
  returns far more ids than CTRL_MAXLEN; pins that truncation happens in
  the loader (`_tokenize_truncated`), to exactly CTRL_MAXLEN tokens, never
  more.
* `test_train_control_is_deterministic_given_same_seed` -- two full
  separate `train_control` calls, same seed, same corpus, same tokenizer
  and model_factory (a *fresh* model each call, both freshly seeded) ->
  identical first probe. A mutant that fails to reseed (or reseeds only
  one of torch/numpy/random) would very likely diverge here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from crucible.latent import config
from crucible.latent import control as control_module
from crucible.latent import corpus
from crucible.latent.control import render_control_input, score_split_control, train_control

# -- synthetic corpus construction --------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _fn_id_for_split(split: str, seed: int, prefix: str) -> str:
    """Search synthetic fn_ids until one hashes into `split` -- exercises
    the real `assign_split` rather than hardcoding a hash value (same
    pattern as tests/latent/test_train.py)."""
    for i in range(10_000):
        candidate = f"{prefix}-{i}"
        if corpus.assign_split(candidate, seed) == split:
            return candidate
    raise AssertionError(f"no synthetic fn_id landed in split={split!r} within 10000 tries")


def _sample_row(i: int, fn_id: str) -> dict:
    """One synthetic (function, input) sample: alternating return/exception
    outcomes (a balanced binary target) whose `args` field is a literal
    "POS"/"NEG" marker word 1:1 with the outcome -- a `render_control_input`
    output the hash-tokenizer stub turns into one distinct, label-correlated
    token, making the task linearly separable for a tiny embedding+linear
    stub model. `snapshots` is deliberately empty: the control arm never
    reads execution-state snapshots (that is B-lite's exclusive input) --
    only `function_src` + `args` (prereg §3)."""
    outcome = "return" if i % 2 == 0 else "exception:ValueError"
    marker = "POS" if outcome == "return" else "NEG"
    return {
        "fn_id": fn_id,
        "function_src": f"def f_{i}(x):\n    return x\n",
        "args": marker,
        "outcome": outcome,
        "return_repr": str(i) if outcome == "return" else None,
        "snapshots": [],
    }


def _build_corpus(tmp_path: Path, *, n_train: int = 16, n_val: int = 4) -> Path:
    """A 20-sample corpus (1 sample per function): `n_train` functions
    hashed into "train", `n_val` into "val" -- none into "test"."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    functions: list[dict] = []
    samples: list[dict] = []
    i = 0
    for _ in range(n_train):
        fn_id = _fn_id_for_split("train", corpus.SPLIT_SEED, prefix=f"tr{i}")
        functions.append({"fn_id": fn_id, "function_src": f"def f_{i}(): pass\n"})
        samples.append(_sample_row(i, fn_id))
        i += 1
    for _ in range(n_val):
        fn_id = _fn_id_for_split("val", corpus.SPLIT_SEED, prefix=f"va{i}")
        functions.append({"fn_id": fn_id, "function_src": f"def f_{i}(): pass\n"})
        samples.append(_sample_row(i, fn_id))
        i += 1

    _write_jsonl(corpus_dir / "functions.jsonl", functions)
    _write_jsonl(corpus_dir / "samples.jsonl", samples)
    (corpus_dir / "gen_stats.json").write_text(json.dumps({
        "target_functions": len(functions), "seed": 0, "candidates": len(functions),
        "parse_fail": 0, "validate_fail": {}, "nondet_rejected": 0, "truncated_rejected": 0,
        "balance_rejected": 0, "harvest_error": 0, "accepted_functions": len(functions),
        "accepted_samples": len(samples), "complete": True,
    }))
    return corpus_dir


# -- tokenizer + model stubs ---------------------------------------------------


def _make_hash_tokenizer(vocab_size: int):
    """Deterministic per-word tokenizer stub: each whitespace-split word ->
    `sha256(word)` mod `vocab_size`. Not a real subword tokenizer -- stands
    in for the `(text: str) -> list[int]` interface `train_control`'s own
    docstring defines. Uses `hashlib` (not the salted builtin `hash()`) so
    it is reproducible run over run, not merely within one process."""

    def tokenize(text: str) -> list[int]:
        ids = []
        for word in text.split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()[:8]
            ids.append(int.from_bytes(digest, "big") % vocab_size)
        return ids

    return tokenize


class _StubOutput:
    """A `.logits`-bearing object -- the HuggingFace `ModelOutput`-style
    branch of the model contract."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class TinyStubModel(nn.Module):
    """2-layer (embedding + linear) stub satisfying `train_control`'s model
    contract: `forward(input_ids, attention_mask) -> _StubOutput(logits)`,
    `logits` a masked-mean-pooled bag-of-tokens score, `(B,)`."""

    def __init__(self, vocab_size: int, d: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.linear = nn.Linear(d, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _StubOutput:
        mask = attention_mask.unsqueeze(-1).to(self.embed.weight.dtype)  # (B, L, 1)
        x = self.embed(input_ids)  # (B, L, d)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, d)
        logits = self.linear(pooled).squeeze(-1)  # (B,)
        return _StubOutput(logits)


VOCAB_SIZE = 64


# -- render_control_input: determinism + exact format --------------------------


def test_render_control_input_format_and_determinism():
    src = "def f(a, b):\n    return a + b\n"
    args = "(1, 2)"
    result = render_control_input(src, args)
    assert result == src + "\nINPUT: " + args
    assert render_control_input(src, args) == result  # pure function, same inputs -> same output


def test_render_control_input_does_not_truncate():
    """render_control_input is pure string concatenation -- truncation is
    entirely the loader's job (see test_tokenize_truncated_pins_ctrl_maxlen),
    never this function's."""
    src = "def f():\n    return 1\n" * 200
    result = render_control_input(src, "()")
    assert result == src + "\nINPUT: ()"


# -- tokenization truncation pin -----------------------------------------------


def test_tokenize_truncated_pins_ctrl_maxlen():
    long_tokenizer = lambda text: list(range(10_000))  # noqa: E731 -- far past CTRL_MAXLEN
    ids = control_module._tokenize_truncated(long_tokenizer, "irrelevant text", config.CTRL_MAXLEN)
    assert len(ids) == config.CTRL_MAXLEN


def test_tokenize_truncated_passes_short_input_through_unchanged():
    short_tokenizer = lambda text: [1, 2, 3]  # noqa: E731
    ids = control_module._tokenize_truncated(short_tokenizer, "hi", max_len=10)
    assert ids == [1, 2, 3]


# -- extract_logits: both output shapes -----------------------------------------


def test_extract_logits_handles_dot_logits_attribute():
    out = _StubOutput(torch.tensor([1.0, 2.0]))
    result = control_module._extract_logits(out)
    assert torch.equal(result, torch.tensor([1.0, 2.0]))


def test_extract_logits_handles_tuple_output_and_squeezes_trailing_dim():
    out = (torch.tensor([[1.0], [2.0]]),)
    result = control_module._extract_logits(out)
    assert result.shape == (2,)
    assert torch.equal(result, torch.tensor([1.0, 2.0]))


# -- the test-split literal guard, reused from train.py -------------------------


def test_load_raises_on_test_split(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    with pytest.raises(ValueError):
        control_module._load(corpus_dir, "test")


def test_load_is_train_pys_load_function_object():
    """Not merely equivalent behavior -- the SAME function object, so the
    guard can never drift between the two arms."""
    from crucible.latent.train import _load as train_load
    assert control_module._load is train_load


def test_load_train_and_val_both_work(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    assert len(control_module._load(corpus_dir, "train")) == 16
    assert len(control_module._load(corpus_dir, "val")) == 4


# -- runs, learns, writes the required artifacts ---------------------------------


def test_train_control_learns_separable_task_and_writes_artifacts(tmp_path, monkeypatch):
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    tokenizer = _make_hash_tokenizer(VOCAB_SIZE)
    model_factory = lambda: TinyStubModel(VOCAB_SIZE)  # noqa: E731

    # Real CTRL_LR (2e-5) is tuned for fine-tuning a large pretrained model,
    # not for a randomly-initialized tiny stub in two epochs -- monkeypatch
    # a larger LR and a smaller BATCH (more optimizer steps per epoch) for
    # this test, exactly as test_train.py's TINY_OVERRIDES uses a larger LR
    # than B-lite's own real config.LR.
    monkeypatch.setattr(control_module.config, "CTRL_MAX_EPOCHS", 2)
    monkeypatch.setattr(control_module.config, "CTRL_LR", 0.2)
    monkeypatch.setattr(control_module.config, "BATCH", 4)

    summary = train_control(
        corpus_dir, out_dir, model_factory=model_factory, tokenizer=tokenizer, device="cpu",
    )

    assert summary["epochs_run"] == 2
    assert summary["stopped_reason"] == "max_epochs"  # PATIENCE=5 default, only 2 evals ran
    assert summary["wall_s"] > 0
    assert summary["best_val_auroc"] is not None
    assert summary["best_val_auroc"] > 0.9  # proves the loop actually trains + evaluates

    on_disk_summary = json.loads((out_dir / "train_summary.json").read_text())
    assert on_disk_summary == summary

    # -- probes.jsonl: exactly 2 lines (one per epoch), {epoch, val_auroc} only
    probe_lines = [
        json.loads(line)
        for line in (out_dir / "probes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(probe_lines) == 2
    for n, record in enumerate(probe_lines, start=1):
        assert set(record) == {"epoch", "val_auroc"}
        assert record["epoch"] == n
        assert isinstance(record["val_auroc"], float)
    assert probe_lines[-1]["val_auroc"] > 0.9

    # -- best.pt: state dict + config snapshot -------------------------------
    ckpt_path = out_dir / "best.pt"
    assert ckpt_path.exists()
    checkpoint = torch.load(ckpt_path, weights_only=False)
    assert "state_dict" in checkpoint
    assert "config" in checkpoint
    assert checkpoint["config"]["CTRL_LR"] == 0.2


# -- NaN/inf loss is an infra failure, not a training result ---------------------


def test_train_control_raises_on_nan_loss(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    tokenizer = _make_hash_tokenizer(VOCAB_SIZE)

    class NanOnceModel(nn.Module):
        """First forward call returns a NaN logit via a TUPLE-shaped output
        (the non-`.logits` branch of `_extract_logits`); later calls (if
        ever reached, which they must not be) are finite."""

        def __init__(self, vocab_size: int, d: int = 16) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d)
            self.linear = nn.Linear(d, 1)
            self.calls = 0

        def forward(self, input_ids, attention_mask):
            self.calls += 1
            mask = attention_mask.unsqueeze(-1).to(self.embed.weight.dtype)
            x = self.embed(input_ids)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            logits = self.linear(pooled).squeeze(-1)
            if self.calls == 1:
                logits = logits * float("nan")
            return (logits,)

    model_factory = lambda: NanOnceModel(VOCAB_SIZE)  # noqa: E731

    with pytest.raises(RuntimeError):
        train_control(
            corpus_dir, out_dir, model_factory=model_factory, tokenizer=tokenizer, device="cpu",
        )


# -- early stop: the strictly-greater comparison pin ------------------------------


def test_train_control_early_stops_before_max_epochs(tmp_path, monkeypatch):
    """A val AUROC that never improves (rigged flat via a monkeypatched
    `_evaluate`) must still exhaust PATIENCE and stop, well before
    CTRL_MAX_EPOCHS -- if the comparison were `>=` instead of `>`, a flat
    score would tie the "best" forever and training would run to
    CTRL_MAX_EPOCHS instead."""
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    tokenizer = _make_hash_tokenizer(VOCAB_SIZE)
    model_factory = lambda: TinyStubModel(VOCAB_SIZE)  # noqa: E731

    def _flat_evaluate(*args, **kwargs):
        return {"val_auroc": 0.5}

    monkeypatch.setattr(control_module, "_evaluate", _flat_evaluate)
    monkeypatch.setattr(control_module.config, "CTRL_MAX_EPOCHS", 100)
    monkeypatch.setattr(control_module.config, "PATIENCE", 2)

    summary = train_control(
        corpus_dir, out_dir, model_factory=model_factory, tokenizer=tokenizer, device="cpu",
    )

    # epoch 1: 0.5 > -inf -> improves, patience resets to 0.
    # epoch 2: 0.5 not > 0.5 -> 1 non-improving epoch.
    # epoch 3: 0.5 not > 0.5 -> 2 non-improving epochs == PATIENCE -> stop.
    assert summary["stopped_reason"] == "early_stop"
    assert summary["epochs_run"] == 3


# -- determinism: same seed, same corpus, same tokenizer -> identical first eval -


def test_train_control_is_deterministic_given_same_seed(tmp_path, monkeypatch):
    corpus_dir = _build_corpus(tmp_path)
    tokenizer = _make_hash_tokenizer(VOCAB_SIZE)

    monkeypatch.setattr(control_module.config, "CTRL_MAX_EPOCHS", 2)
    monkeypatch.setattr(control_module.config, "CTRL_LR", 0.05)
    monkeypatch.setattr(control_module.config, "BATCH", 4)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    train_control(
        corpus_dir, out_a, model_factory=lambda: TinyStubModel(VOCAB_SIZE),
        tokenizer=tokenizer, device="cpu",
    )
    train_control(
        corpus_dir, out_b, model_factory=lambda: TinyStubModel(VOCAB_SIZE),
        tokenizer=tokenizer, device="cpu",
    )

    lines_a = (out_a / "probes.jsonl").read_text().splitlines()
    lines_b = (out_b / "probes.jsonl").read_text().splitlines()
    assert lines_a and lines_b

    first_a = json.loads(lines_a[0])
    first_b = json.loads(lines_b[0])
    assert first_a["epoch"] == first_b["epoch"]
    assert first_a["val_auroc"] == first_b["val_auroc"]


# -- score_split_control: the control-arm mirror of train.score_split ---------


def test_score_split_control_writes_probabilities_for_every_sample_in_split(tmp_path, monkeypatch):
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    tokenizer = _make_hash_tokenizer(VOCAB_SIZE)
    model_factory = lambda: TinyStubModel(VOCAB_SIZE)  # noqa: E731

    monkeypatch.setattr(control_module.config, "CTRL_MAX_EPOCHS", 2)
    monkeypatch.setattr(control_module.config, "CTRL_LR", 0.2)
    monkeypatch.setattr(control_module.config, "BATCH", 4)

    train_control(
        corpus_dir, out_dir, model_factory=model_factory, tokenizer=tokenizer, device="cpu",
    )

    scores_path = tmp_path / "val_scores.json"
    score_split_control(
        model_factory, tokenizer, out_dir / "best.pt", corpus_dir, "val", scores_path,
        device="cpu",
    )

    assert scores_path.exists()
    scores = json.loads(scores_path.read_text())

    val_samples = control_module._load(corpus_dir, "val")
    expected_keys = {f"{s.fn_id}:{s.args}" for s in val_samples}
    assert set(scores) == expected_keys
    assert len(scores) == len(val_samples)
    for prob in scores.values():
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


def test_score_split_control_refuses_test_split_without_allow_test(tmp_path, monkeypatch):
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    tokenizer = _make_hash_tokenizer(VOCAB_SIZE)
    model_factory = lambda: TinyStubModel(VOCAB_SIZE)  # noqa: E731

    monkeypatch.setattr(control_module.config, "CTRL_MAX_EPOCHS", 2)
    monkeypatch.setattr(control_module.config, "CTRL_LR", 0.2)
    monkeypatch.setattr(control_module.config, "BATCH", 4)

    train_control(
        corpus_dir, out_dir, model_factory=model_factory, tokenizer=tokenizer, device="cpu",
    )

    with pytest.raises(ValueError):
        score_split_control(
            model_factory, tokenizer, out_dir / "best.pt", corpus_dir, "test",
            tmp_path / "test_scores.json", device="cpu",
        )
    assert not (tmp_path / "test_scores.json").exists()
