"""RED/GREEN tests for the B-lite training harness (prereg §5.2/§5.5).

No real corpus, no real jina encoder anywhere in this file: a 20-sample
synthetic corpus is written directly in Task 2/3's on-disk jsonl format
(16 functions hashed into "train", 4 into "val" -- never "test"), and
`code_embedder` is a seeded, deterministic-per-text random-projection stub
(a plain function of the text via `sha256`, no shared RNG state with the
training loop) standing in for the frozen jina encoder ops wires in
production.

Everything here runs on CPU with tiny dims, per the 4G-scope test command.

Mutation pins:

* `test_load_raises_on_test_split` -- the literal guard itself, called
  directly (grep-proof: no other call site of `load_split` exists in
  `train.py` for a mutant to route around this one).
* `test_train_blite_raises_on_nan_total_loss` -- monkeypatches `blite_loss`
  to hand back one NaN total loss; a mutant that swallows/skips the
  finiteness check would let training continue silently instead of raising.
* `test_train_blite_is_deterministic_given_same_seed` -- two full separate
  `train_blite` calls, same seed, same corpus, same embedder -> identical
  FIRST probe's val_auroc; a mutant that fails to reseed (or reseeds only
  one of torch/numpy/random) would very likely diverge here.
* `test_train_blite_early_stops_on_flat_val_auroc` -- pins the `>` (not
  `>=`) early-stop comparison: a flat val AUROC must still exhaust patience
  and stop, not be treated as "still improving" by tying the previous best.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from crucible.latent import corpus
from crucible.latent import train as train_module
from crucible.latent.train import train_blite

# -- synthetic corpus construction --------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _fn_id_for_split(split: str, seed: int, prefix: str) -> str:
    """Search synthetic fn_ids until one hashes into `split` -- exercises
    the real `assign_split` rather than hardcoding a hash value (same
    pattern as `tests/latent/test_corpus.py`)."""
    for i in range(10_000):
        candidate = f"{prefix}-{i}"
        if corpus.assign_split(candidate, seed) == split:
            return candidate
    raise AssertionError(f"no synthetic fn_id landed in split={split!r} within 10000 tries")


def _sample_row(i: int, fn_id: str) -> dict:
    """One synthetic (function, input) sample: alternating return/exception
    outcomes (a balanced binary target) and a couple of small, varied
    snapshots (real `encode_state_sequence` input shape)."""
    outcome = "return" if i % 2 == 0 else "exception:ValueError"
    return {
        "fn_id": fn_id,
        "function_src": f"def f_{i}(x):\n    y = x + {i}\n    return y\n",
        "args": f"({i % 3},)",
        "outcome": outcome,
        "return_repr": str(i) if outcome == "return" else None,
        "snapshots": [
            {"line": 1, "locals": [["x", "int", str(i % 3)]]},
            {"line": 2, "locals": [["x", "int", str(i % 3)], ["y", "int", str(i)]]},
        ],
    }


def _build_corpus(tmp_path: Path, *, n_train: int = 16, n_val: int = 4) -> Path:
    """A 20-sample corpus (1 sample per function): `n_train` functions
    hashed into "train", `n_val` into "val" -- none into "test" (this
    module must never be able to read it, so the fixture never bothers
    minting one)."""
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


def _make_code_embedder(d_model: int):
    """A deterministic function of each text (sha256 -> a local
    `torch.Generator` seed) -- stands in for the frozen jina encoder. Uses
    its OWN generator instance, never the global torch RNG, so it neither
    disturbs nor depends on `train_blite`'s own seeding."""

    def embed(texts: list[str]) -> torch.Tensor:
        rows = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()[:8]
            seed = int.from_bytes(digest, "big") % (2**31)
            gen = torch.Generator().manual_seed(seed)
            rows.append(torch.randn(d_model, generator=gen))
        return torch.stack(rows)

    return embed


TINY_OVERRIDES = {
    "D_MODEL": 8,
    "STATE_ENC_D": 8,
    "STATE_ENC_LAYERS": 1,
    "PRED_LAYERS": 1,
    "PRED_HEADS": 2,
    "LAMBDA_ISO": 0.1,
    "N_OUTCOME_CLASSES": 3,
    "LR": 1e-2,
    "BATCH": 4,
    "MAX_STEPS": 30,
    "EVAL_EVERY": 10,
    "PATIENCE": 5,
    "TRAIN_SEED": 0,
}


# -- the test-split literal guard ---------------------------------------------


def test_load_raises_on_test_split(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    with pytest.raises(ValueError):
        train_module._load(corpus_dir, "test")


def test_load_train_and_val_both_work(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    assert len(train_module._load(corpus_dir, "train")) == 16
    assert len(train_module._load(corpus_dir, "val")) == 4


# -- runs, learns, writes the required artifacts ------------------------------


def test_train_blite_runs_learns_and_writes_artifacts(tmp_path, monkeypatch):
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    embedder = _make_code_embedder(TINY_OVERRIDES["D_MODEL"])

    losses: list[float] = []
    orig_blite_loss = train_module.blite_loss

    def _spy_blite_loss(*args, **kwargs):
        total, parts = orig_blite_loss(*args, **kwargs)
        losses.append(parts["total"])
        return total, parts

    monkeypatch.setattr(train_module, "blite_loss", _spy_blite_loss)

    summary = train_blite(
        corpus_dir, out_dir, code_embedder=embedder, device="cpu",
        config_overrides=TINY_OVERRIDES,
    )

    # -- learns: loss at the last step < loss at the first step -------------
    assert len(losses) == TINY_OVERRIDES["MAX_STEPS"]
    assert losses[-1] < losses[0]

    # -- summary --------------------------------------------------------------
    assert summary["steps_run"] == TINY_OVERRIDES["MAX_STEPS"]
    assert summary["stopped_reason"] == "max_steps"  # PATIENCE=5 over only 3 evals
    assert summary["wall_s"] > 0
    assert summary["best_val_auroc"] is not None

    on_disk_summary = json.loads((out_dir / "train_summary.json").read_text())
    assert on_disk_summary == summary

    # -- probes.jsonl: >=1 line, every line carries the three probe keys ----
    probe_lines = [
        json.loads(line)
        for line in (out_dir / "probes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(probe_lines) >= 1
    for record in probe_lines:
        assert set(record) == {"step", "val_auroc", "latent_std_mean", "effective_rank"}
        assert isinstance(record["step"], int)
        assert isinstance(record["val_auroc"], float)
        assert isinstance(record["latent_std_mean"], float)
        assert isinstance(record["effective_rank"], float)

    # -- best.pt: state dict + config snapshot -------------------------------
    ckpt_path = out_dir / "best.pt"
    assert ckpt_path.exists()
    checkpoint = torch.load(ckpt_path, weights_only=False)
    assert "state_dict" in checkpoint
    assert "config" in checkpoint
    assert checkpoint["config"]["D_MODEL"] == TINY_OVERRIDES["D_MODEL"]


# -- NaN/inf loss is an infra failure, not a training result -------------------


def test_train_blite_raises_on_nan_total_loss(tmp_path, monkeypatch):
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    embedder = _make_code_embedder(TINY_OVERRIDES["D_MODEL"])

    orig_blite_loss = train_module.blite_loss
    call_count = {"n": 0}

    def _nan_once(*args, **kwargs):
        call_count["n"] += 1
        total, parts = orig_blite_loss(*args, **kwargs)
        if call_count["n"] == 1:
            total = torch.tensor(float("nan"))
        return total, parts

    monkeypatch.setattr(train_module, "blite_loss", _nan_once)

    with pytest.raises(RuntimeError):
        train_blite(
            corpus_dir, out_dir, code_embedder=embedder, device="cpu",
            config_overrides=TINY_OVERRIDES,
        )


# -- determinism: same seed, same corpus, same embedder -> identical first eval


def test_train_blite_is_deterministic_given_same_seed(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    embedder = _make_code_embedder(TINY_OVERRIDES["D_MODEL"])

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    train_blite(corpus_dir, out_a, code_embedder=embedder, device="cpu", config_overrides=TINY_OVERRIDES)
    train_blite(corpus_dir, out_b, code_embedder=embedder, device="cpu", config_overrides=TINY_OVERRIDES)

    lines_a = (out_a / "probes.jsonl").read_text().splitlines()
    lines_b = (out_b / "probes.jsonl").read_text().splitlines()
    assert lines_a and lines_b

    first_a = json.loads(lines_a[0])
    first_b = json.loads(lines_b[0])
    assert first_a["step"] == first_b["step"]
    assert first_a["val_auroc"] == first_b["val_auroc"]
    assert first_a["latent_std_mean"] == first_b["latent_std_mean"]
    assert first_a["effective_rank"] == first_b["effective_rank"]


# -- early stop: the strictly-greater comparison pin ---------------------------


def test_train_blite_early_stops_on_flat_val_auroc(tmp_path, monkeypatch):
    """A val AUROC that never improves (rigged flat via a monkeypatched
    `_evaluate`) must still exhaust PATIENCE and stop -- if the comparison
    were `>=` instead of `>`, a flat score would tie the "best" forever and
    training would run to MAX_STEPS instead."""
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    embedder = _make_code_embedder(TINY_OVERRIDES["D_MODEL"])

    def _flat_evaluate(*args, **kwargs):
        return {"val_auroc": 0.5, "latent_std_mean": 0.0, "effective_rank": 1.0}

    monkeypatch.setattr(train_module, "_evaluate", _flat_evaluate)

    overrides = dict(TINY_OVERRIDES)
    overrides.update({"MAX_STEPS": 100, "EVAL_EVERY": 5, "PATIENCE": 2})

    summary = train_blite(
        corpus_dir, out_dir, code_embedder=embedder, device="cpu",
        config_overrides=overrides,
    )

    # eval @5: 0.5 > -inf -> improves, patience resets.
    # eval @10: 0.5 not > 0.5 -> 1 non-improving eval.
    # eval @15: 0.5 not > 0.5 -> 2 non-improving evals == PATIENCE -> stop.
    assert summary["stopped_reason"] == "early_stop"
    assert summary["steps_run"] == 15


# -- config_overrides validation ------------------------------------------------


def test_train_blite_rejects_unknown_config_override_key(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    embedder = _make_code_embedder(8)

    with pytest.raises(ValueError):
        train_blite(
            corpus_dir, out_dir, code_embedder=embedder, device="cpu",
            config_overrides={"NOT_A_REAL_KEY": 1},
        )


# -- bf16-by-device-TYPE pin (round-1 review finding) --------------------------


def test_use_bf16_decides_by_device_type_not_string_equality():
    """`device="cuda:0"` is cuda just as much as bare `"cuda"` is -- a
    string-compare (`device == "cuda"`) mutant would silently fall back to
    fp32 for it, violating "bf16 iff cuda" (prereg §5.2). CPU-safe: only
    exercises `torch.device(...).type`, never touches an actual CUDA
    context, so it runs the same whether or not a GPU is present."""
    assert train_module._use_bf16("cuda")
    assert train_module._use_bf16("cuda:0")
    assert not train_module._use_bf16("cpu")


# -- zero-eval fallback checkpoint (round-1 review finding) --------------------


def test_train_blite_saves_fallback_checkpoint_when_max_steps_below_eval_every(tmp_path):
    """MAX_STEPS < EVAL_EVERY -> the loop never hits an eval boundary at
    all, so best.pt would never be written by the improve-on-eval path.
    train_blite must still guarantee best.pt exists (the module's own
    contract) via the end-of-run fallback save."""
    corpus_dir = _build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    embedder = _make_code_embedder(TINY_OVERRIDES["D_MODEL"])

    overrides = dict(TINY_OVERRIDES)
    overrides.update({"MAX_STEPS": 3, "EVAL_EVERY": 100})

    summary = train_blite(
        corpus_dir, out_dir, code_embedder=embedder, device="cpu",
        config_overrides=overrides,
    )

    assert summary["steps_run"] == 3
    assert summary["stopped_reason"] == "max_steps"
    assert summary["best_val_auroc"] is None  # no eval ever ran

    assert (out_dir / "best.pt").exists()
    on_disk_summary = json.loads((out_dir / "train_summary.json").read_text())
    assert on_disk_summary == summary

    # probes.jsonl exists (created fresh at the start of the run) but is
    # empty -- no EVAL_EVERY boundary was ever crossed.
    assert (out_dir / "probes.jsonl").read_text() == ""
