"""Wire the stream stages into one build, and re-run a written stream's visible suites.

*One deterministic function of ``(cfg, recs)``.* ``build_stream`` runs build_units ->
enumerate/sample/make_mutant -> validate_many -> compose -> write, and every source of
randomness on the path is a seeded ``random.Random`` keyed off ``cfg.seed`` (the spec
subsample, the per-unit spec sample, and compose's own splits). So two builds with the
same config and inputs land on the same ``stream_hash`` -- the identity the store names
the directory for. ``limit_units`` samples ``random.Random(f"{seed}:units")`` rather than
taking the first N, or the corpus order would leak into which units were measured.

*make_mutant is single-threaded on purpose.* Its ``warnings.catch_warnings()`` is
process-global (a SyntaxWarning promoted to error in one thread would corrupt another), so
the list comprehension over ``specs`` stays serial; only ``validate_many`` -- whose worker
touches no global warning state -- uses the pool. Nothing here promotes any warning to an
error (R-T12-1): ``validate_mutant``'s plain ``compile()`` and ``make_mutant``'s guard both
depend on SyntaxWarning staying a warning.

*build_stream logs the pre-check but does not gate on it.* The statistical bands
(killing-count, timeout-rate) are meaningless at the handful-of-classes scale the tests and
the smoke build run at, so refusing to write a small stream would break legitimate small
builds. The structural gates that must hold at any size are logged, and the CLI ``precheck``
command -- which the Phase-A run must pass through before it replays a stream -- is the
real gate. A build is never *silently* passed: the pre-check outcome and any failing check
names are in the build log.

*The write is atomic against a crash.* ``store`` never deletes and a half-written directory
looks complete, so a fresh stream is staged in a sibling ``.staging-*`` dir under the same
root and ``os.replace``\\d into place -- an atomic rename on one filesystem, so a reader
sees the stream whole or not at all. A directory that already exists is content-addressed
(its name *is* the hash), so it is overwritten in place via the store's idempotent path
rather than re-staged. ``store``'s interface is untouched; the atomicity lives here.

*The rung is a dispatch, not a label.* ``rung`` already rode through to the manifest and
into ``stream_hash``; at ``stack2`` it also decides what a *task* is. The singles are built
and validated identically at both rungs -- they are what the stacker composes from, and
their verdicts are the provenance R-S25-1's component half is checked against -- but at
``stack2`` only the two-site mutants become tasks. So the singles' mutants and validations
are stored at every rung and ``validated`` (what compose sees) holds the stacked pairs
alone -- so at ``stack2`` the composition census in ``counts`` is over the stacked
candidates only, and the singles' verdicts sit on disk without being census-counted.
An unknown rung is refused before any work rather than falling through to base:
a mis-spelled rung that silently built a rung-0 stream would be a mis-labelled experiment,
and the stream hash would not give it away (the label is what got hashed).
"""
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..sandbox.runner import run_tests
from . import evalplus, stack, store
from .build import build_units
from .compose import Pair, compose
from .families import all_operator_names
from .mutants import Mutant, enumerate_specs, make_mutant, sample_specs
from .precheck import precheck
from .units import Unit
from .validate import validate_many

ALLOWED_RUNGS: tuple[str, ...] = ("base", "stack2")
"""The rungs ``build_stream`` knows how to build. ``base`` is rung 0 (one site per task),
``stack2`` is rung 1 (two sites per task). Closed on purpose -- see the module docstring."""


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 0
    C: int = 200
    n_nov: int = 50
    per_family: int = 6
    max_hidden: int = 100
    limit_units: int | None = None
    jobs: int = 8
    rung: str = "base"
    # Not last, by design: it sits here so ``sources`` stays last as the only non-scalar.
    # Safe because no positional caller reaches past ``rung`` (every call site keywords
    # from here on), so no existing positional argument changes meaning.
    pairs_per_family: int = 4
    sources: tuple[str, ...] = ("humaneval", "mbpp")


def _load_recs(cfg: BuildConfig) -> list[dict]:
    recs: list[dict] = []
    for s in cfg.sources:
        recs += evalplus.load(s)
    return recs


def _limit(recs: list[dict], cfg: BuildConfig) -> list[dict]:
    if cfg.limit_units is None or cfg.limit_units >= len(recs):
        return recs
    rng = random.Random(f"{cfg.seed}:units")
    idx = sorted(rng.sample(range(len(recs)), cfg.limit_units))
    return [recs[i] for i in idx]


def _write_atomic(out_root: Path, manifest, units, mutants, validations, dropped) -> Path:
    """Stage a fresh stream and ``os.replace`` it into place; overwrite an existing one.

    ``build_dropped.jsonl`` (the named build-time unit drops) is written INTO the staged
    dir before the rename, so it lands atomically with the rest of the stream -- and into
    the final dir on the content-addressed early-return path, so both paths carry it.
    """
    d_final = store.stream_dir(out_root, manifest)
    if d_final.exists():                                     # content-addressed: same bytes already here
        d = store.write_stream(out_root, manifest, units, mutants, validations)
        store.write_build_dropped(d, dropped)
        return d
    staging = Path(out_root) / f".staging-{os.getpid()}-{manifest.stream_hash[:12]}"
    try:
        produced = store.write_stream(staging, manifest, units, mutants, validations)
        store.write_build_dropped(produced, dropped)         # into the staged dir, before the rename
        os.replace(produced, d_final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return d_final


def _stacked_tasks(u: Unit, singles: list[Pair], cfg: BuildConfig) -> tuple[list[Pair], int]:
    """One unit's rung-1 tasks -- validated two-site mutants -- and its ``stack-apply`` drops.

    Only *valid* singles are offered to the stacker. That is R-S25-1's component half: each
    of a stacked task's two components is independently a bug the visible suite catches, so
    the composite is a two-bug task rather than a bug plus a no-op.

    One ``random.Random`` serves every family of the unit and is passed through them in
    sorted family order, so its state advances across families and the whole unit's pairing
    is a deterministic function of ``(seed, unit_id)`` -- not of dict insertion order.

    The returned int is every ``compose_pair`` that came back ``None``: apply failures and
    the shared guards alike. That *is* the definition of the ``stack-apply`` census key --
    "pairs that failed to become a stacked mutant" -- and the caller hands the total to
    compose as ``extra_counts`` so the manifest reports it.
    """
    rng = random.Random(f"{cfg.seed}:{u.unit_id}:stack")
    by_fam: dict[str, list[Pair]] = {}
    for m, v in singles:
        if v.valid:
            by_fam.setdefault(m.family, []).append((m, v))
    stacked: list[Mutant] = []
    n_drop = 0
    for fam in sorted(by_fam):
        st, d = stack.stack_unit(u, by_fam[fam], rng=rng, max_pairs=cfg.pairs_per_family)
        stacked += st
        n_drop += d
    return list(zip(stacked, validate_many(u, stacked, jobs=cfg.jobs))), n_drop


def build_stream(cfg: BuildConfig, out_root: Path, *, recs: list[dict] | None = None, log=print) -> Path:
    if cfg.rung not in ALLOWED_RUNGS:
        raise ValueError(f"unknown rung {cfg.rung!r}; known: {list(ALLOWED_RUNGS)}")
    recs = _limit(recs if recs is not None else _load_recs(cfg), cfg)
    units, dropped = build_units(recs, seed=cfg.seed, max_hidden=cfg.max_hidden, jobs=cfg.jobs)
    log(f"units built={len(units)} dropped={len(dropped)}")
    ops = all_operator_names()
    validated, mutants, validations = {}, {}, []
    stack_dropped = 0
    for u in units:
        rng = random.Random(f"{cfg.seed}:{u.unit_id}:specs")
        specs = sample_specs(enumerate_specs(u.module_src, ops), per_family=cfg.per_family, rng=rng)
        ms = [m for m in (make_mutant(u, s) for s in specs) if m is not None]
        vs = validate_many(u, ms, jobs=cfg.jobs)
        mutants.update({m.key: m for m in ms}); validations += vs   # singles: stored at EVERY rung
        tasks, extra = list(zip(ms, vs)), ""
        if cfg.rung == "stack2":
            tasks, n_drop = _stacked_tasks(u, tasks, cfg)                   # only two-site mutants are tasks
            mutants.update({m.key: m for m, _ in tasks}); validations += [v for _, v in tasks]
            stack_dropped += n_drop
            extra = f" stacked={len(tasks)} stack-apply={n_drop}"
        validated[u.unit_id] = tasks
        log(f"{u.unit_id}: specs={len(specs)} mutants={len(ms)} valid={sum(v.valid for v in vs)}{extra}")
    manifest = compose(units, validated, seed=cfg.seed, C=cfg.C, n_nov=cfg.n_nov, rung=cfg.rung,
                       extra_counts={"stack-apply": stack_dropped} if cfg.rung == "stack2" else None)
    d = _write_atomic(out_root, manifest, units, mutants, validations, dropped)
    rep = precheck(manifest, {u.unit_id: u for u in units})
    log(f"precheck ok={rep.ok} failing={[c.name for c in rep.checks if not c.passed]}")
    log(f"stream {manifest.stream_hash[:12]} tasks={len(manifest.tasks)} counts={manifest.counts} -> {d}")
    return d


def smoke(stream_d: Path, n: int = 30, *, log=print) -> dict:
    man = store.read_manifest(stream_d)
    res = {"ran": 0, "killed": 0, "not_killed": 0, "infra": 0}
    for t in man.tasks[:n]:
        u, m = store.read_unit(stream_d, t.unit_id), store.read_mutant(stream_d, t.task_key)
        r = run_tests(u.module_name, m.mutated_src, u.visible_test_src)
        res["ran"] += 1
        key = "infra" if r.infra_error else ("killed" if r.killed else "not_killed")
        res[key] += 1
        log(f"{t.task_key[:10]} {t.unit_id} {t.family} phase={t.phase} -> {key}")
    return res
