"""Compose validated mutants into the seeded, content-hashed stream a run replays.

A *class* is the pair (unit, family) -- the unit of exposure the whole experiment turns
on. Each class contributes exactly two tasks: one in phase 1 (``first``) and its sibling
in phase 2 (``second``), so phase 2 asks the agent about a bug *kind* it has already met
in that same unit. Beside them phase 2 carries ``n_nov`` ``novel`` tasks drawn from units
deliberately **held out** of phase 1. Those are the control: without them a phase-2
improvement could be nothing more than "the agent has seen this file before".

Four things here are load-bearing.

*The two mutants of a class must touch different sites.* If ``second`` sat on the same
span as ``first``, "has the agent learned this bug kind" would collapse into "can the
agent replay the exact patch it just wrote". ``_pick_pair`` therefore walks past every
candidate that shares ``m1``'s span, and a family whose valid mutants all sit on one
site is not a class at all -- it is dropped as ``ineligible-class`` rather than quietly
paired up.

*The hold-out is drawn, not taken.* Units are sorted and then shuffled with the run's
seed before the split into novel and class units, so composition does not inherit
``build_units``' record order and the hold-out is not "the first n_nov by task id" --
which would correlate the control with whatever ordering EvalPlus happens to ship.

*All randomness is seeded and purpose-scoped.* Every draw comes from a
``random.Random(f"{seed}:<purpose>")``; nothing reads module-level random state. Phase 1
and phase 2 are shuffled from their own streams so that presentation order is
reproducible from the seed alone and independent between phases.

*``counts`` names every exclusion reason, zero included.* A reason that is merely absent
from the dict is indistinguishable from a reason nobody looked for, so the closed
vocabulary (``hidden-only``, ``equivalent``, ``infra``, ``syntax`` from Task 12, plus
``ineligible-class`` and ``unit-no-valid`` from here) is seeded to zero. ``dropped``
carries the complement: it names *which* unit or class was excluded, and why.

``stream_hash`` covers the knobs (seed, C, n_nov, rung), the unit sources that survived,
and every task's (key, phase, kind). Change any of them and the stream is a different
stream -- it is the id a run record points at, not a checksum of convenience.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

from .mutants import Mutant, Span
from .units import Unit, sha256_text
from .validate import Validation

Pair = tuple[Mutant, Validation]

EXCLUSION_REASONS: tuple[str, ...] = ("hidden-only", "equivalent", "infra", "syntax",
                                      "ineligible-class", "unit-no-valid")
"""Every way a mutant, a unit or a class can fail to reach the stream. All are counted."""


class NotEnoughClasses(RuntimeError):
    """The corpus cannot supply the requested stream -- raised instead of shipping a short one."""


def class_id(unit_id: str, family: str) -> str:
    """The id of the (unit, family) class. Stable and readable: ``"HumanEval/0|ARITH"``."""
    return f"{unit_id}|{family}"


@dataclass(frozen=True)
class TaskSpec:
    """One task as the stream presents it. Field order is frozen -- callers construct positionally.

    ``task_key`` **is** the mutant's key, carried through unchanged: the task is the bug,
    and the bug is identified by its content hash. ``kills_by_timeout`` and
    ``n_killing_visible`` are copied from the mutant's ``Validation`` so a task can be
    described without re-reading the validation record.
    """

    task_key: str
    unit_id: str
    family: str
    class_id: str
    phase: int
    kind: str
    span: Span
    kills_by_timeout: bool
    n_killing_visible: int

    def to_dict(self) -> dict:
        """JSON-ready form: the nested span tuples become lists so a file round-trip is exact."""
        d = asdict(self)
        d["span"] = [list(self.span[0]), list(self.span[1])]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        """Inverse of :meth:`to_dict`; restores the tuple shape so equality holds."""
        d = dict(d)
        d["span"] = (tuple(d["span"][0]), tuple(d["span"][1]))
        return cls(**d)


@dataclass(frozen=True)
class StreamManifest:
    """The whole stream: what to run, in what order, and the hash that identifies it.

    Field order is frozen -- Task 14's store and Task 16's driver construct it
    positionally. ``classes`` maps ``class_id -> (first_key, second_key)``; ``dropped``
    names every excluded unit (``unit_id``) or class (``class_id``) with its reason.
    """

    stream_hash: str
    seed: int
    C: int
    n_nov: int
    rung: str
    unit_ids: tuple[str, ...]
    tasks: tuple[TaskSpec, ...]
    classes: dict[str, tuple[str, str]]
    dropped: tuple[tuple[str, str], ...]
    counts: dict[str, int] = field(default_factory=dict)

    def phase(self, n: int) -> list[TaskSpec]:
        """The tasks of phase ``n``, in stream order."""
        return [t for t in self.tasks if t.phase == n]

    def to_dict(self) -> dict:
        """JSON-ready form: every tuple becomes a list so a file round-trip is exact."""
        return {"stream_hash": self.stream_hash, "seed": self.seed, "C": self.C, "n_nov": self.n_nov,
                "rung": self.rung, "unit_ids": list(self.unit_ids), "tasks": [t.to_dict() for t in self.tasks],
                "classes": {k: list(v) for k, v in self.classes.items()},
                "dropped": [list(x) for x in self.dropped], "counts": dict(self.counts)}

    @classmethod
    def from_dict(cls, d: dict) -> "StreamManifest":
        """Inverse of :meth:`to_dict`; restores every tuple shape so equality holds."""
        return cls(d["stream_hash"], d["seed"], d["C"], d["n_nov"], d["rung"], tuple(d["unit_ids"]),
                   tuple(TaskSpec.from_dict(t) for t in d["tasks"]),
                   {k: (v[0], v[1]) for k, v in d["classes"].items()},
                   tuple(tuple(x) for x in d["dropped"]), dict(d["counts"]))


def _task(m: Mutant, v: Validation, phase: int, kind: str) -> TaskSpec:
    """Package one validated mutant as a task. The mutant's key becomes the task key."""
    return TaskSpec(m.key, m.unit_id, m.family, class_id(m.unit_id, m.family), phase, kind, m.span,
                    v.kills_by_timeout, v.n_killing_visible)


def _bump(counts: dict[str, int], key: str) -> None:
    """Increment ``counts[key]``, creating it at zero first."""
    counts[key] = counts.get(key, 0) + 1


def _prefer_non_timeout(pairs: list[Pair], rng: random.Random) -> list[Pair]:
    """``pairs`` shuffled with ``rng``, then stably sorted so non-timeout kills come first.

    A mutant the visible suite only catches by hanging is a real task but a poor one --
    its signal is "wait five seconds", not "this assertion fails" -- so it is chosen last
    rather than excluded. The shuffle happens *before* the sort, so ties break randomly
    instead of by input order.
    """
    pool = list(pairs)
    rng.shuffle(pool)
    pool.sort(key=lambda mv: mv[1].kills_by_timeout)
    return pool


def _pick_pair(pairs: list[Pair], rng: random.Random) -> tuple[Pair, Pair] | None:
    """Two mutants of one family at **different** spans, or ``None`` if the family has none.

    The span check is the class's whole point -- see the module docstring. It is kept even
    though ``_build_classes`` only calls this for families that already show two distinct
    spans: the guard there saves an rng draw, this one keeps the invariant local to the
    function that would otherwise break it.
    """
    pool = _prefer_non_timeout(pairs, rng)
    m1 = pool[0]
    for cand in pool[1:]:
        if cand[0].span != m1[0].span:
            return m1, cand
    return None


def _partition_valid(units: list[Unit], validated: dict[str, list[Pair]],
                     ) -> tuple[dict[str, list[Pair]], list[tuple[str, str]], dict[str, int]]:
    """Split the validated mutants into "usable, by unit" and "excluded, counted by reason".

    A unit with no valid mutant at all is itself dropped and named in ``dropped``: a unit
    missing from the stream must be visible in the record, not inferred from a gap.
    """
    valid_by_unit: dict[str, list[Pair]] = {}
    dropped: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for u in units:
        keep = []
        for m, v in validated.get(u.unit_id, []):
            if v.valid:
                keep.append((m, v))
            else:
                _bump(counts, v.reason)
        if keep:
            valid_by_unit[u.unit_id] = keep
        else:
            dropped.append((u.unit_id, "unit-no-valid"))
            _bump(counts, "unit-no-valid")
    return valid_by_unit, dropped, counts


def _build_classes(valid_by_unit: dict[str, list[Pair]], class_units: list[str], C: int, rng: random.Random,
                   ) -> tuple[dict[str, tuple[str, str]], list[TaskSpec], list[TaskSpec], list[tuple[str, str]]]:
    """Take (unit, family) classes from ``class_units`` until ``C`` of them exist.

    Families are visited in sorted order within each unit so the walk is deterministic;
    families absent from a unit simply never appear (on the real corpus EXC yields no
    mutants at all, and that must not be an error). A family whose valid mutants all
    share one span cannot form a class and is dropped as ``ineligible-class``.
    """
    classes: dict[str, tuple[str, str]] = {}
    p1: list[TaskSpec] = []
    p2: list[TaskSpec] = []
    dropped: list[tuple[str, str]] = []
    for uid in class_units:
        by_fam: dict[str, list[Pair]] = {}
        for m, v in valid_by_unit[uid]:
            by_fam.setdefault(m.family, []).append((m, v))
        for fam in sorted(by_fam):
            if len(classes) >= C:
                break
            distinct_sites = len({m.span for m, _ in by_fam[fam]}) >= 2
            pair = _pick_pair(by_fam[fam], rng) if distinct_sites else None
            if pair is None:
                dropped.append((class_id(uid, fam), "ineligible-class"))
                continue
            (m1, v1), (m2, v2) = pair
            classes[class_id(uid, fam)] = (m1.key, m2.key)
            p1.append(_task(m1, v1, 1, "first"))
            p2.append(_task(m2, v2, 2, "second"))
    return classes, p1, p2, dropped


def _pick_novel(valid_by_unit: dict[str, list[Pair]], novel_units: list[str], rng: random.Random) -> list[TaskSpec]:
    """One task per held-out unit: a random valid mutant, non-timeout kills preferred."""
    return [_task(*_prefer_non_timeout(valid_by_unit[uid], rng)[0], 2, "novel") for uid in novel_units]


def _hash(seed: int, C: int, n_nov: int, rung: str, src_hashes: list[str], tasks: tuple[TaskSpec, ...]) -> str:
    """Content hash of the stream: its knobs, the unit sources it uses, and its task list.

    Every argument is part of the identity. The rung is in here because the same tasks
    run under a different budget rung are a different experiment, and the task list is
    ordered because presentation order is part of what the run measures.
    """
    return sha256_text(json.dumps({"seed": seed, "C": C, "n_nov": n_nov, "rung": rung, "units": src_hashes,
                                   "tasks": [(t.task_key, t.phase, t.kind) for t in tasks]}, sort_keys=True))


def compose(units: list[Unit], validated: dict[str, list[Pair]], *,
            seed: int, C: int, n_nov: int, rung: str = "base") -> StreamManifest:
    """Build the stream: ``C`` classes across two phases plus ``n_nov`` held-out novel tasks.

    Raises :class:`NotEnoughClasses` rather than returning a short stream -- a run with
    fewer classes than pre-registered is not the experiment that was registered.
    """
    rng = random.Random(f"{seed}:compose")
    valid_by_unit, dropped, counts = _partition_valid(units, validated)
    candidates = sorted(valid_by_unit)
    rng.shuffle(candidates)
    if len(candidates) < n_nov + 1:
        raise NotEnoughClasses(f"only {len(candidates)} units with valid mutants; need n_nov={n_nov} plus class units")
    novel_units, class_units = candidates[:n_nov], candidates[n_nov:]

    classes, p1, p2, class_dropped = _build_classes(valid_by_unit, class_units, C, rng)
    dropped = dropped + class_dropped
    for _cid, reason in class_dropped:
        _bump(counts, reason)
    for reason in EXCLUSION_REASONS:
        counts.setdefault(reason, 0)                    # None-vs-zero: unobserved is still named
    if len(classes) < C:
        raise NotEnoughClasses(f"eligible classes {len(classes)} < C={C}")
    p2 = p2 + _pick_novel(valid_by_unit, novel_units, rng)

    random.Random(f"{seed}:phase1").shuffle(p1)
    random.Random(f"{seed}:phase2").shuffle(p2)
    tasks = tuple(p1 + p2)
    unit_ids = tuple(sorted({t.unit_id for t in tasks}))
    src_hashes = sorted(u.src_hash for u in units if u.unit_id in unit_ids)
    counts.update({"eligible_classes": len(classes),
                   "valid_mutants": sum(len(v) for v in valid_by_unit.values())})
    return StreamManifest(_hash(seed, C, n_nov, rung, src_hashes, tasks), seed, C, n_nov, rung,
                          unit_ids, tasks, classes, tuple(dropped), counts)
