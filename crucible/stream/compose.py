"""Compose validated mutants into the seeded, content-hashed stream a run replays.

A *class* is the pair (unit, family) -- the unit of exposure the whole experiment turns
on. Each class contributes exactly two tasks: one in phase 1 (``first``) and its sibling
in phase 2 (``second``), so phase 2 asks the agent about a bug *kind* it has already met
in that same unit. Beside them phase 2 carries ``n_nov`` ``novel`` tasks drawn from units
deliberately **held out** of phase 1. Those are the control: without them a phase-2
improvement could be nothing more than "the agent has seen this file before".

Five things here are load-bearing.

*The two mutants of a class must touch different sites.* If ``second`` sat on the same
span as ``first``, "has the agent learned this bug kind" would collapse into "can the
agent replay the exact patch it just wrote". ``_pick_pair`` therefore walks past every
candidate that shares a site with ``m1``, and a family whose valid mutants all sit on one
site is not a class at all -- it is named in ``dropped`` as ``ineligible-class`` rather
than quietly paired up. The comparison is between *site sets* (``_site_set``), not spans:
a rung-1 mutant touches two sites and sharing **either** of them is sharing a site
(spec §4.8.3). At rung 0 a site set is the mutant's own single span, so the rule -- and
every draw it feeds -- is unchanged.

*Which two mutants form a class is a preference; which of them goes first is a coin
flip.* ``_prefer_non_timeout`` picks the pair from the front of a non-timeout-first
ordering, because a mutant the visible suite catches only by hanging is a poor task. But
if that same ordering also decided phase, every timeout mutant in a pair would land in
phase 2 by construction -- measured at a first-vs-second timeout rate of 0.000 vs 1.000
-- and spec §4.8.1(1c) requires those two rates to sit within 2·SE of each other (Task
15's ``timeout-rate-band`` pre-check enforces it). So after the pair is chosen, a seeded
coin decides which member is ``first``. The preference survives; the systematic
confound does not.

*The hold-out is drawn, not taken.* Units are sorted and then shuffled with the run's
seed before the split into novel and class units, so composition does not inherit
``build_units``' record order and the hold-out is not "the first n_nov by task id" --
which would correlate the control with whatever ordering EvalPlus happens to ship.

*All randomness is seeded and purpose-scoped.* Every draw comes from a
``random.Random(f"{seed}:<purpose>")``; nothing reads module-level random state. Phase 1
and phase 2 are shuffled from their own streams so that presentation order is
reproducible from the seed alone and independent between phases.

*``counts`` is a census, and ``dropped`` names every unit and class the census excludes.*
A reason that is merely absent from the dict is indistinguishable from a reason nobody
looked for, so the closed vocabulary (``hidden-only``, ``equivalent``, ``infra``,
``syntax`` from Task 12, plus ``ineligible-class`` and ``unit-no-valid`` from here, plus
``stack-apply`` which the rung-1 builder hands in through ``extra_counts``) is
seeded to zero. ``eligible_classes`` counts every class the corpus *could* have supplied,
over all class units and independent of ``C``; ``classes_taken`` is what the quota walk
actually took. The walk stops as soon as it has ``C`` classes, so the class units it
never reached are counted as ``units-unused`` and named in ``dropped`` as
``unit-unused`` -- a unit missing from the stream is visible in the record, never
inferred from a gap. ``dropped`` is emitted in sorted order within each segment, so it
does not inherit the caller's unit ordering.

``stream_hash`` covers the knobs (seed, C, n_nov, rung), the unit sources that survived,
and every task's (key, phase, kind). Change any of them and the stream is a different
stream -- it is the id a run record points at, not a checksum of convenience. Neither
``counts`` nor ``dropped`` is in it: they describe the composition, they are not it.
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
                                      "ineligible-class", "unit-no-valid", "stack-apply")
"""Reasons a mutant, unit or class fails to reach the stream. All are counted, zero included.

``dropped`` carries one further reason, ``unit-unused`` (a class unit the ``C``-quota walk
never reached). It is counted under the census key ``units-unused`` rather than here,
alongside ``eligible_classes`` and ``valid_mutants``, because it is a property of the
quota rather than of the mutant.

``stack-apply`` runs the other way: it is counted here but never *named* in ``dropped``.
It is the rung-1 count of mutant pairs that failed composition, and a pair is not a
stream entity -- it has no id to name, and both of its components may still serve
elsewhere in the stream. compose never sees those failures itself; the builder passes
the total in through ``extra_counts``. Seeding it to zero at every rung keeps "no pair
failed" distinct from "nobody counted".
"""


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

    ``span2`` is the second site of a rung-1 stacked task and ``None`` for a single-site
    one. It is trailing and defaulted so every existing positional construction still
    means what it meant, and so a pre-rung-1 manifest still loads.
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
    span2: Span | None = None

    def to_dict(self) -> dict:
        """JSON-ready form: the nested span tuples become lists so a file round-trip is exact."""
        d = asdict(self)
        d["span"] = [list(self.span[0]), list(self.span[1])]
        d["span2"] = None if self.span2 is None else [list(self.span2[0]), list(self.span2[1])]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        """Inverse of :meth:`to_dict`; restores the tuple shape so equality holds.

        ``span2`` is read with ``.get``: a manifest written before rung 1 existed carries
        no such key at all, and it must load as the single-site task it is.
        """
        d = dict(d)
        d["span"] = (tuple(d["span"][0]), tuple(d["span"][1]))
        s2 = d.get("span2")
        d["span2"] = None if s2 is None else (tuple(s2[0]), tuple(s2[1]))
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
    """Package one validated mutant as a task. The mutant's key becomes the task key.

    A stacked mutant's top-level ``span`` is already its *early* component's, so ``span2``
    carries the late one and the task reports both sites it touches. A single-site mutant
    has no components and reports ``None``.
    """
    return TaskSpec(m.key, m.unit_id, m.family, class_id(m.unit_id, m.family), phase, kind, m.span,
                    v.kills_by_timeout, v.n_killing_visible,
                    m.components[1].span if m.components else None)


def _bump(counts: dict[str, int], key: str) -> None:
    """Increment ``counts[key]``, creating it at zero first."""
    counts[key] = counts.get(key, 0) + 1


def _by_family(pairs: list[Pair]) -> dict[str, list[Pair]]:
    """Group one unit's valid mutants by family. Absent families simply never appear --
    on the real corpus EXC yields no mutants at all, and that must not be an error."""
    by: dict[str, list[Pair]] = {}
    for m, v in pairs:
        by.setdefault(m.family, []).append((m, v))
    return by


def _site_set(m: Mutant) -> frozenset[Span]:
    """The sites a mutant touches: its components' spans, or its own span for a single."""
    return frozenset(c.span for c in m.components) if m.components else frozenset((m.span,))


def _is_eligible(pairs: list[Pair]) -> bool:
    """A family forms a class only if two of its valid mutants touch **disjoint** site sets.

    At rung 0 every site set is one span, so this is exactly the old "two of them sit at
    different spans" -- same answer, same order, and no rng either way. At rung 1 a mutant
    touches two sites and sharing either one disqualifies the pairing (spec §4.8.3).
    """
    sets = [_site_set(m) for m, _ in pairs]
    return any(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))


def _prefer_non_timeout(pairs: list[Pair], rng: random.Random) -> list[Pair]:
    """``pairs`` shuffled with ``rng``, then stably sorted so non-timeout kills come first.

    A mutant the visible suite only catches by hanging is a real task but a poor one --
    its signal is "wait five seconds", not "this assertion fails" -- so it is chosen last
    rather than excluded. The shuffle happens *before* the sort, so ties break randomly
    instead of by input order. This orders *selection* only; phase assignment is a
    separate draw (see ``_build_classes``).
    """
    pool = list(pairs)
    rng.shuffle(pool)
    pool.sort(key=lambda mv: mv[1].kills_by_timeout)
    return pool


def _pick_pair(pairs: list[Pair], rng: random.Random) -> tuple[Pair, Pair] | None:
    """Two mutants of one family on **disjoint** sites, or ``None`` if the family has none.

    The site check is the class's whole point -- see the module docstring. It is kept even
    though ``_build_classes`` only calls this for families ``_is_eligible`` accepted: the
    guard there saves an rng draw, this one keeps the invariant local to the function that
    would otherwise break it.

    Unlike ``_is_eligible`` the walk is not exhaustive -- it keeps the preferred head and
    tries only what follows it. At rung 0 that cannot fail on an eligible family (two
    distinct spans exist, so whatever the head is, one of them differs from it); at rung 1
    a head can overlap every remaining candidate, and that family yields no class.
    """
    pool = _prefer_non_timeout(pairs, rng)
    m1 = pool[0]
    for cand in pool[1:]:
        if not (_site_set(cand[0]) & _site_set(m1[0])):
            return m1, cand
    return None


def _partition_valid(units: list[Unit], validated: dict[str, list[Pair]],
                     ) -> tuple[dict[str, list[Pair]], list[str], dict[str, int]]:
    """Split the validated mutants into "usable, by unit" and "excluded, counted by reason".

    Returns ``(valid_by_unit, units with no valid mutant, counts)``. A unit with nothing
    valid is named to the caller so it reaches ``dropped``: a unit missing from the stream
    must be visible in the record, not inferred from a gap.
    """
    valid_by_unit: dict[str, list[Pair]] = {}
    no_valid: list[str] = []
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
            no_valid.append(u.unit_id)
            _bump(counts, "unit-no-valid")
    return valid_by_unit, no_valid, counts


def _class_census(valid_by_unit: dict[str, list[Pair]], class_units: list[str],
                  ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Every (unit, family) class over **all** class units, split eligible / ineligible.

    A census, not a tally: it does not stop at ``C`` and it draws no randomness, so adding
    it cannot shift the walk's rng sequence. Sorted by ``(unit_id, family)`` so the
    ``dropped`` entries derived from it are independent of the caller's ordering.
    """
    eligible: list[tuple[str, str]] = []
    ineligible: list[tuple[str, str]] = []
    for uid in class_units:
        for fam, pairs in _by_family(valid_by_unit[uid]).items():
            (eligible if _is_eligible(pairs) else ineligible).append((uid, fam))
    return sorted(eligible), sorted(ineligible)


def _build_classes(valid_by_unit: dict[str, list[Pair]], class_units: list[str], C: int, rng: random.Random,
                   ) -> tuple[dict[str, tuple[str, str]], list[TaskSpec], list[TaskSpec], list[str]]:
    """Take (unit, family) classes until ``C`` of them exist; report the units never reached.

    Families are visited in sorted order within each unit so the walk is deterministic.
    Which member of a pair becomes ``first`` is a seeded coin flip drawn here, after the
    pair is fixed -- see the module docstring and spec §4.8.1(1c).
    """
    classes: dict[str, tuple[str, str]] = {}
    p1: list[TaskSpec] = []
    p2: list[TaskSpec] = []
    for i, uid in enumerate(class_units):
        if len(classes) >= C:
            return classes, p1, p2, class_units[i:]
        by_fam = _by_family(valid_by_unit[uid])
        for fam in sorted(by_fam):
            if len(classes) >= C:
                break
            if not _is_eligible(by_fam[fam]):
                continue
            pair = _pick_pair(by_fam[fam], rng)
            if pair is None:                                    # rung 0: unreachable; rung 1: see _pick_pair
                continue
            first, second = pair if rng.random() < 0.5 else (pair[1], pair[0])
            classes[class_id(uid, fam)] = (first[0].key, second[0].key)
            p1.append(_task(*first, 1, "first"))
            p2.append(_task(*second, 2, "second"))
    return classes, p1, p2, []


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


def _dropped(no_valid: list[str], ineligible: list[tuple[str, str]], unused: list[str]) -> tuple[tuple[str, str], ...]:
    """Every excluded unit and class, each segment sorted so the caller's order cannot leak in."""
    return tuple([(uid, "unit-no-valid") for uid in sorted(no_valid)]
                 + [(class_id(uid, fam), "ineligible-class") for uid, fam in ineligible]
                 + [(uid, "unit-unused") for uid in sorted(unused)])


def compose(units: list[Unit], validated: dict[str, list[Pair]], *,
            seed: int, C: int, n_nov: int, rung: str = "base",
            extra_counts: dict[str, int] | None = None) -> StreamManifest:
    """Build the stream: ``C`` classes across two phases plus ``n_nov`` held-out novel tasks.

    Raises :class:`NotEnoughClasses` rather than returning a short stream -- a run with
    fewer classes than pre-registered is not the experiment that was registered. That is
    checked **twice**, because the two site tests are not the same test. The pre-walk
    check is over ``_is_eligible``, which asks whether *some* two of a family's mutants
    are disjoint; the walk itself is anchored on ``_pick_pair``'s preferred head, which
    only scans what follows it. For singles the two always agree. At rung 1 a head can
    overlap every remaining candidate, so an eligible family can still yield no class and
    the walk can end short with the pre-check satisfied -- hence the post-walk guard. It
    draws no randomness and cannot fire at rung 0, so the rung-0 stream is untouched.

    ``extra_counts`` merges census keys the *builder* observed and compose never could --
    rung 1's ``stack-apply`` -- into ``counts`` after the closed vocabulary is seeded, so
    the manifest can report them without compose having to know how they happened.
    """
    rng = random.Random(f"{seed}:compose")
    valid_by_unit, no_valid, counts = _partition_valid(units, validated)
    candidates = sorted(valid_by_unit)
    rng.shuffle(candidates)
    if len(candidates) < n_nov + 1:
        raise NotEnoughClasses(f"only {len(candidates)} units with valid mutants; need n_nov={n_nov} plus class units")
    novel_units, class_units = candidates[:n_nov], candidates[n_nov:]

    eligible, ineligible = _class_census(valid_by_unit, class_units)
    if len(eligible) < C:
        raise NotEnoughClasses(f"eligible classes {len(eligible)} < C={C}")
    classes, p1, p2, unused = _build_classes(valid_by_unit, class_units, C, rng)
    if len(classes) < C:
        raise NotEnoughClasses(f"classes taken {len(classes)} < C={C} (eligible {len(eligible)}; the anchored "
                               f"pair walk can fall short of the existential census at rung 1)")
    p2 = p2 + _pick_novel(valid_by_unit, novel_units, rng)

    random.Random(f"{seed}:phase1").shuffle(p1)
    random.Random(f"{seed}:phase2").shuffle(p2)
    tasks = tuple(p1 + p2)
    unit_ids = tuple(sorted({t.unit_id for t in tasks}))
    src_hashes = sorted(u.src_hash for u in units if u.unit_id in unit_ids)
    counts["ineligible-class"] = len(ineligible)
    for reason in EXCLUSION_REASONS:
        counts.setdefault(reason, 0)                    # None-vs-zero: unobserved is still named
    counts.update(extra_counts or {})                   # builder-side census keys, e.g. stack-apply
    counts.update({"eligible_classes": len(eligible), "classes_taken": len(classes),
                   "units-unused": len(unused),
                   "valid_mutants": sum(len(v) for v in valid_by_unit.values())})
    return StreamManifest(_hash(seed, C, n_nov, rung, src_hashes, tasks), seed, C, n_nov, rung,
                          unit_ids, tasks, classes, _dropped(no_valid, ineligible, unused), counts)
