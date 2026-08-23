"""Tests for stream composition: phases, classes, the novel hold-out, and the manifest hash.

Nothing here touches the sandbox -- ``Unit``/``Mutant``/``Validation`` are built by hand,
so every test is a pure function of its fixtures and runs in milliseconds.

The fixtures do more work than they look. ``_world`` gives every unit **three** valid
ARITH mutants: two that share a span (line 2, tags ``a`` and ``a2``) and one on another
line (tag ``b``). The same-span sibling is what makes "a class must touch two different
sites" observable -- a pair-picker that simply takes the first two candidates it sees
will hand back two mutants of one span as soon as the shuffle puts them first, and
``test_pairs_never_share_a_site`` sweeps seeds until it does. CMP is the mirror image:
one valid mutant plus one ``equivalent``, so it can never form a class and always lands
in ``dropped`` as ``ineligible-class``. Only two of the eight families appear at all,
which is also the fixture for "compose tolerates a family that is simply absent" -- on
the real corpus EXC yields zero mutants. ``_timeout_world`` and ``_coin_world`` isolate
the two halves of the timeout rule: which mutants are *chosen*, and which of the chosen
pair goes *first*.

Task 15 imports ``_world`` from this module; keep its name and its default behaviour.
"""

import json
from dataclasses import fields, replace

import pytest

from crucible.stream.compose import NotEnoughClasses, StreamManifest, TaskSpec, class_id, compose
from crucible.stream.mutants import Mutant
from crucible.stream.units import Unit, sha256_text
from crucible.stream.validate import Validation


def _unit(i):
    src = f"def f{i}(a, b):\n    return a + b\n"
    return Unit(f"X/{i}", f"unit_x_{i}", f"f{i}", src, "v", "h", sha256_text(src), 1, 1, ())


def _mut(u, fam, line, tag, timeout=False):
    key = sha256_text(f"{u.unit_id}:{fam}:{line}:{tag}")
    m = Mutant(u.unit_id, key, "Op", 0, fam, ((line, 1), (line, 2)), "src", "diff")
    v = Validation(key, True, "killed-visible", timeout, 1, ("test_v0",))
    return (m, v)


def _equivalent(u):
    """One mutant no test can see: never valid, and the fixture's ``equivalent`` count."""
    return (Mutant(u.unit_id, "eq" + u.unit_id, "Op", 0, "CMP", ((2, 1), (2, 2)), "s", "d"),
            Validation("eq" + u.unit_id, False, "equivalent", False, 0, ()))


def _hidden_only(u):
    """A mutant only the hidden suite catches: not valid, so a unit of these is unusable."""
    return (Mutant(u.unit_id, "ho" + u.unit_id, "Op", 0, "ARITH", ((2, 1), (2, 2)), "s", "d"),
            Validation("ho" + u.unit_id, False, "hidden-only", False, 0, ()))


def _world(n_units=6, n_dead=0):
    """``n_units`` usable units, plus ``n_dead`` whose every mutant is invalid.

    ``n_dead`` defaults to 0, so ``_world(n)`` is exactly the fixture Task 15 imports.
    """
    units = [_unit(i) for i in range(n_units + n_dead)]
    validated = {}
    for u in units[:n_units]:
        validated[u.unit_id] = [_mut(u, "ARITH", 2, "a"), _mut(u, "ARITH", 2, "a2"), _mut(u, "ARITH", 3, "b"),
                                _mut(u, "CMP", 2, "c"), _equivalent(u)]
    for u in units[n_units:]:
        validated[u.unit_id] = [_hidden_only(u)]
    return units, validated


def _timeout_world(n_units=2):
    """Every unit's ARITH family holds one timeout mutant and two that kill outright.

    Two non-timeout mutants at distinct spans exist, so a pair can be formed without the
    timeout one -- and must be.
    """
    units = [_unit(i) for i in range(n_units)]
    validated = {u.unit_id: [_mut(u, "ARITH", 2, "a", timeout=True), _mut(u, "ARITH", 3, "b"),
                             _mut(u, "ARITH", 4, "c")] for u in units}
    return units, validated


def _coin_world(n_units=6):
    """Every eligible class has exactly one timeout and one non-timeout member.

    The pair is forced, so the only freedom left is which member becomes ``first`` --
    which is precisely what the phase coin flip decides.
    """
    units = [_unit(i) for i in range(n_units)]
    validated = {u.unit_id: [_mut(u, "ARITH", 2, "t", timeout=True), _mut(u, "ARITH", 3, "n")] for u in units}
    return units, validated


def test_compose_shapes_and_invariants():
    units, validated = _world()
    m = compose(units, validated, seed=0, C=3, n_nov=2)
    p1, p2 = m.phase(1), m.phase(2)
    assert len(p1) == 3 and all(t.kind == "first" for t in p1)
    assert sum(t.kind == "second" for t in p2) == 3 and sum(t.kind == "novel" for t in p2) == 2
    novel_units = {t.unit_id for t in p2 if t.kind == "novel"}
    assert novel_units.isdisjoint({t.unit_id for t in p1})
    for cid, (k1, k2) in m.classes.items():
        t1 = next(t for t in p1 if t.task_key == k1)
        t2 = next(t for t in p2 if t.task_key == k2)
        assert t1.span != t2.span and t1.class_id == t2.class_id == cid
    assert m.counts["equivalent"] == 6 and "ineligible-class" in m.counts
    assert StreamManifest.from_dict(m.to_dict()) == m
    # Every task is one of that unit's mutants, carried over field for field: the task
    # key *is* the mutant key, never a fresh id minted here.
    for t in m.tasks:
        src = {mu.key: (mu, v) for mu, v in validated[t.unit_id]}
        assert t.task_key in src
        mu, v = src[t.task_key]
        assert (t.family, t.span, t.class_id) == (mu.family, mu.span, class_id(mu.unit_id, mu.family))
        assert (t.kills_by_timeout, t.n_killing_visible) == (v.kills_by_timeout, v.n_killing_visible)
    assert m.seed == 0 and m.C == 3 and m.n_nov == 2 and m.rung == "base"
    assert m.unit_ids == tuple(sorted({t.unit_id for t in m.tasks}))
    assert len(m.classes) == 3 and m.counts["classes_taken"] == 3
    assert m.counts["eligible_classes"] == 4 and m.counts["ineligible-class"] == 4   # census over 4 class units
    assert m.counts["units-unused"] == 1
    # Zero, not absent: every reason in the closed vocabulary is named even when unobserved.
    for reason in ("hidden-only", "equivalent", "infra", "syntax", "ineligible-class", "unit-no-valid"):
        assert reason in m.counts
    assert m.counts["infra"] == 0 and m.counts["syntax"] == 0 and m.counts["unit-no-valid"] == 0
    assert m.counts["valid_mutants"] == 4 * 6


def test_compose_is_deterministic_and_seed_sensitive():
    units, validated = _world()
    a = compose(units, validated, seed=0, C=3, n_nov=2)
    b = compose(units, validated, seed=0, C=3, n_nov=2)
    c = compose(units, validated, seed=1, C=3, n_nov=2)
    assert a.stream_hash == b.stream_hash and [t.task_key for t in a.tasks] == [t.task_key for t in b.tasks]
    assert a.stream_hash != c.stream_hash
    assert a == b


def test_unit_partition_is_seed_dependent():
    """Which units are held out is *drawn* from the seed, never "the first n_nov sorted".

    A composition that skipped the seeded shuffle of the candidate units would hand back
    the same hold-out set for every seed -- and the stream hash would still differ,
    because task *order* is seeded separately. So this is the assertion that pins the
    draw itself.
    """
    units, validated = _world(8)
    held_out = {frozenset(t.unit_id for t in compose(units, validated, seed=s, C=3, n_nov=2).phase(2)
                          if t.kind == "novel") for s in range(8)}
    assert len(held_out) > 1


def test_composition_is_independent_of_input_order():
    """``build_units`` returns units in record order; nothing in the manifest may inherit it.

    The world carries dead units on purpose: ``dropped``'s ``unit-no-valid`` segment is
    built by walking the caller's list, so without a sort it would flip under reversal.
    """
    units, validated = _world(6, n_dead=2)
    a = compose(units, validated, seed=0, C=3, n_nov=2)
    assert a.counts["unit-no-valid"] == 2 and a.counts["units-unused"] == 1
    reordered = list(reversed(units))
    b = compose(reordered, {u.unit_id: validated[u.unit_id] for u in reordered}, seed=0, C=3, n_nov=2)
    assert a == b
    # Grouped by segment, sorted within each: unit ids by id, classes by (unit_id, family).
    segments = ("unit-no-valid", "ineligible-class", "unit-unused")
    reasons = [r for _x, r in a.dropped]
    assert reasons == sorted(reasons, key=segments.index)
    for reason in ("unit-no-valid", "unit-unused"):
        ids = [x for x, r in a.dropped if r == reason]
        assert ids == sorted(ids)
    inel = [tuple(cid.rsplit("|", 1)) for cid, r in a.dropped if r == "ineligible-class"]
    assert inel == sorted(inel)


def test_not_enough_classes_raises():
    units, validated = _world(3)
    with pytest.raises(NotEnoughClasses):
        compose(units, validated, seed=0, C=5, n_nov=1)


def test_not_enough_units_for_the_hold_out_raises():
    units, validated = _world(2)
    with pytest.raises(NotEnoughClasses):
        compose(units, validated, seed=0, C=1, n_nov=2)


def test_prefers_non_timeout_mutants_for_the_pair():
    """The preference is over *which* two mutants form the class, not over their phases.

    Every unit here has two non-timeout mutants at distinct spans, so a timeout mutant is
    never needed -- and must never be picked, for either member. Sweeping seeds removes
    the pool shuffle's luck from the assertion.
    """
    units, validated = _timeout_world(2)
    for seed in range(20):
        m = compose(units, validated, seed=seed, C=1, n_nov=1)
        by_key = {t.task_key: t for t in m.tasks}
        for k1, k2 in m.classes.values():
            assert by_key[k1].kills_by_timeout is False and by_key[k2].kills_by_timeout is False
        assert all(t.kills_by_timeout is False for t in m.phase(2) if t.kind == "novel")


def test_first_second_timeout_assignment_is_balanced():
    """Phase is a coin flip, so ``kills_by_timeout`` cannot pile up in one phase.

    Ordering the pair non-timeout-first *and* letting that order decide the phase would
    put every timeout mutant in phase 2 by construction (measured: 0.000 vs 1.000), and
    spec 4.8.1(1c) requires the two rates to sit within 2*SE -- Task 15's
    ``timeout-rate-band`` pre-check enforces exactly that. Here each class is forced to
    hold one timeout and one non-timeout member, so the observed fraction *is* the coin.
    """
    units, validated = _coin_world(6)
    first_timeout = first_total = 0
    for seed in range(200):
        m = compose(units, validated, seed=seed, C=3, n_nov=1)
        by_key = {t.task_key: t for t in m.tasks}
        for k1, k2 in m.classes.values():
            assert by_key[k1].kills_by_timeout != by_key[k2].kills_by_timeout   # fixture sanity
        firsts = [t for t in m.phase(1)]
        first_total += len(firsts)
        first_timeout += sum(t.kills_by_timeout for t in firsts)
    assert first_total == 600
    fraction = first_timeout / first_total          # observed 0.5400 at this fixture and sweep
    assert 0.35 <= fraction <= 0.65, fraction


def test_pairs_never_share_a_site():
    """Both mutants of a class must sit at different spans, however the pool shuffles."""
    units, validated = _world()
    for seed in range(20):
        m = compose(units, validated, seed=seed, C=3, n_nov=2)
        by_key = {t.task_key: t for t in m.tasks}
        for cid, (k1, k2) in m.classes.items():
            assert by_key[k1].span != by_key[k2].span, (seed, cid)
            assert k1 != k2


def test_novel_units_are_held_out_of_phase_one():
    units, validated = _world(8)
    for seed in range(20):
        m = compose(units, validated, seed=seed, C=3, n_nov=2)
        novel = {t.unit_id for t in m.phase(2) if t.kind == "novel"}
        assert len(novel) == 2
        assert novel.isdisjoint({t.unit_id for t in m.phase(1)})
        assert novel.isdisjoint({t.unit_id for t in m.phase(2) if t.kind == "second"})


def test_stream_hash_changes_with_rung():
    units, validated = _world()
    a = compose(units, validated, seed=0, C=3, n_nov=2)
    b = compose(units, validated, seed=0, C=3, n_nov=2, rung="hard")
    assert [t.task_key for t in a.tasks] == [t.task_key for t in b.tasks]   # only the rung differs
    assert b.rung == "hard" and a.stream_hash != b.stream_hash


def test_stream_hash_changes_when_a_task_changes():
    units, validated = _world()
    base = compose(units, validated, seed=0, C=3, n_nov=2)
    bumped = {uid: [(replace(m, key=m.key + "z"), replace(v, mutant_key=v.mutant_key + "z")) for m, v in pairs]
              for uid, pairs in validated.items()}
    other = compose(units, bumped, seed=0, C=3, n_nov=2)
    assert [t.task_key for t in other.tasks] != [t.task_key for t in base.tasks]
    assert other.stream_hash != base.stream_hash


def test_counts_are_a_census_not_a_quota():
    """``eligible_classes`` describes the corpus; ``classes_taken`` describes this stream.

    Reporting ``len(classes)`` as ``eligible_classes`` would make the key always equal
    ``C`` and say nothing at all -- and the class units the quota walk never reached would
    be missing from ``unit_ids`` *and* from ``dropped``, i.e. inferable only from a gap.
    """
    units, validated = _world(8)
    m = compose(units, validated, seed=0, C=3, n_nov=2)
    class_units = {u.unit_id for u in units} - {t.unit_id for t in m.phase(2) if t.kind == "novel"}
    assert len(class_units) == 6
    assert m.counts["eligible_classes"] == 6 > m.C           # one ARITH class per class unit
    assert m.counts["classes_taken"] == len(m.classes) == 3
    assert m.counts["ineligible-class"] == 6                 # one CMP class per class unit, census
    unused = [uid for uid, reason in m.dropped if reason == "unit-unused"]
    assert m.counts["units-unused"] == len(unused) == 3
    assert set(unused) <= class_units and not set(unused) & set(m.unit_ids)
    # Every class unit is accounted for: it either contributed a class or is named unused.
    assert class_units == {t.unit_id for t in m.phase(1)} | set(unused)
    named_ineligible = {cid for cid, reason in m.dropped if reason == "ineligible-class"}
    assert named_ineligible == {class_id(uid, "CMP") for uid in class_units}


def test_dropped_names_every_excluded_unit_and_class():
    units, validated = _world(5, n_dead=1)
    dead = units[-1]
    m = compose(units, validated, seed=0, C=3, n_nov=2)
    assert (dead.unit_id, "unit-no-valid") in m.dropped
    assert dead.unit_id not in m.unit_ids
    assert m.counts["unit-no-valid"] == 1 and m.counts["hidden-only"] == 1
    ineligible = [cid for cid, reason in m.dropped if reason == "ineligible-class"]
    assert ineligible and all(cid.endswith("|CMP") for cid in ineligible)
    assert m.counts["ineligible-class"] == len(ineligible) == 3
    assert m.counts["units-unused"] == 0                      # the walk reached every class unit


def test_dicts_round_trip_through_json():
    units, validated = _world()
    m = compose(units, validated, seed=0, C=3, n_nov=2)
    d = json.loads(json.dumps(m.to_dict()))
    assert set(d) == {f.name for f in fields(StreamManifest)}
    back = StreamManifest.from_dict(d)
    assert back == m
    assert isinstance(back.tasks, tuple) and isinstance(back.unit_ids, tuple) and isinstance(back.dropped, tuple)
    assert all(isinstance(x, tuple) for x in back.dropped)
    assert all(isinstance(v, tuple) for v in back.classes.values())
    t = m.tasks[0]
    td = json.loads(json.dumps(t.to_dict()))
    assert set(td) == {f.name for f in fields(TaskSpec)}
    assert TaskSpec.from_dict(td) == t and isinstance(TaskSpec.from_dict(td).span[0], tuple)
