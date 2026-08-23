"""Tests for stream composition: phases, classes, the novel hold-out, and the manifest hash.

Nothing here touches the sandbox -- ``Unit``/``Mutant``/``Validation`` are built by hand,
so every test is a pure function of its fixtures and runs in milliseconds.

The fixture does more work than it looks. ``_world`` gives every unit **three** valid
ARITH mutants: two that share a span (line 2, tags ``a`` and ``a2``) and one on another
line (tag ``b``). The same-span sibling is what makes "a class must touch two different
sites" observable -- a pair-picker that simply takes the first two candidates it sees
will hand back two mutants of one span as soon as the shuffle puts them first, and
``test_pairs_never_share_a_site`` sweeps seeds until it does. CMP is the mirror image:
one valid mutant plus one ``equivalent``, so it can never form a class and always lands
in ``dropped`` as ``ineligible-class``. Only two of the eight families appear at all,
which is also the fixture for "compose tolerates a family that is simply absent" -- on
the real corpus EXC yields zero mutants.

Task 15 imports ``_world`` from this module; keep its name and signature.
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


def _world(n_units=6):
    units = [_unit(i) for i in range(n_units)]
    validated = {}
    for u in units:
        validated[u.unit_id] = [_mut(u, "ARITH", 2, "a"), _mut(u, "ARITH", 2, "a2"), _mut(u, "ARITH", 3, "b"),
                                _mut(u, "CMP", 2, "c"), _equivalent(u)]
    return units, validated


def _timeout_world(n_units=2):
    """Every unit's ARITH family holds one timeout mutant and two that kill outright."""
    units = [_unit(i) for i in range(n_units)]
    validated = {u.unit_id: [_mut(u, "ARITH", 2, "a", timeout=True), _mut(u, "ARITH", 3, "b"),
                             _mut(u, "ARITH", 4, "c")] for u in units}
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
    assert len(m.classes) == 3 and m.counts["eligible_classes"] == 3
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
    """``build_units`` returns units in record order; the manifest must not depend on it."""
    units, validated = _world()
    a = compose(units, validated, seed=0, C=3, n_nov=2)
    reordered = list(reversed(units))
    b = compose(reordered, {u.unit_id: validated[u.unit_id] for u in reordered}, seed=0, C=3, n_nov=2)
    assert a == b


def test_not_enough_classes_raises():
    units, validated = _world(3)
    with pytest.raises(NotEnoughClasses):
        compose(units, validated, seed=0, C=5, n_nov=1)


def test_not_enough_units_for_the_hold_out_raises():
    units, validated = _world(2)
    with pytest.raises(NotEnoughClasses):
        compose(units, validated, seed=0, C=1, n_nov=2)


def test_prefers_non_timeout_mutants_for_m1():
    units, validated = _world(2)
    u = units[0]
    validated[u.unit_id] = [_mut(u, "ARITH", 2, "a", timeout=True), _mut(u, "ARITH", 3, "b"), _mut(u, "ARITH", 4, "c")]
    m = compose(units, validated, seed=3, C=1, n_nov=1)
    first = m.phase(1)[0]
    assert first.kills_by_timeout is False
    # Whichever unit the draw makes the class unit, and whichever it holds out, a task
    # whose failure mode is "wait for the timeout" is the last resort -- for the novel
    # pick too. Sweeping seeds removes the shuffle's luck from the assertion.
    tu, tv = _timeout_world(2)
    for seed in range(20):
        m = compose(tu, tv, seed=seed, C=1, n_nov=1)
        assert m.phase(1)[0].kills_by_timeout is False
        assert all(t.kills_by_timeout is False for t in m.phase(2) if t.kind == "novel")


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


def test_dropped_names_every_excluded_unit_and_class():
    units, validated = _world()
    dead = units[0]
    validated[dead.unit_id] = [(Mutant(dead.unit_id, "hk", "Op", 0, "ARITH", ((2, 1), (2, 2)), "s", "d"),
                               Validation("hk", False, "hidden-only", False, 0, ()))]
    m = compose(units, validated, seed=0, C=3, n_nov=2)
    assert (dead.unit_id, "unit-no-valid") in m.dropped
    assert dead.unit_id not in m.unit_ids
    assert m.counts["unit-no-valid"] == 1 and m.counts["hidden-only"] == 1
    ineligible = [cid for cid, reason in m.dropped if reason == "ineligible-class"]
    assert ineligible and all(cid.endswith("|CMP") for cid in ineligible)
    assert m.counts["ineligible-class"] == len(ineligible)


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
