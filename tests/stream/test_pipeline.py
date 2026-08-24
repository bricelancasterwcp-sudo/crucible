import gzip, json, pathlib
import pytest
from crucible.stream.pipeline import build_stream, BuildConfig, smoke
from crucible.stream import store

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
def _recs():
    out = []
    for n in ("mini_humaneval.jsonl.gz", "mini_mbpp.jsonl.gz"):
        with gzip.open(FIX / n, "rt") as fh:
            out += [json.loads(l) for l in fh]
    return out

def test_build_stream_is_deterministic_and_structural_prechecks_pass(tmp_path):
    # n_nov=0 so all three fixture units are class units (is_pos has BOOL+SDL, add2 has ARITH ⇒ C=2 is reachable).
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    d1 = build_stream(cfg, tmp_path / "a", recs=_recs(), log=lambda *a: None)
    d2 = build_stream(cfg, tmp_path / "b", recs=_recs(), log=lambda *a: None)
    m1, m2 = store.read_manifest(d1), store.read_manifest(d2)
    assert m1.stream_hash == m2.stream_hash and len(m1.tasks) == 4
    from crucible.stream.precheck import precheck
    units = {u: store.read_unit(d1, u) for u in m1.unit_ids}
    rep = precheck(m1, units)
    by = {c.name: c for c in rep.checks}
    # The statistical bands are meaningless at n=2; the structural checks must hold at any size.
    for name in ("family-distribution-identical", "novel-disjoint", "distinct-sites", "counts-named"):
        assert by[name].passed, by[name]

def test_smoke_reports_all_killed(tmp_path):
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    d = build_stream(cfg, tmp_path, recs=_recs(), log=lambda *a: None)
    res = smoke(d, n=3, log=lambda *a: None)
    assert res["ran"] == 3 and res["killed"] == 3 and res["infra"] == 0


def test_limit_samples_the_seeded_subset_not_the_first_n():
    # Guards the invariant: limit_units samples with random.Random(f"{seed}:units"),
    # never "the first N". A recs[:n] mutation returns the sorted first N and fails here.
    from crucible.stream.pipeline import _limit
    recs = [{"task_id": f"T/{i}"} for i in range(20)]
    cfg = BuildConfig(seed=0, limit_units=5)
    got = _limit(recs, cfg)
    ids = [r["task_id"] for r in got]
    assert len(ids) == 5
    assert ids != [f"T/{i}" for i in range(5)]          # not the first N
    assert [r["task_id"] for r in _limit(recs, cfg)] == ids   # deterministic for a fixed seed


def test_smoke_counts_a_surviving_mutant_as_not_killed(tmp_path, monkeypatch):
    # A mutant the visible suite does not kill must land in not_killed, never killed.
    # A "always killed" mutation of smoke's classifier flips this test red.
    from crucible.stream import pipeline
    from crucible.sandbox.report import TestReport
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    d = build_stream(cfg, tmp_path, recs=_recs(), log=lambda *a: None)
    survivor = TestReport(("t",), (), (), (), 0.0, None)   # only a pass ⇒ not killed, not infra
    monkeypatch.setattr(pipeline, "run_tests", lambda *a, **k: survivor)
    res = smoke(d, n=1, log=lambda *a: None)
    assert res["killed"] == 0 and res["not_killed"] == 1 and res["infra"] == 0


def test_dropped_dict_round_trips():
    # Dropped crosses a file boundary (build_dropped.jsonl), so it carries the same
    # to_dict/from_dict pair every persisted dataclass in the store does.
    from crucible.stream.build import Dropped
    d = Dropped("HumanEval/42", "canonical-syntax:invalid syntax")
    assert Dropped.from_dict(d.to_dict()) == d
    assert d.to_dict() == {"unit_id": "HumanEval/42", "reason": "canonical-syntax:invalid syntax"}


def test_build_stream_persists_build_time_unit_drops(tmp_path):
    # A canonical that does not compile is dropped at build time; its identity must reach
    # disk as provenance, not survive only as dropped={N} in the build log. build_dropped.jsonl
    # is parallel to validations.jsonl: every drop is NAMED.
    from crucible.stream.build import Dropped
    recs = _recs()
    bad = dict(recs[0]); bad["task_id"] = "HumanEval/9999"; bad["canonical_solution"] = "    return a -\n"
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    d = build_stream(cfg, tmp_path, recs=recs + [bad], log=lambda *a: None)

    assert (d / "build_dropped.jsonl").exists()
    dropped = store.read_build_dropped(d)
    assert [x.unit_id for x in dropped] == ["HumanEval/9999"]
    assert dropped[0].reason.startswith("canonical-syntax:")
    # Round-trips to the same list[Dropped] on re-read, and the dropped unit is not a unit.
    assert store.read_build_dropped(d) == dropped
    assert all(isinstance(x, Dropped) for x in dropped)
    assert "HumanEval/9999" not in store.read_manifest(d).unit_ids


def test_build_time_drops_do_not_change_stream_hash(tmp_path):
    # The hash covers seed/C/n_nov/rung/surviving-unit src_hashes/tasks -- never the drops.
    # A non-compiling canonical never becomes a unit, so the surviving units are identical
    # with or without it appended, and so is the stream_hash.
    recs = _recs()
    bad = dict(recs[0]); bad["task_id"] = "HumanEval/9999"; bad["canonical_solution"] = "    return a -\n"
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    clean = build_stream(cfg, tmp_path / "clean", recs=recs, log=lambda *a: None)
    withdrop = build_stream(cfg, tmp_path / "withdrop", recs=recs + [bad], log=lambda *a: None)
    assert store.read_manifest(clean).stream_hash == store.read_manifest(withdrop).stream_hash
    assert store.read_build_dropped(clean) == []
    assert [x.unit_id for x in store.read_build_dropped(withdrop)] == ["HumanEval/9999"]


def _stack_recs():
    """The fixture corpus plus one unit with enough same-family sites to *stack*.

    A rung-1 class needs two site-disjoint stacked mutants, i.e. FOUR distinct spans in one
    (unit, family) group. The three shipped fixture units top out at two spans per family --
    one pair, one stacked mutant, no class -- so the corpus is extended here in this file's
    existing idiom (copy a rec, replace its body) rather than by relaxing what the rung-1
    test asserts. ``addn`` gives ARITH six spans and SDL five, i.e. two eligible rung-1
    classes, and its CONST pair cannot compose (see the stack-apply assertion below), so the
    same build also exercises a nonzero ``stack-apply`` census.
    """
    recs = _recs()
    r = dict(recs[0])
    r["task_id"] = "HumanEval/1000"
    r["entry_point"] = "addn"
    r["prompt"] = 'def addn(a: int, b: int) -> int:\n    """Sum a few ways."""\n'
    r["canonical_solution"] = ("    p = a + b\n    q = a - b\n    r = a * 3\n"
                               "    s = b + 7\n    return p + q + r + s\n")
    r["base_input"] = [[1, 2], [5, 3], [-4, 9], [11, 6]]
    r["plus_input"] = [[2, 7], [8, 1]]
    return recs + [r]


def test_base_rung_never_stacks(tmp_path, monkeypatch):
    # The rung branch, from the other side: at rung "base" the stacking layer is not
    # merely unused, it is never called. An inverted branch dies here rather than in a
    # census assertion that could be read as a tuning difference.
    from crucible.stream import stack
    def boom(*a, **k):
        raise AssertionError("stack_unit called at rung base")
    monkeypatch.setattr(stack, "stack_unit", boom)
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    d = build_stream(cfg, tmp_path, recs=_recs(), log=lambda *a: None)
    man = store.read_manifest(d)
    assert man.rung == "base" and all(t.span2 is None for t in man.tasks)


def test_unknown_rung_is_refused_before_any_work(tmp_path):
    # ALLOWED_RUNGS is the closed vocabulary; an unknown rung is a caller error, not a
    # silent fall-through to base. The check is first, so nothing is built and nothing is
    # written -- with C at its default this would otherwise die as NotEnoughClasses instead.
    from crucible.stream.pipeline import ALLOWED_RUNGS
    assert ALLOWED_RUNGS == ("base", "stack2")
    with pytest.raises(ValueError):
        build_stream(BuildConfig(seed=0, rung="tower"), tmp_path, recs=_recs(), log=lambda *a: None)
    assert list(tmp_path.iterdir()) == []


def test_stack2_composes_only_two_site_tasks_and_keeps_the_singles_on_disk(tmp_path):
    # Every task at rung 1 is a two-site mutant: span2 set, two components, and the
    # components' spans are exactly the two sites the task reports.
    from crucible.stream.precheck import precheck
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=6, max_hidden=2, jobs=2, rung="stack2")
    d = build_stream(cfg, tmp_path / "a", recs=_stack_recs(), log=lambda *a: None)
    man = store.read_manifest(d)
    assert man.rung == "stack2" and len(man.tasks) == 4
    task_mutants = {t.task_key: store.read_mutant(d, t.task_key) for t in man.tasks}
    for t in man.tasks:
        assert t.span2 is not None and t.span2 != t.span
        m = task_mutants[t.task_key]
        assert len(m.components) == 2
        assert {c.span for c in m.components} == {t.span, t.span2}

    # A real rung-1 stream must clear the gate a run replays it through -- including
    # two-site-at-stack2, which only a genuinely stacked manifest can pass.
    rep = precheck(man, {u: store.read_unit(d, u) for u in man.unit_ids})
    assert rep.ok, [c for c in rep.checks if not c.passed]

    # stack-apply is the builder-side census key: pairs that failed to become a stacked
    # mutant. addn's CONST pair always fails -- NumberReplacer yields two mutation
    # positions per literal, so re-selecting the early component by exact span on the
    # intermediate source finds two hits, not one -- so this stream's count is >= 1 and a
    # dropped extra_counts shows up as 0 here.
    assert man.counts["stack-apply"] >= 1

    # Singles are provenance, not tasks (R-S25-1): a stacked task's two components must be
    # checkable, so the single-site mutants it was composed from stay on disk with their
    # verdicts -- and not one of them is a task. "Single" is read off the mutant itself
    # (no components), not off "is not a task": most stacked mutants are not tasks either.
    task_keys = {t.task_key for t in man.tasks}
    vals = {v.mutant_key: v for v in store.read_validations(d)}
    assert task_keys <= set(vals)
    stored = [store.read_mutant(d, k) for k in vals]
    singles = [m for m in stored if not m.components]
    assert singles and not (task_keys & {m.key for m in singles})
    # The provenance has to be usable, not merely present: EVERY component of EVERY task
    # must be findable on disk as a single-site mutant its own verdict calls valid. That is
    # R-S25-1's component half, checked the way a reader would check it -- by the component
    # coordinates, which are stated against the original source and so match the single's
    # (operator, occurrence, span) exactly. Storing only the invalid singles, or only the
    # first unit's, leaves this subset short.
    valid_single_sites = {(m.operator, m.occurrence, m.span) for m in singles if vals[m.key].valid}
    for t in man.tasks:
        comps = {(c.operator, c.occurrence, c.span) for c in task_mutants[t.task_key].components}
        assert comps <= valid_single_sites, (t.task_key, comps - valid_single_sites)
    # and what compose was offered was the stacked mutants alone -- the census counts them,
    # not the far larger pool of valid singles they were built from.
    assert man.counts["valid_mutants"] == sum(vals[m.key].valid for m in stored if m.components)
    assert man.counts["valid_mutants"] < sum(vals[m.key].valid for m in singles)

    # Spec §9 at rung 1: same seed, same inputs => same stream. Rung 0 is pinned by
    # test_build_stream_is_deterministic_...; rung 1 adds a whole stacking layer (a
    # per-unit rng threaded through families in sorted order, plus a second validate_many
    # pass whose kills_by_timeout feed pair selection), and a stream that is only
    # *usually* the same stream is not a reproducible experiment. A sibling out_root is
    # required, not cosmetic: same root and _write_atomic takes the content-addressed
    # early return, which would compare a hash to itself.
    d2 = build_stream(cfg, tmp_path / "b", recs=_stack_recs(), log=lambda *a: None)
    assert store.read_manifest(d2).stream_hash == man.stream_hash
