import gzip, json, pathlib
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
