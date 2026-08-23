from dataclasses import replace
from crucible.stream.precheck import precheck, two_proportion_band, mean_band
from tests.stream.test_compose import _world
from crucible.stream.compose import compose

def _full_counts(m):
    c = dict(m.counts)
    for k in ("hidden-only", "equivalent", "infra", "syntax", "ineligible-class", "unit-no-valid", "eligible_classes", "valid_mutants"):
        c.setdefault(k, 0)
    return replace(m, counts=c)

def test_precheck_passes_on_a_well_formed_stream():
    units, validated = _world(8)
    m = _full_counts(compose(units, validated, seed=0, C=4, n_nov=2))
    rep = precheck(m, {u.unit_id: u for u in units})
    assert rep.ok, [c for c in rep.checks if not c.passed]
    assert {c.name for c in rep.checks} == {"family-distribution-identical", "killing-count-band", "unit-length-identical",
                                            "timeout-rate-band", "novel-disjoint", "distinct-sites", "counts-named"}

def test_precheck_fails_when_a_second_task_shares_the_site():
    units, validated = _world(8)
    m = _full_counts(compose(units, validated, seed=0, C=4, n_nov=2))
    p1 = m.phase(1)[0]
    bad_tasks = tuple(replace(t, span=p1.span) if (t.phase == 2 and t.class_id == p1.class_id) else t for t in m.tasks)
    rep = precheck(replace(m, tasks=bad_tasks), {u.unit_id: u for u in units})
    assert not rep.ok and not next(c for c in rep.checks if c.name == "distinct-sites").passed

def test_precheck_fails_when_counts_missing():
    units, validated = _world(8)
    m = compose(units, validated, seed=0, C=4, n_nov=2)
    m2 = replace(m, counts={k: v for k, v in m.counts.items() if k != "equivalent"})
    rep = precheck(m2, {u.unit_id: u for u in units})
    assert not next(c for c in rep.checks if c.name == "counts-named").passed

def test_bands():
    d, band, ok = two_proportion_band(10, 100, 12, 100)
    assert ok and abs(d) < band
    d, band, ok = two_proportion_band(10, 100, 60, 100)
    assert not ok
    assert mean_band([1, 2, 3, 4], [1, 2, 3, 4])[2] and not mean_band([1, 1, 1, 1, 1], [9, 9, 9, 9, 9])[2]
