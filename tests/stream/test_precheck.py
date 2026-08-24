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
                                            "timeout-rate-band", "novel-disjoint", "distinct-sites", "counts-named",
                                            "two-site-at-stack2"}

def test_precheck_fails_when_a_second_task_shares_the_site():
    units, validated = _world(8)
    m = _full_counts(compose(units, validated, seed=0, C=4, n_nov=2))
    p1 = m.phase(1)[0]
    bad_tasks = tuple(replace(t, span=p1.span) if (t.phase == 2 and t.class_id == p1.class_id) else t for t in m.tasks)
    rep = precheck(replace(m, tasks=bad_tasks), {u.unit_id: u for u in units})
    assert not rep.ok and not next(c for c in rep.checks if c.name == "distinct-sites").passed

def test_distinct_sites_rejects_overlapping_site_sets():
    # good stacked class: {2,3} vs {4,5} -- disjoint, passes.
    # bad stacked class: {2,3} vs {2,5} -- shares line 2, is named.
    units, validated = _world(8)
    m = _full_counts(compose(units, validated, seed=0, C=4, n_nov=2))
    by_key = {t.task_key: t for t in m.tasks}
    span = lambda line: ((line, 1), (line, 2))
    cids = list(m.classes.items())
    good_cid, (good_k1, good_k2) = cids[0]
    bad_cid, (bad_k1, bad_k2) = cids[1]
    updated = {
        good_k1: replace(by_key[good_k1], span=span(2), span2=span(3)),
        good_k2: replace(by_key[good_k2], span=span(4), span2=span(5)),
        bad_k1: replace(by_key[bad_k1], span=span(2), span2=span(3)),
        bad_k2: replace(by_key[bad_k2], span=span(5), span2=span(2)),
    }
    new_tasks = tuple(updated.get(t.task_key, t) for t in m.tasks)
    rep = precheck(replace(m, tasks=new_tasks), {u.unit_id: u for u in units})
    check = next(c for c in rep.checks if c.name == "distinct-sites")
    assert not rep.ok
    assert not check.passed
    assert bad_cid in check.detail
    assert good_cid not in check.detail

    # existing single-site manifest fixture (span2=None throughout) still passes.
    rep_unmodified = precheck(m, {u.unit_id: u for u in units})
    assert next(c for c in rep_unmodified.checks if c.name == "distinct-sites").passed

def test_precheck_fails_when_counts_missing():
    units, validated = _world(8)
    m = compose(units, validated, seed=0, C=4, n_nov=2)
    m2 = replace(m, counts={k: v for k, v in m.counts.items() if k != "equivalent"})
    rep = precheck(m2, {u.unit_id: u for u in units})
    assert not next(c for c in rep.checks if c.name == "counts-named").passed

def test_precheck_fails_when_counts_missing_stack_apply():
    units, validated = _world(8)
    m = compose(units, validated, seed=0, C=4, n_nov=2)
    m2 = replace(m, counts={k: v for k, v in m.counts.items() if k != "stack-apply"})
    rep = precheck(m2, {u.unit_id: u for u in units})
    assert not next(c for c in rep.checks if c.name == "counts-named").passed

def test_bands():
    d, band, ok = two_proportion_band(10, 100, 12, 100)
    assert ok and abs(d) < band
    d, band, ok = two_proportion_band(10, 100, 60, 100)
    assert not ok
    assert mean_band([1, 2, 3, 4], [1, 2, 3, 4])[2] and not mean_band([1, 1, 1, 1, 1], [9, 9, 9, 9, 9])[2]


def _stack2(m):
    """``m`` re-labelled rung 1, with every task given a plausible second site.

    The rung is a *label* on the manifest -- it is what enters ``stream_hash`` -- so a
    manifest can claim ``stack2`` while carrying single-site tasks. That is precisely the
    mislabelling ``two-site-at-stack2`` exists to catch, so the fixture builds the honest
    version here and each test breaks it in one place.
    """
    tasks = tuple(replace(t, span2=((t.span[0][0] + 10, 1), (t.span[0][0] + 10, 2))) for t in m.tasks)
    return replace(m, rung="stack2", tasks=tasks)


def test_two_site_check_is_a_no_op_below_stack2():
    # Every rung-0 task has span2 None by construction. The check must pass anyway --
    # dropping its rung guard turns every rung-0 stream red, which is what this pins.
    units, validated = _world(8)
    m = _full_counts(compose(units, validated, seed=0, C=4, n_nov=2))
    assert m.rung == "base" and all(t.span2 is None for t in m.tasks)
    rep = precheck(m, {u.unit_id: u for u in units})
    assert rep.ok
    assert next(c for c in rep.checks if c.name == "two-site-at-stack2").passed


def test_two_site_check_passes_on_an_honest_stack2_manifest():
    units, validated = _world(8)
    m = _stack2(_full_counts(compose(units, validated, seed=0, C=4, n_nov=2)))
    rep = precheck(m, {u.unit_id: u for u in units})
    assert rep.ok, [c for c in rep.checks if not c.passed]


def test_two_site_check_names_a_stack2_task_with_no_second_site():
    # One task's span2 nulled: the report must go not-ok, EXACTLY this check must be the
    # one that failed, and the offending task must be named rather than merely counted.
    units, validated = _world(8)
    m = _stack2(_full_counts(compose(units, validated, seed=0, C=4, n_nov=2)))
    victim = m.tasks[0]
    broken = replace(m, tasks=(replace(victim, span2=None),) + m.tasks[1:])
    rep = precheck(broken, {u.unit_id: u for u in units})
    assert not rep.ok
    failed = [c.name for c in rep.checks if not c.passed]
    assert failed == ["two-site-at-stack2"]
    assert victim.task_key in next(c for c in rep.checks if c.name == "two-site-at-stack2").detail
