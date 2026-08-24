"""Task 13 -- the lens: reduce ``TaskRecord``s (one per arm-attempt) to the success rates
E1/E2 read. Pure -- builds ``TaskRecord`` objects directly, no sandbox -- so UNWRAPPED.

The load-bearing property is honest measurement: ``succ_overall`` is the mean of
``hidden_pass`` over ONLY the records that were actually scored (``hidden_pass is not
None``). Infra / not-measured records are EXCLUDED from the denominator -- never counted
as failures. ``infra_rate`` reports them separately. Each check below is mutation-pinned:

* Counting ``hidden_pass is None`` as ``False`` in ``succ_overall`` (denominator = ALL
  records) breaks ``test_succ_overall_excludes_infra`` -- 2/5 != 2/3. THE KEY MUTATION.
* Dropping the per-kind measured-only filter breaks ``test_per_kind_rates_are_measured_only``
  -- ``second`` becomes 1/2 instead of 1/1.
"""
import json
from dataclasses import fields

import pytest

from crucible.run.lens import ArmLens, build_lens
from crucible.run.records import TaskRecord


def _task(**kw) -> TaskRecord:
    base = dict(
        task_key="k1", arm="A0", unit_id="HumanEval/0", family="fam", phase=1,
        kind="first", landed=True, status="ok", confidence=0.9,
        visible_reward=0.5, executions_charged=3, hidden_pass=True,
        tampered=False, infra_error=None, tokens=128, wall_s=2.5, gpu_s=0.75,
    )
    base.update(kw)
    return TaskRecord(**base)


def _mixed() -> list[TaskRecord]:
    """Five attempts by arm A0. Measured: r1,r2,r3 (2 pass). Infra (None): r4,r5.

    Hand-computed truth:
      succ_overall = 2/3   (2 pass of 3 MEASURED; the 2 None are excluded, not failed)
      infra_rate   = 2/5   (r4 None+infra_error, r5 None)
      succ_phase1  = 1/2   (first: r1 True, r2 False)
      succ_second  = 1/1   (second: r3 True; r5 None excluded)
      succ_novel   = 0/0   -> 0.0 (novel: r4 None excluded -> empty denominator)
      landing_rate = 4/5   (only r4 did not land)
      abstain_rate = 1/5   (only r5 abstained)
    """
    return [
        _task(task_key="k1", kind="first", hidden_pass=True, landed=True, status="ok"),
        _task(task_key="k2", kind="first", hidden_pass=False, landed=True, status="ok"),
        _task(task_key="k3", kind="second", hidden_pass=True, landed=True, status="ok"),
        _task(task_key="k4", kind="novel", hidden_pass=None, landed=False,
              status="infra_error", infra_error="oom"),
        _task(task_key="k5", kind="second", hidden_pass=None, landed=True,
              status="abstain"),
    ]


def test_succ_overall_excludes_infra():
    # THE honest-measurement invariant, and the KEY mutation pin.
    lens = build_lens(_mixed())

    assert lens.succ_overall == pytest.approx(2 / 3)  # NOT 2/5
    assert lens.infra_rate == pytest.approx(2 / 5)
    assert lens.n == 5
    assert lens.arm == "A0"


def test_per_kind_rates_are_measured_only():
    lens = build_lens(_mixed())

    assert lens.succ_phase1 == pytest.approx(1 / 2)
    assert lens.succ_second == pytest.approx(1.0)  # 1/1 -- r5's None is excluded
    assert lens.succ_novel == pytest.approx(0.0)  # empty measured denominator -> 0.0


def test_landing_and_abstain_rates():
    lens = build_lens(_mixed())

    assert lens.landing_rate == pytest.approx(4 / 5)
    assert lens.abstain_rate == pytest.approx(1 / 5)


def test_infra_rate_counts_infra_error_even_when_hidden_pass_set():
    # A record with hidden_pass measured but infra_error set still counts as infra.
    recs = [
        _task(task_key="a", hidden_pass=True, infra_error=None),
        _task(task_key="b", hidden_pass=False, infra_error="flaky sandbox"),
    ]
    lens = build_lens(recs)

    assert lens.infra_rate == pytest.approx(1 / 2)


def test_empty_list_is_all_zero_no_crash():
    lens = build_lens([])

    assert lens.n == 0
    assert lens.arm == ""
    for rate in (lens.succ_overall, lens.succ_phase1, lens.succ_second,
                 lens.succ_novel, lens.landing_rate, lens.abstain_rate,
                 lens.infra_rate):
        assert rate == 0.0


def test_all_measured_all_pass_is_one():
    recs = [_task(task_key=f"k{i}", hidden_pass=True) for i in range(4)]
    lens = build_lens(recs)

    assert lens.succ_overall == pytest.approx(1.0)
    assert lens.infra_rate == pytest.approx(0.0)


def test_build_lens_rejects_mixed_arms():
    recs = [_task(task_key="a", arm="A0"), _task(task_key="b", arm="A1")]
    with pytest.raises(ValueError):
        build_lens(recs)


def test_arm_lens_round_trips_through_dict():
    lens = build_lens(_mixed())
    assert ArmLens.from_dict(lens.to_dict()) == lens


def test_arm_lens_to_dict_is_complete():
    # Dropping any field from to_dict makes this FAIL.
    lens = build_lens(_mixed())
    assert set(lens.to_dict()) == {f.name for f in fields(ArmLens)}


# --- S3 (Task 11): adapter lineage --------------------------------------------------------
#
# ``adapter_ids`` is the DISTINCT set of adapters that stamped this arm's records, in
# first-seen (attempt) order -- the lens's answer to "which adapters served this run".
# ``None`` (the base model, before the first accepted sleep) is not an adapter and never
# appears. Order is attempt order, never sorted, so the sequence reads as the run's
# adapter history.

def test_arm_lens_defaults_adapter_ids_to_empty():
    assert build_lens(_mixed()).adapter_ids == ()


def test_adapter_ids_are_distinct_and_in_first_seen_order():
    recs = [
        _task(task_key="k1", adapter_id=None),
        _task(task_key="k2", adapter_id="ad-b"),
        _task(task_key="k3", adapter_id="ad-b"),
        _task(task_key="k4", adapter_id="ad-a"),
        _task(task_key="k5", adapter_id="ad-b"),
    ]
    assert build_lens(recs).adapter_ids == ("ad-b", "ad-a")


def test_arm_lens_round_trips_with_adapter_ids():
    lens = build_lens([_task(adapter_id="ad-b")])
    back = ArmLens.from_dict(json.loads(json.dumps(lens.to_dict())))
    assert back == lens and back.adapter_ids == ("ad-b",)
