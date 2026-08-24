"""Round-trip and completeness tests for the S2 record schema.

The load-bearing property is honest measurement: ``hidden_pass`` is ``bool | None`` and
``None`` -- an attempt that was never scored -- must survive both the JSON round-trip and
the real filesystem round-trip WITHOUT being coerced to ``False``. Dropping the field from
``to_dict`` or coercing ``None`` in ``from_dict`` is what the mutation check breaks.
"""
import json
from dataclasses import fields

import pytest

from crucible.run.records import (
    ExecRecord,
    TaskRecord,
    read_task_records,
    write_records,
)


def _exec(**kw) -> ExecRecord:
    base = dict(
        task_key="k1", arm="A0", node_id="n7", visible_reward=0.5,
        charged=True, wall_s=1.25, infra_error=None,
    )
    base.update(kw)
    return ExecRecord(**base)


def _task(**kw) -> TaskRecord:
    base = dict(
        task_key="k1", arm="A0", unit_id="HumanEval/0", family="fam", phase=1,
        kind="first", landed=True, status="ok", confidence=0.9,
        visible_reward=0.5, executions_charged=3, hidden_pass=True,
        tampered=False, infra_error=None, tokens=128, wall_s=2.5, gpu_s=0.75,
    )
    base.update(kw)
    return TaskRecord(**base)


def test_exec_record_round_trips_through_json():
    e = _exec()
    assert ExecRecord.from_dict(json.loads(json.dumps(e.to_dict()))) == e


def test_exec_record_allows_infra_error():
    e = _exec(charged=False, infra_error="oom", visible_reward=0.0)
    back = ExecRecord.from_dict(json.loads(json.dumps(e.to_dict())))
    assert back == e and back.infra_error == "oom" and back.charged is False


def test_task_record_round_trips_through_json():
    t = _task()
    assert TaskRecord.from_dict(json.loads(json.dumps(t.to_dict()))) == t


@pytest.mark.parametrize("hp", [True, False, None])
def test_task_record_hidden_pass_round_trips_through_json(hp):
    t = _task(hidden_pass=hp)
    back = TaskRecord.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back == t
    assert back.hidden_pass is hp  # None must NOT become False


def test_task_record_carries_tamper_and_optional_fields():
    t = _task(tampered=True, infra_error="sandbox died", tokens=None, gpu_s=None,
              hidden_pass=None, status="infra_error", landed=False)
    back = TaskRecord.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back == t
    assert back.tampered is True and back.tokens is None and back.gpu_s is None


def test_exec_record_to_dict_is_complete():
    assert set(_exec().to_dict()) == {f.name for f in fields(ExecRecord)}


def test_task_record_to_dict_is_complete():
    # Dropping hidden_pass from to_dict makes this FAIL (mutation check, Step 5).
    assert set(_task().to_dict()) == {f.name for f in fields(TaskRecord)}


def test_write_and_read_records_round_trip_through_filesystem(tmp_path):
    task_recs = [
        _task(),
        _task(task_key="k2", hidden_pass=None, status="infra_error",
              infra_error="oom", tokens=None, gpu_s=None, landed=False),
        _task(task_key="k3", hidden_pass=False, tampered=True),
    ]
    exec_recs = [_exec(), _exec(task_key="k2", charged=False, infra_error="oom")]

    write_records(tmp_path, task_recs, exec_recs)

    assert (tmp_path / "task_records.jsonl").exists()
    assert (tmp_path / "exec_records.jsonl").exists()

    back = read_task_records(tmp_path)
    assert back == task_recs
    # The honest-measurement invariant survives the disk trip.
    assert back[1].hidden_pass is None
    assert back[2].hidden_pass is False


def test_written_lines_are_sorted_key_json(tmp_path):
    write_records(tmp_path, [_task()], [_exec()])
    line = (tmp_path / "task_records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert line == json.dumps(json.loads(line), sort_keys=True)


# --- S3 (Task 11): the two trailing fields A_full stamps ---------------------------------
#
# ``retrieved_ids`` is the memory column E1 is measured on (empty tuple <=> no-hit, the
# None-vs-zero discipline applied to a sequence) and ``adapter_id`` is the sleep lineage the
# lens reduces. Both are TRAILING and DEFAULTED, so every S2-era positional construction --
# and every S2-era jsonl line, written before either field existed -- still means what it
# meant. ``test_task_record_to_dict_is_complete`` above catches a field dropped from
# ``to_dict`` automatically; these pin the tuple shape and the backward-compatible read.

def test_task_record_defaults_the_s3_fields_to_empty_and_none():
    t = _task()
    assert t.retrieved_ids == () and t.adapter_id is None


def test_task_record_s3_fields_round_trip_through_json():
    t = _task(retrieved_ids=("sem-a", "sem-b"), adapter_id="ad-0123456789abcdef")
    back = TaskRecord.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back == t
    assert back.retrieved_ids == ("sem-a", "sem-b")   # a tuple, not the list JSON carries


def test_task_record_reads_an_s2_era_line_that_predates_the_s3_fields():
    d = _task().to_dict()
    d.pop("retrieved_ids"); d.pop("adapter_id")
    back = TaskRecord.from_dict(d)
    assert back.retrieved_ids == () and back.adapter_id is None
    assert back.hidden_pass is True                   # nothing else shifted


def test_s3_fields_survive_the_filesystem_round_trip(tmp_path):
    recs = [_task(retrieved_ids=("sem-a",), adapter_id="ad-1"),
            _task(task_key="k2", retrieved_ids=(), adapter_id=None)]
    write_records(tmp_path, recs, [])
    assert read_task_records(tmp_path) == recs
