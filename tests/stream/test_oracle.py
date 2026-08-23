import pytest
from crucible.sandbox.exec import ExecResult
from crucible.stream.oracle import OracleError, compute_expected

MOD = "def f(x):\n    if x == 'boom':\n        raise ValueError('no')\n    if x == 'hang':\n        while True: pass\n    return [x, x]\n"


def test_expected_values_and_reasons():
    exp = compute_expected("unit_t", MOD, "f", [[1], ["boom"], ["hang"], [2.5]], per_input_timeout_s=1.0)
    by = {e.index: e for e in exp}
    assert by[0].ok and by[0].value_repr == "[1, 1]"
    assert not by[1].ok and by[1].reason == "raised:ValueError"
    assert not by[2].ok and by[2].reason == "timeout"
    assert by[3].ok and by[3].value_repr == "[2.5, 2.5]"


def test_non_roundtrip_and_long_values_are_dropped():
    mod = "def g(n):\n    if n == 0:\n        return float('nan')\n    return list(range(n))\n"
    exp = compute_expected("unit_g", mod, "g", [[0], [5000]], max_repr=100)
    assert exp[0].reason == "no-roundtrip" and exp[1].reason == "repr-too-long"


def test_driver_crash_raises_oracle_error_not_json_error():
    # The driver prints its result as JSON on stdout; a module that dies at import prints
    # nothing, so a bare json.loads would raise JSONDecodeError with no diagnosis (R-T7-1).
    with pytest.raises(OracleError) as ei:
        compute_expected("unit_bad", "raise RuntimeError('import boom')\n", "f", [[1]])
    err = ei.value
    assert err.module_name == "unit_bad" and err.returncode != 0 and not err.timed_out
    assert "import boom" in err.stderr_tail


def test_driver_wall_cap_raises_oracle_error_flagged_timed_out():
    # A hang at import time is outside the per-input alarm; only the wall cap ends it.
    with pytest.raises(OracleError) as ei:
        compute_expected("unit_slow", "import time\ntime.sleep(120)\n", "f", [[1]], wall_cap_s=3.0)
    assert ei.value.timed_out and ei.value.returncode is None


def test_module_name_must_be_an_identifier():
    # module_name is interpolated into the driver source as an import statement.
    with pytest.raises(ValueError):
        compute_expected("os; import subprocess", MOD, "f", [[1]])


def _fake_exec(stdout: str):
    """Stand in for a sandbox run that returned ``stdout`` cleanly (no subprocess)."""
    def _run(argv, files, **kw):
        return ExecResult(0, stdout, "", 0.01, False, "/nonexistent")
    return _run


def test_truncated_stdout_raises_oracle_error_not_json_error(monkeypatch):
    # execute() reads back at most 1 MiB per stream and appends a note; the JSON is then
    # cut mid-record. That must be a named oracle failure, not a JSONDecodeError.
    cut = '[{"index": 0, "ok": true, "value_repr": "1", "reason": null}, {"index": 1'
    monkeypatch.setattr("crucible.stream.oracle.execute", _fake_exec(cut))
    with pytest.raises(OracleError):
        compute_expected("unit_t", MOD, "f", [[1], [2]])


def test_record_count_mismatch_raises_oracle_error(monkeypatch):
    # Callers zip expected against inputs; a short or misordered result would silently
    # pair one input's expectation with another input's arguments.
    short = '[{"index": 0, "ok": true, "value_repr": "1", "reason": null}]'
    monkeypatch.setattr("crucible.stream.oracle.execute", _fake_exec(short))
    with pytest.raises(OracleError):
        compute_expected("unit_t", MOD, "f", [[1], [2]])
