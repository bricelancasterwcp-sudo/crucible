"""Expected outputs of a canonical solution, computed by running it inside the sandbox.

The oracle is the only place a *value* enters the instrument. Everything downstream --
the generated test file, and therefore every kill and every survival -- is derived from
what this module says the canonical returned, so the rules here are about refusing to
guess.

Three of them are load-bearing:

*One subprocess per unit, one alarm per input.* All inputs for a unit are driven by a
single sandboxed process (``driver.py``), which arms ``signal.setitimer`` around each
call. A unit that hangs on input 7 therefore costs one timeout, not one process, and the
other inputs are still measured.

*Only values that survive ``eval(repr(v)) == v`` become tests.* The generated test file
compares against a literal, so a value whose ``repr`` does not reconstruct it (``nan``,
an object with a default ``<... at 0x...>`` repr) would produce a test that fails for the
canonical itself. Such inputs are dropped with a reason (``no-roundtrip``), never guessed
at. The same applies to a ``repr`` longer than ``max_repr`` (``repr-too-long``).

*A driver that produced no result is not a measurement.* ``execute()`` reads back at most
1 MiB of stdout, kills the process group on the wall cap, and returns rc=127 for a missing
interpreter -- so the JSON this module parses can be absent, truncated or preceded by a
crash. Any of those raise :class:`OracleError` (ruling R-T7-1) carrying what was known at
the time, rather than fabricating per-input reasons the driver never reported or letting a
bare ``JSONDecodeError`` escape. The caller (``stream/build.py``) drops the unit.
"""
from __future__ import annotations

import json
import keyword
import sys
from dataclasses import dataclass

from ..sandbox.exec import execute

# Runs inside the sandbox: reads inputs.json, calls the entry point once per input, and
# prints one JSON record per input. Kept as a template (not a file) so the module name,
# entry point, timeout and repr cap are frozen into the child at build time.
_DRIVER = r'''
import json, signal, sys, math
import {module} as M
fn = getattr(M, {entry!r})
inputs = json.load(open("inputs.json"))
TO = {timeout}
MAXR = {max_repr}
def _alarm(*a): raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
out = []
for i, args in enumerate(inputs):
    rec = {{"index": i, "ok": False, "value_repr": None, "reason": None}}
    try:
        signal.setitimer(signal.ITIMER_REAL, TO)
        try:
            v = fn(*args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        r = repr(v)
        if len(r) > MAXR:
            rec["reason"] = "repr-too-long"
        else:
            try:
                back = eval(r)
                same = (back == v) if not isinstance(v, float) else (back == v and not math.isnan(v))
            except Exception:
                same = False
            if same:
                rec["ok"] = True; rec["value_repr"] = r
            else:
                rec["reason"] = "no-roundtrip"
    except TimeoutError:
        rec["reason"] = "timeout"
    except BaseException as e:
        rec["reason"] = "raised:" + type(e).__name__
    out.append(rec)
print(json.dumps(out))
'''

_STDERR_TAIL = 400


class OracleError(RuntimeError):
    """The driver produced no usable result, so *no* input of this unit was measured.

    Distinct from a per-input reason: those describe a measured call. This describes the
    instrument failing, and carries what was observable (``returncode``, ``timed_out``,
    the stderr tail) so the caller can name the drop instead of inferring it.
    """

    def __init__(self, module_name: str, returncode: int | None, timed_out: bool,
                 stderr_tail: str, detail: str) -> None:
        super().__init__(f"oracle driver for {module_name!r} produced no result "
                         f"(rc={returncode}, timed_out={timed_out}): {detail} | stderr: {stderr_tail}")
        self.module_name = module_name
        self.returncode = returncode
        self.timed_out = timed_out
        self.stderr_tail = stderr_tail
        self.detail = detail


@dataclass(frozen=True)
class Expected:
    """What the canonical did for one input: a usable value, or why there is none."""

    index: int
    ok: bool
    value_repr: str | None
    reason: str | None


def _check_module_name(module_name: str) -> None:
    """Reject a name that cannot be an ``import`` target in the driver source.

    ``module_name`` is interpolated into the driver text (the entry point is not -- it
    goes in as a ``repr``), so a non-identifier is either a crash or an injection. A
    caller bug, so it raises rather than returning drops.
    """
    if not module_name.isidentifier() or keyword.iskeyword(module_name):
        raise ValueError(f"module_name must be a Python identifier and not a keyword; got {module_name!r}")


def compute_expected(module_name: str, module_src: str, entry_point: str, inputs: list[list], *,
                     per_input_timeout_s: float = 5.0, wall_cap_s: float = 60.0,
                     max_repr: int = 2000) -> list[Expected]:
    """Run ``entry_point`` on every input inside the sandbox and report what came back.

    One :class:`Expected` per input, in input order. Raises :class:`OracleError` when the
    driver itself failed (crash, wall cap, truncated or unparseable stdout) -- that is an
    unmeasured unit, not a unit whose inputs all failed.
    """
    _check_module_name(module_name)
    driver = _DRIVER.format(module=module_name, entry=entry_point, timeout=per_input_timeout_s,
                            max_repr=max_repr)
    files = {f"{module_name}.py": module_src, "driver.py": driver, "inputs.json": json.dumps(inputs)}
    res = execute([sys.executable, "driver.py"], files, wall_cap_s=wall_cap_s)
    if res.timed_out or res.returncode != 0:
        raise OracleError(module_name, res.returncode, res.timed_out, res.stderr[-_STDERR_TAIL:],
                          "driver did not exit cleanly")
    return _parse(module_name, res.stdout, res.stderr, len(inputs))


def _parse(module_name: str, stdout: str, stderr: str, n_inputs: int) -> list[Expected]:
    """Decode the driver's JSON, refusing anything that is not one record per input.

    Callers zip these against the inputs, so a short, long or misordered result would
    silently attach one input's expected value to another input's arguments.
    """
    tail = stderr[-_STDERR_TAIL:]
    try:
        records = json.loads(stdout)
    except ValueError as exc:  # JSONDecodeError: empty, noisy or 1 MiB-truncated stdout
        raise OracleError(module_name, 0, False, tail, f"stdout is not JSON: {exc}") from exc
    if not isinstance(records, list) or len(records) != n_inputs:
        raise OracleError(module_name, 0, False, tail,
                          f"expected {n_inputs} records, got {len(records) if isinstance(records, list) else type(records).__name__}")
    try:
        expected = [Expected(r["index"], r["ok"], r["value_repr"], r["reason"]) for r in records]
    except (TypeError, KeyError) as exc:
        raise OracleError(module_name, 0, False, tail, f"malformed record: {exc}") from exc
    if [e.index for e in expected] != list(range(n_inputs)):
        raise OracleError(module_name, 0, False, tail, "records are not in input order")
    return expected
