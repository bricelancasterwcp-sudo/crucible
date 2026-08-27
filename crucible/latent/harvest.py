"""Sensorium-backed harvest runner for the B-lite corpus (prereg §4/§5.1).

`harvest(function_src, args_literal, workdir)` runs one (function, args)
sample TWICE, each time as its own `sensorium run --focus <module>:<fn>`
subprocess (resource-limited, wall-clock-capped), and reads each run's
SQLite trace back through sensorium's own `store` package -- never by
re-parsing stdout/stderr, and never by guessing at the trace schema.

Honesty rules this module exists to enforce (see docs/findings and
`rigorous-experiments`):

* An outcome the recorder could not measure is reported as such, never
  invented. A subprocess that blows its wall clock comes back as
  `outcome="timeout"`, not as a return value that happens to be missing.
* Truncation is counted, not swallowed. `truncated=True` whenever sensorium
  itself marked a captured value truncated/unread (`meta["truncated_count"]
  > 0`), whenever our own MAX_SNAPSHOTS cap dropped trailing snapshots,
  whenever our own 64-char `VALUE_REPR_CAP` cuts a locals repr sensorium
  itself did not already truncate (a rendering choice beyond what the brief
  states, made for the same reason: a value_repr this module cut is a
  value_repr this module truncated, whichever cap did it), or whenever the
  run timed out (its state is inherently partial).
* Determinism is actually checked by running twice and comparing, not
  assumed. Two runs disagreeing on `outcome` OR on the hash of
  (return value, snapshot sequence) come back `deterministic=False`. The
  hash covers the return value as well as the snapshot sequence
  deliberately: a function whose only statement is `return random.random()`
  binds no local variable, so sensorium's --focus tier (which reports
  per-line LOCAL DELTAS, see `sensorium.record.tracer`) emits zero LINE
  events for it on any run -- the snapshot sequence alone is identically
  empty both times regardless of the return value's randomness. Hashing
  the snapshot sequence alone would silently pass a genuinely
  nondeterministic sample as deterministic, which is exactly the failure
  mode "nondeterminism is detected not ignored" bans.

Reading sensorium's own store schema (worth restating here since it drives
every reader function below): CALL/RETURN/RAISE/UNWIND events are recorded
unconditionally for any traced code (any file under the subprocess's cwd,
minus stdlib/site-packages) -- so the outcome and return value are visible
even for a sample function with no --focus match. LINE events (per-line
LOCAL DELTAS, not full snapshots) are ONLY emitted for code objects that
match `--focus module:qualname`, and only when something actually changed
since the frame's previous LINE event -- see `_build_snapshots` for how the
full per-line locals state is folded back out of those deltas.

Isolation across calls (round-3 CRITICAL fix, found by another
implementer's real-harvest smoke and confirmed against production data --
`runs/blite-corpus/samples.jsonl`'s 1000 rows were all byte-identical
copies of the corpus's first successful harvest): every call gets its OWN
subdirectory of the caller's `workdir` (`workdir/call-<uuid4 hex>`), used
for the script, `SENSORIUM_DIR`, the subprocess cwd, and the trace read --
so callers may keep passing ONE shared scratch directory across thousands
of calls (the real usage in `crucible.latent.gen`'s corpus-generation
passes) without their sensorium run-ids (fixed constants, "run-a"/"run-b")
ever colliding. Independently, `_execute_once` checks the subprocess's exit
code and raises `HarvestError` rather than falling through to a store read
on anything outside {0, 1} (0 = clean target return, 1 = the target's own
uncaught exception -- `sensorium run` reports that via the same exit code,
so this is NOT a bare "nonzero" check, which would misfire on every
legitimate exception-outcome sample) -- belt-and-suspenders: even if a
future refactor ever reintroduces a shared run-id by some other path, a
refused/failed `sensorium run` can no longer be silently read as a stale
prior result.

Containment (read before trusting this as a sandbox): the recorded
subprocess is NOT filesystem-sandboxed. `_preexec_rlimit` caps RLIMIT_AS
(virtual memory) only, and there is no RLIMIT_FSIZE cap here the way
`crucible.sandbox.exec`'s untrusted-code path has one -- because sensorium
itself must be free to write its own SQLite trace file into `workdir`
during the run, and capping file size there risks truncating the very
trace this module depends on to report anything at all. So a sample body
that does `open(path, "w")`, `shutil.rmtree`, or similar writes to the real
filesystem, unbounded, for as long as `EXEC_TIMEOUT_S` allows -- rlimits
here cover memory and wall clock, nothing else. Containment for this
harvester rests entirely on Task 2's corpus validator rejecting samples
that reference filesystem/network/process builtins BY NAME before they
ever reach `harvest()`, not on anything in this module. The threat model is
therefore "our own generator's output, occasionally buggy," not an
adversarial sample; if the B-lite corpus ever has to ingest
external/untrusted code, this module is not sufficient on its own and
`crucible.sandbox.exec`'s fuller isolation (network block, FSIZE cap,
process-group kill) should be used instead.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from sensorium.store.reader import Trace

from crucible.latent.config import (
    EXEC_RLIMIT_AS_MB,
    EXEC_TIMEOUT_S,
    MAX_SNAPSHOTS,
    VALUE_REPR_CAP,
)

_RUNNER_FILENAME = "harvest_target.py"
_RUN_ID_A = "run-a"
_RUN_ID_B = "run-b"


@dataclass(frozen=True)
class Snapshot:
    """One LINE event's fully-folded locals state, name-sorted."""

    line: int
    locals: tuple[tuple[str, str, str], ...]   # (name, type_name, value_repr<=64ch)


@dataclass(frozen=True)
class HarvestResult:
    outcome: str                     # "return" | "exception:<TypeName>" | "timeout"
    return_repr: str | None          # repr of the return value, None unless outcome=="return"
    snapshots: tuple[Snapshot, ...]  # per-line locals states, in execution order, truncated to MAX_SNAPSHOTS
    truncated: bool                  # snapshots dropped beyond MAX_SNAPSHOTS, a value_repr cut at
                                      # VALUE_REPR_CAP, sensorium marked truncation, OR a timeout
    deterministic: bool              # two independent recorded runs agreed (outcome + state-sequence hash)


class HarvestError(RuntimeError):
    """Environment problem, not a sample-execution outcome (e.g. no sensorium console script)."""


def harvest(function_src: str, args_literal: str, workdir: Path) -> HarvestResult:
    """Run one (function_src, args_literal) sample twice and report what happened.

    `function_src` must define exactly one top-level function; it is written
    into `workdir` verbatim (no validation beyond `ast.parse` to find its
    name -- the no-import/no-syntax-error validator is Task 2's job, and
    harvest stays permissive so the nondeterminism sample, which legitimately
    imports `random`, still runs).

    `workdir` is resolved to an ABSOLUTE path first thing (round-2 live-fire
    fix, found via the corpus run's audit trail): the `sensorium run`
    subprocess's cwd IS `workdir`, so a RELATIVE `workdir` made every path
    derived from it -- the script path, `SENSORIUM_DIR`, the trace path this
    function itself checks afterward -- ambiguous about which process's cwd
    it resolves against. A relative `SENSORIUM_DIR` inherited by the child
    resolves a SECOND time against the child's own (already-workdir) cwd at
    the moment sensorium's `paths.trace_root()` creates it, landing the
    store at `workdir/workdir/.sensorium` -- a path `_execute_once`'s
    `trace_path.exists()` check, run back in THIS process against THIS
    process's (unrelated) cwd, never looks at. Every real call site in this
    repo already passed an absolute path (pytest's `tmp_path`), which is why
    this went unnoticed until a live corpus run passed a path relative to
    its own launch directory.

    Every call additionally gets its OWN subdirectory of `workdir`
    (`call-<uuid4 hex>`; round-3 CRITICAL fix, see the module docstring) --
    a plain `uuid4` is infrastructure, not measurement, so no determinism
    property of this module depends on it. `workdir` itself may therefore be
    shared safely across thousands of `harvest()` calls (the real usage:
    `crucible.latent.gen`'s corpus-generation passes call this against one
    scratch directory for an entire multi-hour run) without their fixed
    "run-a"/"run-b" sensorium run-ids ever colliding. The per-call
    subdirectory is removed again once this call returns (`finally`,
    best-effort) -- cleanup beyond the fix itself, added because a
    multi-hour run sharing one `workdir` would otherwise accumulate one
    subdirectory (script + sensorium store) per call forever.
    """
    workdir = Path(workdir).resolve()
    call_dir = workdir / f"call-{uuid4().hex}"
    call_dir.mkdir(parents=True, exist_ok=True)
    try:
        fname = _function_name(function_src)
        script_path = _write_runner_script(function_src, fname, args_literal, call_dir)
        focus = f"{script_path.stem}:{fname}"
        sensorium_dir = call_dir / ".sensorium"

        run_a = _execute_once(script_path, focus, call_dir, sensorium_dir, _RUN_ID_A, fname)
        run_b = _execute_once(script_path, focus, call_dir, sensorium_dir, _RUN_ID_B, fname)

        outcome_a, return_repr_a, snapshots_a, truncated_a = run_a
        outcome_b, return_repr_b, snapshots_b, _truncated_b = run_b
        deterministic = (
            outcome_a == outcome_b
            and _state_hash(return_repr_a, snapshots_a) == _state_hash(return_repr_b, snapshots_b)
        )
        return HarvestResult(
            outcome=outcome_a,
            return_repr=return_repr_a,
            snapshots=snapshots_a,
            truncated=truncated_a,
            deterministic=deterministic,
        )
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)


# -- sample script construction ---------------------------------------------
def _function_name(function_src: str) -> str:
    tree = ast.parse(function_src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    raise HarvestError("function_src defines no top-level function")


def _write_runner_script(function_src: str, fname: str, args_literal: str, workdir: Path) -> Path:
    """Define the function, read nothing, print nothing, call it.

    `args_literal` is embedded via `repr()` so it lands as a Python string
    LITERAL in the generated script; the script itself decodes it with
    `ast.literal_eval` at run time (never `eval`), exactly as the brief's
    mechanics specify.
    """
    body = function_src if function_src.endswith("\n") else function_src + "\n"
    body += "import ast\n"
    body += f"{fname}(*ast.literal_eval({args_literal!r}))\n"
    path = workdir / _RUNNER_FILENAME
    path.write_text(body)
    return path


# -- subprocess execution -----------------------------------------------------
def _sensorium_exe() -> str:
    """The `sensorium` console script for THIS interpreter's venv.

    Resolved next to `sys.executable` first (works whether or not the venv's
    bin/ is on PATH) and falls back to a PATH search.
    """
    candidate = Path(sys.executable).with_name("sensorium")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("sensorium")
    if found:
        return found
    raise HarvestError(
        "no `sensorium` console script found next to "
        f"{sys.executable} or on PATH -- run "
        "`.venv/bin/pip install -e /path/to/sensorium` (or the uv "
        "equivalent) first")


def _preexec_rlimit() -> None:
    """preexec_fn: cap the recorded subprocess's virtual address space."""
    limit = EXEC_RLIMIT_AS_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _execute_once(script_path: Path, focus: str, workdir: Path, sensorium_dir: Path,
                   run_id: str, fname: str) -> tuple[str, str | None, tuple[Snapshot, ...], bool]:
    """One `sensorium run` subprocess. Returns (outcome, return_repr, snapshots, truncated)."""
    env = os.environ.copy()
    # Scoped inside workdir, deliberately: without this every harvest() call
    # would write into the caller's real ~/.sensorium/traces and never clean
    # up. workdir's own lifecycle (tmp_path in tests, the corpus builder's
    # scratch dir in later tasks) is what bounds these files.
    env["SENSORIUM_DIR"] = str(sensorium_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [_sensorium_exe(), "run", "--focus", focus, "--run-id", run_id,
           "--", str(script_path)]
    try:
        proc = subprocess.run(
            cmd, cwd=str(workdir), env=env, preexec_fn=_preexec_rlimit,
            timeout=EXEC_TIMEOUT_S, capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        # The subprocess is `sensorium run`, and the recorded target runs
        # IN that same process (boot.py uses runpy, not a further fork) --
        # so killing it on timeout kills the hang directly. Its trace is
        # necessarily partial (uninstall/finalize never ran -- meta stays
        # "incomplete": true and no events were flushed for a batch under
        # sensorium's 512-row threshold), so it is not worth opening: report
        # the timeout itself rather than an empty snapshot sequence dressed
        # up as a measurement.
        return "timeout", None, (), True

    # Round-3 CRITICAL fix: a refused/failed `sensorium run` must never fall
    # through to a store read. 0 is a clean target return and 1 is the
    # TARGET's own uncaught exception -- `sensorium run` reports that via
    # its own exit code too (`boot.run_target`'s `except BaseException:
    # exit_status = 1`, verified live), so both are legitimate and the
    # store is read normally for either. Anything else -- 2
    # (`boot.TargetError`: an invalid run id, an unresolvable target, OR a
    # run-id COLLISION when a trace already exists at this path -- the
    # exact bug that silently replayed call 1's result for every call
    # after it, pre-fix), a negative returncode (killed by signal), or any
    # other value -- means `sensorium run` itself refused or crashed before
    # producing a trustworthy trace, and `trace_path.exists()` below cannot
    # tell a fresh trace from a stale one left by a PRIOR call that used
    # this same path. `stderr[-200:]` names the failure without risking an
    # unbounded message (sensorium's own error lines are short, but a crash
    # inside the interpreter itself is not bounded by anything sensorium
    # controls).
    if proc.returncode not in (0, 1):
        tail = (proc.stderr or "")[-200:]
        raise HarvestError(
            f"sensorium run exited {proc.returncode} for run_id={run_id!r}: {tail}")

    trace_path = sensorium_dir / "traces" / f"{run_id}.db"
    if not trace_path.exists():
        # Process exited (any status) without ever creating a trace file --
        # unreachable via boot.run_target's own contract short of the
        # recorder failing to install monitoring at all. Honest, not hidden.
        raise HarvestError(f"sensorium produced no trace at {trace_path}")

    trace = Trace.open(trace_path)
    try:
        return _read_result(trace, fname)
    finally:
        trace._c.close()   # Trace exposes no public close(); this is its own connection.


def _read_result(trace: Trace, fname: str) -> tuple[str, str | None, tuple[Snapshot, ...], bool]:
    meta = trace.meta
    if meta.get("incomplete", True):
        # The recorder itself never finished cleanly (e.g. the RLIMIT_AS cap
        # aborted the interpreter rather than raising a catchable
        # MemoryError). Not literally OUR wall-clock timeout, but the same
        # honest answer applies: this run's outcome is not something we
        # measured, so it is not reported as one.
        return "timeout", None, (), True

    # Frame/snapshot lookup happens ONCE, ahead of the outcome branch, and is
    # shared by both the exception and return paths below (final review
    # CRITICAL): an `exception:*` outcome still keeps the LINE-event
    # snapshots recorded before the raise -- sensorium's own tracer fires a
    # frame's LINE event for a line BEFORE that line executes (see
    # `_build_snapshots`' docstring and `harvest_target.py`'s runner), so the
    # snapshot for the very line that raises still reflects real pre-raise
    # locals, not anything invented after the fact. Discarding these
    # unconditionally on the exception path (the pre-fix behavior) made
    # `outcome != "return"` synonymous with an EMPTY snapshot sequence --
    # label 0 (`crucible.latent.gen.binary_label`) would then be readable
    # straight off `len(snapshots) == 0`, independent of anything the model
    # is supposed to have learned. `return_repr` stays `None` for an
    # exception regardless (there is no return value to report).
    code_id = _code_id_for(trace, fname)
    frame = _first_frame(trace, code_id)
    snapshots, snap_truncated = _build_snapshots(trace, frame.id if frame is not None else None)
    truncated = snap_truncated or _meta_truncated(meta)

    uncaught = meta.get("uncaught")
    if uncaught is not None:
        return f"exception:{uncaught['type']}", None, snapshots, truncated

    return_repr = None
    if frame is not None and frame.return_event_id is not None:
        ev = trace.event(frame.return_event_id)
        if ev is not None and ev.payload is not None:
            return_repr = _capture_repr(ev.payload.get("value", {"k": "none"}))

    return "return", return_repr, snapshots, truncated


def _meta_truncated(meta: dict) -> bool:
    """sensorium's own truncation counter for this run (capture.capture_stats,
    diffed across the run by boot._finalize_meta) -- values sensorium itself
    could not fully capture, in ANY event, not just the ones harvest reads."""
    return bool(meta.get("truncated_count"))


def _code_id_for(trace: Trace, fname: str) -> int | None:
    for c in trace.codes():
        if c.qualname == fname:
            return c.id
    return None


def _first_frame(trace: Trace, code_id: int | None):
    if code_id is None:
        return None
    frames = trace.frames(code_id=code_id)
    # frames() is ORDER BY id -- the first activation, i.e. the outermost
    # call for a non-recursive sample, which is every sample harvest() runs.
    return frames[0] if frames else None


# -- LINE-delta folding --------------------------------------------------------
def _build_snapshots(trace: Trace, frame_id: int | None) -> tuple[tuple[Snapshot, ...], bool]:
    """Fold sensorium's per-line locals DELTAS back into full per-line state.

    Each LINE event carries only `deltas` (names whose value changed since
    the frame's previous LINE event) and `unbound` (names that went out of
    scope) -- see the module docstring on tracer.py's `_on_line`. This keeps
    a running `cur` dict and emits one full, name-sorted Snapshot per event.

    Two independent truncation sources are folded into the single
    `truncated` flag this returns, on top of what sensorium itself marks
    (`_capture_marks_truncated`, from sensorium's own 200-char str/repr and
    8-item sample caps): our own MAX_SNAPSHOTS cap (dropped trailing
    snapshots, applied once at the end) and our own `VALUE_REPR_CAP`
    (64 chars) cutting a `value_repr` sensorium's own caps left intact --
    e.g. a 200-char string is under sensorium's 200-char str cap but its
    quoted repr is over 64 chars, so a value_repr this module itself had to
    cut is reported as truncated the same as one sensorium already cut,
    even though the brief's mechanics note only spells out sensorium's own
    marker, the MAX_SNAPSHOTS cap, and the wall-clock timeout.
    """
    if frame_id is None:
        return (), False
    events = trace.events(kind="LINE", frame_id=frame_id)
    cur: dict[str, dict] = {}
    snapshots: list[Snapshot] = []
    truncated = False
    for ev in events:
        payload = ev.payload or {}
        if payload.get("unread"):
            # The whole frame's locals could not be read at this line at
            # all (a hostile f_locals mapping) -- sensorium still wrote the
            # event to say so; `cur` is left as-is (see tracer.py).
            truncated = True
        for name, cap in payload.get("deltas", {}).items():
            cur[name] = cap
            if _capture_marks_truncated(cap):
                truncated = True
        for name in payload.get("unbound", []):
            cur.pop(name, None)

        entries = []
        for name in sorted(cur):
            cap = cur[name]
            full_repr = _capture_repr(cap)
            value_repr = full_repr[:VALUE_REPR_CAP]
            if len(full_repr) > VALUE_REPR_CAP:
                truncated = True
            entries.append((name, _capture_type_name(cap), value_repr))
        snapshots.append(Snapshot(line=ev.line, locals=tuple(entries)))

    if len(snapshots) > MAX_SNAPSHOTS:
        snapshots = snapshots[:MAX_SNAPSHOTS]
        truncated = True
    return tuple(snapshots), truncated


# -- capture-dict interpretation ----------------------------------------------
# sensorium's `capture_value` (sensorium.record.capture) hands back a small
# tagged dict per value -- {"k": "none"|"bool"|"num"|"str"|"seq"|"map"|"obj"|
# "unread", ...} -- never a live Python object (that is the whole point: the
# recorder must never hold a reference back into the observed program). The
# functions below turn that tagged dict into the (type_name, value_repr) pair
# harvest reports, entirely from what sensorium already captured -- nothing
# here re-derives a value sensorium did not hand back.
def _capture_type_name(cap: dict) -> str:
    k = cap.get("k")
    if k == "none":
        return "NoneType"
    if k == "bool":
        return "bool"
    if k == "num":
        return "float" if isinstance(cap.get("v"), float) else "int"
    if k == "str":
        return "str"
    return cap.get("type", k or "?")   # seq / map / obj / unread all carry "type"


def _capture_repr(cap: dict) -> str:
    """Best-effort `repr()` reconstruction from a sensorium capture dict.

    Exact for none/bool/num/str (sensorium hands back the real value for
    these). For seq/map, sensorium only ever captured a `len` and a sample
    of up to CAPS["sample"] items (never the full container), so the repr
    built here is itself a sample -- marked by an appended `...` exactly
    when sensorium's own `trunc` flag says the sample fell short of `len`.
    For obj, sensorium already computed `repr()` (capped at 200 chars) at
    capture time; that string is used as-is.
    """
    k = cap.get("k")
    if k == "none":
        return "None"
    if k in ("bool", "num"):
        return repr(cap.get("v"))
    if k == "str":
        r = repr(cap.get("v", ""))
        return r + "..." if cap.get("trunc") else r
    if k == "obj":
        return cap.get("repr", f"<{cap.get('type', '?')}>")
    if k == "seq":
        return _seq_repr(cap)
    if k == "map":
        return _map_repr(cap)
    if k == "unread":
        return f"<{cap.get('type', '?')} unread>"
    return repr(cap)


_SEQ_BRACKETS = {"list": "[]", "tuple": "()", "set": "{}", "frozenset": "{}"}


def _seq_repr(cap: dict) -> str:
    t = cap.get("type", "list")
    sample = cap.get("sample")
    if sample is None:
        return f"<{t} len={cap.get('len')}>"
    items = ", ".join(_capture_repr(x) for x in sample)
    if t == "tuple" and len(sample) == 1 and not cap.get("trunc"):
        items += ","
    open_c, close_c = _SEQ_BRACKETS.get(t, "[]")
    trailer = ", ..." if cap.get("trunc") else ""
    return f"{open_c}{items}{trailer}{close_c}"


def _map_repr(cap: dict) -> str:
    sample = cap.get("sample")
    if sample is None:
        return f"<dict len={cap.get('len')}>"
    items = ", ".join(f"{_capture_repr(k)}: {_capture_repr(v)}" for k, v in sample)
    trailer = ", ..." if cap.get("trunc") else ""
    return f"{{{items}{trailer}}}"


def _capture_marks_truncated(cap) -> bool:
    if not isinstance(cap, dict):
        return False
    if cap.get("trunc") or cap.get("unread"):
        return True
    if cap.get("k") == "seq":
        return any(_capture_marks_truncated(x) for x in cap.get("sample") or [])
    if cap.get("k") == "map":
        return any(_capture_marks_truncated(k) or _capture_marks_truncated(v)
                   for k, v in cap.get("sample") or [])
    return False


# -- determinism ---------------------------------------------------------------
def _state_hash(return_repr: str | None, snapshots: tuple[Snapshot, ...]) -> str:
    payload = [return_repr, [[s.line, [list(row) for row in s.locals]] for s in snapshots]]
    blob = json.dumps(payload, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
