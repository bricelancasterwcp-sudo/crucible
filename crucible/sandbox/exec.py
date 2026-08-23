"""Isolated subprocess execution: the one place crucible runs untrusted code.

Pattern borrowed from mini-swe-agent's LocalEnvironment (MIT): subprocess + timeout, but with
a fresh workdir, socket blocking, resource limits, file-backed capture and process-group kill.

Threat model (S1, ruling R-T2-3)
--------------------------------
The network block is a Python-level shim (``sitecustomize.py`` on ``PYTHONPATH``), not OS
isolation. It stops *accidental* network use by generated single-function code -- an
``import requests``, a stray ``socket.create_connection`` -- which is the failure mode S1
actually has. It is not an adversary barrier: a unit that shells out to ``curl``, or calls
``connect`` through ``ctypes``, still reaches the network. No S1 code path produces such a
unit (units are single functions the proposer writes, run by pytest). OS-level isolation
(network namespace / bubblewrap) is deferred -- see ``docs/CARRIED-DEBT.md``.

Similarly, a grandchild that calls ``setsid()`` leaves the process group and survives the
wall-cap kill. Since output is captured to files rather than inherited pipes it can no
longer stall ``execute()`` (that was ruling R-T2-1), but it is not reaped; killing an
escaped subtree needs a cgroup or PID namespace, deferred with the same ruling.

Capture is bounded twice: ``RLIMIT_FSIZE`` (16 MiB) stops the child at the kernel, and only
the first 1 MiB of each stream is read back into memory, decoded with ``errors="replace"``
so untrusted bytes can never raise inside the instrument.
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass

SITECUSTOMIZE_SRC = '''\
# crucible sandbox: block outbound network without breaking socket class hierarchy.
import socket as _s
def _blocked(*a, **k):
    raise OSError("network disabled in crucible sandbox")
_s.socket.connect = _blocked
_s.socket.connect_ex = _blocked
_s.create_connection = _blocked
_s.getaddrinfo = _blocked
'''

_CLEAN_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TERM")

# Reserved filenames inside the workdir: stdout/stderr are captured to files, not pipes.
_STDOUT_NAME = ".crucible-stdout"
_STDERR_NAME = ".crucible-stderr"

# Kernel-side cap on anything the child writes, capture files included.
_FSIZE_CAP_BYTES = 16 << 20
# How much of each capture file is read back into the caller's memory.
_READ_CAP_BYTES = 1 << 20
_TRUNCATION_NOTE = "\n[crucible: truncated at {n} bytes]"
# Grace period for reaping an already-SIGKILLed process group.
_REAP_S = 5.0


@dataclass(frozen=True)
class ExecResult:
    returncode: int | None
    stdout: str
    stderr: str
    wall_s: float
    timed_out: bool
    workdir: str


def _prlimit_path() -> str | None:
    """Absolute path to util-linux ``prlimit``, or None when it is not installed."""
    return shutil.which("prlimit")


def _child_limits(mem_limit_bytes: int) -> Callable[[], None]:
    """preexec_fn fallback for hosts without ``prlimit``. Not thread-safe -- see _wrap_limits."""
    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_CAP_BYTES, _FSIZE_CAP_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return _apply


def _wrap_limits(argv: list[str], mem_limit_bytes: int) -> tuple[list[str], Callable[[], None] | None]:
    """Return (argv to exec, preexec_fn).

    Limits are applied by a ``prlimit`` wrapper process when available: ``preexec_fn`` runs
    between fork and exec and is not thread-safe, and later tasks call ``execute()`` from
    thread pools (R-T2-1). The fallback keeps single-threaded hosts without util-linux working.
    """
    prlimit = _prlimit_path()
    if prlimit is None:
        return list(argv), _child_limits(mem_limit_bytes)
    wrapper = [prlimit, f"--as={mem_limit_bytes}", f"--fsize={_FSIZE_CAP_BYTES}", "--core=0", "--"]
    return wrapper + list(argv), None


def _env(workdir: str, extra: dict[str, str] | None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _CLEAN_ENV_KEYS if k in os.environ}
    if extra:
        env.update(extra)
    # Sandbox keys land after env_extra, so a caller can never weaken them (R-T2-4).
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": workdir,
        "HOME": workdir,
        "TMPDIR": workdir,
    })
    env.pop("PYTEST_ADDOPTS", None)
    return env


def _write_files(workdir: str, files: dict[str, str]) -> None:
    for name, src in {**files, "sitecustomize.py": SITECUSTOMIZE_SRC}.items():
        path = os.path.join(workdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)


def _capture_paths(workdir: str) -> tuple[str, str]:
    return os.path.join(workdir, _STDOUT_NAME), os.path.join(workdir, _STDERR_NAME)


def _read_capped(path: str, cap: int = _READ_CAP_BYTES) -> str:
    """First ``cap`` bytes of a capture file as text; never raises on untrusted bytes (R-T2-2)."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(cap + 1)
    except OSError:
        return ""
    text = data[:cap].decode("utf-8", errors="replace")
    if len(data) > cap:
        return text + _TRUNCATION_NOTE.format(n=cap)
    return text


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group, then reap the direct child."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=_REAP_S)
    except subprocess.TimeoutExpired:
        pass


def execute(argv: list[str], files: dict[str, str], *, wall_cap_s: float = 60.0,
            mem_limit_bytes: int = 4 << 30, env_extra: dict[str, str] | None = None,
            keep: bool = False) -> ExecResult:
    """Run ``argv`` in a fresh temp dir seeded with ``files``.

    The workdir is the child's cwd, HOME, TMPDIR and PYTHONPATH; it is removed unless
    ``keep``. ``.crucible-stdout`` / ``.crucible-stderr`` are reserved names inside it.
    On the wall cap the process group is SIGKILLed, ``timed_out`` is True and
    ``returncode`` is None.
    """
    workdir = tempfile.mkdtemp(prefix="crucible-")
    try:
        _write_files(workdir, files)
        out_path, err_path = _capture_paths(workdir)
        child_argv, preexec = _wrap_limits(argv, mem_limit_bytes)
        env = _env(workdir, env_extra)
        t0 = time.monotonic()
        with open(out_path, "wb") as out_fh, open(err_path, "wb") as err_fh:
            proc = subprocess.Popen(
                child_argv, cwd=workdir, env=env,
                stdin=subprocess.DEVNULL, stdout=out_fh, stderr=err_fh,
                start_new_session=True, preexec_fn=preexec,
            )
            try:
                rc: int | None = proc.wait(timeout=wall_cap_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                rc, timed_out = None, True
        wall_s = time.monotonic() - t0
        return ExecResult(rc, _read_capped(out_path), _read_capped(err_path),
                          wall_s, timed_out, workdir)
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
