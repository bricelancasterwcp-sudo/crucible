"""Isolated subprocess execution: the one place crucible runs untrusted code.

Pattern borrowed from mini-swe-agent's LocalEnvironment (MIT): subprocess + timeout, but with
a fresh workdir, socket blocking, an address-space limit, and process-group kill.
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import tempfile
import time
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


@dataclass(frozen=True)
class ExecResult:
    returncode: int | None
    stdout: str
    stderr: str
    wall_s: float
    timed_out: bool
    workdir: str


def _child_limits(mem_limit_bytes: int):
    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return _apply


def _env(workdir: str, extra: dict[str, str] | None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _CLEAN_ENV_KEYS if k in os.environ}
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": workdir,
        "HOME": workdir,
        "TMPDIR": workdir,
    })
    if extra:
        env.update(extra)
    env.pop("PYTEST_ADDOPTS", None)
    return env


def execute(argv: list[str], files: dict[str, str], *, wall_cap_s: float = 60.0,
            mem_limit_bytes: int = 4 << 30, env_extra: dict[str, str] | None = None,
            keep: bool = False) -> ExecResult:
    workdir = tempfile.mkdtemp(prefix="crucible-")
    try:
        for name, src in {**files, "sitecustomize.py": SITECUSTOMIZE_SRC}.items():
            path = os.path.join(workdir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
        t0 = time.monotonic()
        proc = subprocess.Popen(
            argv, cwd=workdir, env=_env(workdir, env_extra),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True, preexec_fn=_child_limits(mem_limit_bytes),
        )
        try:
            out, err = proc.communicate(timeout=wall_cap_s)
            timed_out, rc = False, proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            timed_out, rc = True, None
        return ExecResult(rc, out, err, time.monotonic() - t0, timed_out, workdir)
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
