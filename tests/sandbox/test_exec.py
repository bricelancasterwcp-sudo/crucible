import sys, time

import pytest
from crucible.sandbox.exec import execute, ExecResult

PY = sys.executable

def test_runs_a_script_and_captures_stdout():
    r = execute([PY, "main.py"], {"main.py": "print('hi')"})
    assert isinstance(r, ExecResult)
    assert r.returncode == 0 and r.stdout.strip() == "hi" and not r.timed_out

def test_wall_cap_kills_infinite_loop():
    t0 = time.monotonic()
    r = execute([PY, "-c", "while True: pass"], {}, wall_cap_s=1.0)
    assert r.timed_out and r.returncode is None
    assert time.monotonic() - t0 < 5.0

def test_network_is_blocked():
    src = "import socket\ntry:\n    socket.create_connection(('93.184.216.34', 80), timeout=2)\n    print('OPEN')\nexcept OSError as e:\n    print('BLOCKED', e)\n"
    r = execute([PY, "net.py"], {"net.py": src})
    assert "BLOCKED" in r.stdout and "OPEN" not in r.stdout

def test_no_bytecode_written_and_addopts_not_inherited(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "--this-would-break")
    src = "import os,sys\nprint(os.environ.get('PYTHONDONTWRITEBYTECODE'), 'PYTEST_ADDOPTS' in os.environ, sys.flags.dont_write_bytecode)"
    r = execute([PY, "e.py"], {"e.py": src})
    assert r.stdout.split() == ["1", "False", "1"]

def test_memory_limit_is_applied():
    src = "import resource\nprint(resource.getrlimit(resource.RLIMIT_AS)[0])"
    r = execute([PY, "m.py"], {"m.py": src}, mem_limit_bytes=4 << 30)
    assert int(r.stdout.strip()) == 4 << 30

def test_workdir_is_removed_unless_keep():
    import os
    r = execute([PY, "-c", "pass"], {})
    assert not os.path.exists(r.workdir)
    r2 = execute([PY, "-c", "pass"], {}, keep=True)
    assert os.path.exists(r2.workdir)
    import shutil
    shutil.rmtree(r2.workdir)

def test_detached_grandchild_does_not_defeat_wall_cap():
    src = ("import os, time\n"
           "if os.fork() == 0:\n    os.setsid(); time.sleep(8); os._exit(0)\n"
           "while True: pass\n")
    t0 = time.monotonic()
    r = execute([PY, "g.py"], {"g.py": src}, wall_cap_s=1.0)
    assert r.timed_out and time.monotonic() - t0 < 4.0

def test_non_utf8_output_does_not_raise():
    r = execute([PY, "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfe\\x00bad'); sys.stdout.buffer.flush()"], {})
    assert r.returncode == 0 and "bad" in r.stdout

def test_output_flood_is_bounded_and_terminates():
    t0 = time.monotonic()
    r = execute([PY, "-c", "while True: print('x' * 1000)"], {}, wall_cap_s=10.0)
    assert time.monotonic() - t0 < 12.0
    assert len(r.stdout) <= (1 << 20) + 64
    assert r.timed_out or r.returncode != 0

def test_env_extra_cannot_override_sandbox_keys():
    src = "import os, sys\nprint(os.environ['PYTHONPATH'] == os.getcwd(), sys.flags.dont_write_bytecode)"
    r = execute([PY, "e2.py"], {"e2.py": src}, env_extra={"PYTHONPATH": "/nonexistent", "PYTHONDONTWRITEBYTECODE": "0"})
    assert r.stdout.split() == ["True", "1"]

@pytest.mark.parametrize("use_prlimit", [True, False])
def test_rlimits_apply_via_prlimit_and_via_fallback(monkeypatch, use_prlimit):
    from crucible.sandbox import exec as sandbox_exec
    if not use_prlimit:
        monkeypatch.setattr(sandbox_exec, "_prlimit_path", lambda: None)
    src = ("import resource\n"
           "print(resource.getrlimit(resource.RLIMIT_AS)[0], resource.getrlimit(resource.RLIMIT_FSIZE)[0])")
    r = execute([PY, "m.py"], {"m.py": src}, mem_limit_bytes=4 << 30)
    assert r.stdout.split() == [str(4 << 30), str(16 << 20)]
