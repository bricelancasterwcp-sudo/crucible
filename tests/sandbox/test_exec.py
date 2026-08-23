import sys, time
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
