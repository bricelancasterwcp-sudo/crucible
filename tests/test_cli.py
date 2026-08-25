import gzip, json, pathlib, subprocess, sys

import pytest


def _recs():
    fix = pathlib.Path(__file__).resolve().parent / "fixtures"
    out = []
    for n in ("mini_humaneval.jsonl.gz", "mini_mbpp.jsonl.gz"):
        with gzip.open(fix / n, "rt") as fh:
            out += [json.loads(l) for l in fh]
    return out


def test_cli_help_runs():
    out = subprocess.run([sys.executable, "-m", "crucible.cli", "stream", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "build" in out.stdout and "precheck" in out.stdout


def test_cli_precheck_exits_zero_on_ok_and_nonzero_on_bad(tmp_path):
    from crucible.cli import main
    from crucible.stream.pipeline import build_stream, BuildConfig
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    d = build_stream(cfg, tmp_path / "good", recs=_recs(), log=lambda *a: None)
    assert main(["stream", "precheck", str(d)]) == 0

    # Tamper the census so precheck's counts-named gate fails ⇒ report.ok is False ⇒ exit 1.
    mpath = d / "manifest.json"
    man = json.loads(mpath.read_text())
    man["counts"].pop("equivalent")
    mpath.write_text(json.dumps(man))
    assert main(["stream", "precheck", str(d)]) == 1


def test_cli_arm_help_lists_pilot_and_run():
    out = subprocess.run([sys.executable, "-m", "crucible.cli", "arm", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "pilot" in out.stdout and "run" in out.stdout


def test_cli_build_passes_rung_and_pairs_per_family_through(tmp_path, monkeypatch):
    # The two new build knobs must reach BuildConfig -- --rung by name and
    # --pairs-per-family as the keyword that keeps the positional construction valid.
    # build_stream is stubbed because the CLI's own build path loads the real corpus.
    import crucible.stream.pipeline as pipeline
    from crucible.cli import main
    seen = {}
    def fake_build(cfg, out, **kw):
        seen["cfg"] = cfg
        return out
    monkeypatch.setattr(pipeline, "build_stream", fake_build)
    rc = main(["stream", "build", "--rung", "stack2", "--pairs-per-family", "3", "--out", str(tmp_path)])
    assert rc == 0
    assert seen["cfg"].rung == "stack2" and seen["cfg"].pairs_per_family == 3
    # The defaults are unchanged for a plain build.
    assert main(["stream", "build", "--out", str(tmp_path)]) == 0
    assert seen["cfg"].rung == "base" and seen["cfg"].pairs_per_family == 4


def test_cli_build_rejects_an_unknown_rung(tmp_path):
    # --rung carries pipeline.ALLOWED_RUNGS as argparse choices, so a typo is refused at
    # parse time (SystemExit 2) rather than reaching build_stream's ValueError.
    from crucible.cli import main
    from crucible.stream.pipeline import ALLOWED_RUNGS
    assert set(ALLOWED_RUNGS) == {"base", "stack2"}
    with pytest.raises(SystemExit) as e:
        main(["stream", "build", "--rung", "tower", "--out", str(tmp_path)])
    assert e.value.code == 2


def test_cli_build_reports_not_enough_classes_as_exit_2_at_stack2(tmp_path, monkeypatch):
    # compose refuses a short rung-1 stream; the CLI must still turn that into one line
    # and exit 2, never a traceback. The handler is rung-agnostic -- this pins it at stack2.
    import crucible.stream.pipeline as pipeline
    from crucible.cli import main
    from crucible.stream.compose import NotEnoughClasses
    def boom(cfg, out, **kw):
        raise NotEnoughClasses("classes taken 1 < C=2")
    monkeypatch.setattr(pipeline, "build_stream", boom)
    assert main(["stream", "build", "--rung", "stack2", "--C", "2", "--out", str(tmp_path)]) == 2


# --- S3 (Task 11): the A_full run flags -----------------------------------------------
#
# Two claims, both wiring: (1) ``--arm A_full`` grows ``--memory-db`` and
# ``--sleep-threshold`` and both reach the hooks, (2) EVERY OTHER ARM constructs NO memory
# store at all. (2) is the mutation pin that matters -- an A_noMem run that opened an organ
# (even a fresh, unused one) would mean the two arms no longer differ by exactly the
# pre-registered column, and nothing in an A_noMem record would show it.

@pytest.fixture(scope="module")
def _stream(tmp_path_factory):
    from crucible.stream.pipeline import BuildConfig, build_stream
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    return build_stream(cfg, tmp_path_factory.mktemp("cli-stream"), recs=_recs(),
                        log=lambda *a: None)


class _NoProposer:
    """Stands in for a served proposer: identity is asserted at construction in the real one."""

    def __init__(self, model):
        self.model = model


def _stub_run(monkeypatch, seen):
    """Replace the real ``run_arm`` and proposer construction; capture what the CLI wired."""
    import crucible.cli as cli
    import crucible.run.driver as driver_module
    monkeypatch.setattr(cli, "_proposer_or_none", lambda url, model, chat=False: _NoProposer(model))

    def fake_run_arm(cfg, stream_dir, keys, proposer, value, out_dir, *, log=print, hooks=None):
        seen.update(cfg=cfg, value=value, hooks=hooks, out_dir=out_dir, keys=keys,
                    proposer=proposer)
        return out_dir / cfg.name

    monkeypatch.setattr(driver_module, "run_arm", fake_run_arm)


def test_cli_arm_run_a_full_wires_the_memory_db_and_sleep_threshold(_stream, tmp_path, monkeypatch):
    from crucible.cli import main
    from crucible.run.arm import ARMS
    from crucible.run.full import AdapterProposer
    from crucible.value.online import OnlineValue
    seen = {}
    _stub_run(monkeypatch, seen)

    db = tmp_path / "custom" / "memory.sqlite3"
    rc = main(["arm", "run", str(_stream), "--arm", "A_full", "--base-url", "http://x",
               "--out", str(tmp_path / "runs"), "--memory-db", str(db),
               "--sleep-threshold", "4"])

    assert rc == 0
    assert seen["hooks"] is not None
    assert isinstance(seen["value"], OnlineValue)      # A_full runs value v1, not the constant
    assert db.exists()                                 # the organ was opened where asked
    assert seen["hooks"].sleep_threshold == 4
    # The driver must generate through the RE-POINTABLE proposer, or an accepted adapter
    # would never serve a single task while the records still claimed it did (review C1).
    assert seen["proposer"] is seen["hooks"].proposer
    assert isinstance(seen["proposer"], AdapterProposer)
    assert seen["proposer"].base_model == ARMS["A_full"].model


def test_cli_arm_run_a_full_defaults_the_memory_db_under_the_arm_dir(_stream, tmp_path, monkeypatch):
    from crucible.cli import main
    seen = {}
    _stub_run(monkeypatch, seen)

    rc = main(["arm", "run", str(_stream), "--arm", "A_full", "--base-url", "http://x",
               "--out", str(tmp_path / "runs")])

    assert rc == 0
    assert (tmp_path / "runs" / "A_full" / "memory.sqlite3").exists()
    assert seen["hooks"].sleep_threshold == 16         # spec S5 / R-S3-3 default


@pytest.mark.parametrize("arm,retrieval,sleep", [("A_full", True, True),
                                                 ("A_mem_nosleep", True, False),
                                                 ("A_sleep_nomem", False, True),
                                                 ("A_mem_exactonly", True, False)])
def test_cli_arm_run_full_family_wires_the_declared_switches(_stream, tmp_path, monkeypatch,
                                                             arm, retrieval, sleep):
    """The exploratory ablations run through the SAME wiring as A_full -- OnlineValue,
    AdapterProposer, an organ of their own -- differing only by the FULL_FAMILY switches.
    MUTATION: swap a tuple in FULL_FAMILY and the wrong arm sleeps (or retrieves)."""
    from crucible.cli import main
    from crucible.run.full import AdapterProposer
    from crucible.value.online import OnlineValue
    seen = {}
    _stub_run(monkeypatch, seen)

    rc = main(["arm", "run", str(_stream), "--arm", arm, "--base-url", "http://x",
               "--out", str(tmp_path / "runs")])

    assert rc == 0
    hooks = seen["hooks"]
    assert (hooks.retrieval_enabled, hooks.sleep_enabled) == (retrieval, sleep)
    assert hooks.retrieval_mode == {"A_full": "full", "A_mem_nosleep": "full",
                                    "A_sleep_nomem": "off", "A_mem_exactonly": "exact"}[arm]
    assert isinstance(seen["value"], OnlineValue)
    assert seen["proposer"] is hooks.proposer            # re-pointable proposer (C1)
    assert isinstance(seen["proposer"], AdapterProposer)
    assert (tmp_path / "runs" / arm / "memory.sqlite3").exists()   # its OWN organ


def test_cli_arm_run_a_nomem_never_constructs_a_memory_store(_stream, tmp_path, monkeypatch):
    """MUTATION (c): hooks (and therefore an organ) built for a non-A_full arm."""
    import crucible.memory.store as store_module
    import crucible.run.full as full_module
    from crucible.cli import main
    from crucible.value.model import ConstantValue
    seen = {}
    _stub_run(monkeypatch, seen)

    def boom(*a, **kw):
        raise AssertionError("A_noMem must never construct a MemoryStore")

    # Both names: the one A_full's wiring actually looks up (``full`` imported it at module
    # level) and the defining module, so no construction path can slip past the boom.
    monkeypatch.setattr(full_module, "MemoryStore", boom)
    monkeypatch.setattr(store_module, "MemoryStore", boom)
    rc = main(["arm", "run", str(_stream), "--arm", "A_noMem", "--base-url", "http://x",
               "--out", str(tmp_path / "runs")])

    assert rc == 0
    assert seen["hooks"] is None
    assert seen["proposer"] is not None and not hasattr(seen["proposer"], "base_model")
    assert isinstance(seen["value"], ConstantValue)    # A_noMem keeps v0 (spec S6)


def test_cli_arm_run_rejects_an_unknown_arm(_stream, tmp_path):
    from crucible.cli import main
    assert main(["arm", "run", str(_stream), "--arm", "A_nope", "--base-url", "http://x"]) == 2


def test_cli_arm_run_b_mem_wires_store_only_hooks(_stream, tmp_path, monkeypatch):
    """B_mem differs from the frozen B_search by the store ALONE (prereg §3): MemHooks,
    ConstantValue (never OnlineValue), the plain proposer (never AdapterProposer)."""
    from crucible.cli import main
    from crucible.run.full import AdapterProposer, MemHooks
    from crucible.value.model import ConstantValue
    seen = {}
    _stub_run(monkeypatch, seen)
    rc = main(["arm", "run", str(_stream), "--arm", "B_mem", "--base-url", "http://x",
               "--out", str(tmp_path / "runs")])
    assert rc == 0
    assert isinstance(seen["hooks"], MemHooks)
    assert isinstance(seen["value"], ConstantValue)
    assert not isinstance(seen["proposer"], AdapterProposer)
    assert (tmp_path / "runs" / "B_mem" / "memory.sqlite3").exists()
