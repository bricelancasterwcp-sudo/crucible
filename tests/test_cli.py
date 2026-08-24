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
