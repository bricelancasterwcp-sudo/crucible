import gzip, json, pathlib, subprocess, sys


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
