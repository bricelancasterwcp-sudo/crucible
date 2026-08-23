"""``crucible`` command line: build a stream, pre-check one on disk, or smoke-test one.

Three subcommands under ``stream``. ``build`` composes a stream and writes it, printing the
directory it landed in. ``precheck`` reads a written stream back and exits non-zero unless
every structural gate passes -- so a shell pipeline (or a Phase-A run) can refuse a stream
that does not match itself across phases without parsing the JSON it also prints.
``smoke`` re-applies the first ``--n`` tasks' mutants against their visible suites and
prints the kill census.

Nothing here promotes a warning to an error (R-T12-1): no ``-W``, no ``PYTHONWARNINGS``,
no ``warnings`` filter. The build path depends on SyntaxWarning staying a warning.

``NotEnoughClasses`` -- a corpus that cannot supply the requested ``C`` classes -- is a real
build failure, but it is the operator's mis-sizing, not a crash: it is caught and turned
into a one-line message and exit 2 rather than a traceback.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="crucible")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stream").add_subparsers(dest="scmd", required=True)
    b = s.add_parser("build")
    b.add_argument("--seed", type=int, default=0); b.add_argument("--C", type=int, default=200)
    b.add_argument("--n-nov", type=int, default=50); b.add_argument("--per-family", type=int, default=6)
    b.add_argument("--max-hidden", type=int, default=100); b.add_argument("--limit-units", type=int, default=None)
    b.add_argument("--jobs", type=int, default=8); b.add_argument("--rung", default="base")
    b.add_argument("--out", type=Path, default=Path("streams"))
    pc = s.add_parser("precheck"); pc.add_argument("dir", type=Path)
    sm = s.add_parser("smoke"); sm.add_argument("dir", type=Path); sm.add_argument("--n", type=int, default=30)
    a = p.parse_args(argv)
    if a.scmd == "build":
        from crucible.stream.compose import NotEnoughClasses
        from crucible.stream.pipeline import BuildConfig, build_stream
        cfg = BuildConfig(a.seed, a.C, a.n_nov, a.per_family, a.max_hidden, a.limit_units, a.jobs, a.rung)
        try:
            print(build_stream(cfg, a.out))
        except NotEnoughClasses as e:
            print(f"build failed: {e}", file=sys.stderr); return 2
        return 0
    if a.scmd == "precheck":
        from crucible.stream import store
        from crucible.stream.precheck import precheck
        man = store.read_manifest(a.dir)
        rep = precheck(man, {u: store.read_unit(a.dir, u) for u in man.unit_ids})
        print(json.dumps(rep.to_dict(), indent=1)); return 0 if rep.ok else 1
    if a.scmd == "smoke":
        from crucible.stream.pipeline import smoke
        print(json.dumps(smoke(a.dir, a.n))); return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
