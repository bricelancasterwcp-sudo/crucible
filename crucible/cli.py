"""``crucible`` command line: build/precheck/smoke a stream, or run the arms over one.

``stream`` groups the S1 subcommands. ``build`` composes a stream and writes it, printing the
directory it landed in. ``precheck`` reads a written stream back and exits non-zero unless
every structural gate passes -- so a shell pipeline (or a Phase-A run) can refuse a stream
that does not match itself across phases without parsing the JSON it also prints. ``smoke``
re-applies the first ``--n`` tasks' mutants against their visible suites and prints the kill
census.

``arm`` groups the S2 run subcommands, each talking to a served proposer over ``--base-url``.
``pilot`` runs the ceiling pilot (``A_noMem`` over ``--n`` phase-1 tasks) and prints the
:class:`~crucible.run.pilot.PilotVerdict` as JSON -- p0 and whether the stream is too easy
(spec §4.8.4). ``run`` runs one named arm over a chosen task set and prints where the records
landed. The proposer's served identity is asserted on construction, so a mismatched or
unreachable server is turned into a one-line message and a non-zero exit -- never a traceback.

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

DEFAULT_PILOT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # ARMS["A_noMem"].model (amendment A2)
PROPOSER_ERROR_EXIT = 3                    # served identity mismatch / unreachable server


def _add_stream(sub) -> None:
    """The S1 ``stream`` subcommands: build / precheck / smoke.

    ``--rung``'s choices are ``pipeline.ALLOWED_RUNGS`` itself, imported here rather than
    restated, so a rung added there is offered here the same day and a typo is refused at
    parse time. The import is local because it pulls the whole build stack (cosmic-ray);
    every other heavy import in this module is deferred the same way.
    """
    from crucible.stream.pipeline import ALLOWED_RUNGS
    s = sub.add_parser("stream").add_subparsers(dest="scmd", required=True)
    b = s.add_parser("build")
    b.add_argument("--seed", type=int, default=0); b.add_argument("--C", type=int, default=200)
    b.add_argument("--n-nov", type=int, default=50); b.add_argument("--per-family", type=int, default=6)
    b.add_argument("--max-hidden", type=int, default=100); b.add_argument("--limit-units", type=int, default=None)
    b.add_argument("--jobs", type=int, default=8); b.add_argument("--rung", default="base", choices=ALLOWED_RUNGS)
    b.add_argument("--pairs-per-family", type=int, default=4)   # rung-1 pairing cap, per (unit, family)
    b.add_argument("--out", type=Path, default=Path("streams"))
    pc = s.add_parser("precheck"); pc.add_argument("dir", type=Path)
    sm = s.add_parser("smoke"); sm.add_argument("dir", type=Path); sm.add_argument("--n", type=int, default=30)


def _add_arm(sub) -> None:
    """The S2 ``arm`` subcommands: pilot (ceiling) / run (one arm over a task set).

    ``SLEEP_THRESHOLD_DEFAULT`` is imported here rather than restated, the same way
    ``_add_stream`` imports ``ALLOWED_RUNGS``: the number the CLI advertises is the number
    the sleep trigger actually uses, and the import stays local to the one function that
    needs it (this module defers every import out of its top level).
    """
    from crucible.sleep.loop import SLEEP_THRESHOLD_DEFAULT
    a = sub.add_parser("arm").add_subparsers(dest="acmd", required=True)
    pl = a.add_parser("pilot"); pl.add_argument("stream_dir", type=Path)
    pl.add_argument("--base-url", required=True); pl.add_argument("--model", default=DEFAULT_PILOT_MODEL)
    pl.add_argument("--n", type=int, default=30); pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--out", type=Path, default=Path("runs"))
    # Serving surface defaults to the ARM's requirement (ArmConfig.chat), not a fixed value:
    # an instruct proposer must be chat-served, a base proposer raw-served. Default None here
    # means "use the arm's chat flag"; --chat/--no-chat force an override. See arm.py A2.
    pl.add_argument("--chat", action=argparse.BooleanOptionalAction, default=None)
    rn = a.add_parser("run"); rn.add_argument("stream_dir", type=Path)
    rn.add_argument("--arm", required=True); rn.add_argument("--base-url", required=True)
    rn.add_argument("--tasks", default="phase1"); rn.add_argument("--out", type=Path, default=Path("runs"))
    rn.add_argument("--chat", action=argparse.BooleanOptionalAction, default=None)
    # A_full only (every other arm ignores both). --memory-db defaults to None here, not to a
    # path, because the real default (``<out>/<arm>/memory.sqlite3``) depends on two other
    # flags; resolving it at parse time would bake in whatever --out happened to be declared
    # first. --sleep-threshold's default is the spec's N (R-S3-3), imported rather than
    # restated so the CLI cannot drift from the trigger it configures; the S3 exit smoke
    # overrides it to 4 (spec S9).
    rn.add_argument("--memory-db", type=Path, default=None)
    rn.add_argument("--sleep-threshold", type=int, default=SLEEP_THRESHOLD_DEFAULT)


def _run_stream(a) -> int:
    """Dispatch the parsed ``stream`` subcommand; the S1 behaviour, unchanged."""
    if a.scmd == "build":
        from crucible.stream.compose import NotEnoughClasses
        from crucible.stream.pipeline import BuildConfig, build_stream
        # Positional through ``rung``, as BuildConfig's frozen field order allows; the
        # rung-1 knob rides in by keyword so inserting it cannot silently shift the rest.
        cfg = BuildConfig(a.seed, a.C, a.n_nov, a.per_family, a.max_hidden, a.limit_units, a.jobs, a.rung,
                          pairs_per_family=a.pairs_per_family)
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


def _proposer_or_none(base_url: str, model: str, chat: bool = False):
    """A ``VLLMProposer`` for ``(base_url, model)``, or ``None`` after a one-line error.

    ``chat`` selects the chat-completions serving surface (required for an instruct proposer;
    see arm.py amendment A2). Construction asserts served identity, so a mismatched checkpoint
    or an unreachable server raises ``IdentityMismatch`` (connection failures are folded into it
    upstream). Caught here and printed as one line -- the caller returns
    :data:`PROPOSER_ERROR_EXIT`, never a traceback.
    """
    from crucible.proposer.client import VLLMProposer
    from crucible.proposer.identity import IdentityMismatch
    try:
        return VLLMProposer(base_url, model, chat=chat)
    except (IdentityMismatch, OSError) as e:
        print(f"proposer error: {e}", file=sys.stderr)
        return None


def _arm_pilot(a) -> int:
    """Run the ceiling pilot and print its verdict as JSON."""
    from crucible.run.arm import ARMS
    from crucible.run.pilot import ceiling_pilot
    from crucible.value.model import ConstantValue
    # The pilot always runs A_noMem, so its serving surface is A_noMem's; --chat/--no-chat override.
    chat = ARMS["A_noMem"].chat if a.chat is None else a.chat
    proposer = _proposer_or_none(a.base_url, a.model, chat)
    if proposer is None:
        return PROPOSER_ERROR_EXIT
    verdict = ceiling_pilot(a.stream_dir, a.out, proposer, ConstantValue(), n=a.n, seed=a.seed)
    print(json.dumps(verdict.to_dict(), sort_keys=True))
    return 0


def _task_keys(manifest, tasks: str) -> list[str]:
    """The task_keys ``--tasks`` selects: ``all`` manifest tasks, ``phase1`` only, or from a file."""
    if tasks == "all":
        return [t.task_key for t in manifest.tasks]
    if tasks == "phase1":
        return [t.task_key for t in manifest.tasks if t.kind == "first"]
    lines = Path(tasks).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def _arm_run(a) -> int:
    """Run one named arm over the chosen task set and print where the records landed.

    A_full -- and ONLY A_full -- gets the memory organ, value v1 and the sleep loop, wired as
    the driver's ``hooks``. Every other arm passes ``hooks=None`` and keeps v0's
    ``ConstantValue`` (spec S6: A_noMem's pilot already ran on it; arms differ by exactly the
    pre-registered column). The gate is the arm NAME because ``ArmConfig`` is deliberately
    memory-free -- see ``crucible.run.arm``'s ARMS comment -- and it is a hard gate: nothing
    on the non-A_full path so much as opens a store.
    """
    from crucible.run.arm import ARMS
    from crucible.run.driver import run_arm
    from crucible.run.full import FULL_ARM, build_full_hooks
    from crucible.stream import store
    from crucible.value.model import ConstantValue
    if a.arm not in ARMS:
        print(f"unknown arm {a.arm!r}; known: {sorted(ARMS)}", file=sys.stderr); return 2
    cfg = ARMS[a.arm]
    # Serving surface follows the arm (instruct -> chat, base -> raw); --chat/--no-chat override.
    chat = cfg.chat if a.chat is None else a.chat
    proposer = _proposer_or_none(a.base_url, cfg.model, chat)
    if proposer is None:
        return PROPOSER_ERROR_EXIT
    keys = _task_keys(store.read_manifest(a.stream_dir), a.tasks)
    value, hooks = ConstantValue(), None
    if cfg.name == FULL_ARM:
        from crucible.value.online import OnlineValue
        value = OnlineValue()
        hooks = build_full_hooks(cfg, a.stream_dir, a.out, base_url=a.base_url, value=value,
                                 chat=chat, proposer=proposer, memory_db=a.memory_db,
                                 sleep_threshold=a.sleep_threshold)
        # The SAME client, wrapped so an accepted adapter is what the next request asks for.
        # Handing run_arm the unwrapped one would leave the arm on the base weights forever
        # while its records still claimed an adapter (review C1).
        proposer = hooks.proposer
    out_path = run_arm(cfg, a.stream_dir, keys, proposer, value, a.out, hooks=hooks)
    print(f"records written to {out_path}"); return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="crucible")
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_stream(sub)
    _add_arm(sub)
    a = p.parse_args(argv)
    if a.cmd == "arm":
        return _arm_pilot(a) if a.acmd == "pilot" else _arm_run(a)
    return _run_stream(a)


if __name__ == "__main__":
    sys.exit(main())
