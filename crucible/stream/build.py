"""Turn one EvalPlus record into a :class:`Unit` -- or say, by name, why it cannot be one.

This is where the task stream stops being data and becomes an instrument. Every step
either produces something a mutant can be measured against or refuses to produce
anything, and the refusal is always recorded as a reason rather than a silent gap.

Three rules carry the weight.

*The canonical must pass its own generated tests, in the sandbox, before the unit
exists.* The oracle computed the expected values inside one process; the runner then
executes the rendered file against the same canonical inside a different one. That
second run is not a formality -- it is the only thing that catches a canonical whose
output depends on the process rather than on its arguments (a pid, a clock, iteration
over a set of strings under a fresh ``PYTHONHASHSEED``). Such a unit would fail every
candidate alike, and a suite of them reads as a magnificent mutation score. What the
self-check cannot catch is a canonical that is merely *wrong*: the oracle derived the
expectations from it, so wrong-but-deterministic is self-consistent by construction.
That limit is real and is not papered over here.

*A failed unit comes back as a drop, not as an exception.* ``build_units`` maps it across a
``ThreadPoolExecutor``; an exception escaping one record aborts the map and throws away
every other record's completed work. So the two failures the pipeline can legitimately
hand back -- an oracle driver that produced no result, a render that was handed
misaligned arguments -- are caught here and returned as drops (ruling R-T8-1). Genuine
caller bugs (a record missing ``task_id``, a module name that is not an identifier)
still raise: they are not properties of the unit and must not be laundered into a drop
reason.

*Order is record order.* ``pool.map`` yields in submission order however the threads
finish, and the outputs are left that way -- units in record order, drops in record
order -- because downstream tasks pair them back against the records they came from.
Sorting the results afterwards would have made the output a function of the unit ids
instead (ruling R-T8-2).

Hidden inputs are sampled with a seeded ``random.Random`` keyed on ``seed``, the task id
and the literal ``"hidden"``, never by taking the first N: "first N" is a different
subset of EvalPlus's extra inputs, not a sample of them, and it makes the hidden suite a
function of dataset order. Same record and seed therefore always give an identical
``Unit``, down to ``src_hash``.
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from ..sandbox.runner import run_tests
from .evalplus import full_source
from .oracle import OracleError, compute_expected
from .testgen import render_tests
from .units import Unit, module_name_for, sha256_text, strip_docstrings

# A unit whose *canonical* needs longer than this for the visible suite cannot be run
# against K mutants inside any sane per-unit budget, so it is dropped rather than left to
# blow the wall cap later, where it would look like a hang the mutant caused. Compared
# against ``TestReport.wall_s``, which covers the runner's collect probe as well as pytest
# (ruling R-T3-6) -- the same total a mutant run will pay, which is what makes it the
# right thing to budget against.
MAX_CANONICAL_VISIBLE_WALL_S = 20.0


@dataclass(frozen=True)
class Dropped:
    """A record that did not become a unit, and the reason it did not.

    Same shape as the ``Unit`` it stands in for as far as callers are concerned: it
    carries ``unit_id``, so a partition of results can always be traced back to records.
    The reason is a short tag with a colon-separated detail (``canonical-syntax:...``,
    ``oracle-error:...``, ``canonical-fails-visible:...``) -- never an empty string and
    never ``None``.
    """

    unit_id: str
    reason: str


def _select_hidden(rec: dict, seed: int, max_hidden: int) -> list[list]:
    """The hidden inputs for this record: all of them, or a seeded sample of them.

    ``rng.sample`` over *indices*, then ``sorted``, so the chosen inputs keep their
    dataset order and the selection depends only on ``(seed, task_id)`` -- not on how
    many units were built before this one, and not on dataset order.
    """
    plus = list(rec["plus_input"])
    if len(plus) <= max_hidden:
        return plus
    rng = random.Random(f"{seed}:{rec['task_id']}:hidden")
    idx = sorted(rng.sample(range(len(plus)), max_hidden))
    return [plus[i] for i in idx]


def _self_check(module_name: str, module_src: str, visible_src: str, hidden_src: str,
                n_hidden: int) -> str | None:
    """Run the generated suites against the canonical; return a drop reason or ``None``.

    ``infra_error`` is reported in place of the failing test names when it is what went
    wrong, because "our generated file would not collect" and "the canonical failed four
    tests" are different facts and a drop reason that conflated them would be a guess.
    The hidden run is guarded by ``n_hidden``: a hidden file with no tests in it collects
    to nothing, which the runner correctly calls an infra error, and blaming the unit for
    that would be false.
    """
    rv = run_tests(module_name, module_src, visible_src)
    if not rv.all_passed:
        return f"canonical-fails-visible:{rv.infra_error or (rv.failed + rv.timed_out + rv.errored)}"
    if rv.wall_s > MAX_CANONICAL_VISIBLE_WALL_S:
        return f"visible-too-slow:{rv.wall_s:.1f}s"
    if n_hidden:
        rh = run_tests(module_name, module_src, hidden_src)
        if not rh.all_passed:
            return f"canonical-fails-hidden:{rh.infra_error or (rh.failed + rh.timed_out + rh.errored)}"
    return None


def build_unit(rec: dict, *, seed: int, max_hidden: int = 100) -> Unit | Dropped:
    """Build one unit from one EvalPlus record, or return why it was dropped."""
    uid, entry = rec["task_id"], rec["entry_point"]
    try:
        module_src = strip_docstrings(full_source(rec))
        compile(module_src, uid, "exec")
    except SyntaxError as e:
        return Dropped(uid, f"canonical-syntax:{e.msg}")
    mod = module_name_for(uid)
    atol = float(rec.get("atol") or 0)
    vis_in, hid_in = list(rec["base_input"]), _select_hidden(rec, seed, max_hidden)
    try:
        # A driver that crashed, hung or printed no JSON measured *nothing* -- including
        # the common case of a canonical whose module will not import in the sandbox.
        exp_v = compute_expected(mod, module_src, entry, vis_in)
        exp_h = compute_expected(mod, module_src, entry, hid_in)
    except OracleError as e:
        return Dropped(uid, f"oracle-error:{e.detail or e}")
    try:
        vis_src, drop_v = render_tests(mod, entry, vis_in, exp_v, prefix="v", atol=atol)
        hid_src, drop_h = render_tests(mod, entry, hid_in, exp_h, prefix="h", atol=atol)
    except ValueError as e:
        return Dropped(uid, f"render-error:{e}")
    n_v, n_h = len(vis_in) - len(drop_v), len(hid_in) - len(drop_h)
    if n_v < 1:
        # Every visible input was unrenderable. Running the empty file would collect no
        # tests, which is an infra error, not a verdict about this unit.
        return Dropped(uid, "no-visible-tests")
    reason = _self_check(mod, module_src, vis_src, hid_src, n_h)
    if reason is not None:
        return Dropped(uid, reason)
    return Unit(uid, mod, entry, module_src, vis_src, hid_src, sha256_text(module_src), n_v, n_h,
                tuple(drop_v + drop_h))


def build_units(recs: list[dict], *, seed: int, max_hidden: int = 100, jobs: int = 8,
                progress: Callable[[Unit | Dropped], None] | None = None,
                ) -> tuple[list[Unit], list[Dropped]]:
    """Build every record, in parallel, and partition the results in record order.

    ``progress`` (when given) is called once per record with that record's ``Unit`` or
    ``Dropped``, in record order -- so a caller counting drops sees the same sequence the
    returned lists describe, not whatever order the threads happened to finish in.
    """
    units: list[Unit] = []
    dropped: list[Dropped] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for res in pool.map(lambda r: build_unit(r, seed=seed, max_hidden=max_hidden), recs):
            (units if isinstance(res, Unit) else dropped).append(res)
            if progress:
                progress(res)
    return units, dropped
