"""Codec-landing pre-check (spec §4.7): before trusting a served model, prove it rewrites.

The whole S2 codec bet (spec §4.4) is that a full-module rewrite almost always *lands* --
parses back into a runnable module -- so "did the codec produce source" stops being an
experimental variable that could confound the pass-rate the arms measure. A model that
does NOT answer in the codec (wrong format, chatty prose, truncated blocks) would quietly
poison a run. So before an arm draws real attempts, this gate asks the cheaper question
first: draw ``n`` smoke tasks, prompt the served model once each, and measure the fraction
whose output lands. Clear ``>= 0.95`` and the model is trusted; miss it and the operator
falls back to the §2 baseline proposer (Qwen2.5-Coder-1.5B) -- a recorded manual decision,
deliberately not automated here.

Three choices mirror the driver (Task 12), for the same reasons:

* *The smoke prompt is the REAL prompt.* Each drawn task is reconstructed with the
  MUTANT's buggy ``module_src`` (``dataclasses.replace``), then a free (uncharged) symptom
  ``run`` learns the visible-suite failure and ``build_prompt`` assembles exactly what a
  real attempt would see. Landing is measured on the true prompt distribution, not a
  fabricated one -- otherwise the pre-check would clear a model that only lands on prompts
  the run never sends.

* *The tasks are DRAWN, seeded, never "first N".* ``random.Random(f"{seed}:landing")``
  samples the smoke set, so it moves with the seed and is not pinned to composition's
  shuffle order.

* *The gate is INCLUSIVE.* ``passes`` is ``rate >= 0.95`` (§4.7 says "at least 95%"), so a
  model landing exactly 0.95 is trusted, not rejected on a boundary technicality.

``LandingResult`` is frozen and ``to_dict``-serialisable so the verdict lands in a run's
record as data, not as a line in a log.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from pathlib import Path

from crucible.proposer.codec import landing_rate
from crucible.proposer.prompt import build_prompt
from crucible.sandbox.task_run import run
from crucible.stream import store
from crucible.stream.compose import StreamManifest, TaskSpec
from crucible.stream.units import Unit

LANDING_GATE = 0.95   # §4.7: a served model is trusted only if >= 95% of smoke rewrites land


@dataclass(frozen=True)
class LandingResult:
    """The §4.7 verdict for one served model: its smoke landing rate and whether it passes.

    ``model`` is the proposer's served id; ``n`` the number of smoke tasks drawn; ``rate``
    the fraction of those that landed; ``passes`` iff ``rate >= LANDING_GATE``. Frozen and
    JSON-native so it round-trips into a run record.
    """

    model: str
    n: int
    rate: float
    passes: bool

    def to_dict(self) -> dict:
        """JSON-ready form. All fields are JSON-native scalars, so this is exact."""
        return {"model": self.model, "n": self.n, "rate": self.rate, "passes": self.passes}


def _verdict(rate: float, n: int, model: str) -> LandingResult:
    """Apply the §4.7 gate: ``passes`` iff ``rate >= 0.95`` (INCLUSIVE -- see module docstring)."""
    return LandingResult(model=model, n=n, rate=rate, passes=rate >= LANDING_GATE)


def _mutated_unit(stream_dir: Path, task: TaskSpec) -> Unit:
    """The per-task unit a real attempt would repair: canonical unit + the MUTANT's module.

    Same reconstruction as the driver -- only ``module_src`` is swapped for the mutant's
    buggy source, so the smoke prompt embeds the same bug an attempt would see.
    """
    unit = store.read_unit(stream_dir, task.unit_id)
    mutant = store.read_mutant(stream_dir, task.task_key)
    return dataclasses.replace(unit, module_src=mutant.mutated_src)


def _draw_keys(manifest: StreamManifest, n: int, seed: int) -> list[str]:
    """``n`` smoke task_keys drawn with ``random.Random(f"{seed}:landing")`` -- never first N.

    Raises rather than returning a short set when the stream has fewer than ``n`` tasks: a
    pre-check run on fewer smoke tasks than requested is not the pre-check that was asked for.
    """
    keys = [t.task_key for t in manifest.tasks]
    if n > len(keys):
        raise ValueError(f"stream has {len(keys)} tasks; cannot draw n={n} smoke tasks")
    return random.Random(f"{seed}:landing").sample(keys, n)


def _smoke_output(stream_dir: Path, task: TaskSpec, proposer, seed: int) -> str:
    """One smoke completion's recovered source: free symptom -> real prompt -> one generate.

    The symptom ``run`` is uncharged (a pre-check, like the driver's free symptom); the
    single ``generate(n=1)`` output's ``text`` is already codec-decoded module source, which
    ``landing_rate`` re-checks for landing.
    """
    unit = _mutated_unit(stream_dir, task)
    symptom = run(unit, unit.module_src, None)              # free symptom, never charged
    prompt = build_prompt(unit, symptom)
    return proposer.generate(prompt, n=1, seed=seed)[0].text


def landing_precheck(stream_dir: Path, proposer, *, n: int = 30, seed: int = 0) -> LandingResult:
    """Measure ``proposer``'s codec-landing rate over ``n`` seeded smoke tasks (spec §4.7).

    Draws ``n`` smoke task_keys from the stream at ``stream_dir``, prompts ``proposer`` once
    each on the real repair prompt, and returns the landing rate + the ``>= 0.95`` verdict.
    ``model`` is taken from ``proposer.model``. On failure the operator falls back to the §2
    baseline proposer -- recorded by hand, not automated here.
    """
    stream_dir = Path(stream_dir)
    manifest = store.read_manifest(stream_dir)
    by_key = {t.task_key: t for t in manifest.tasks}
    keys = _draw_keys(manifest, n, seed)
    outputs = [_smoke_output(stream_dir, by_key[k], proposer, seed) for k in keys]
    return _verdict(landing_rate(outputs), n, proposer.model)
