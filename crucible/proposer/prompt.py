"""The repair prompt a frozen proposer sees, in the full-module-rewrite codec.

The codec is deliberate (spec S4.4). Rather than ask the model for a diff, a patch, or
the single repaired function -- each of which fails to parse or fails to apply some of
the time -- the prompt asks for the *entire* corrected module inside one fenced
``python`` block. Almost every completion then parses, and "did the codec land" stops
being an experimental variable that could confound the thing we actually measure.

The prompt is assembled from three things the agent is *allowed* to see: the mutated
module source, the visible test file, and the *symptom* -- the ``TestReport`` from one
free execution of the visible suite (which visible tests failed, and why). On refinement
an optional ``feedback`` note about the prior attempt is appended.

Honest-measurement rule: the agent only ever sees the visible suite, so this template
mentions no other suite and never implies more tests exist than the ones printed here.
The withheld evaluation set is not named, described, or alluded to -- the model must earn
a real repair from the visible evidence alone, not pattern-match to a leaked target.

``build_prompt`` is a pure function of its arguments: no randomness, no clock, no
environment. The same ``(unit, symptom, feedback, memory)`` always yields byte-identical
text, so a run is reproducible from its recorded inputs.
"""
from __future__ import annotations

from crucible.sandbox.report import TestReport
from crucible.stream.units import Unit

MODULE_FENCE = "```python"  # the codec: the model returns one fenced python block = the whole module

_PREAMBLE = (
    "You are repairing a single Python function. Exactly one function in the module below "
    "has been altered and is now wrong. Return the COMPLETE corrected module inside one "
    f"{MODULE_FENCE} block, and nothing else -- no prose, no explanation, no partial diff."
)

_INSTRUCTION = (
    "Now output the entire fixed module as one complete "
    f"{MODULE_FENCE} ... ``` block. Reproduce every line the module needs to run -- "
    "imports, every function, all of it -- not just the line you changed. "
    "Output only that block."
)


def _render_symptom(symptom: TestReport) -> str:
    """One short block describing what the visible suite reported for this module."""
    if symptom.infra_error is not None:
        return f"the run could not produce a verdict: {symptom.infra_error}"
    lines: list[str] = []
    if symptom.failed:
        lines.append("failed: " + ", ".join(symptom.failed))
    if symptom.timed_out:
        lines.append("timed out: " + ", ".join(symptom.timed_out))
    if symptom.errored:
        lines.append("errored: " + ", ".join(symptom.errored))
    if not lines:
        lines.append("no visible test failures were reported")
    return "\n".join(lines)


def build_prompt(unit: Unit, symptom: TestReport, *, feedback: str | None = None,
                 memory: str | None = None) -> str:
    """Assemble the deterministic repair prompt for ``unit`` given its ``symptom``.

    On refinement, ``feedback`` (what the previous attempt got wrong) is appended so the
    model can correct course. Returns byte-identical text for identical inputs.

    ``memory`` is the S3 retrieved-memory block (``crucible.memory.retrieve``), which already
    carries its own ``## Prior experience with this code`` header; it is inserted as its own
    section between the Symptom and the instruction -- after the evidence the agent must
    reason from, before the codec order it must obey. ``None`` means *no memory organ*, and
    it is load-bearing that ``None`` adds NOTHING: A_noMem passes ``memory=None`` and its
    prompt must stay byte-for-byte the S2 text, or the arms differ by more than the one
    pre-registered column and the comparison stops being a comparison.
    """
    parts: list[str] = [
        _PREAMBLE,
        "",
        "## Module under repair",
        MODULE_FENCE,
        unit.module_src.rstrip("\n"),
        "```",
        "",
        "## Visible tests (run against the module above)",
        MODULE_FENCE,
        unit.visible_test_src.rstrip("\n"),
        "```",
        "",
        "## Symptom",
        "The visible test suite was executed once and reported:",
        _render_symptom(symptom),
        "",
    ]
    if memory is not None:
        parts += [memory.rstrip("\n"), ""]
    parts.append(_INSTRUCTION)
    if feedback is not None:
        parts += ["", "## Feedback on your previous attempt", feedback]
    return "\n".join(parts) + "\n"
