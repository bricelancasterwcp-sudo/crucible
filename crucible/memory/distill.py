"""The mechanical distiller (ruling R-S3-2): verified episode -> ``SemanticItem``, no LLM.

*Mechanical, not generative.* Every field of the produced ``SemanticItem`` is either
copied straight off the episode (``unit_id``, ``family``, ``class_id``), computed by a
pure function of the caller's inputs (``landed_diff`` via ``difflib``, ``item_id`` via
``content_id``), or a fixed literal (``status``, ``version``, ``verification_method``).
There is no proposer call, no summarisation, no judgment anywhere in this module --
that is the whole point of R-S3-2: a later exploratory variant may add proposer
self-distillation behind a flag, but this instrument's semantic writer never invokes a
model.

*The guard is the contract.* ``distill`` REFUSES (raises ``ValueError``) an episode that
is not verified, or one with ``landed_module is None`` -- either would mint a lesson
from an attempt that never actually produced a working fix. ``verified`` is read as the
dataclass field, not recomputed from ``hidden_pass``/tamper evidence (``schema.py``'s
``episode_verified`` is the one place that derivation happens; by the time an episode
reaches this module its ``verified`` field is already the caller's considered answer).

*The landed diff is against the MUTATED source, not the original.* ``mutated_src`` (the
bug the agent actually saw) plays the ``a`` side; ``episode.landed_module`` (the patch
that passed re-execution) plays the ``b`` side. ``_unified`` mirrors
``stream/mutants.py``'s ``_unified`` idiom byte-for-byte (fixed ``a/<module>.py ->
b/<module>.py`` headers, **no** dates -- difflib would otherwise stamp a timestamp and
the same fix would mint a different diff text, hence a different lesson, on every run).
It is reimplemented here rather than imported: ``stream.mutants`` pulls in
``cosmic_ray``, and this leaf module (like ``schema.py``'s redefinition of ``Span``)
should stay importable without that dependency. The module name for the header comes
from ``stream.units.module_name_for(episode.unit_id)`` -- reused, not reimplemented,
since ``EpisodicRecord`` does not itself carry a ``module_name`` field.

*``render_lesson`` is a pure function of the item.* No clock, no randomness -- calling
it twice on the same ``SemanticItem`` produces identical bytes, which is what lets the
lesson text participate in a deterministic prompt (pre-reg §8, instrument honesty).

*Inferred, not pinned by the brief:* ``confidence=1.0`` at mint time (a freshly
distilled lesson comes from a hidden-suite-verified, untampered episode -- full
confidence until falsification or a later calibration pass revises it; Task 8's
``Calibrator`` is a separate, later-consulted signal, not something this module reads)
and ``source_locator=f"episode:{episode.item_id}"`` (self-sufficient: ``distill`` is not
given a run id, so the locator names the one thing it does have -- the cited episode --
rather than inventing run context it was never handed).
"""
from __future__ import annotations

import difflib
import json

from ..stream.units import module_name_for
from .schema import EpisodicRecord, SemanticItem, Span, content_id

LESSON_TEMPLATE = """### Prior verified fix in this code (family {family})
The altered region was at spans {spans}. The repair that passed re-execution:
```diff
{landed_diff}
```
Visible tests that flipped from failing to passing: {flipped_tests}."""
"""The ONE fixed lesson template (R-S3-2) -- a spec-locked prompt surface pinned
byte-for-byte by the task brief. The four placeholders (``family``, ``spans``,
``landed_diff``, ``flipped_tests``) are the entire contract; the surrounding prose is
not free-form and must not be reworded for taste, the same way a wire-format string
literal is not reformatted. ``spans`` is rendered in JSON list form and the two test
tuples are rendered ``", ".join``-ed -- see ``render_lesson``. ``killing_tests`` is
carried on the item for later falsification (Task 5) but does not appear in this
template at all."""


def _unified(module_name: str, a: str, b: str) -> str:
    """Unified diff with fixed headers and **no** dates -- mirrors ``stream.mutants._unified``."""
    return "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                        fromfile=f"a/{module_name}.py", tofile=f"b/{module_name}.py", n=3))


def distill(episode: EpisodicRecord, *, mutated_src: str, spans: tuple[Span, ...],
            flipped_tests: tuple[str, ...], killing_tests: tuple[str, ...], now: str) -> SemanticItem:
    """Mechanically template one verified episode into a ``SemanticItem``. See module docstring.

    Raises ``ValueError`` if ``episode.verified`` is not ``True`` or
    ``episode.landed_module`` is ``None`` -- both checked, independently, before anything
    else runs.
    """
    if not episode.verified:
        raise ValueError(f"distill refuses a non-verified episode: {episode.item_id}")
    if episode.landed_module is None:
        raise ValueError(f"distill refuses an episode with no landed_module: {episode.item_id}")

    module_name = module_name_for(episode.unit_id)
    landed_diff = _unified(module_name, mutated_src, episode.landed_module)
    cited_episode_id = episode.item_id
    item_id = content_id("semantic", {"cited_episode_id": cited_episode_id})

    return SemanticItem(
        item_id=item_id,
        unit_id=episode.unit_id,
        family=episode.family,
        class_id=episode.class_id,
        cited_episode_id=cited_episode_id,
        mutated_spans=tuple(spans),
        landed_diff=landed_diff,
        flipped_tests=tuple(flipped_tests),
        killing_tests=tuple(killing_tests),
        created_at=now,
        confidence=1.0,
        status="active",
        version=1,
        source_locator=f"episode:{cited_episode_id}",
        valid_at=now,
        invalid_at=None,
        expired_at=None,
        last_verified_at=None,
        falsified_by=None,
        verification_method="mechanical-template",
    )


def render_lesson(item: SemanticItem) -> str:
    """Render ``LESSON_TEMPLATE`` for ``item``. Pure function of ``item`` -- byte-stable."""
    spans_json = json.dumps(item.to_dict()["mutated_spans"])
    return LESSON_TEMPLATE.format(
        family=item.family,
        spans=spans_json,
        landed_diff=item.landed_diff,
        flipped_tests=", ".join(item.flipped_tests),
    )
