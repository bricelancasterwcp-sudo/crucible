"""The deterministic lexical symptom scorer (Phase-C prereg §4.2, task-2 brief).

*What this is, and what it deliberately is not.* This module scores how well a candidate
lesson's text overlaps a query's text using plain binary (set-membership) cosine
similarity over tokens -- no embeddings, no external model, no dependency outside the
standard library. Two calls against the same inputs return the same float, always: no
clock is read, no randomness is drawn, and every ranking ends in an ``item_id`` tie-break
so set/dict iteration order never leaks into an observable ordering. That determinism is
the point -- this scorer sits under a pre-registered gate, and a gate whose ranking could
silently reorder itself between two runs on identical data is not a gate.

*Why unit-local test-NAME tokens (``test\\w*``) are dropped.* A lesson and a query both
carry text that mentions the tests involved (a failing test's name in the symptom, a
flipped/killing test's name folded into the diff context elsewhere in the pipeline).
Those names are minted per-unit (``test_v0``, ``test_h1``, ...) -- they are the *label* a
specific unit's test harness happens to use, not a description of the failure's shape.
Two unrelated units can each have a ``test_v0``; leaving that token in would manufacture a
same-name "match" between them that says nothing about whether the underlying symptom is
actually similar. Dropping every token matching ``test\\w*`` removes that unit-local noise
while leaving every other diagnostic word (``return``, ``assert``, ``off_by_one``, ...)
intact, since those genuinely do carry cross-unit signal. This is why ``tokenize``'s split
boundary is ``[^a-z0-9_]`` rather than strictly non-alphanumeric: an identifier like
``test_v0`` must survive tokenizing as one token so the ``test\\w*`` filter can catch it
whole -- splitting on the underscore too would fragment it into ``test``/``v0`` and let
the harness-specific suffix leak through anyway, defeating the filter's purpose (task-2
brief interface comment reads "split on non-alphanumeric"; underscore is read as staying
attached to the word it's part of, the same way Python's own ``\\w`` treats it).

*Why family rides as a TOKEN, not a weight.* The obvious alternative is a hand-tuned bonus
added to the score when ``query.family == lesson.family``. That is a magic number this
module explicitly avoids: instead, ``lesson_text``/``query_text`` fold the family string
into the text itself, so it becomes just another token subject to the same set-overlap
math as everything else. A shared family token contributes exactly one unit of overlap,
on the same footing as any other shared word -- no separate knob to mistune, no second
code path to keep in sync with the rest of the scorer, and the boost falls out of the
tokenizer/scorer pair mechanically rather than being asserted by a constant.

*Where tau lives.* ``TAU`` below is the retrieval-abstention threshold from spec §4.4
(LOCK-C) -- the score below which a candidate is treated as "no good match" rather than
retrieved. It is declared here, at ``None``, because this module's job is to produce
scores, not to decide the cutoff: the value is set exactly once, at the LOCK-C commit that
freezes it against the pre-registered calibration run, and never edited before or after
that commit. Any caller that reads ``TAU`` before the lock sees ``None`` and must treat
that as "not yet decided" -- never coerce it to a default.
"""
from __future__ import annotations

import math
import re

from .schema import SemanticItem

# Retrieval-abstention threshold, spec §4.4 / LOCK-C. Set ONCE, at the lock commit that
# freezes it against the pre-registered calibration run -- never before, never edited
# after. See the module docstring's "where tau lives" paragraph.
TAU: float | None = None

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_TEST_NAME_RE = re.compile(r"test\w*")
_MIN_TOKEN_LEN = 2

_SYMPTOM_HEADER = "\n## Symptom\n"
_SECTION_HEADER_RE = re.compile(r"\n## ")


def tokenize(text: str) -> frozenset[str]:
    """Lowercase; split on runs of non-word characters (``[^a-z0-9_]``, so an identifier
    like ``test_v0`` survives as one token, not three); drop tokens shorter than
    ``_MIN_TOKEN_LEN``; drop any token matching ``test\\w*`` (see module docstring)."""
    candidates = _TOKEN_RE.findall(text.lower())
    return frozenset(
        token for token in candidates
        if len(token) >= _MIN_TOKEN_LEN and not _TEST_NAME_RE.fullmatch(token)
    )


def score(query_tokens: frozenset[str], lesson_tokens: frozenset[str]) -> float:
    """Binary cosine similarity: ``|Q ∩ L| / sqrt(|Q| * |L|)``; ``0.0`` when either side
    is empty (an empty set has no direction to be similar in, so the ratio is undefined
    rather than meaningfully zero -- this guard makes the undefined case an explicit,
    deterministic ``0.0`` instead of a ``ZeroDivisionError``)."""
    if not query_tokens or not lesson_tokens:
        return 0.0
    overlap = len(query_tokens & lesson_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(lesson_tokens))


def lesson_text(item: SemanticItem, episode_symptom: str) -> str:
    """The lesson side's raw text: the landed diff, the episode's own symptom text, and
    the item's family -- family folded in as a token (see module docstring)."""
    return item.landed_diff + "\n" + episode_symptom + "\n" + item.family


def query_text(module_src: str, symptom_text: str, family: str) -> str:
    """The query side's raw text: the module under repair's source, the current
    symptom text, and the query's family -- symmetric with ``lesson_text``."""
    return module_src + "\n" + symptom_text + "\n" + family


def rank(
    query_tokens: frozenset[str],
    candidates: list[tuple[SemanticItem, frozenset[str]]],
) -> list[tuple[float, SemanticItem]]:
    """Score every candidate against ``query_tokens`` and sort by ``(-score,
    item.item_id)`` -- highest score first, ties broken ascending by ``item_id`` so the
    result is identical across runs and independent of the candidates' input order or
    any set/dict iteration order upstream."""
    scored = [(score(query_tokens, lesson_tokens), item) for item, lesson_tokens in candidates]
    return sorted(scored, key=lambda pair: (-pair[0], pair[1].item_id))


def symptom_section(root_prompt: str) -> str:
    """The text between the ``## Symptom`` header and the next ``## `` header (or the end
    of the string), stripped of leading/trailing whitespace; ``""`` when no ``## Symptom``
    header is present at all."""
    start = root_prompt.find(_SYMPTOM_HEADER)
    if start == -1:
        return ""
    content_start = start + len(_SYMPTOM_HEADER)
    next_header = _SECTION_HEADER_RE.search(root_prompt, content_start)
    content_end = next_header.start() if next_header else len(root_prompt)
    return root_prompt[content_start:content_end].strip()
