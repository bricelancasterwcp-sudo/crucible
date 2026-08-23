"""Codec: recover a Python module from a proposer's completion, and decide if it LANDS.

The proposer answers in the full-module-rewrite codec (spec S4.4/S4.7): the corrected
module is returned inside one ``python`` fenced block. This module turns that raw
completion text back into module source and answers the S4.7 question -- did the
submission *land*? A submission LANDS iff the recovered text is non-empty and parses as a
Python module. Landing is a codec property, deliberately upstream of whether any test
passes, so "the codec produced runnable source" never confounds the pass-rate we measure.
The landing rate over a batch (``landing_rate``) is the S4.7 pre-check gate (>= 95%).

Extraction rule: take the LAST ``python`` block. Models frequently restate a wrong
version and then print the corrected one, so the final block is the model's final answer.
With no fenced block at all we fall back to the whole completion, which lands only if the
bare text itself parses.

R-T11-1 lesson (frozen): the parse check runs ``compile()`` under a suppressed
``SyntaxWarning`` filter. A construct like ``x is 'a'`` raises ``SyntaxWarning`` at compile
time, which becomes a hard error under ``-W error``. Landing must be a property of the
*source*, never of the process's warning filters, so we neutralise the filter for the
duration of the check. ``compile(..., "exec")`` only compiles -- it never executes the
candidate, and (unlike ``ast.parse``) it is what actually emits the ``SyntaxWarning`` the
guard exists to contain.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

# One fenced ```python block: the tag, optional trailing spaces/tabs, a newline, then the
# body up to the closing ```. Body is non-greedy; DOTALL lets a body span many lines.
# `findall` yields every block in document order -- we take the last.
_FENCE_RE = re.compile(r"```python[^\S\r\n]*\r?\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Landed:
    """Outcome of decoding one completion.

    ``ok``          -- the submission landed (non-empty source that parses).
    ``module_src``  -- the recovered source (the last fenced block, or the whole text on
                       fallback); kept even when ``ok`` is False so callers can inspect it.
    ``reason``      -- why it did *not* land, or ``None`` when it did:
                         ``None``        landed;
                         ``"empty"``     recovered source was blank;
                         ``"syntax"``    a fenced block was found but does not parse;
                         ``"no-fence"``  no ```python block, and the whole-text fallback
                                         did not parse.
    """

    ok: bool
    module_src: str | None
    reason: str | None


def _parses(src: str) -> bool:
    """True iff ``src`` compiles as a Python module, independent of the process's -W flags.

    ``compile(..., "exec")`` compiles only -- it never executes ``src``. ``SyntaxWarning``
    (e.g. ``x is 'a'``) is suppressed for the duration so it can never be promoted to an
    error under ``-W error`` and mistaken for a landing failure (R-T11-1).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            compile(src, "<landing>", "exec")
        except (SyntaxError, ValueError):
            return False
    return True


def extract_module(text: str) -> Landed:
    """Recover module source from a completion and decide whether it lands (S4.7)."""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        src = blocks[-1]  # the LAST block is the model's final answer
        had_fence = True
    else:
        src = text  # no fenced block -- fall back to the raw completion
        had_fence = False
    if not src.strip():
        return Landed(False, src, "empty")
    if _parses(src):
        return Landed(True, src, None)
    return Landed(False, src, "syntax" if had_fence else "no-fence")


def landing_rate(texts: list[str]) -> float:
    """Fraction of ``texts`` that land -- the S4.7 pre-check statistic. Empty input -> 0.0."""
    if not texts:
        return 0.0
    return sum(extract_module(t).ok for t in texts) / len(texts)
