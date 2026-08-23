"""A ``Candidate``: one proposer's repair attempt, plus the scores search ranks it by.

A candidate is the model's *full-module rewrite* after codec extraction -- not a diff --
so ``text`` is fed to the sandbox verbatim as the module source. The two scores are the
proposer's own confidence signals: ``mean_logprob`` is the length-normalised log-prob of
the generated tokens, and ``self_certainty`` a separate self-consistency score. Both are
``None`` when the serving path did not return logprobs, so search must treat absent scores
as unknown rather than as zero.

Round-tripping matters because candidates cross a file boundary (they land in the record).
``to_dict``/``from_dict`` are exact inverses: the dataclass holds only JSON-native scalars,
so ``asdict`` and ``cls(**d)`` suffice with no tuple/enum coercion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Candidate:
    """One candidate module rewrite and the proposer confidence scores search ranks by."""

    text: str
    mean_logprob: float | None
    self_certainty: float | None

    def to_dict(self) -> dict:
        """JSON-ready form. All fields are JSON-native, so a plain ``asdict`` is exact."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        """Inverse of :meth:`to_dict`; the fields need no coercion to restore equality."""
        return cls(**d)
