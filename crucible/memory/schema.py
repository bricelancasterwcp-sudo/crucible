"""The memory organ's two content-stores, as typed records -- schema ported, no code vendored.

Field provenance (pre-reg §9, binding): **MemOS**'s (Apache-2.0) `TextualMemoryMetadata`
contributes ``confidence``, ``source_locator``, ``status``, ``version`` -- its provenance
metadata for a single memory item. **Graphiti**'s (Apache-2.0) bi-temporal fact edges
contribute ``valid_at`` / ``invalid_at`` / ``expired_at`` -- when a fact became true, when
it stopped being true, and when the store retired it, three different questions MemOS
does not separate. **MIRIX**'s (Apache-2.0) ``skill_experience`` credibility/evidence/
lineage model contributes the shape of ``last_verified_at`` / ``falsified_by`` /
``verification_method`` -- our own fields, not theirs verbatim, because MIRIX's lineage is
skill-shaped and ours is execution-shaped (a lesson is falsified by *re-running* the test
it cites, not by a credibility score decaying). Only the field *shapes* were read; no
MemOS/Graphiti/MIRIX source is imported or copied here -- see ``THIRD_PARTY.md``.

Two things here are load-bearing.

*Keys are identities, not descriptions.* ``content_id`` hashes ``{"kind": kind,
**identity_fields}`` -- never a human-readable summary, a prompt, or a diff. An
``EpisodicRecord``'s identity is ``{"task_key", "arm"}`` (spec: one episode per task per
arm -- a re-run of the same task by the same arm is the *same* episode, overwritten, not
duplicated); a ``SemanticItem``'s identity is ``{"cited_episode_id"}`` (one lesson per
verified episode, per ruling R-S3-2's mechanical-template writer). Everything else on
either record -- the prompt, the diff, the confidence -- describes the record; none of it
is allowed to participate in its id, or two writes that only differ in a description
field would silently mint two rows for what is the same fact.

*``verified`` is derived, never stored as an independent truth.* ``episode_verified``
implements the pre-reg success definition (hidden-suite pass, untampered) as a pure
function of its two inputs so no caller can construct an ``EpisodicRecord`` whose
``verified`` flag disagrees with its own ``hidden_pass``/tamper evidence -- the field on
the dataclass is there for the store to filter on cheaply, but this function is the one
place its value is decided. ``hidden_pass=None`` (the hidden suite has not run yet) is
distinct from ``hidden_pass=False`` (it ran and failed) and neither is verified -- the
None-vs-zero discipline that runs through this codebase (see ``stream/compose.py``'s
``EXCLUSION_REASONS`` seeding).

Like ``stream/units.py``, every id minted here is a content hash of text, never a
description of one; like ``stream/compose.py``'s ``TaskSpec``, both dataclasses below are
frozen with a positional field order downstream tasks construct against, and their
``to_dict``/``from_dict`` pair converts every tuple to a list and back so a JSON round
trip through a file or a SQLite TEXT column is exact.

No clock is read anywhere in this module: ``created_at``, ``valid_at`` and every other
timestamp field is a plain string the caller supplies, so nothing here depends on when it
happens to run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from ..stream.units import sha256_text

# Redefined rather than imported from stream.mutants: that module pulls in cosmic_ray,
# and this is a leaf schema module that should stay importable without it.
Span = tuple[tuple[int, int], tuple[int, int]]


def content_id(kind: str, identity_fields: dict) -> str:
    """Content-addressed id: sha256 of the canonical (sorted-key) JSON of ``{"kind": kind, **identity_fields}``.

    Insertion order of ``identity_fields`` never matters -- ``json.dumps(..., sort_keys=True)``
    fixes the key order before hashing, so two callers who assemble the same identity in a
    different order still land on the same id. ``kind`` separates id spaces that could
    otherwise collide: an episode and a semantic item whose identity fields happen to
    coincide in value must not mint the same id.
    """
    return sha256_text(json.dumps({"kind": kind, **identity_fields}, sort_keys=True))


def episode_verified(hidden_pass: bool | None, tampered: bool) -> bool:
    """The pre-reg success definition: hidden suite passed, and the landing was untampered.

    ``hidden_pass is True`` -- not merely truthy -- so ``None`` (hidden suite never ran)
    is explicitly excluded rather than coerced. See the module docstring's None-vs-zero
    note.
    """
    return hidden_pass is True and not tampered


@dataclass(frozen=True)
class EpisodicRecord:
    """One (task, arm) attempt, written mechanically by the driver for every attempt, pass or fail.

    Field order is frozen -- the store (Task 2) and every later task construct this
    positionally. ``item_id`` is minted by ``content_id("episode", {"task_key": ...,
    "arm": ...})`` before construction -- never hand-assigned, per the module docstring's
    identity-not-description rule. ``verified`` is the caller's call to
    ``episode_verified`` at construction time, not recomputed here; the dataclass does not
    re-derive it because a frozen field either holds the value the caller decided or it
    holds nothing -- there is no third option that reads a clock or re-runs a test.
    """

    item_id: str
    task_key: str
    arm: str
    unit_id: str
    family: str
    class_id: str
    phase: int
    kind: str
    root_prompt: str
    landed_module: str | None
    visible_reward: float
    executions_charged: int
    hidden_pass: bool | None
    verified: bool
    memory_item_ids: tuple[str, ...]
    created_at: str
    confidence: float
    status: str
    version: int
    source_locator: str
    valid_at: str
    invalid_at: str | None
    expired_at: str | None
    last_verified_at: str | None
    falsified_by: str | None
    verification_method: str

    def to_dict(self) -> dict:
        """JSON-ready form: ``memory_item_ids`` becomes a list so a file round-trip is exact."""
        d = asdict(self)
        d["memory_item_ids"] = list(self.memory_item_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicRecord":
        """Inverse of :meth:`to_dict`; restores the tuple shape so equality holds."""
        d = dict(d)
        d["memory_item_ids"] = tuple(d["memory_item_ids"])
        return cls(**d)


@dataclass(frozen=True)
class SemanticItem:
    """One lesson, mechanically templated from a single verified episode (ruling R-S3-2).

    Field order is frozen -- the store (Task 2) and the lesson renderer (Task 3) construct
    this positionally. ``item_id`` is minted by ``content_id("semantic",
    {"cited_episode_id": ...})`` before construction -- one lesson per verified episode, so
    re-templating the same episode overwrites rather than duplicates. ``mutated_spans`` is
    a tuple of ``Span`` (the same ``((line, col), (line, col))`` shape as
    ``stream/compose.py``'s ``TaskSpec.span``/``span2``, generalised to a tuple so a
    rung-1 stacked mutant's two sites both fit) rather than the mutant object itself --
    only the spans, since the item cites its episode for everything else about the fix.
    """

    item_id: str
    unit_id: str
    family: str
    class_id: str
    cited_episode_id: str
    mutated_spans: tuple[Span, ...]
    landed_diff: str
    flipped_tests: tuple[str, ...]
    killing_tests: tuple[str, ...]
    created_at: str
    confidence: float
    status: str
    version: int
    source_locator: str
    valid_at: str
    invalid_at: str | None
    expired_at: str | None
    last_verified_at: str | None
    falsified_by: str | None
    verification_method: str

    def to_dict(self) -> dict:
        """JSON-ready form: every tuple (spans, nested included) becomes a list."""
        d = asdict(self)
        d["mutated_spans"] = [[list(span[0]), list(span[1])] for span in self.mutated_spans]
        d["flipped_tests"] = list(self.flipped_tests)
        d["killing_tests"] = list(self.killing_tests)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticItem":
        """Inverse of :meth:`to_dict`; restores every tuple shape so equality holds."""
        d = dict(d)
        d["mutated_spans"] = tuple((tuple(span[0]), tuple(span[1])) for span in d["mutated_spans"])
        d["flipped_tests"] = tuple(d["flipped_tests"])
        d["killing_tests"] = tuple(d["killing_tests"])
        return cls(**d)
