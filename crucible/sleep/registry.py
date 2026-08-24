"""The adapter registry: one JSONL row per trained-and-gated adapter, append-only.

Each sleep cycle (a later task) trains one candidate adapter and puts it through a
regression gate; win or lose, the outcome is recorded here -- an accepted adapter that
never got written down would be indistinguishable from one that was never trained, and a
rejected one that got silently dropped would make "why does the server still have the old
adapter loaded" impossible to answer from the file alone. Every ``record`` call appends
exactly one line; nothing here ever rewrites or deletes a prior line -- the same
append-only discipline ``crucible.stream.store``'s ``validations.jsonl`` uses, for the same
reason (a dropped outcome must not look like an outcome that never happened).

*A crash DURING a ``record`` call's write is tolerated; a crash that corrupts an already-
completed line is not (fix, review finding 2).* A process killed mid-``write()`` can leave
a torn, unparseable FINAL line on disk -- that ``record`` call never actually committed a
row, so ``_read_all`` drops an unparseable final line silently rather than raising (see its
own docstring). This is not the same claim as "the whole training history survives a crash
mid-cycle": it survives a crash mid-*append*, because the append that was interrupted never
counted as a row in the first place. An unparseable line that is NOT the last one is a
different, worse failure -- some earlier, already-completed write left the file corrupted
underneath later writes that still landed -- and ``_read_all`` still raises loudly on that,
exactly as it did before this fix.

*``adapter_id`` is identity, not description (the same rule ``schema.py``'s
``content_id`` follows).* ``adapter_id_for`` mints it as ``"ad-" + episode_set_hash[:16]``
-- a prefix of the sha256 ``select.episode_set_hash`` computes over the exact
verified-episode pairs that trained the adapter -- so two calls that trained on the same
episode set (same content, same order) always mint the same adapter id, and a caller can
recognise "we already trained this exact set" without re-hashing anything itself. This
module does not compute ``episode_set_hash`` -- ``record`` takes it as a plain string
argument, and ``adapter_id_for`` only derives the id from an already-computed digest, so
``select.py`` stays the single place that hash is defined.

*``latest_accepted`` skips rejected rows -- it does not just return the last row.* A
sleep cycle that trains a candidate and then rejects it at the regression gate still
writes a row (``accepted=False``); a caller asking "what adapter is the server allowed to
be running" must never get that rejected candidate back. The scan walks the file in
REVERSE write order and returns the first ``accepted=True`` row's id, so a later accepted
adapter always wins over an earlier one, and any number of trailing rejected rows since
the last acceptance are correctly invisible to it.

No clock is read anywhere in this module: ``created_at`` is caller-supplied, matching
every other timestamp field in this codebase (see ``crucible.memory.store``'s module
docstring).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterRecord:
    """One row of the registry -- exactly what ``AdapterRegistry.record`` writes and reads back."""

    adapter_id: str
    episode_set_hash: str
    base_digest: str
    accepted: bool
    created_at: str

    def to_dict(self) -> dict:
        """JSON-ready form. Every field is a JSON-native scalar, so ``asdict`` is exact."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AdapterRecord":
        """Inverse of :meth:`to_dict`; no field needs coercion to restore equality."""
        return cls(**d)


def adapter_id_for(episode_set_hash: str) -> str:
    """``"ad-" + episode_set_hash[:16]`` -- the deterministic, content-addressed adapter id.

    Deterministic from the data (identity-not-description): the same episode-set hash
    always mints the same adapter id, so a caller can tell "we already trained this exact
    verified-episode set" without keeping any state of its own.
    """
    return "ad-" + episode_set_hash[:16]


class AdapterRegistry:
    """Append-only JSONL ledger of trained adapters at ``path``.

    ``path``'s parent directories are created on construction
    (``mkdir(parents=True, exist_ok=True)``); the file itself is created lazily by the
    first ``record`` call -- an ``AdapterRegistry`` over a path nothing has been recorded
    to yet is a legitimate, empty registry, not an error (``latest_accepted`` returns
    ``None`` for it).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, adapter_id: str, episode_set_hash: str, base_digest: str,
               accepted: bool, created_at: str) -> None:
        """Append one row. One JSON object per line, keys sorted, UTF-8 -- the S1 store
        convention (``crucible.stream.store``, ``crucible.run.records``): the same row
        written twice is the same bytes.
        """
        rec = AdapterRecord(adapter_id=adapter_id, episode_set_hash=episode_set_hash,
                             base_digest=base_digest, accepted=accepted, created_at=created_at)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")

    def _read_all(self) -> list[AdapterRecord]:
        """Every row in write order, or ``[]`` if nothing has been recorded yet.

        Tolerates exactly one failure mode: the FINAL non-blank line failing to parse as
        JSON -- what a crash mid-``record()`` write leaves behind. That call's row never
        actually committed, so there is nothing to recover and the torn tail is dropped,
        the same way a database WAL discards an incomplete final entry (see the module
        docstring). A non-blank line BEFORE the final one that fails to parse is a
        different, worse thing: a completed write left that byte range malformed while
        later writes still landed after it -- real corruption, not a torn append -- so
        that case still raises ``json.JSONDecodeError`` rather than silently skipping data
        loss in the middle of the ledger.
        """
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        records: list[AdapterRecord] = []
        last_index = len(lines) - 1
        for i, line in enumerate(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if i == last_index:
                    break  # torn write: this record() call never completed -- nothing to recover
                raise
            records.append(AdapterRecord.from_dict(payload))
        return records

    def latest_accepted(self) -> str | None:
        """The most recently recorded ``accepted=True`` adapter id, or ``None`` if none yet.

        Walks the file in reverse write order -- see the module docstring's "skips
        rejected rows" note.
        """
        for rec in reversed(self._read_all()):
            if rec.accepted:
                return rec.adapter_id
        return None
