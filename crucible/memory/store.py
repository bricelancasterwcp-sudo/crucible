"""The memory organ's SQLite store: three typed tables, one payload column each.

*The payload column IS the record.* Every row is ``item_id TEXT PRIMARY KEY, unit_id
TEXT, family TEXT, verified INTEGER, payload TEXT`` where ``payload`` is
``json.dumps(record.to_dict())``. The four non-payload columns exist only so retrieval
can filter with a SQL ``WHERE`` instead of deserializing every row to check a field --
they are never the source of truth and a caller must never be able to lose a field by
going through this store. Reading is always ``from_dict(json.loads(payload))``: the
exact inverse of the write, so no field --  not even ones no index touches -- can be
silently dropped between a write and the next read.

*Honest storage: this store never decides what "relevant" means.* ``mark_falsified``
sets ``falsified_by`` on a row's payload, but ``semantic_for``/``semantic_family``/
``episodes`` keep returning that row -- filtering falsified or stale items out is the
retriever's job (a later task in this plan), not the store's. A store that quietly hid
falsified rows would make replaying an old decision impossible to distinguish from a row
that was never falsified in the first place.

*Single-threaded by design.* One :class:`MemoryStore` is opened per arm run and used
from that run's single driver thread only -- there is no background writer, no worker
pool touching this file, so the connection is opened with sqlite3's default
``check_same_thread=True`` deliberately: a second thread trying to use this connection is
a bug in the caller, and the default lets sqlite3 catch that loudly instead of silently
racing on shared state.

*Procedural exists, but nothing writes it (R-S3-1).* The procedural table is created with
the same column layout as ``semantic`` because a later slice may populate it without a
schema migration, but S3 has no ``write_procedural`` method at all -- skills defer to a
later slice, per the design spec's ruling. Its absence is itself the pin: a caller
reaching for ``store.write_procedural(...)`` gets ``AttributeError``, not a method that
silently no-ops.

*``item_id`` is minted by the caller, not this module.* Task 1's ``content_id("episode",
...)`` / ``content_id("semantic", ...)`` mint the id before a record is ever constructed;
this store just persists whatever ``rec.item_id`` / ``item.item_id`` already is as the
primary key. ``write_episode``/``write_semantic`` are ``INSERT OR REPLACE``, so writing a
record that shares an existing ``item_id`` overwrites that row in place -- the mechanism
behind "one episode per (task_key, arm)" and "one lesson per cited_episode_id" that Task
1's docstring describes as the identity contract, not something this module re-derives.

No clock is read anywhere in this module: ``mark_verified``'s timestamp is supplied by
the caller, matching every other timestamp field in the schema.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schema import EpisodicRecord, SemanticItem

_TABLES = ("episodic", "semantic", "procedural")

# Every table shares this layout (see the module docstring's "payload IS the record").
# ``verified`` is only ever set from a real field for the episodic table (EpisodicRecord
# has a ``verified`` field; SemanticItem does not) -- semantic/procedural rows always
# store 0 there since nothing filters on it for those tables.
_CREATE_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS {table} ("
    "item_id TEXT PRIMARY KEY, "
    "unit_id TEXT NOT NULL, "
    "family TEXT NOT NULL, "
    "verified INTEGER NOT NULL DEFAULT 0, "
    "payload TEXT NOT NULL)"
)
_CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_{table}_unit_family ON {table}(unit_id, family)"


class MemoryStore:
    """One SQLite file per arm run, holding the episodic, semantic, and procedural tables.

    ``__init__`` creates the file (if it doesn't exist) and all three tables (if they
    don't exist) idempotently -- opening an already-populated db is a no-op on the
    schema and leaves every prior row untouched. See the module docstring for the
    single-threaded and honest-storage guarantees.
    """

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        for table in _TABLES:
            self._conn.execute(_CREATE_TABLE_SQL.format(table=table))
            self._conn.execute(_CREATE_INDEX_SQL.format(table=table))
        self._conn.commit()

    def write_episode(self, rec: EpisodicRecord) -> None:
        """INSERT OR REPLACE by ``item_id`` -- a re-write of the same identity overwrites."""
        self._conn.execute(
            "INSERT OR REPLACE INTO episodic (item_id, unit_id, family, verified, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (rec.item_id, rec.unit_id, rec.family, int(rec.verified), json.dumps(rec.to_dict())),
        )
        self._conn.commit()

    def write_semantic(self, item: SemanticItem) -> None:
        """INSERT OR REPLACE by ``item_id``. ``verified`` indexed column is unused here (always 0)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO semantic (item_id, unit_id, family, verified, payload) "
            "VALUES (?, ?, ?, 0, ?)",
            (item.item_id, item.unit_id, item.family, json.dumps(item.to_dict())),
        )
        self._conn.commit()

    def episodes(self, verified_only: bool = False) -> list[EpisodicRecord]:
        """All episodes in insertion order, or only the verified ones if ``verified_only``."""
        sql = "SELECT payload FROM episodic"
        if verified_only:
            sql += " WHERE verified = 1"
        sql += " ORDER BY rowid"
        rows = self._conn.execute(sql).fetchall()
        return [EpisodicRecord.from_dict(json.loads(row[0])) for row in rows]

    def semantic_for(self, unit_id: str, family: str) -> list[SemanticItem]:
        """Exact class match: lessons filed under this exact (unit_id, family) pair.

        Honest storage: falsified items are included -- see the module docstring.
        """
        rows = self._conn.execute(
            "SELECT payload FROM semantic WHERE unit_id = ? AND family = ? ORDER BY rowid",
            (unit_id, family),
        ).fetchall()
        return [SemanticItem.from_dict(json.loads(row[0])) for row in rows]

    def semantic_family(self, family: str) -> list[SemanticItem]:
        """Family-wide: every lesson filed under this family, any unit_id. Superset of ``semantic_for``."""
        rows = self._conn.execute(
            "SELECT payload FROM semantic WHERE family = ? ORDER BY rowid",
            (family,),
        ).fetchall()
        return [SemanticItem.from_dict(json.loads(row[0])) for row in rows]

    def mark_falsified(self, item_id: str, falsified_by: str) -> None:
        """Set ``falsified_by`` on whichever table holds ``item_id``. Row is kept, not deleted."""
        self._patch_payload(item_id, "falsified_by", falsified_by)

    def mark_verified(self, item_id: str, at: str) -> None:
        """Set ``last_verified_at`` on whichever table holds ``item_id``. ``at`` is caller-supplied."""
        self._patch_payload(item_id, "last_verified_at", at)

    def _patch_payload(self, item_id: str, field: str, value: str) -> None:
        """Patch one payload field in place, searching episodic then semantic.

        Both dataclasses carry ``falsified_by``/``last_verified_at`` by the same name, so
        this can patch either table's JSON without knowing which class a row holds --
        the payload dict is the unit of work, not the dataclass. ``item_id`` spaces
        never collide across kinds (Task 1's ``content_id`` hashes ``kind`` in), so at
        most one table can hold a given id.
        """
        for table in ("episodic", "semantic"):
            row = self._conn.execute(f"SELECT payload FROM {table} WHERE item_id = ?", (item_id,)).fetchone()
            if row is None:
                continue
            payload = json.loads(row[0])
            payload[field] = value
            self._conn.execute(
                f"UPDATE {table} SET payload = ? WHERE item_id = ?",
                (json.dumps(payload), item_id),
            )
            self._conn.commit()
            return
        raise KeyError(f"item_id not found in episodic or semantic: {item_id}")

    def verified_count(self) -> int:
        """Total verified episodes -- the sleep trigger's running total."""
        (count,) = self._conn.execute("SELECT COUNT(*) FROM episodic WHERE verified = 1").fetchone()
        return count

    def count_verified_since(self, marker: int) -> int:
        """``verified_count() - marker``, clamped at 0.

        A convenience form of the sleep trigger's own arithmetic: a caller snapshots
        ``verified_count()`` at the last accepted sleep as ``marker`` and calls this to
        find out how many NEW verified episodes have landed since. Clamped so a stale or
        future marker never reports a negative count.
        """
        return max(0, self.verified_count() - marker)

    def close(self) -> None:
        self._conn.close()
