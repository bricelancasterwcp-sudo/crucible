"""The records a run writes: one ``ExecRecord`` per charged execution, one ``TaskRecord``
per (arm, task) attempt -- and the jsonl writers/readers that put them on disk.

These two dataclasses are the seam between the driver (Task 12), which fills them in, and
the lens (Task 13), which reduces the ``TaskRecord``s to success-by-phase for E1/E2. They
are frozen so a record cannot be edited after the attempt it describes is over.

*``hidden_pass`` is ``bool | None``, and the ``None`` is load-bearing.* It is THE OUTCOME
the driver computes by running the hidden test -- but only when the attempt actually got
that far. ``None`` means the attempt was never scored: infra died, or it was not measured.
Coercing that ``None`` to ``False`` would silently turn "we don't know" into "it failed"
and bias every rate the lens computes, so ``from_dict`` restores it verbatim (``cls(**d)``,
no coercion) and ``to_dict`` never drops it. That property is what the mutation check bites.

``ExecRecord``'s fields are all JSON-native scalars (``str``/``float``/``int``/``bool``/
``None``), so its ``to_dict`` is a plain ``asdict`` and its ``from_dict`` a plain
``cls(**d)``. ``TaskRecord`` carries ONE tuple (``retrieved_ids``, added for S3's A_full),
so its pair reshapes exactly that field and nothing else. The jsonl files follow the S1
store convention (``crucible/stream/store.py``): one JSON object per line,
``sort_keys=True`` so identical records are identical bytes, UTF-8 pinned so the bytes do
not depend on the writer's locale.

*The two S3 fields are trailing, defaulted, and read with ``.get``.* ``retrieved_ids`` and
``adapter_id`` are stamped only by the A_full hooks (:mod:`crucible.run.full`); every S2
arm leaves them at their defaults, and an S2-era ``task_records.jsonl`` line -- written
before either field existed -- still loads, because ``from_dict`` reads both with ``.get``
rather than requiring the key. That matters on the resume path: the driver reads back the
records an earlier process wrote and would otherwise refuse a run it started itself.
``retrieved_ids == ()`` means "nothing was retrieved" and ``adapter_id is None`` means "the
base model served this attempt" -- the None-vs-zero discipline applied to both.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TASK_RECORDS_FILE = "task_records.jsonl"
EXEC_RECORDS_FILE = "exec_records.jsonl"


@dataclass(frozen=True)
class ExecRecord:
    """One charged execution -- a single node run the budget paid for.

    ``charged`` records whether this execution counted against the arm's budget; an
    infra-failed run (``infra_error`` set) is written down as provenance even when it did
    not charge, so "why did the budget move" is answerable from the file alone.
    """

    task_key: str
    arm: str
    node_id: str
    visible_reward: float
    charged: bool
    wall_s: float
    infra_error: str | None

    def to_dict(self) -> dict:
        """JSON-ready form. All fields are JSON-native scalars, so ``asdict`` is exact."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExecRecord":
        """Inverse of :meth:`to_dict`; no field needs coercion to restore equality."""
        return cls(**d)


@dataclass(frozen=True)
class TaskRecord:
    """One task attempt by one arm -- the unit E1/E2 reduce over.

    ``hidden_pass`` is the driver-computed outcome: ``True``/``False`` when the hidden test
    was actually run, ``None`` when the attempt was never scored (infra failure, or not
    measured). ``tampered`` flags an attempt that reached the hidden oracle illegitimately.
    ``tokens`` and ``gpu_s`` are ``None`` when the serving path did not report them.

    ``retrieved_ids`` (the memory items whose text was in this attempt's prompt) and
    ``adapter_id`` (the sleep-trained adapter that served it, ``None`` for the base model)
    are stamped by the A_full hooks alone -- see the module docstring.
    """

    task_key: str
    arm: str
    unit_id: str
    family: str
    phase: int
    kind: str
    landed: bool
    status: str
    confidence: float
    visible_reward: float
    executions_charged: int
    hidden_pass: bool | None
    tampered: bool
    infra_error: str | None
    tokens: int | None
    wall_s: float
    gpu_s: float | None
    retrieved_ids: tuple[str, ...] = ()
    adapter_id: str | None = None

    def to_dict(self) -> dict:
        """JSON-ready form. Every field is carried -- ``hidden_pass`` included, ``None`` and
        all -- with ``retrieved_ids`` as a list so a file round trip is exact."""
        d = asdict(self)
        d["retrieved_ids"] = list(self.retrieved_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskRecord":
        """Inverse of :meth:`to_dict`. ``hidden_pass`` is restored verbatim -- never coerced
        to ``False`` -- keeping "not measured" distinct from "failed"; the two S3 fields are
        read with ``.get`` so an S2-era line still loads (see the module docstring)."""
        d = dict(d)
        d["retrieved_ids"] = tuple(d.get("retrieved_ids", ()))
        d["adapter_id"] = d.get("adapter_id")
        return cls(**d)


def write_records(path: Path, task_recs: list[TaskRecord], exec_recs: list[ExecRecord]) -> None:
    """Write both record files under ``path``: ``task_records.jsonl`` + ``exec_records.jsonl``.

    One JSON object per line, ``sort_keys=True``, UTF-8 -- the S1 store convention -- so the
    files are appendable, greppable, and byte-stable. Order is the caller's list order,
    never sorted, so the file traces straight back to attempt order.
    """
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    _write_jsonl(d / TASK_RECORDS_FILE, (r.to_dict() for r in task_recs))
    _write_jsonl(d / EXEC_RECORDS_FILE, (r.to_dict() for r in exec_recs))


def _write_jsonl(path: Path, dicts) -> None:
    """One object per line, keys sorted -- the same records written twice are the same bytes."""
    with open(path, "w", encoding="utf-8") as fh:
        for obj in dicts:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")


def read_task_records(path: Path) -> list[TaskRecord]:
    """Every ``TaskRecord`` under ``path``, in write order -- the exact records written."""
    p = Path(path) / TASK_RECORDS_FILE
    with open(p, encoding="utf-8") as fh:
        return [TaskRecord.from_dict(json.loads(line)) for line in fh if line.strip()]
