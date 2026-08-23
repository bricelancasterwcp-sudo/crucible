"""Write a composed stream to a directory, and read it back exactly.

A stream is the thing a Phase-A run replays, so it has to survive the trip to disk
without changing. Everything here is in service of that one property.

*The directory is named for the stream, not for the run.* ``stream_dir`` is
``root / stream_hash[:12]`` -- the identity a run record points at. Two composions that
produce the same stream land on the same directory and, because the JSON is written with
``sort_keys=True``, the same bytes; a different stream cannot collide with them, because
the name *is* the content hash. Re-writing a stream over itself is therefore an in-place
overwrite of identical files, which is why ``write_stream`` uses ``exist_ok=True``
throughout rather than refusing or clearing the directory first.

*Units are keyed by module name, not by unit id.* A real unit id is an EvalPlus task id
(``"HumanEval/0"``) and the slash is not a directory. ``module_name_for`` -- the same
mapping the sandbox uses to name the module it imports -- turns it into
``unit_humaneval_0``, and ``read_unit`` maps through it again, so the reader takes the
same unit id the manifest carries.

*``validations.jsonl`` holds every verdict, valid or not.* The invalid ones are the
provenance for the exclusions Task 13 counts: an ``equivalent`` mutant that is written
down is a measurement, one that is dropped is indistinguishable from a mutant nobody
made. It is JSONL rather than a JSON array so the file stays appendable and greppable.

*``build_dropped.jsonl`` names the records that never became units.* Same argument as the
invalid validations, one stage earlier: a canonical that would not compile, failed its own
visible self-check or oracle-errored is provenance about *which* inputs the instrument could
not measure, and "dropped 7" in a log throws that identity away. It is written at the same
atomic seam as the rest of the stream (see ``pipeline._write_atomic``) in ``build_units``'
record order, never sorted, so it does not leak caller order and does not touch ``stream_hash``.

Text IO is pinned to UTF-8 everywhere. The default would be the locale's encoding, which
would make the bytes on disk -- and so a re-read of a content-hashed source -- depend on
the environment that happened to write them.
"""
from __future__ import annotations

import json
from pathlib import Path

from .build import Dropped
from .compose import StreamManifest
from .mutants import Mutant
from .units import Unit, module_name_for
from .validate import Validation


def stream_dir(root: Path, manifest: StreamManifest) -> Path:
    """Where ``manifest``'s stream lives under ``root``: the first 12 chars of its hash."""
    return Path(root) / manifest.stream_hash[:12]


def write_stream(root: Path, manifest: StreamManifest, units: list[Unit], mutants: dict[str, Mutant],
                 validations: list[Validation]) -> Path:
    """Write the whole stream under ``root`` and return its directory.

    Layout: ``manifest.json``; ``units/<module_name>/{module.py,test_visible.py,
    test_hidden.py,unit.json}``; ``mutants/<key>.json``; ``validations.jsonl``. The three
    ``.py`` files are the sources verbatim, so the unit directory is directly runnable and
    readable by a human; ``unit.json`` is the record the reader reconstructs from.
    """
    d = stream_dir(root, manifest)
    (d / "units").mkdir(parents=True, exist_ok=True)
    (d / "mutants").mkdir(exist_ok=True)
    _write_json(d / "manifest.json", manifest.to_dict(), indent=1)
    for u in units:
        ud = d / "units" / u.module_name
        ud.mkdir(exist_ok=True)
        (ud / "module.py").write_text(u.module_src, encoding="utf-8")
        (ud / "test_visible.py").write_text(u.visible_test_src, encoding="utf-8")
        (ud / "test_hidden.py").write_text(u.hidden_test_src, encoding="utf-8")
        _write_json(ud / "unit.json", u.to_dict())
    for key, m in mutants.items():
        _write_json(d / "mutants" / f"{key}.json", m.to_dict())
    with open(d / "validations.jsonl", "w", encoding="utf-8") as fh:
        for v in validations:
            fh.write(json.dumps(v.to_dict(), sort_keys=True) + "\n")
    return d


def _write_json(path: Path, obj: dict, *, indent: int | None = None) -> None:
    """One JSON file, keys sorted -- the same stream written twice is the same bytes."""
    path.write_text(json.dumps(obj, indent=indent, sort_keys=True), encoding="utf-8")


def read_manifest(d: Path) -> StreamManifest:
    """The stream's manifest, tuple shapes and all."""
    return StreamManifest.from_dict(json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8")))


def read_unit(d: Path, unit_id: str) -> Unit:
    """The unit the manifest calls ``unit_id`` -- mapped to its on-disk ``module_name``."""
    path = Path(d) / "units" / module_name_for(unit_id) / "unit.json"
    return Unit.from_dict(json.loads(path.read_text(encoding="utf-8")))


def read_mutant(d: Path, key: str) -> Mutant:
    """The mutant with content key ``key``."""
    return Mutant.from_dict(json.loads((Path(d) / "mutants" / f"{key}.json").read_text(encoding="utf-8")))


def read_validations(d: Path) -> list[Validation]:
    """Every verdict in the stream, in write order -- the invalid ones included."""
    with open(Path(d) / "validations.jsonl", encoding="utf-8") as fh:
        return [Validation.from_dict(json.loads(line)) for line in fh if line.strip()]


def write_build_dropped(d: Path, dropped: list[Dropped]) -> None:
    """Write the build-time unit drops as provenance, one JSON object per line.

    Parallel to ``validations.jsonl``: a canonical that could not become a unit (does not
    compile, fails its own visible self-check, oracle-errored) is NAMED on disk by id and
    reason, not discarded to a ``dropped={N}`` count in the build log. Order is
    ``build_units``' record order -- the caller's list is written as-is, never sorted --
    so the file is deterministic and traces straight back to the records.
    """
    with open(Path(d) / "build_dropped.jsonl", "w", encoding="utf-8") as fh:
        for x in dropped:
            fh.write(json.dumps(x.to_dict(), sort_keys=True) + "\n")


def read_build_dropped(d: Path) -> list[Dropped]:
    """Every build-time unit drop in the stream, in write order.

    Backward-compatible: a stream written before this file existed simply has no drops,
    so a missing file reads as an empty list rather than an error.
    """
    path = Path(d) / "build_dropped.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [Dropped.from_dict(json.loads(line)) for line in fh if line.strip()]
