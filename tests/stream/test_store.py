"""Tests for the on-disk stream store: layout, round-trip fidelity, and byte-stability.

Nothing here touches the sandbox -- ``Unit``/``Mutant``/``Validation``/``StreamManifest``
are built by hand and written to a ``tmp_path``, so every test is a real filesystem
round-trip that still runs in milliseconds.

Three of these tests exist to pin behaviour that a reader could mistake for incidental.

*The slash.* Real unit ids are EvalPlus task ids (``"HumanEval/0"``), which cannot be a
directory name. Units are therefore keyed on disk by ``module_name_for(unit_id)``, and
``test_read_unit_maps_slash_bearing_ids`` fails with ``FileNotFoundError`` the moment a
reader indexes by the raw id instead.

*The invalid validations.* ``validations.jsonl`` carries **every** verdict, including the
mutants that never became tasks. That is provenance: "this mutant was equivalent" and
"nobody looked at this mutant" are different facts, and only the file distinguishes them.

*The key order.* A stream directory is named for a content hash, so two writes of the
same stream must produce the same bytes -- even when the manifest's ``counts``/``classes``
dicts were built in a different insertion order by a different code path. That is what
``sort_keys=True`` buys, and what ``test_manifest_bytes_ignore_dict_key_order`` measures.
"""

from pathlib import Path

from crucible.stream import store
from crucible.stream.compose import StreamManifest, TaskSpec
from crucible.stream.mutants import Mutant
from crucible.stream.units import Unit, module_name_for, sha256_text
from crucible.stream.validate import Validation


def test_write_and_read_round_trip(tmp_path: Path):
    u = Unit("X/0", "unit_x_0", "f", "def f():\n    return 1\n", "v", "h", sha256_text("s"), 1, 1, ())
    m = Mutant("X/0", "k1", "Op", 0, "ARITH", ((1, 1), (1, 2)), "def f():\n    return 2\n", "d")
    v = Validation("k1", True, "killed-visible", False, 1, ("test_v0",))
    t = TaskSpec("k1", "X/0", "ARITH", "X/0|ARITH", 1, "first", ((1, 1), (1, 2)), False, 1)
    man = StreamManifest("abc123def456789", 0, 1, 0, "base", ("X/0",), (t,), {"X/0|ARITH": ("k1", "k1")}, (), {"x": 1})
    d = store.write_stream(tmp_path, man, [u], {"k1": m}, [v])
    assert d == tmp_path / "abc123def456"
    assert store.read_manifest(d) == man and store.read_unit(d, "X/0") == u
    assert store.read_mutant(d, "k1") == m and store.read_validations(d) == [v]
    assert (d / "units" / "unit_x_0" / "module.py").read_text() == u.module_src


def _unit(unit_id: str) -> Unit:
    src = f"def f(a, b):\n    return a + b  # {unit_id}\n"
    return Unit(unit_id, module_name_for(unit_id), "f", src, f"# visible {unit_id}\n",
                f"# hidden {unit_id}\n", sha256_text(src), 2, 3, (("v1", "unhashable"),))


def _mutant(unit_id: str, key: str) -> Mutant:
    return Mutant(unit_id, key, "AddSub", 1, "ARITH", ((2, 11), (2, 12)), "def f(a, b):\n    return a - b\n", "diff")


def _manifest(stream_hash: str, unit_ids: tuple[str, ...], tasks: tuple[TaskSpec, ...],
              classes: dict[str, tuple[str, str]], counts: dict[str, int]) -> StreamManifest:
    return StreamManifest(stream_hash, 7, len(classes), 0, "base", unit_ids, tasks, classes,
                          (("Y/9", "unit-no-valid"),), counts)


def test_read_unit_maps_slash_bearing_ids(tmp_path: Path):
    """``"HumanEval/0"`` is not a path -- units are keyed by ``module_name_for`` on both sides."""
    u = _unit("HumanEval/0")
    man = _manifest("0123456789abcdef", ("HumanEval/0",), (), {}, {})
    d = store.write_stream(tmp_path, man, [u], {}, [])

    assert (d / "units" / "unit_humaneval_0" / "unit.json").exists()
    assert store.read_unit(d, "HumanEval/0") == u
    assert (d / "units" / "unit_humaneval_0" / "module.py").read_text(encoding="utf-8") == u.module_src
    assert (d / "units" / "unit_humaneval_0" / "test_visible.py").read_text(encoding="utf-8") == u.visible_test_src
    assert (d / "units" / "unit_humaneval_0" / "test_hidden.py").read_text(encoding="utf-8") == u.hidden_test_src


def test_validations_keep_the_invalid_ones(tmp_path: Path):
    """``validations.jsonl`` is provenance: an excluded mutant is recorded, not dropped."""
    kept = Validation("k-kept", True, "killed-visible", False, 2, ("test_v0", "test_v1"))
    excluded = Validation("k-equiv", False, "equivalent", False, 0, ())
    infra = Validation("k-infra", False, "infra", False, 0, ())
    man = _manifest("feedfacecafe0000", ("X/0",), (), {}, {})

    d = store.write_stream(tmp_path, man, [_unit("X/0")], {}, [kept, excluded, infra])

    assert (d / "validations.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert store.read_validations(d) == [kept, excluded, infra]


def test_manifest_bytes_ignore_dict_key_order(tmp_path: Path):
    """Same stream, dicts built in a different order -> byte-identical ``manifest.json``."""
    tasks = (TaskSpec("k1", "X/0", "ARITH", "X/0|ARITH", 1, "first", ((2, 11), (2, 12)), False, 1),
             TaskSpec("k2", "X/0", "ARITH", "X/0|ARITH", 2, "second", ((2, 4), (2, 5)), True, 2))
    classes = {"X/0|ARITH": ("k1", "k2"), "X/1|CMP": ("k3", "k4")}
    counts = {"valid_mutants": 4, "eligible_classes": 2, "equivalent": 1}
    forward = _manifest("aaaabbbbcccc0001", ("X/0", "X/1"), tasks, classes, counts)
    reversed_ = _manifest("aaaabbbbcccc0001", ("X/0", "X/1"), tasks,
                          dict(reversed(list(classes.items()))), dict(reversed(list(counts.items()))))
    assert forward == reversed_

    a = store.write_stream(tmp_path / "a", forward, [], {}, [])
    b = store.write_stream(tmp_path / "b", reversed_, [], {}, [])

    assert (a / "manifest.json").read_bytes() == (b / "manifest.json").read_bytes()


def test_rewriting_the_same_stream_is_byte_identical(tmp_path: Path):
    """Re-writing a stream over its own directory overwrites in place and changes nothing."""
    u, m, v = _unit("X/0"), _mutant("X/0", "k1"), Validation("k1", True, "killed-visible", False, 1, ("test_v0",))
    man = _manifest("0f0f0f0f0f0f0f0f", ("X/0",), (), {}, {"valid_mutants": 1})

    first = store.write_stream(tmp_path, man, [u], {"k1": m}, [v])
    before = {p.relative_to(first): p.read_bytes() for p in sorted(first.rglob("*")) if p.is_file()}
    second = store.write_stream(tmp_path, man, [u], {"k1": m}, [v])
    after = {p.relative_to(second): p.read_bytes() for p in sorted(second.rglob("*")) if p.is_file()}

    assert second == first and after == before
    assert store.read_mutant(second, "k1") == m


def test_stream_dir_is_the_hash_prefix(tmp_path: Path):
    """The directory name is the stream's identity: the first 12 chars of its content hash."""
    man = _manifest("0123456789abcdef0123", (), (), {}, {})
    assert store.stream_dir(tmp_path, man) == tmp_path / "0123456789ab"
    assert store.stream_dir(str(tmp_path), man) == tmp_path / "0123456789ab"
