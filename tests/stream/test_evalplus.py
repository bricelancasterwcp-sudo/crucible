import gzip, hashlib, json, pathlib, shutil
import pytest
from crucible.stream import evalplus as ep

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"

def _seed(tmp_path, monkeypatch):
    ds = {}
    for name, fx in (("humaneval", "mini_humaneval.jsonl.gz"), ("mbpp", "mini_mbpp.jsonl.gz")):
        src = FIX / fx
        dst = tmp_path / ep.DATASETS[name].filename
        shutil.copy(src, dst)
        ds[name] = ep.Dataset(url="file://unused", sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                              filename=ep.DATASETS[name].filename)
    monkeypatch.setattr(ep, "DATASETS", ds)
    return tmp_path

def test_load_reads_records_from_cache(tmp_path, monkeypatch):
    cache = _seed(tmp_path, monkeypatch)
    rows = ep.load("humaneval", cache=cache)
    assert [r["task_id"] for r in rows] == ["HumanEval/0", "HumanEval/1"]
    assert ep.load("mbpp", cache=cache)[0]["entry_point"] == "first"

def test_fetch_rejects_bad_digest(tmp_path, monkeypatch):
    cache = _seed(tmp_path, monkeypatch)
    bad = dict(ep.DATASETS); bad["humaneval"] = ep.Dataset("file://unused", "0" * 64, bad["humaneval"].filename)
    monkeypatch.setattr(ep, "DATASETS", bad)
    with pytest.raises(ep.DigestMismatch):
        ep.fetch("humaneval", cache=cache)

def test_full_source_and_source_of(tmp_path, monkeypatch):
    cache = _seed(tmp_path, monkeypatch)
    he = ep.load("humaneval", cache=cache)[0]
    assert ep.full_source(he).startswith("def add2(") and "return a + b" in ep.full_source(he)
    assert ep.source_of("HumanEval/0") == "humaneval" and ep.source_of("Mbpp/2") == "mbpp"

def test_real_dataset_table_is_pinned():
    # The pinned digests are the provenance of the whole stream; they must not drift silently.
    assert ep.DATASETS["humaneval"].sha256 == "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
    assert ep.DATASETS["mbpp"].sha256 == "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63"
