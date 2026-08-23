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


# --- Fix round 1: R-T5-1 (cache_dir env handling) --------------------------------

def test_cache_dir_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_CACHE", raising=False)
    assert ep.cache_dir() == pathlib.Path.home() / ".cache" / "crucible"

def test_cache_dir_treats_blank_env_as_unset(monkeypatch):
    # An empty var is an UNSET var, not "cache into the CWD".
    for blank in ("", "   ", "\t\n"):
        monkeypatch.setenv("CRUCIBLE_CACHE", blank)
        assert ep.cache_dir() == pathlib.Path.home() / ".cache" / "crucible"

def test_cache_dir_expands_user_and_vars(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CACHE", "~/x")
    assert ep.cache_dir() == pathlib.Path.home() / "x"
    monkeypatch.setenv("CRUCIBLE_CACHE", "$HOME/y")
    assert ep.cache_dir() == pathlib.Path.home() / "y"

def test_cache_dir_honours_a_real_path(monkeypatch, tmp_path):
    monkeypatch.setenv("CRUCIBLE_CACHE", str(tmp_path / "z"))
    assert ep.cache_dir() == tmp_path / "z"


# --- Fix round 1: R-T5-2 (atomic download) ---------------------------------------

class _FakeResponse:
    """Stands in for urlopen's response: a context manager with .read()."""

    def __init__(self, payload=b"", *, fail=False):
        self._payload, self._fail = payload, fail

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        if self._fail:
            raise OSError("connection reset mid-transfer")
        return self._payload

def _pin_only(monkeypatch, fx="mini_humaneval.jsonl.gz"):
    """Pin DATASETS to the fixture's digest but seed NO file: forces the download branch."""
    payload = (FIX / fx).read_bytes()
    ds = dict(ep.DATASETS)
    ds["humaneval"] = ep.Dataset("https://example.invalid/he.jsonl.gz",
                                 hashlib.sha256(payload).hexdigest(), ds["humaneval"].filename)
    monkeypatch.setattr(ep, "DATASETS", ds)
    return payload

def test_fetch_downloads_then_promotes_atomically(tmp_path, monkeypatch):
    payload = _pin_only(monkeypatch)
    monkeypatch.setattr(ep.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse(payload))
    path = ep.fetch("humaneval", cache=tmp_path)
    assert path == tmp_path / ep.DATASETS["humaneval"].filename
    assert path.read_bytes() == payload
    assert not path.with_suffix(path.suffix + ".part").exists()

def test_fetch_leaves_no_file_behind_when_download_fails(tmp_path, monkeypatch):
    _pin_only(monkeypatch)
    monkeypatch.setattr(ep.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse(fail=True))
    with pytest.raises(OSError):
        ep.fetch("humaneval", cache=tmp_path)
    path = tmp_path / ep.DATASETS["humaneval"].filename
    assert not path.exists()
    assert not path.with_suffix(path.suffix + ".part").exists()
    assert list(tmp_path.iterdir()) == []

def test_digest_mismatch_message_names_the_file_and_the_remedy(tmp_path, monkeypatch):
    cache = _seed(tmp_path, monkeypatch)
    bad = dict(ep.DATASETS); bad["humaneval"] = ep.Dataset("file://unused", "0" * 64, bad["humaneval"].filename)
    monkeypatch.setattr(ep, "DATASETS", bad)
    with pytest.raises(ep.DigestMismatch) as exc:
        ep.fetch("humaneval", cache=cache)
    msg = str(exc.value)
    assert str(cache / bad["humaneval"].filename) in msg
    assert msg.endswith("delete this file and re-run fetch")
