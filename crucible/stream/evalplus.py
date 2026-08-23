"""The EvalPlus dataset loader: the only place the task stream touches the network.

The seed units of the Phase-A stream are EvalPlus HumanEval+ / MBPP+ records
(Apache-2.0; see ``THIRD_PARTY.md``). Each release is pinned by *content digest*,
not by URL alone: a release asset that silently changes under a fixed tag would
change every downstream task key, so ``fetch`` verifies sha256 on every call --
including calls that hit an already-cached file -- and raises rather than returning
a file whose provenance it cannot vouch for.

Downloading happens only when the cache file is missing. Tests seed the cache
directly and monkeypatch ``DATASETS``, so the suite never touches the network.
``load`` returns records in file order; nothing here samples or shuffles.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    """A pinned release asset: where it lives, what it must hash to, what we call it."""

    url: str
    sha256: str
    filename: str


DATASETS: dict[str, Dataset] = {
    "humaneval": Dataset(
        "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
        "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101", "HumanEvalPlus-v0.1.10.jsonl.gz"),
    "mbpp": Dataset(
        "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
        "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63", "MbppPlus-v0.2.0.jsonl.gz"),
}


class DigestMismatch(RuntimeError):
    """A dataset file's sha256 does not match its pin -- the file is not the dataset."""


def _cache_override() -> Path | None:
    """The cache dir named by ``$CRUCIBLE_CACHE``, or None if the var says nothing.

    An empty or whitespace-only value is an *unset* var, not a request to cache into
    the current working directory -- the same none-vs-zero rule the rest of the tree
    follows (ruling R-T5-1). A set value gets ``$VAR`` and ``~`` expanded, because a
    shell-style path that arrives unexpanded would otherwise create a literal
    ``~``-named directory next to wherever the process happened to start.
    """
    raw = os.environ.get("CRUCIBLE_CACHE")
    if raw is None or not raw.strip():
        return None
    return Path(os.path.expandvars(raw)).expanduser()


def cache_dir() -> Path:
    override = _cache_override()
    return override if override is not None else Path.home() / ".cache" / "crucible"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str, path: Path) -> None:
    """Download ``url`` to ``path`` atomically: nothing appears at ``path`` unless it is whole.

    Writing straight to the final path creates/truncates it before the first byte
    arrives, so a dropped connection, ^C or OOM leaves a partial file that ``exists()``
    -- and every later fetch then fails the digest check forever (ruling R-T5-2). The
    body lands in a ``.part`` sibling that is promoted with ``os.replace`` only once it
    is complete, and is removed on any failure, ``KeyboardInterrupt`` included.
    """
    part = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
            out.write(resp.read())
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def fetch(name: str, *, cache: Path | None = None) -> Path:
    ds = DATASETS[name]
    cache = cache or cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / ds.filename
    if not path.exists():
        _download(ds.url, path)
    got = _sha256(path)
    if got != ds.sha256:
        raise DigestMismatch(
            f"{name}: expected {ds.sha256}, got {got} at {path} -- delete this file and re-run fetch")
    return path


def load(name: str, *, cache: Path | None = None) -> list[dict]:
    path = fetch(name, cache=cache)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def full_source(rec: dict) -> str:
    return rec["prompt"] + rec["canonical_solution"]


def source_of(task_id: str) -> str:
    head = task_id.split("/")[0].lower()
    return {"humaneval": "humaneval", "mbpp": "mbpp"}[head]
