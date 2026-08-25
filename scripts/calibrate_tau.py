#!/usr/bin/env python3
"""Tau calibration script -- the ranker-sanity gate for LOCK-C (Phase-C prereg §4.4/§7).

*What this measures.* ``retrieve_symptom``'s cross-unit path (``crucible/memory/
retrieve.py``) ranks candidate lessons by the lexical scorer in ``crucible.memory.
symmatch`` and keeps only those scoring ``>= tau``. Before ``tau`` can be locked, we need
evidence the scorer actually separates "this lesson is about the same bug shape" from
"this lesson is about something else" -- otherwise a chosen tau is just a number, not a
calibrated threshold. This script scores every (episode, lesson) pair it can find across
one or more already-run memory dbs, buckets each pair as *unrelated* (different unit AND
different family -- genuinely nothing in common) or *related* (same ``class_id``, i.e.
same unit and same family -- the case the scorer is supposed to rank highly), and reports
whether the unrelated distribution's tail sits below the related distribution's middle.
Pairs that are neither (e.g. same family, different unit, different class) are ignored --
they are not the comparison this gate is about.

*Self-citation pairs are excluded, not just another "related" pair.* A lesson mechanically
templated from episode E (``lesson.cited_episode_id == E.item_id``) trivially shares E's
own text -- scoring it against E's own query would be closer to "does this scorer echo its
own input" than "does this scorer generalize to other episodes of the same shape". That
pair is dropped from the comparison entirely, in both directions of the join.

*Which text each side gets scored on, exactly (spec §4.4 restated).* The query side is
``query_text(unit_src, symptom_section(episode.root_prompt), episode.family)`` where
``unit_src`` comes from the stream (the module this episode was attempting to repair) --
injected via ``unit_src_for`` so this stays testable without a real stream on disk. The
lesson side is ``lesson_text(lesson, symptom_section(cited_episode.root_prompt))`` using
the LESSON's OWN cited episode, not the episode being paired against it -- a lesson
templated from episode X describes X's symptom, regardless of which episode Y it is
currently being scored against. If that cited episode no longer resolves in this db
(``store.episode_by_id`` returns ``None``), the symptom text is ``""`` -- None-safe, never
a crash (the None-vs-zero discipline this codebase uses throughout).

*Percentiles, exactly.* ``statistics.quantiles(vals, n=100)`` returns 99 cut points
dividing ``vals`` into 100 equal-sized groups (the default ``method="exclusive"``); cut
point ``quantiles[i]`` (0-indexed) is the ``(i + 1)``-th percentile. So ``p50`` is
``quantiles[49]``, ``p90`` is ``quantiles[89]``, ``p95`` (the tau candidate) is
``quantiles[94]``, and ``p99`` is ``quantiles[98]``. This is deliberately spelled out
because ``statistics.quantiles`` raises for fewer than two data points, and a script whose
tau feeds a pre-registered lock has to be explicit about which exact index produced which
exact number -- silently reaching for the wrong index would still return *a* float, just
the wrong one.

*``median_related`` uses ``statistics.median``, not the ``quantiles``-based ``p50``.* The
sanity gate only needs the related distribution's central tendency, and ``median`` is
defined (and correct) for as few as one data point where ``quantiles`` is not (it demands
>= 2, an exclusive-method requirement, not a data-scarcity judgment call this script makes
itself). ``related_summary``'s own ``p50``/``p90``/``p99`` are still ``quantiles``-based
like ``unrelated_summary``'s, and so still need >= 2 related pairs -- with exactly one
related pair, ``median_related`` is a real number while ``related_summary`` is all
``null``. Two different guards for two different functions' two different minimums, both
honestly represented as ``null`` rather than a fabricated 0.0 (never fake a number).

*Sanity verdict.* ``"PASS"`` iff both ``tau_p95_unrelated`` and ``median_related`` are
real numbers AND ``median_related > tau_p95_unrelated`` -- the unrelated distribution's
tail sits below where the related distribution centers, i.e. a tau near the unrelated p95
would not routinely swallow related pairs. Any other outcome is ``"FAIL"``; a FAIL caused
by too little data (either distribution has fewer values than its stat needs) carries a
``"reason"`` key naming which count was short, so the record distinguishes "the scorer
doesn't separate these" from "there wasn't enough data to tell".

*Determinism.* No randomness, no clock. DBs are visited in the order given on argv;
within a db, episodes come from ``store.episodes()`` (insertion order) and lessons from
``store.semantic_all()`` (item_id order, per that method's own docstring) -- both already
deterministic store orders, so the nested-loop pair order is fixed by argv + those two
orders alone. ``unit_src_for`` results are memoized per ``unit_id`` across the whole run
(safe because the CLI wires every db to the SAME ``--stream``, so a given unit_id names
the same module source everywhere it appears) purely to avoid re-reading the stream for
every episode that shares a unit -- it does not change the reported numbers.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# Make `crucible` importable when invoked as `python scripts/calibrate_tau.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crucible.memory.store import MemoryStore  # noqa: E402
from crucible.memory.symmatch import (  # noqa: E402
    lesson_text,
    query_text,
    score,
    symptom_section,
    tokenize,
)
from crucible.stream import store as stream_store  # noqa: E402

_NULL_SUMMARY = {"p50": None, "p90": None, "p99": None}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Tau calibration: score every (episode, lesson) pair in the given "
                    "memory dbs and check the ranker-sanity gate (Phase-C prereg §4.4/§7)."
    )
    ap.add_argument("--stream", required=True, help="Stream directory (as written by "
                    "crucible.stream.store.write_stream) that every db's episodes were run against.")
    ap.add_argument("db_paths", nargs="+", help="One or more memory.sqlite3 paths, in the "
                    "order to report them (argv order is the deterministic iteration order).")
    return ap.parse_args(argv)


def unit_src_from_stream(stream_dir: Path) -> Callable[[str], str]:
    """Build the CLI's real ``unit_src_for``: ``unit_id -> module_src`` via a stream read."""
    def _load(unit_id: str) -> str:
        return stream_store.read_unit(stream_dir, unit_id).module_src
    return _load


def calibrate(db_paths: Sequence[str], unit_src_for: Callable[[str], str]) -> dict:
    """See the module docstring for the full policy. Pure function of the dbs' current
    contents plus whatever ``unit_src_for`` returns -- no writes, no clock, so two calls
    against the same inputs return an equal dict."""
    unrelated_scores: list[float] = []
    related_scores: list[float] = []
    src_cache: dict[str, str] = {}

    def _src(unit_id: str) -> str:
        if unit_id not in src_cache:
            src_cache[unit_id] = unit_src_for(unit_id)
        return src_cache[unit_id]

    for db_path in db_paths:
        store = MemoryStore(Path(db_path))
        try:
            episodes = store.episodes()
            lessons = [item for item in store.semantic_all() if item.falsified_by is None]
            if not lessons:
                continue
            for episode in episodes:
                query_tokens = tokenize(query_text(
                    _src(episode.unit_id), symptom_section(episode.root_prompt), episode.family,
                ))
                for lesson in lessons:
                    if lesson.cited_episode_id == episode.item_id:
                        continue  # self-citation: excluded from the comparison entirely
                    cited = store.episode_by_id(lesson.cited_episode_id)
                    lesson_symptom = symptom_section(cited.root_prompt) if cited is not None else ""
                    lesson_tokens = tokenize(lesson_text(lesson, lesson_symptom))
                    pair_score = score(query_tokens, lesson_tokens)

                    if lesson.unit_id != episode.unit_id and lesson.family != episode.family:
                        unrelated_scores.append(pair_score)
                    elif lesson.class_id == episode.class_id:
                        related_scores.append(pair_score)
                    # else: neither unrelated nor related -- not this gate's comparison, ignored
        finally:
            store.close()

    return _summarize(unrelated_scores, related_scores)


def _quantiles_or_none(vals: list[float]) -> list[float] | None:
    """``statistics.quantiles(vals, n=100)``, or ``None`` if there are fewer than the two
    data points that function requires (see module docstring's percentile note)."""
    if len(vals) < 2:
        return None
    return statistics.quantiles(vals, n=100)


def _summary(quantiles: list[float] | None) -> dict:
    if quantiles is None:
        return dict(_NULL_SUMMARY)
    return {"p50": quantiles[49], "p90": quantiles[89], "p99": quantiles[98]}


def _summarize(unrelated: list[float], related: list[float]) -> dict:
    n_unrelated = len(unrelated)
    n_related = len(related)

    unrelated_quantiles = _quantiles_or_none(unrelated)
    related_quantiles = _quantiles_or_none(related)

    tau_p95_unrelated = unrelated_quantiles[94] if unrelated_quantiles is not None else None
    median_related = statistics.median(related) if n_related >= 1 else None

    result = {
        "n_unrelated": n_unrelated,
        "n_related": n_related,
        "tau_p95_unrelated": tau_p95_unrelated,
        "median_related": median_related,
        "unrelated_summary": _summary(unrelated_quantiles),
        "related_summary": _summary(related_quantiles),
    }

    if tau_p95_unrelated is None or median_related is None:
        reasons = []
        if tau_p95_unrelated is None:
            reasons.append(f"n_unrelated={n_unrelated} < 2 (tau_p95_unrelated needs >= 2 unrelated pairs)")
        if median_related is None:
            reasons.append(f"n_related={n_related} == 0 (no related pairs to take a median of)")
        result["sanity"] = "FAIL"
        result["reason"] = "; ".join(reasons)
        return result

    result["sanity"] = "PASS" if median_related > tau_p95_unrelated else "FAIL"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = calibrate(args.db_paths, unit_src_from_stream(Path(args.stream)))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
