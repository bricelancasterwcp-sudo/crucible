"""Structural pre-checks (spec §4.8.1-3): does a composed stream match itself across phases?

Task 16 runs :func:`precheck` before it writes or replays a stream and refuses any stream
whose report ``.ok`` is False. Each of the seven named checks appears in the report whether
it passed or not -- a check that is merely absent is indistinguishable from one nobody ran,
so the report is a census of every gate, not a list of the failures.

The point of the phased design is that phase 2 must differ from phase 1 in exactly one
thing: whether the agent has met the bug *kind* in that unit before. Everything else has
to be balanced by construction, and these checks are what prove it after the fact --
matched family mix (``family-distribution-identical``), matched difficulty
(``killing-count-band``, ``unit-length-identical``), and no systematic timeout confound
(``timeout-rate-band``, the consumer of Task 13's seeded phase coin flip). ``novel-disjoint``
guards the control; ``distinct-sites`` guards each class's whole reason to exist; and
``counts-named`` guards that the census the composer emitted is complete.

Two degenerate statistics are defined explicitly rather than left to raise.

*A band with too little data does not pass.* ``mean_band`` needs n>=2 per side to have a
variance at all (ledger ruling R5), so with fewer it returns ``(nan, nan, False)`` -- the
check fails rather than silently accepting an unmeasured difference.

*Zero standard error means "identical or bust".* ``two_proportion_band`` floors the pooled
variance at ``1e-12`` so a run with no timeouts on either side yields a tiny-but-positive
band: the point difference is exactly 0, so it passes, and any nonzero difference against a
zero SE fails. ``mean_band`` gives its zero-variance comparison the same ``+1e-12`` slack,
so equal means pass and unequal means fail. In both cases the rule is: SE 0 => the band
passes iff the point difference is exactly 0.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

from .compose import StreamManifest
from .units import Unit

REQUIRED_COUNTS = ("hidden-only", "equivalent", "infra", "syntax", "ineligible-class", "unit-no-valid",
                   "eligible_classes", "valid_mutants")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PrecheckReport:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}


def two_proportion_band(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, bool]:
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), False)
    p1, p2, p = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
    d = p1 - p2
    return (d, 2 * se, abs(d) <= 2 * se)


def mean_band(xs: list[float], ys: list[float]) -> tuple[float, float, bool]:
    if len(xs) < 2 or len(ys) < 2:
        return (float("nan"), float("nan"), False)
    def var(v):
        mu = sum(v) / len(v); return sum((x - mu) ** 2 for x in v) / (len(v) - 1)
    d = sum(xs) / len(xs) - sum(ys) / len(ys)
    se = math.sqrt(var(xs) / len(xs) + var(ys) / len(ys))
    return (d, 2 * se, abs(d) <= 2 * se + 1e-12)


def precheck(manifest: StreamManifest, units_by_id: dict[str, Unit]) -> PrecheckReport:
    first = [t for t in manifest.tasks if t.kind == "first"]
    second = [t for t in manifest.tasks if t.kind == "second"]
    novel = [t for t in manifest.tasks if t.kind == "novel"]
    checks: list[Check] = []

    fam1, fam2 = Counter(t.family for t in first), Counter(t.family for t in second)
    checks.append(Check("family-distribution-identical", fam1 == fam2, f"{dict(fam1)} vs {dict(fam2)}"))

    d, band, ok = mean_band([t.n_killing_visible for t in first], [t.n_killing_visible for t in second])
    checks.append(Check("killing-count-band", ok, f"diff={d:.3f} band={band:.3f}"))

    len1 = Counter(len(units_by_id[t.unit_id].module_src) for t in first)
    len2 = Counter(len(units_by_id[t.unit_id].module_src) for t in second)
    checks.append(Check("unit-length-identical", len1 == len2, "multiset of unit lengths"))

    d, band, ok = two_proportion_band(sum(t.kills_by_timeout for t in first), len(first),
                                      sum(t.kills_by_timeout for t in second), len(second))
    checks.append(Check("timeout-rate-band", ok, f"diff={d:.3f} band={band:.3f}"))

    inter = {t.unit_id for t in novel} & {t.unit_id for t in first}
    checks.append(Check("novel-disjoint", not inter, f"overlap={sorted(inter)}"))

    by_key = {t.task_key: t for t in manifest.tasks}
    bad = [cid for cid, (k1, k2) in manifest.classes.items()
           if k1 not in by_key or k2 not in by_key or by_key[k1].span == by_key[k2].span]
    checks.append(Check("distinct-sites", not bad, f"same-site classes={bad[:5]}"))

    missing = [k for k in REQUIRED_COUNTS if k not in manifest.counts]
    checks.append(Check("counts-named", not missing, f"missing={missing}"))
    return PrecheckReport(tuple(checks))
