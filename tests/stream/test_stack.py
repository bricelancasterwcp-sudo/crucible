"""Tests for two-site mutant composition: span overlap, the wrong-site trap, and the
span-partition pairing layer that feeds it."""

import random

import pytest

from crucible.stream.mutants import Mutant, MutantSpec, enumerate_specs, make_mutant
from crucible.stream.stack import compose_pair, sample_pairs, spans_overlap, stack_unit
from crucible.stream.units import Unit, sha256_text
from crucible.stream.validate import Validation


def _unit(src, name="unit_s"):
    return Unit("S/0", name, "f", src, "v", "h", sha256_text(src), 1, 1, ())


def _single(u, operator, occurrence):
    spec = next(s for s in enumerate_specs(u.module_src, [operator]) if s.occurrence == occurrence)
    m = make_mutant(u, spec)
    assert m is not None
    return m


def test_spans_overlap_is_inclusive_and_catches_nesting():
    assert spans_overlap(((1, 0), (1, 4)), ((1, 4), (1, 8)))       # shared boundary point
    assert spans_overlap(((1, 0), (3, 0)), ((2, 0), (2, 5)))       # nested
    assert not spans_overlap(((1, 0), (1, 4)), ((2, 0), (2, 4)))   # disjoint lines


def test_compose_pair_flips_both_sites():
    u = _unit("def f(a, b, c, d):\n    x = a + b\n    y = c + d\n    return x, y\n")
    op = "ReplaceBinaryOperator_Add_Sub"
    ma, mb = _single(u, op, 0), _single(u, op, 1)
    st = compose_pair(u, ma, mb)
    assert st is not None
    assert "a - b" in st.mutated_src and "c - d" in st.mutated_src
    assert [c.span for c in st.components] == sorted([ma.span, mb.span])   # file order
    assert (st.operator, st.occurrence, st.span) == (
        st.components[0].operator, st.components[0].occurrence, st.components[0].span)
    assert st.key == sha256_text(u.src_hash + "\n" + st.diff) and st.unit_id == u.unit_id
    # The key equality above is tautological on its own -- it re-derives the key from the
    # diff the mutant carries. What makes it an identity is the diff BASE: the one hashed
    # diff is against the ORIGINAL source, so BOTH component edits show up as removals.
    # Diffing against the intermediate (late.mutated_src) would hide the late edit.
    assert "-    x = a + b" in st.diff and "-    y = c + d" in st.diff


def test_compose_pair_survives_operator_created_match():
    # The trap: applying Add->Sub at line 3 creates a NEW Sub->Add match there, shifting
    # original-source occurrence indices for Sub->Add. Span-matching must still land the
    # early component on line 2.
    u = _unit("def f(a, b, c, d):\n    x = a - b\n    y = c + d\n    return x, y\n")
    early = _single(u, "ReplaceBinaryOperator_Sub_Add", 0)         # line 2: a - b
    late = _single(u, "ReplaceBinaryOperator_Add_Sub", 0)          # line 3: c + d
    st = compose_pair(u, early, late)
    assert st is not None
    assert "a + b" in st.mutated_src and "c - d" in st.mutated_src


def test_compose_pair_argument_order_does_not_matter():
    u = _unit("def f(a, b, c, d):\n    x = a + b\n    y = c + d\n    return x, y\n")
    op = "ReplaceBinaryOperator_Add_Sub"
    ma, mb = _single(u, op, 0), _single(u, op, 1)
    assert compose_pair(u, ma, mb) == compose_pair(u, mb, ma)


def test_compose_pair_refuses_overlap_and_missing_span():
    u = _unit("def f(a, b, c, d):\n    x = a + b\n    y = c + d\n    return x, y\n")
    op = "ReplaceBinaryOperator_Add_Sub"
    ma, mb = _single(u, op, 0), _single(u, op, 1)
    assert compose_pair(u, ma, ma) is None                          # same span = overlap
    import dataclasses
    ghost = dataclasses.replace(ma, span=((2, 8), (2, 13)), components=())
    assert compose_pair(u, dataclasses.replace(ghost, span=((1, 0), (1, 1))), mb) is None  # span not found


def test_compose_pair_drops_a_composite_equal_to_a_single(monkeypatch):
    u = _unit("def f(a, b, c, d):\n    x = a + b\n    y = c + d\n    return x, y\n")
    op = "ReplaceBinaryOperator_Add_Sub"
    ma, mb = _single(u, op, 0), _single(u, op, 1)
    import crucible.stream.stack as stack
    monkeypatch.setattr(stack, "mutate_code", lambda src, op_, occ: src)   # second apply is a no-op
    assert compose_pair(u, ma, mb) is None


def test_compose_pair_drops_an_ambiguous_span_match():
    # Not in the brief; added because the brief's six tests do not kill a weakening of
    # `_match_occurrence` to `hits[0] if hits else None`. NumberReplacer yields TWO
    # positions at the SAME span for one literal (n+1 and n-1), so an early component
    # there cannot be re-selected by span alone -- taking the first hit would silently
    # compose `a + 2` where the component meant `a + 0`. Spec §4: "Not exactly one match
    # => drop the pair, reason stack-apply" -- never guessed.
    u = _unit("def f(a, b):\n    x = a + 1\n    y = b + 2\n    return x, y\n")
    early = _single(u, "NumberReplacer", 1)   # line 2 literal `1` -> `0`
    late = _single(u, "NumberReplacer", 2)    # line 3 literal `2` -> `3`
    assert early.span < late.span and "a + 0" in early.mutated_src
    assert early.span == _single(u, "NumberReplacer", 0).span   # the ambiguity, pinned
    assert compose_pair(u, early, late) is None


@pytest.mark.parametrize("sentinel", ["original", "early_single", "noncompiling", "no_match"])
def test_compose_pair_drops_a_composite_that_is_not_a_new_bug(monkeypatch, sentinel):
    # Sibling of the brief's no-op test above, which pins only the `late.mutated_src`
    # member of the "differs from" guard. These cover the other two members, the compile
    # guard, and the `stacked is None` branch -- each of which otherwise survives deletion.
    u = _unit("def f(a, b, c, d):\n    x = a + b\n    y = c + d\n    return x, y\n")
    op = "ReplaceBinaryOperator_Add_Sub"
    ma, mb = _single(u, op, 0), _single(u, op, 1)
    assert ma.span < mb.span      # so `ma` is the EARLY component after internal ordering
    returns = {
        "original": u.module_src,       # the two edits cancelled -- not a bug at all
        "early_single": ma.mutated_src,  # the late edit vanished -- equals the early single
        "noncompiling": "def f(:\n",    # composite does not compile -- fails for the wrong reason
        "no_match": None,               # the operator did not match after all
    }[sentinel]
    import crucible.stream.stack as stack
    monkeypatch.setattr(stack, "mutate_code", lambda src, op_, occ: returns)
    assert compose_pair(u, ma, mb) is None
    assert compose_pair(u, mb, ma) is None   # argument order cannot smuggle it past


def _valid(m, timeout=False):
    return (m, Validation(m.key, True, "killed-visible", timeout, 1, ("test_v0",)))


def _four_site_unit():
    u = _unit("def f(a, b, c, d, e, g, h, i):\n    w = a + b\n    x = c + d\n"
              "    y = e + g\n    z = h + i\n    return w, x, y, z\n")
    op = "ReplaceBinaryOperator_Add_Sub"
    return u, [_single(u, op, k) for k in range(4)]


def test_sample_pairs_never_reuses_a_span_and_is_seeded():
    u, singles = _four_site_unit()
    pairs = sample_pairs(singles, rng=random.Random("0:t"), max_pairs=4)
    assert len(pairs) == 2
    spans = [s for a, b in pairs for s in (a.span, b.span)]
    assert len(set(spans)) == 4                                    # pairwise-disjoint site-sets
    again = sample_pairs(singles, rng=random.Random("0:t"), max_pairs=4)
    assert [(a.key, b.key) for a, b in pairs] == [(a.key, b.key) for a, b in again]


def test_sample_pairs_respects_cap():
    u, singles = _four_site_unit()
    assert len(sample_pairs(singles, rng=random.Random("0:t"), max_pairs=1)) == 1


def test_stack_unit_builds_valid_composites_and_counts_apply_drops(monkeypatch):
    u, singles = _four_site_unit()
    stacked, dropped = stack_unit(u, [_valid(m) for m in singles], rng=random.Random("0:t"), max_pairs=4)
    assert len(stacked) == 2 and dropped == 0
    assert all(len(m.components) == 2 for m in stacked)
    import crucible.stream.stack as stack
    monkeypatch.setattr(stack, "compose_pair", lambda *_: None)     # every pair fails to apply
    stacked2, dropped2 = stack_unit(u, [_valid(m) for m in singles], rng=random.Random("0:t"), max_pairs=4)
    assert stacked2 == [] and dropped2 == 2


def test_stack_unit_needs_two_distinct_spans():
    u = _unit("def f(a, b):\n    return a + b\n")
    m = _single(u, "ReplaceBinaryOperator_Add_Sub", 0)
    assert stack_unit(u, [_valid(m)], rng=random.Random("0:t"), max_pairs=4) == ([], 0)


class _NoShuffle:
    """An rng stand-in whose ``shuffle`` is the identity, so an ordering assertion is exact.

    A seeded ``random.Random`` would also work, but only for a seed hand-picked to permute
    the pool the *wrong* way -- which pins the test to the seed rather than to the sort.
    """

    @staticmethod
    def shuffle(seq):
        return None


def test_sample_pairs_refuses_two_overlapping_spans():
    # Not in the brief; added because the brief's tests draw from four spans that are
    # pairwise disjoint anyway, so deleting `spans_overlap` from the pairing walk survives
    # them. Adjacent decorators are a REAL intra-family case: RemoveDecorator spans run
    # line-start to line-start, so @d on line 4 ends exactly where @d on line 5 begins, and
    # spans_overlap is inclusive-end on purpose. Distinct spans are not enough -- the walk
    # must ask, and here the honest answer is "no pair".
    u = _unit("def d(x):\n    return x\n\n@d\n@d\ndef f():\n    return 1\n")
    a, b = _single(u, "RemoveDecorator", 0), _single(u, "RemoveDecorator", 1)
    assert a.span != b.span and spans_overlap(a.span, b.span)   # the shared boundary, pinned
    assert sample_pairs([a, b], rng=random.Random("0:t"), max_pairs=4) == []


def test_stack_unit_prefers_a_non_timeout_member_at_a_shared_span(monkeypatch):
    # Not in the brief; added because nothing in it observes the non-timeout preference, so
    # deleting `pool.sort(...)` from stack_unit survives. NumberReplacer emits two positions
    # at ONE literal's span (n-1 and n+1), so only one of them can represent that span in a
    # pair -- and per compose's `_prefer_non_timeout` it must be the one the visible suite
    # kills by assertion, not the one it kills by hanging.
    u = _unit("def f(a, b):\n    x = a + 1\n    y = b + 2\n    return x, y\n")
    slow, fast = _single(u, "NumberReplacer", 0), _single(u, "NumberReplacer", 1)
    other = _single(u, "NumberReplacer", 2)
    assert slow.span == fast.span != other.span and slow.key != fast.key
    seen = []
    import crucible.stream.stack as stack
    monkeypatch.setattr(stack, "compose_pair", lambda unit, a, b: seen.append((a, b)) or None)
    # The timeout kill is listed FIRST and the shuffle is the identity: only the stable
    # sort can demote it. (CONST cannot really compose -- see the ambiguity test above --
    # so what is asserted is the pair handed to compose_pair, not its result.)
    stack_unit(u, [_valid(slow, timeout=True), _valid(fast), _valid(other)],
               rng=_NoShuffle, max_pairs=4)
    assert [m.key for m in seen[0]] == [fast.key, other.key]


def test_sample_pairs_will_not_reuse_a_partner_for_a_span_it_skipped():
    # Not in the brief, and the sibling of its `used.update` check: that one pins the WRITE
    # to `used`, this one the READ on the second member. Three decorators, the first two
    # adjacent (overlapping): the walk skips the overlapping partner, pairs #1 with #3, and
    # then reaches #2 -- still unused, and disjoint from #3. Only `b not in used` stops it
    # from spending #3 twice, which would hand compose two "classes" sharing a site.
    u = _unit("def d(x):\n    return x\n\n@d\n@d\ndef f():\n    return 1\n\n@d\ndef g():\n    return 2\n")
    ms = [_single(u, "RemoveDecorator", k) for k in range(3)]
    assert spans_overlap(ms[0].span, ms[1].span) and not spans_overlap(ms[1].span, ms[2].span)
    pairs = sample_pairs(ms, rng=_NoShuffle, max_pairs=4)     # identity shuffle: order is the layout above
    assert [(a.span, b.span) for a, b in pairs] == [(ms[0].span, ms[2].span)]


def test_sample_pairs_draw_does_not_depend_on_caller_order():
    # Not in the brief; without `sorted(by_span)` the shuffle is applied to dict-insertion
    # order, so the same seed and the same spans would draw different pairs depending on how
    # the caller happened to list its mutants -- the same normalisation compose applies to
    # units before its seeded shuffle.
    u, singles = _four_site_unit()
    a = sample_pairs(singles, rng=random.Random("0:t"), max_pairs=4)
    b = sample_pairs(list(reversed(singles)), rng=random.Random("0:t"), max_pairs=4)
    assert [(x.key, y.key) for x, y in a] == [(x.key, y.key) for x, y in b]


def test_stack_unit_breaks_ties_at_a_shared_span_randomly(monkeypatch):
    # Not in the brief; pins the shuffle that runs BEFORE the stable sort. Two equally-good
    # (both non-timeout) mutants share one span, so one of them is dropped -- and which one
    # must be a coin flip, not "whichever the caller listed first", or every such choice in
    # the run would inherit the grouping order. Swept over seeds, per compose's tests.
    u = _unit("def f(a, b):\n    x = a + 1\n    y = b + 2\n    return x, y\n")
    tied = [_single(u, "NumberReplacer", 0), _single(u, "NumberReplacer", 1)]
    other = _single(u, "NumberReplacer", 2)
    assert tied[0].span == tied[1].span != other.span
    seen = []
    import crucible.stream.stack as stack
    monkeypatch.setattr(stack, "compose_pair", lambda unit, a, b: seen.append((a, b)) or None)
    picked = set()
    for seed in range(20):
        seen.clear()
        stack_unit(u, [_valid(m) for m in tied + [other]], rng=random.Random(f"{seed}:t"), max_pairs=4)
        picked.add(next(m.key for m in seen[0] if m.span == tied[0].span))
    assert picked == {tied[0].key, tied[1].key}


def test_sample_pairs_partition_is_drawn_not_fixed_by_file_order():
    # Not in the brief; without `rng.shuffle(spans)` the partition is always "site 1 with
    # site 2, site 3 with site 4". Which sites get stacked would then be a fixed function of
    # the source layout rather than a seeded draw, and every stacked mutant would join
    # adjacent sites by construction -- a locality confound baked into the rung. All three
    # pairings of four disjoint sites must be reachable.
    u, singles = _four_site_unit()
    partitions = {frozenset(frozenset((a.span, b.span)) for a, b in
                            sample_pairs(singles, rng=random.Random(f"{seed}:t"), max_pairs=4))
                  for seed in range(20)}
    assert len(partitions) == 3
