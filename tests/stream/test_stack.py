"""Tests for two-site mutant composition: span overlap and the wrong-site trap."""

from crucible.stream.mutants import Mutant, MutantSpec, enumerate_specs, make_mutant
from crucible.stream.stack import compose_pair, spans_overlap
from crucible.stream.units import Unit, sha256_text


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
