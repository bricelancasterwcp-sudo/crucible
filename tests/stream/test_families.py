from crucible.stream import families as F


def test_every_core_operator_is_mapped_or_excluded():
    assert F.check_complete() == []


def test_family_examples():
    assert F.family_of("ReplaceBinaryOperator_Add_Sub") == "ARITH"
    assert F.family_of("ReplaceComparisonOperator_Lt_GtE") == "CMP"
    assert F.family_of("ReplaceUnaryOperator_Delete_Not") == "UNARY"
    assert F.family_of("AddNot") == "BOOL" and F.family_of("ReplaceOrWithAnd") == "BOOL"
    assert F.family_of("NumberReplacer") == "CONST"
    assert F.family_of("ZeroIterationForLoop") == "FLOW"
    assert F.family_of("RemoveDecorator") == "EXC" and F.family_of("ExceptionReplacer") == "EXC"
    assert F.family_of("StatementDeletion") == "SDL"
    assert F.family_of("VariableReplacer") is None and F.family_of("Nonsense") is None
    assert F.family_of("core/AddNot") is None  # names arrive WITHOUT the core/ prefix


def test_operators_by_family_covers_all_families_and_counts():
    by = F.operators_by_family()
    assert set(by) == set(F.FAMILIES)
    assert len(by["ARITH"]) == 132 and len(by["CMP"]) == 56 and len(by["UNARY"]) == 12
    assert sum(len(v) for v in by.values()) == 213 - len(F.EXCLUDED) + 1


def test_all_operator_names_is_sorted_and_complete():
    names = F.all_operator_names()
    assert names == sorted(names)
    assert len(names) == len(set(names)) == 214
    assert F.SDL_OPERATOR in names
    assert all(not n.startswith("core/") for n in names)


def test_operators_by_family_lists_are_sorted_and_disjoint():
    by = F.operators_by_family()
    seen: set[str] = set()
    for fam, names in by.items():
        assert names == sorted(names), fam
        assert not (seen & set(names))
        seen |= set(names)
    assert seen.isdisjoint(F.EXCLUDED)
