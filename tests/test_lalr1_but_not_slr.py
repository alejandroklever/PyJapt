import pytest

from pyjapt import Grammar


def grammar():
    g = Grammar()

    S = g.add_non_terminal("S", True)
    (A,) = g.add_non_terminals("A")

    g.add_terminals("a b c d")

    S %= "A a"
    S %= "b A c"
    S %= "d c"
    S %= "b d a"
    A %= "d"

    return g


def test_slr():
    g = grammar()
    parser = g.get_parser("slr")
    assert parser.has_conflicts


@pytest.mark.lalr1
def test_lalr():
    g = grammar()
    parser = g.get_parser("lalr1")
    assert not parser.has_conflicts
