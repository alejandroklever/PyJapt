from pyjapt import Grammar


def grammar():
    g = Grammar()

    S = g.add_non_terminal("S", True)
    A, B = g.add_non_terminals("A B")

    g.add_terminals("a b c d")

    S %= "A a"
    S %= "b A c"
    S %= "B c"
    S %= "b B a"
    A %= "d"
    B %= "d"

    return g


def test_lalr():
    g = grammar()
    parser = g.get_parser("lalr1")
    assert parser.has_conflicts


def test_lr1():
    g = grammar()
    parser = g.get_parser("lr1")
    assert not parser.has_conflicts
