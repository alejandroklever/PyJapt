import pytest

from pyjapt import Grammar, Lexer
from pyjapt.parsing import RuleList


def grammar():
    g = Grammar()

    S = g.add_non_terminal("S", True)
    A, B = g.add_non_terminals("A B")

    g.add_terminals("a b c d")

    S %= "A a"
    S %= "B a"
    A %= "A c"
    A %= "d"
    B %= "B c"
    B %= "d"

    return g

def test_lalr():
    g = grammar()

    lexer = g.get_lexer()
    parser = g.get_parser('lalr1')

    tokens = lexer('ab')
    result = parser(tokens)

    assert result == None