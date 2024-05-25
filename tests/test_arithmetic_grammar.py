from typing import Literal
import pytest

from pyjapt import Grammar
from pyjapt.typing import RuleList, Lexer, SLRParser, LR1Parser, LALR1Parser


tests = [
    ("1 + 2 * 5 - 4", 7),
    ("(1 - 2 + 45) * 3 / 2", 66),
    ("(2 + 2) * 2 + 2", 10),
    ("((3 + 4) * 5) - 6 / 2", 32),
    ("7 * (2 / 3)", 0),
    ("(2 + 4) / (1 + 1) + 2 * 2", 7),
    ("(8 - 5) * (3 + 2) / 4", 3),
    ("8 / 2 * (2 + 2)", 16),
    ("", None),
]


def get_artithmetic_expressions_grammar() -> Grammar:
    g = Grammar()
    expr = g.add_non_terminal("expr", True)
    term, fact = g.add_non_terminals("term fact")
    g.add_terminals("+ - / * ( )")
    g.add_terminal("int", regex=r"\d+")

    @g.terminal("whitespace", r" +")
    def whitespace(lexer: Lexer):
        lexer.column += len(lexer.token.lex)
        lexer.position += len(lexer.token.lex)

    @g.production("expr -> ")
    def empty_expression(s: RuleList):
        s.force_parsing_error()

    expr %= "expr + term", lambda s: s[1] + s[3]
    expr %= "expr - term", lambda s: s[1] - s[3]
    expr %= "term", lambda s: s[1]

    term %= "term * fact", lambda s: s[1] * s[3]
    term %= "term / fact", lambda s: s[1] // s[3]
    term %= "fact", lambda s: s[1]

    fact %= "( expr )", lambda s: s[2]
    fact %= "int", lambda s: int(s[1])

    return g


def parse(parser_name: Literal['slr', 'lr1', 'lalr1'], text: str):
    g = get_artithmetic_expressions_grammar()
    lexer = g.get_lexer()
    parser = g.get_parser(parser_name)

    return parser(lexer(text))

@pytest.mark.slr
@pytest.mark.parametrize("test,expected", tests)
def test_slr(test, expected):
    assert parse("slr", test) == expected, "Bad Parsing"


@pytest.mark.lr1
@pytest.mark.parametrize("test,expected", tests)
def test_lr1(test, expected):
    assert parse("lr1", test) == expected, "Bad Parsing"


@pytest.mark.lalr1
@pytest.mark.parametrize("test,expected", tests)
def test_lalr1(test, expected):
    assert parse("lalr1", test) == expected, "Bad Parsing"
