from pyjapt import Grammar
from pyjapt.parsing import RuleList


def parse_arithmetic_expression(text: str):
    g = Grammar()
    expr = g.add_non_terminal("expr", True)
    term, fact = g.add_non_terminals("term fact")
    g.add_terminals("+ - / * ( )")
    g.add_terminal("int", regex=r"\d+")

    @g.terminal("whitespace", r" +")
    def whitespace(_lexer):
        _lexer.column += len(_lexer.token.lex)
        _lexer.position += len(_lexer.token.lex)

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

    lexer = g.get_lexer()
    parser = g.get_parser("slr")

    return parser(lexer(text))


def test_lexer(text: str):
    g = Grammar()
    g.add_non_terminal("expr", True)
    g.add_non_terminals("term fact")
    g.add_terminals("+ - / * ( )")
    g.add_terminal("int", regex=r"\d+")

    @g.terminal("whitespace", r" +")
    def whitespace(_lexer):
        _lexer.column += len(_lexer.token.lex)
        _lexer.position += len(_lexer.token.lex)

    @g.production("expr -> expr + term", "expr -> expr - term", "expr -> term")
    def exp_term(rules):
        if len(rules) > 2:
            if rules[2] == "+":
                return rules[1] + rules[3]
            elif rules[2] == "-":
                return rules[1] - rules[3]
            raise ValueError(rules[2])

        if len(rules) == 2:
            return rules[1]

    @g.production("term -> term / fact", "term -> term * fact", "term -> fact")
    def term_fact(rules):
        if len(rules) > 2:
            if rules[2] == "*":
                return rules[1] * rules[3]
            elif rules[2] == "/" and rules[3] != 0:
                return rules[1] // rules[3]
            raise ValueError(rules[2])

        if len(rules) == 2:
            return rules[1]

    @g.production("fact -> int", "fact -> ( expr )")
    def fact(rules):
        if len(rules) > 2:
            return rules[2]

        if len(rules) == 2:
            return float(rules[1])

    parser = g.get_parser("slr")
    lexer = g.get_lexer()

    value = parser(lexer(text))
    return value


def test_arithmetic_grammar():
    tests = [
        ("1 + 2 * 5 - 4", 7),
        ("(1 - 2 + 45) * 3 / 2", 66),
        ("(2 + 2) * 2 + 2", 10),
        ("", None),
    ]
    for text, result in tests:
        assert parse_arithmetic_expression(text) == result, "Bad Parsing"
