import re

PARSER_TEMPLATE = """from abc import ABC
from pyjapt.parsing import ShiftReduceParser
from %s import %s


class %s(ShiftReduceParser, ABC):
    def __init__(self, verbose=False):
        self.G = G
        self.verbose = verbose
        self.action = self.__action_table()
        self.goto = self.__goto_table()
        self._errors = []

    @staticmethod
    def __action_table():
        return %s

    @staticmethod
    def __goto_table():
        return %s
"""

LEXER_TEMPLATE = """import re

from pyjapt.lexing import Token, Lexer
from %s import %s


class %s(Lexer):
    def __init__(self):
        self.lineno = 1
        self.column = 1
        self.position = 0
        self.token = Token('', '', 0, 0)
        self.pattern = re.compile(r'%s')
        self.token_rules = %s
        self.error_handler = %s
        self.contain_errors = False
        self.eof = '%s'
    
    def __call__(self, text):
        return %s
"""


class LRParserSerializer:
    @staticmethod
    def build(parser, parser_class_name, grammar_module_name, grammar_variable_name):
        action, goto = LRParserSerializer._build_parsing_tables(parser, grammar_variable_name)
        content = PARSER_TEMPLATE % (grammar_module_name, grammar_variable_name, parser_class_name, action, goto)
        try:
            with open('parsertab.py', 'x') as f:
                f.write(content)
        except FileExistsError:
            with open('parsertab.py', 'w') as f:
                f.write(content)

    @staticmethod
    def _build_parsing_tables(parser, variable_name):
        s1 = '{\n'
        for (state, symbol), (act, tag) in parser.action.items():
            s1 += f'            ({state}, {variable_name}["{symbol}"]): '

            if act == 'SHIFT':
                s1 += f'("{act}", {tag}),\n'
            elif act == 'REDUCE':
                s1 += f'("{act}", {variable_name}["{repr(tag)}"]),\n'
            else:
                s1 += f'("{act}", None),\n'

        s1 += '        }'

        s2 = '{\n'
        for (state, symbol), dest in parser.goto.items():
            s2 += f'            ({state}, {variable_name}["{symbol}"]): {dest},\n'
        s2 += '        }'

        return s1, s2


class LexerSerializer:
    @staticmethod
    def build(grammar, lexer_class_name, grammar_module_name, grammar_variable_name):
        f1 = filter(lambda x: x[1][2] is not None, grammar.terminal_rules.items())
        f2 = filter(lambda x: x[1][2] is None and not x[1][1], grammar.terminal_rules.items())
        f3 = filter(lambda x: x[1][2] is None and x[1][1], grammar.terminal_rules.items())

        ruled_tokens = list(f1)
        not_literal_tokens = sorted(f2, key=lambda x: len(x[1][0]), reverse=True)
        literal_tokens = sorted(f3, key=lambda x: len(x[1][0]), reverse=True)

        pattern = re.compile('|'.join(
            ['(?P<%s>%s)' % (name, regex) for name, (regex, _, _) in ruled_tokens] +
            ['(?P<%s>%s)' % (name, regex) for name, (regex, _, _) in not_literal_tokens] +
            ['(%s)' % regex for _, (regex, _, _) in literal_tokens]
        )).pattern

        token_rules = f"{{key: rule for key, (_, _, rule) in {grammar_variable_name}.terminal_rules.items() if rule " \
                      f"is not None}}"

        error_handler = f"{grammar_variable_name}.lexical_error_handler if " \
                        f"{grammar_variable_name}.lexical_error_handler is not None else self.error "

        call_return = f"[Token(t.lex, {grammar_variable_name}[t.token_type], t.line, t.column) for t in " \
                      f"self.tokenize(text)] "

        content = LEXER_TEMPLATE % (
            grammar_module_name, grammar_variable_name, lexer_class_name, pattern, token_rules, error_handler,
            grammar.EOF.name, call_return,
        )

        try:
            with open('lexertab.py', 'x') as f:
                f.write(content)
        except FileExistsError:
            with open('lexertab.py', 'w') as f:
                f.write(content)
