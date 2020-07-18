import json
import re
import sys
from typing import List, FrozenSet, Optional, Tuple, Iterable, Callable, Dict

from .lexing import Lexer, Token
from .serialization import LRParserSerializer, LexerSerializer


class GrammarError(Exception):
    def __init__(self, *args):
        super().__init__(args)
        self.text = args[0]


class Symbol:
    def __init__(self, name: str, grammar: 'Grammar'):
        self.name: str = name
        self.grammar: 'Grammar' = grammar

    @property
    def IsEpsilon(self):
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return repr(self.name)

    def __add__(self, other):
        if isinstance(other, Symbol):
            return Sentence(self, other)

        raise TypeError(other)

    def __or__(self, other):
        if isinstance(other, Sentence):
            return SentenceList(Sentence(self), other)
        raise TypeError(other)

    def __len__(self):
        return 1


class NonTerminal(Symbol):
    def __init__(self, name: str, grammar: 'Grammar'):
        super().__init__(name, grammar)
        self.productions: List['Production'] = []
        self.error_productions: List['Production'] = []

    def __mod__(self, other):
        if isinstance(other, str):
            if other:
                p = Production(self, Sentence(*(self.grammar[s] for s in other.split())))
            else:
                p = Production(self, self.grammar.EPSILON)
            self.grammar.add_production(p)
            return self

        if isinstance(other, Symbol):
            p = Production(self, Sentence(other))
            self.grammar.add_production(p)
            return self

        if isinstance(other, Sentence):
            p = Production(self, other)
            self.grammar.add_production(p)
            return self

        if isinstance(other, tuple):
            assert len(other) > 1

            if isinstance(other[0], str):
                if other[0]:
                    other = Sentence(*(self.grammar[s] for s in other[0].split())), other[1]
                else:
                    other = self.grammar.EPSILON, other[1]

            if isinstance(other[0], Symbol) and not isinstance(other[0], Sentence):
                other[0] = Sentence(other[0])

            if isinstance(other[0], Sentence):
                p = Production(self, other[0], other[1])
            else:
                raise TypeError("Valid types for a production are 'Symbol', 'Sentence' or 'str'")

            self.grammar.add_production(p)
            return self

        if isinstance(other, SentenceList):
            for s in other:
                p = Production(self, s)
                self.grammar.add_production(p)
            return self

        raise TypeError(other)

    @property
    def IsTerminal(self):
        return False

    @property
    def IsNonTerminal(self):
        return True

    @property
    def IsEpsilon(self):
        return False


class Terminal(Symbol):
    def __init__(self, name: str, grammar: 'Grammar'):
        super().__init__(name, grammar)

    @property
    def IsTerminal(self):
        return True

    @property
    def IsNonTerminal(self):
        return False

    @property
    def IsEpsilon(self):
        return False


class ErrorTerminal(Terminal):
    def __init__(self, G):
        super().__init__('error', G)


class EOF(Terminal):
    def __init__(self, G):
        super().__init__('$', G)


class Sentence:
    def __init__(self, *args):
        self.symbols = tuple(x for x in args if not x.IsEpsilon)
        self.hash = hash(self.symbols)

    def __len__(self):
        return len(self.symbols)

    def __add__(self, other) -> 'Sentence':
        if isinstance(other, Symbol):
            return Sentence(*(self.symbols + (other,)))

        if isinstance(other, Sentence):
            return Sentence(*(self.symbols + other.symbols))

        raise TypeError(other)

    def __or__(self, other) -> 'SentenceList':
        if isinstance(other, Sentence):
            return SentenceList(self, other)

        if isinstance(other, Symbol):
            return SentenceList(self, Sentence(other))

        raise TypeError(other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return ("%s " * len(self.symbols) % tuple(self.symbols)).strip()

    def __iter__(self):
        return iter(self.symbols)

    def __contains__(self, item):
        return item in self.symbols

    def __getitem__(self, index):
        return self.symbols[index]

    def __eq__(self, other) -> bool:
        return self.symbols == other.symbols

    def __hash__(self):
        return self.hash

    @property
    def IsEpsilon(self):
        return False


class SentenceList:
    def __init__(self, *args):
        self._sentences = list(args)

    def Add(self, symbol):
        if not symbol and (symbol is None or not symbol.IsEpsilon):
            raise ValueError(symbol)

        self._sentences.append(symbol)

    def __iter__(self):
        return iter(self._sentences)

    def __or__(self, other):
        if isinstance(other, Sentence):
            self.Add(other)
            return self

        if isinstance(other, Symbol):
            return self | Sentence(other)


class Epsilon(Terminal, Sentence):
    def __init__(self, grammar: 'Grammar'):
        super().__init__('epsilon', grammar)

    def __str__(self):
        return "e"

    def __repr__(self):
        return 'epsilon'

    def __iter__(self):
        yield from ()

    def __len__(self):
        return 0

    def __add__(self, other):
        return other

    def __eq__(self, other):
        return isinstance(other, (Epsilon,))

    def __hash__(self):
        return hash("")

    @property
    def IsEpsilon(self):
        return True


class Production:
    def __init__(self, non_terminal: NonTerminal, sentence: Sentence,
                 rule: Optional[Callable[['RuleList'], object]] = None):
        self.left: NonTerminal = non_terminal
        self.right: Sentence = sentence
        self.rule: Optional[Callable[['RuleList'], object]] = rule

    @property
    def IsEpsilon(self):
        return self.right.IsEpsilon

    def __str__(self):
        return '%s := %s' % (self.left, self.right)

    def __repr__(self):
        return '%s -> %s' % (self.left, self.right)

    def __iter__(self):
        yield self.left
        yield self.right

    def __eq__(self, other):
        return isinstance(other, Production) and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash((self.left, self.right))


class Grammar:
    def __init__(self):
        self.productions: List[Production] = []
        self.non_terminals: List[NonTerminal] = []
        self.terminals: List[Terminal] = []
        self.start_symbol: Optional[NonTerminal] = None

        self.ERROR: ErrorTerminal = ErrorTerminal(self)
        self.EPSILON: Epsilon = Epsilon(self)
        self.EOF: EOF = EOF(self)

        self.lexical_error_handler = None  # type: Optional[Callable[[Lexer], None]]
        self.terminal_rules = {}  # type: Dict[str, Tuple[str, bool, Optional[Callable[[Lexer], Optional[Token]]]]]
        self.production_dict = {}  # type: Dict[str, Production]
        self.symbol_dict = {'$': self.EOF, 'error': self.ERROR}  # type: Dict[str, Symbol]

    def add_terminal(self, name: str, regex: str = None,
                     rule: Optional[Callable[[Lexer], Optional[Token]]] = None) -> Terminal:
        """
        Create a terminal for the grammar with its respective regular expression

        If not regex is given then the regular expression will be the literal name of the Terminal

        If the regex is given then de param name most be a valid python identifier

        All terminals with a not given regex will be marked as `literals` and then it will have a higher priority in
        the lexer recognition above the others, and the reserved set will be sorted by the length of the regular
        expression by descending

        The given rule must be a function

        :param name: str

        :param regex: str

        :param rule: function to handle the arrival of a token during the lexical process

        :return: Terminal
        """
        name = name.strip()
        if not name:
            raise Exception("Empty name")

        assert name not in self.symbol_dict, f'Terminal {name} already defined in grammar'

        literal = False
        if regex is None:
            regex = re.escape(name)
            literal = True

        term = Terminal(name, self)
        self.terminals.append(term)
        self.symbol_dict[name] = term
        self.terminal_rules[name] = regex, literal, rule
        return term

    def add_terminal_error(self):
        self.terminals.append(self.ERROR)

    def add_terminals(self, names: str) -> Tuple[Terminal, ...]:
        return tuple(self.add_terminal(x) for x in names.strip().split())

    def add_non_terminal(self, name: str, start_symbol: bool = False) -> NonTerminal:
        """
        Add and return a new non-terminal, use the start_symbol parameter for define the started symbol in th grammar,
        if two non-terminal are marked as started symbol an AssertionError will be raised

        :param name: str

        :param start_symbol: bool

        :return: NonTerminal created
        """
        name = name.strip()
        if not name:
            raise Exception("Empty name")

        assert name not in self.symbol_dict, f'Non-Terminal {name} already defined in grammar'

        term = NonTerminal(name, self)

        if start_symbol:
            if self.start_symbol is None:
                self.start_symbol = term
            else:
                raise Exception("Cannot define more than one start symbol.")

        self.non_terminals.append(term)
        self.symbol_dict[name] = term
        return term

    def add_non_terminals(self, names: str) -> Tuple[NonTerminal, ...]:
        return tuple(self.add_non_terminal(x) for x in names.strip().split())

    def add_production(self, production: Production):
        production.left.productions.append(production)
        self.productions.append(production)
        self.production_dict[repr(production)] = production

    def production(self, production: str):
        """
        Return a function to decorate a method that will be used for as production rule

        :param production: is a string representing the production to attach the decorated function,
                       the string has the form '<non-terminal name> -> <symbols separated with white spaces>'

        :return: a function to decorate the production
        """

        def decorator(rule: Optional[Callable[['RuleList'], object]]):
            head, body = production.split('->')
            head = self[head.strip()]
            head %= body.strip(), rule
            return rule

        return decorator

    def terminal(self, name: str, regex: str):
        """
        Return a function to decorate a method that will be used for as terminal rule in tokenizer process

        :param name: the name of a terminal

        :param regex: the regex of the terminal

        :return: a function to decorate the production
        """

        def decorator(rule: Optional[Callable[[Lexer], Optional[Token]]]):
            self.add_terminal(name, regex, rule)
            return rule

        return decorator

    def lexical_error(self, handler: Callable[[Lexer], None]):
        self.lexical_error_handler = handler
        return handler

    def augmented_grammar(self, force: bool = False):
        if not self.is_augmented_grammar or force:

            G = self.copy()
            # S, self.startSymbol, SS = self.startSymbol, None, self.NonTerminal('S\'', True)
            S = G.start_symbol
            G.start_symbol = None
            SS = G.add_non_terminal('S\'', True)
            SS %= S + G.EPSILON, lambda x: x

            return G
        else:
            return self.copy()

    def copy(self):
        G = Grammar()
        G.productions = self.productions.copy()
        G.non_terminals = self.non_terminals.copy()
        G.terminals = self.terminals.copy()
        G.start_symbol = self.start_symbol
        G.EPSILON = self.EPSILON
        G.ERROR = self.ERROR
        G.EOF = self.EOF
        G.symbol_dict = self.symbol_dict.copy()

        return G

    def serialize_lexer(self, class_name: str, grammar_module_name: str, grammar_variable_name: str = 'G'):
        LexerSerializer.build(self, class_name, grammar_module_name, grammar_variable_name)

    @staticmethod
    def serialize_parser(parser, class_name: str, grammar_module_name: str, grammar_variable_name: str = 'G'):
        LRParserSerializer.build(parser, class_name, grammar_module_name, grammar_variable_name)

    def to_json(self):

        productions = []

        for p in self.productions:
            head = p.left.name

            body = []

            for s in p.right:
                body.append(s.Name)

            productions.append({'Head': head, 'Body': body})

        d = {'NonTerminals': [symb.name for symb in self.non_terminals],
             'Terminals': [symb.name for symb in self.terminals],
             'Productions': productions}

        # [{'Head':p.Left.Name, "Body": [s.Name for s in p.Right]} for p in self.Productions]
        return json.dumps(d)

    @property
    def is_augmented_grammar(self):
        augmented = 0
        for left, _ in self.productions:
            if self.start_symbol == left:
                augmented += 1
        if augmented <= 1:
            return True
        else:
            return False

    @staticmethod
    def from_json(data):
        data = json.loads(data)

        G = Grammar()
        dic = {'epsilon': G.EPSILON}

        for term in data['Terminals']:
            dic[term] = G.add_terminal(term)

        for noTerm in data['NonTerminals']:
            dic[noTerm] = G.add_non_terminal(noTerm)

        for p in data['Productions']:
            head = p['Head']
            dic[head] %= Sentence(*[dic[term] for term in p['Body']])

        return G

    def __getitem__(self, item):
        try:
            try:
                return self.symbol_dict[item]
            except KeyError:
                return self.production_dict[item]
        except KeyError:
            return None

    def __str__(self):

        mul = '%s, '

        ans = 'Non-Terminals:\n\t'

        nonterminals = mul * (len(self.non_terminals) - 1) + '%s\n'

        ans += nonterminals % tuple(self.non_terminals)

        ans += 'Terminals:\n\t'

        terminals = mul * (len(self.terminals) - 1) + '%s\n'

        ans += terminals % tuple(self.terminals)

        ans += 'Productions:\n\t'

        ans += str(self.productions)

        return ans


class Item:
    def __init__(self, production: Production, pos: int, lookaheads: Iterable[Symbol] = None):
        if lookaheads is None:
            lookaheads = []
        self.production: Production = production
        self.pos: int = pos
        self.lookaheads: FrozenSet[Symbol] = frozenset(lookaheads)

    def __str__(self):
        s = str(self.production.left) + " -> "
        if len(self.production.right) > 0:
            for i, _ in enumerate(self.production.right):
                if i == self.pos:
                    s += "."
                s += str(self.production.right[i]) + " "
            if self.pos == len(self.production.right):
                s += "."
        else:
            s += "."
        s += ", " + str(self.lookaheads)[10:-1]
        return s

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
                (self.pos == other.pos) and
                (self.production == other.production) and
                (set(self.lookaheads) == set(other.lookaheads))
        )

    def __hash__(self):
        return hash((self.production, self.pos, self.lookaheads))

    @property
    def IsReduceItem(self) -> bool:
        return len(self.production.right) == self.pos

    @property
    def NextSymbol(self) -> Optional[Symbol]:
        if self.pos < len(self.production.right):
            return self.production.right[self.pos]
        else:
            return None

    def NextItem(self) -> Optional['Item']:
        if self.pos < len(self.production.right):
            return Item(self.production, self.pos + 1, self.lookaheads)
        else:
            return None

    def Preview(self, skip=1) -> List[Symbol]:
        unseen = self.production.right[self.pos + skip:]
        return [unseen + (lookahead,) for lookahead in self.lookaheads]

    def Center(self) -> 'Item':
        return Item(self.production, self.pos)


class RuleList:
    def __init__(self, parser, rules):
        self.__parser = parser
        self.__list = rules

    def __iter__(self):
        return iter(self.__list)

    def __getitem__(self, item):
        return self.__list[item]

    def __setitem__(self, key, value):
        self.__list[key] = value

    def lineno(self, index):
        pass

    def success(self):
        pass

    def warning(self):
        pass

    def error(self, index, message):
        self.__parser.set_error(self[index].line, self[index].column, message)
