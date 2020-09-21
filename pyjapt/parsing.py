import json
import re
import sys
from typing import List, FrozenSet, Optional, Tuple, Iterable, Callable, Dict, Set

from pyjapt.automata import State
from pyjapt.lexing import Lexer, Token
from pyjapt.serialization import LRParserSerializer, LexerSerializer
from pyjapt.utils import ContainerSet

TerminalRule = Callable[[Lexer], Optional[Token]]
ProductionRule = Callable[['RuleList'], object]


class GrammarError(Exception):
    def __init__(self, *args):
        super().__init__(args)
        self.text = args[0]


class Symbol:
    def __init__(self, name: str, grammar: Optional['Grammar']):
        self.name: str = name
        self.grammar: 'Grammar' = grammar

    @property
    def is_epsilon(self) -> bool:
        return False

    @property
    def is_terminal(self) -> bool:
        return False

    @property
    def is_non_terminal(self) -> bool:
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

    def __mod__(self, other) -> 'NonTerminal':
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
                other = Sentence(other[0]), other[1]

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
    def is_non_terminal(self) -> bool:
        return True


class Terminal(Symbol):
    def __init__(self, name: str, grammar: 'Grammar'):
        super().__init__(name, grammar)

    @property
    def is_terminal(self) -> bool:
        return True


class PropagationTerminal(Symbol):
    def __init__(self):
        super().__init__('#', None)


class ErrorTerminal(Terminal):
    def __init__(self, grammar: 'Grammar'):
        super().__init__('error', grammar)


class EOF(Terminal):
    def __init__(self, grammar: 'Grammar'):
        super().__init__('$', grammar)


class Sentence:
    def __init__(self, *args):
        self.symbols = tuple(x for x in args if not x.is_epsilon)
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
    def is_epsilon(self):
        return False


class SentenceList:
    def __init__(self, *args):
        self._sentences = list(args)

    def add(self, symbol):
        if not symbol and (symbol is None or not symbol.is_epsilon):
            raise ValueError(symbol)

        self._sentences.append(symbol)

    def __iter__(self):
        return iter(self._sentences)

    def __or__(self, other):
        if isinstance(other, Sentence):
            self.add(other)
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
    def is_epsilon(self):
        return True


class Production:
    def __init__(self, non_terminal: NonTerminal, sentence: Sentence,
                 rule: Optional[Callable[['RuleList'], object]] = None):
        self.left: NonTerminal = non_terminal
        self.right: Sentence = sentence
        self.rule: Optional[Callable[['RuleList'], object]] = rule

    @property
    def is_epsilon(self) -> bool:
        return self.right.is_epsilon

    def __str__(self):
        return '%s -> %s' % (self.left, self.right)

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

    def add_terminal(self, name: str, regex: str = None, rule: Optional[TerminalRule] = None) -> Terminal:
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

    def production(self, *productions: str) -> Callable[[ProductionRule], ProductionRule]:
        """
        Return a function to decorate a method that will be used for as production rule

        :param productions: is a sequence of strings representing the production to attach the decorated function,
                       the string has the form '<non-terminal name> -> <symbols separated with white spaces>'

        :return: a function to decorate the production
        """

        def decorator(rule: ProductionRule) -> ProductionRule:
            for production in productions:
                head, body = production.split('->')
                head = self[head.strip()]
                head %= body.strip(), rule
            return rule

        return decorator

    def terminal(self, name: str, regex: str) -> Callable[[TerminalRule], TerminalRule]:
        """
        Return a function to decorate a method that will be used for as terminal rule in tokenizer process

        :param name: the name of a terminal

        :param regex: the regex of the terminal

        :return: a function to decorate the production
        """

        def decorator(rule: TerminalRule) -> TerminalRule:
            self.add_terminal(name, regex, rule)
            return rule

        return decorator

    def lexical_error(self, handler: Callable[[Lexer], None]):
        self.lexical_error_handler = handler
        return handler

    def augmented_grammar(self, force: bool = False):
        if not self.is_augmented_grammar or force:
            grammar = self.copy()
            start_symbol = grammar.start_symbol
            grammar.start_symbol = None
            new_start_symbol = grammar.add_non_terminal('S\'', True)
            new_start_symbol %= start_symbol + grammar.EPSILON, lambda x: x
            return grammar
        else:
            return self.copy()

    def copy(self):
        grammar = Grammar()
        grammar.productions = self.productions.copy()
        grammar.non_terminals = self.non_terminals.copy()
        grammar.terminals = self.terminals.copy()
        grammar.start_symbol = self.start_symbol
        grammar.EPSILON = self.EPSILON
        grammar.ERROR = self.ERROR
        grammar.EOF = self.EOF
        grammar.symbol_dict = self.symbol_dict.copy()

        return grammar

    def get_lexer(self) -> Lexer:
        items = self.terminal_rules.items()
        f1 = filter(lambda x: x[1][2] is not None, items)
        f2 = filter(lambda x: x[1][2] is None and not x[1][1], items)
        f3 = filter(lambda x: x[1][2] is None and x[1][1], items)

        ruled_tokens = list(f1)
        not_literal_tokens = sorted(f2, key=lambda x: len(x[1][0]), reverse=True)
        literal_tokens = sorted(f3, key=lambda x: len(x[1][0]), reverse=True)

        table = (
                [(name, regex) for name, (regex, _, _) in ruled_tokens] +
                [(name, regex) for name, (regex, _, _) in not_literal_tokens] +
                [(None, regex) for _, (regex, _, _) in literal_tokens]
        )

        return Lexer(table, self.EOF.name,
                     {s: r for s, (_, _, r) in items if r is not None}, self.lexical_error_handler)

    def get_parser(self, name: str, verbose: bool = False):
        if name == 'slr':
            return SLRParser(self, verbose)

        if name == 'lalr1':
            return LALR1Parser(self, verbose)

        if name == 'lr1':
            return LR1Parser(self, verbose)

        raise ValueError(name)  # TODO: create a custom error

    def serialize_lexer(self, class_name: str, grammar_module_name: str, grammar_variable_name: str = 'G'):
        LexerSerializer.build(self.get_lexer(), class_name, grammar_module_name, grammar_variable_name)

    def serialize_parser(self, parser_type: str, class_name: str, grammar_module_name: str,
                         grammar_variable_name: str = 'G'):
        LRParserSerializer.build(self.get_parser(parser_type), class_name, grammar_module_name, grammar_variable_name)

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

    def to_json(self):

        productions = []

        for p in self.productions:
            head = p.left.name

            body = []

            for s in p.right:
                body.append(s.Name)

            productions.append({'Head': head, 'Body': body})

        d = {'NonTerminals': [symbol.name for symbol in self.non_terminals],
             'Terminals': [symbol.name for symbol in self.terminals],
             'Productions': productions}

        # [{'Head':p.Left.Name, "Body": [s.Name for s in p.Right]} for p in self.Productions]
        return json.dumps(d)

    @staticmethod
    def from_json(data):
        data = json.loads(data)

        grammar = Grammar()
        dic = {'epsilon': grammar.EPSILON}

        for term in data['Terminals']:
            dic[term] = grammar.add_terminal(term)

        for noTerm in data['NonTerminals']:
            dic[noTerm] = grammar.add_non_terminal(noTerm)

        for p in data['Productions']:
            head = p['Head']
            dic[head] %= Sentence(*[dic[term] for term in p['Body']])

        return grammar

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
        non_terminals = mul * (len(self.non_terminals) - 1) + '%s\n'
        ans += non_terminals % tuple(self.non_terminals)
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
    def is_reduce_item(self) -> bool:
        return len(self.production.right) == self.pos

    @property
    def next_symbol(self) -> Optional[Symbol]:
        if self.pos < len(self.production.right):
            return self.production.right[self.pos]
        else:
            return None

    def next_item(self) -> Optional['Item']:
        if self.pos < len(self.production.right):
            return Item(self.production, self.pos + 1, self.lookaheads)
        else:
            return None

    def preview(self, skip: int = 1) -> List[Symbol]:
        unseen = self.production.right[self.pos + skip:]
        return [unseen + (lookahead,) for lookahead in self.lookaheads]

    def center(self) -> 'Item':
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

    def __len__(self):
        return len(self.__list)

    def set_error(self, index: int, message: str):
        self.__parser.set_error(self[index].line, self[index].column, message)


##############################
# Compute Firsts and Follows #
##############################
def compute_local_first(firsts, alpha):
    first_alpha = ContainerSet()

    try:
        alpha_is_epsilon = alpha.is_epsilon
    except AttributeError:
        alpha_is_epsilon = False

    if alpha_is_epsilon:
        first_alpha.set_epsilon()
    else:
        for symbol in alpha:
            first_symbol = firsts[symbol]
            first_alpha.update(first_symbol)
            if not first_symbol.contains_epsilon:
                break
        else:
            first_alpha.set_epsilon()

    return first_alpha


def compute_firsts(grammar: Grammar):
    firsts = {}
    change = True

    for terminal in grammar.terminals:
        firsts[terminal] = ContainerSet(terminal)

    for non_terminal in grammar.non_terminals:
        firsts[non_terminal] = ContainerSet()

    while change:
        change = False

        # P: X -> alpha
        for production in grammar.productions:
            x, alpha = production

            first_x = firsts[x]

            try:
                first_alpha = firsts[alpha]
            except KeyError:
                first_alpha = firsts[alpha] = ContainerSet()

            local_first = compute_local_first(firsts, alpha)

            change |= first_alpha.hard_update(local_first)
            change |= first_x.hard_update(local_first)

    return firsts


def compute_follows(grammar: Grammar, firsts):
    follows = {}
    change = True

    local_firsts = {}

    # init Follow(Vn)
    for non_terminal in grammar.non_terminals:
        follows[non_terminal] = ContainerSet()
    follows[grammar.start_symbol] = ContainerSet(grammar.EOF)

    while change:
        change = False

        # P: X -> alpha
        for production in grammar.productions:
            x, alpha = production

            follow_x = follows[x]

            for i, symbol_y in enumerate(alpha):
                # X -> zeta Y beta
                if symbol_y.is_non_terminal:
                    follow_y = follows[symbol_y]
                    try:
                        first_beta = local_firsts[alpha, i]
                    except KeyError:
                        first_beta = local_firsts[alpha, i] = compute_local_first(firsts, alpha[i + 1:])
                    # First(beta) - { epsilon } subset of Follow(Y)
                    change |= follow_y.update(first_beta)
                    # beta ->* epsilon or X -> zeta Y ? Follow(X) subset of Follow(Y)
                    if first_beta.contains_epsilon:
                        change |= follow_y.update(follow_x)
    # Follow(Vn)
    return follows


#########################
# LR0 AUTOMATA BUILDING #
#########################
def closure_lr0(items: Iterable[Item]):
    closure = set(items)

    pending = set(items)
    while pending:
        current = pending.pop()
        symbol = current.next_symbol

        if current.is_reduce_item or symbol.is_terminal:
            continue

        new_items = set(Item(p, 0) for p in symbol.productions)  # if Item(p, 0) not in closure]
        pending |= new_items - closure
        closure |= new_items
    return frozenset(closure)


def goto_lr0(items: Iterable[Item], symbol: Symbol) -> FrozenSet[Item]:
    return frozenset(item.next_item() for item in items if item.next_symbol == symbol)


def build_lr0_automaton(grammar: Grammar, just_kernel: bool = False) -> State:
    assert len(grammar.start_symbol.productions) == 1, 'Grammar must be augmented'

    start_production = grammar.start_symbol.productions[0]
    start_item = Item(start_production, 0)
    start = frozenset([start_item])

    if not just_kernel:
        automaton = State(closure_lr0(start), True)
    else:
        automaton = State(start, True)

    pending = [start]
    visited = {start: automaton}

    symbols = grammar.terminals + grammar.non_terminals
    while pending:
        current = pending.pop()
        current_state = visited[current]
        current_closure = current_state.state if not just_kernel else closure_lr0(current)
        for symbol in symbols:
            kernel = goto_lr0(current_closure, symbol)

            if kernel == frozenset():
                continue

            try:
                next_state = visited[kernel]
            except KeyError:
                next_state = visited[kernel] = State(closure_lr0(kernel), True) if not just_kernel else State(kernel,
                                                                                                              True)
                pending.append(kernel)

            current_state.add_transition(symbol.name, next_state)
    return automaton


#########################
# LR1 AUTOMATA BUILDING #
#########################
def compress(items: Iterable[Item]) -> Set[Item]:
    centers = {}

    for item in items:
        center = item.center()
        try:
            lookaheads = centers[center]
        except KeyError:
            centers[center] = lookaheads = set()
        lookaheads.update(item.lookaheads)

    return set(Item(x.production, x.pos, set(lookaheads)) for x, lookaheads in centers.items())


def expand(item: Item, firsts):
    next_symbol = item.next_symbol
    if next_symbol is None or not next_symbol.is_non_terminal:
        return []

    lookaheads = ContainerSet()

    for preview in item.preview():
        local_first = compute_local_first(firsts, preview)
        lookaheads.update(local_first)

    assert not lookaheads.contains_epsilon

    return [Item(p, 0, lookaheads) for p in next_symbol.productions]


def closure_lr1(items, firsts):
    closure = set(items)
    pending = set(items)
    while pending:
        new_items = set(expand(pending.pop(), firsts))
        pending |= new_items - closure
        closure |= new_items
    return compress(closure)


def goto_lr1(items, symbol, firsts=None, just_kernel=False):
    assert just_kernel or firsts is not None, '`firsts` must be provided if `just_kernel=False`'
    items = frozenset(item.next_item() for item in items if item.next_symbol == symbol)
    return items if just_kernel else closure_lr1(items, firsts)


def build_lr1_automaton(grammar, firsts=None):
    assert len(grammar.start_symbol.productions) == 1, 'Grammar must be augmented'

    if not firsts:
        firsts = compute_firsts(grammar)
    firsts[grammar.EOF] = ContainerSet(grammar.EOF)

    start_production = grammar.start_symbol.productions[0]
    start_item = Item(start_production, 0, lookaheads=(grammar.EOF,))
    start = frozenset([start_item])

    closure = closure_lr1(start, firsts)
    automaton = State(frozenset(closure), True)

    pending = [start]
    visited = {start: automaton}

    symbols = grammar.terminals + grammar.non_terminals
    while pending:
        current = pending.pop()
        current_state = visited[current]

        current_closure = current_state.state
        for symbol in symbols:
            kernel = goto_lr1(current_closure, symbol, just_kernel=True)

            if kernel == frozenset():
                continue

            try:
                next_state = visited[kernel]
            except KeyError:
                goto = closure_lr1(kernel, firsts)
                visited[kernel] = next_state = State(frozenset(goto), True)
                pending.append(kernel)
            current_state.add_transition(symbol.name, next_state)

    return automaton


###########################
# LALR1 AUTOMATA BUILDING #
###########################
def determining_lookaheads(state, propagate, table, firsts):
    for item_kernel in state.state:
        closure = closure_lr1([Item(item_kernel.production, item_kernel.pos, ('#',))], firsts)
        for item in closure:
            if item.is_reduce_item:
                continue

            next_state = state.get(item.next_symbol.name)
            next_item = item.next_item().center()
            if '#' in item.lookaheads:
                propagate[state, item_kernel].append((next_state, next_item))
            table[next_state, next_item].extend(item.lookaheads - {'#'})


def build_lalr1_automaton(grammar, firsts=None):
    automaton = build_lr0_automaton(grammar, just_kernel=True)

    if not firsts:
        firsts = compute_firsts(grammar)
    firsts['#'] = ContainerSet('#')
    firsts[grammar.EOF] = ContainerSet(grammar.EOF)

    table = {(state, item): ContainerSet() for state in automaton for item in state.state}
    propagate = {(state, item): [] for state in automaton for item in state.state}

    for state in automaton:
        determining_lookaheads(state, propagate, table, firsts)
    del firsts['#']

    start_item = list(automaton.state).pop()
    table[automaton, start_item] = ContainerSet(grammar.EOF)

    change = True
    while change:
        change = False
        for from_state, from_item in propagate:
            for to_state, to_item in propagate[from_state, from_item]:
                change |= table[to_state, to_item].extend(table[from_state, from_item])

    for state in automaton:
        for item in state.state:
            item.lookaheads = frozenset(table[state, item])

    for state in automaton:
        state.state = frozenset(closure_lr1(state.state, firsts))

    return automaton


#############################
# SLR & LR1 & LALR1 Parsers #
#############################
class ShiftReduceParser:
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'
    OK = 'OK'
    contains_errors = False

    def __init__(self, grammar: Grammar, verbose: bool = False):
        self.grammar = grammar
        self.augmented_grammar = grammar.augmented_grammar(True)
        self.firsts = compute_firsts(self.augmented_grammar)
        self.follows = compute_follows(self.augmented_grammar, self.firsts)
        self.automaton = self._build_automaton()
        self.conflicts = []
        self.verbose = verbose

        self.action = {}
        self.goto = {}
        self.shift_reduce_count = 0
        self.reduce_reduce_count = 0
        self._errors = []
        self._build_parsing_table()

        if self.conflicts:
            sys.stderr.write(f"Warning: {self.shift_reduce_count} Shift-Reduce Conflicts\n")
            sys.stderr.write(f"Warning: {self.reduce_reduce_count} Reduce-Reduce Conflicts\n")

    ##############
    # Errors API #
    ##############
    @property
    def errors(self, clean: bool = True):
        return [(m if clean else (r, c, m)) for r, c, m in sorted(self._errors)]

    def set_error(self, line, column, message):
        self._errors.append((line, column, message))

    #############
    #    End    #
    #############

    def _build_parsing_table(self):
        grammar = self.augmented_grammar
        automaton = self.automaton

        for i, node in enumerate(automaton):
            node.id = i

        for node in automaton:
            for item in node.state:
                if item.is_reduce_item:
                    if item.production.left == grammar.start_symbol:
                        self._register(self.action, (node.id, grammar.EOF), (self.OK, None))
                    else:
                        for lookahead in self._lookaheads(item):
                            self._register(self.action, (node.id, lookahead), (self.REDUCE, item.production))
                else:
                    symbol = item.next_symbol
                    if symbol.is_terminal:
                        self._register(self.action, (node.id, symbol), (self.SHIFT, node.get(symbol.name).id))
                    else:
                        self._register(self.goto, (node.id, symbol), node.get(symbol.name).id)

    def _register(self, table, key, value):
        if key in table and table[key] != value:
            action, tag = table[key]
            if action != value[0]:
                if action == self.SHIFT:
                    table[key] = value  # By default shifting if exists a Shift-Reduce Conflict
                self.shift_reduce_count += 1
                self.conflicts.append(('SR', value[1], tag))
            else:
                self.reduce_reduce_count += 1
                self.conflicts.append(('RR', value[1], tag))
        else:
            table[key] = value

    def _build_automaton(self):
        raise NotImplementedError()

    def _lookaheads(self, item):
        raise NotImplementedError()

    def __call__(self, tokens: List[Token]):
        """
        Parse the given TokenList

        :param tokens: List[Token]
        :return: Any
        """
        inserted_error = False
        stack: list = [0]  # The order in stack is [init state] + [symbol, rule, state, ...]
        cursor = 0

        while True:
            if cursor >= len(tokens):
                return

            state = stack[-1]
            lookahead = tokens[cursor]

            if isinstance(lookahead.token_type, str):
                # making the token_type always a terminal
                lookahead.token_type = self.grammar[lookahead.token_type]

            if self.verbose:
                prev = ' '.join([s.name for s in stack if isinstance(s, Symbol)])
                post = ' '.join([tokens[i].lex for i in range(cursor, len(tokens))])
                print(f'{prev} <-> {post}')
                print()

            ##########################
            # Error Handling Section #
            ##########################
            if (state, lookahead.token_type) not in self.action:
                self.contains_errors = True

                if (state, self.grammar.ERROR) in self.action:
                    if self.verbose:
                        print(f'Inserted error token {lookahead,}')

                    inserted_error = True
                    lookahead = Token(lookahead.lex, self.grammar.ERROR, lookahead.line, lookahead.column)
                else:
                    # If an error insertion fails then the parsing process enter into a panic mode recovery
                    sys.stderr.write(
                        f'{lookahead.line, lookahead.column} - SyntacticError: ERROR at or near "{lookahead.lex}"\n')

                    while (state, lookahead.token_type) not in self.action:
                        cursor += 1
                        if cursor >= len(tokens):
                            return
                        lookahead = tokens[cursor]

                    continue
            #######
            # End #
            #######

            action, tag = self.action[state, lookahead.token_type]

            if action == self.SHIFT:
                # in this case tag is an integer
                if self.verbose:
                    print(f'Shift: {lookahead.lex, tag}')

                if not inserted_error:
                    # the rule of an error token is it lexeme
                    stack += [lookahead.token_type, lookahead.lex, tag]
                    cursor += 1
                else:
                    # the rule of an error token is the self token
                    stack += [lookahead.token_type, lookahead, tag]
            elif action == self.REDUCE:
                # in this case tag is a Production
                if self.verbose:
                    print(f'Reduce: {repr(tag)}')

                head, body = tag

                rules = RuleList(self, [None] * (len(body) + 1))
                for i, s in enumerate(reversed(body), 1):
                    state, rules[-i], symbol = stack.pop(), stack.pop(), stack.pop()
                    assert s == symbol, f'ReduceError: in production "{repr(tag)}". Expected {s} instead of {s}'

                if tag.rule is not None:
                    rules[0] = tag.rule(rules)

                state = stack[-1]
                goto = self.goto[state, head]
                stack += [head, rules[0], goto]
            elif action == self.OK:
                return stack[2]
            else:
                raise Exception(f'ParsingError: invalid action {action}')

            inserted_error = False


class SLRParser(ShiftReduceParser):
    def _build_automaton(self):
        return build_lr0_automaton(self.augmented_grammar)

    def _lookaheads(self, item):
        return self.follows[item.production.left]


class LR1Parser(ShiftReduceParser):
    def _build_automaton(self):
        return build_lr1_automaton(self.augmented_grammar, firsts=self.firsts)

    def _lookaheads(self, item):
        return item.lookaheads


class LALR1Parser(LR1Parser):
    def _build_automaton(self):
        return build_lalr1_automaton(self.augmented_grammar, firsts=self.firsts)
