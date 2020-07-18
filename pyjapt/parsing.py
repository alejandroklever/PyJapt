import sys

from pyjapt.automata import State
from pyjapt.grammar import Item, RuleList, Symbol
from pyjapt.lexing import Token
from pyjapt.utils import ContainerSet


##############################
# Compute Firsts and Follows #
##############################
def compute_local_first(firsts, alpha):
    first_alpha = ContainerSet()

    try:
        alpha_is_epsilon = alpha.IsEpsilon
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


def compute_firsts(G):
    firsts = {}
    change = True

    for terminal in G.terminals:
        firsts[terminal] = ContainerSet(terminal)

    for nonterminal in G.non_terminals:
        firsts[nonterminal] = ContainerSet()

    while change:
        change = False

        # P: X -> alpha
        for production in G.productions:
            X, alpha = production

            first_X = firsts[X]

            try:
                first_alpha = firsts[alpha]
            except KeyError:
                first_alpha = firsts[alpha] = ContainerSet()

            local_first = compute_local_first(firsts, alpha)

            change |= first_alpha.hard_update(local_first)
            change |= first_X.hard_update(local_first)

    return firsts


def compute_follows(G, firsts):
    follows = {}
    change = True

    local_firsts = {}

    # init Follow(Vn)
    for nonterminal in G.non_terminals:
        follows[nonterminal] = ContainerSet()
    follows[G.start_symbol] = ContainerSet(G.EOF)

    while change:
        change = False

        # P: X -> alpha
        for production in G.productions:
            X = production.left
            alpha = production.right

            follow_X = follows[X]

            for i, symbol_Y in enumerate(alpha):
                # X -> zeta Y beta
                if symbol_Y.IsNonTerminal:
                    follow_Y = follows[symbol_Y]
                    try:
                        first_beta = local_firsts[alpha, i]
                    except KeyError:
                        first_beta = local_firsts[alpha, i] = compute_local_first(firsts, alpha[i + 1:])
                    # First(beta) - { epsilon } subset of Follow(Y)
                    change |= follow_Y.update(first_beta)
                    # beta ->* epsilon or X -> zeta Y ? Follow(X) subset of Follow(Y)
                    if first_beta.contains_epsilon:
                        change |= follow_Y.update(follow_X)
    # Follow(Vn)
    return follows


#########################
# LR0 AUTOMATA BUILDING #
#########################
def closure_lr0(items):
    closure = set(items)

    pending = set(items)
    while pending:
        current = pending.pop()
        symbol = current.NextSymbol

        if current.IsReduceItem or symbol.IsTerminal:
            continue

        new_items = set(Item(p, 0) for p in symbol.productions)  # if Item(p, 0) not in closure]
        pending |= new_items - closure
        closure |= new_items
    return frozenset(closure)


def goto_lr0(items, symbol):
    return frozenset(item.NextItem() for item in items if item.NextSymbol == symbol)


def build_lr0_automaton(G, just_kernel=False):
    assert len(G.start_symbol.productions) == 1, 'Grammar must be augmented'

    start_production = G.start_symbol.productions[0]
    start_item = Item(start_production, 0)
    start = frozenset([start_item])

    if not just_kernel:
        automaton = State(closure_lr0(start), True)
    else:
        automaton = State(start, True)

    pending = [start]
    visited = {start: automaton}

    symbols = G.terminals + G.non_terminals
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
def compress(items):
    centers = {}

    for item in items:
        center = item.Center()
        try:
            lookaheads = centers[center]
        except KeyError:
            centers[center] = lookaheads = set()
        lookaheads.update(item.lookaheads)

    return set(Item(x.production, x.pos, set(lookaheads)) for x, lookaheads in centers.items())


def expand(item, firsts):
    next_symbol = item.NextSymbol
    if next_symbol is None or not next_symbol.IsNonTerminal:
        return []

    lookaheads = ContainerSet()

    for preview in item.Preview():
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
    items = frozenset(item.NextItem() for item in items if item.NextSymbol == symbol)
    return items if just_kernel else closure_lr1(items, firsts)


def build_lr1_automaton(G, firsts=None):
    assert len(G.start_symbol.productions) == 1, 'Grammar must be augmented'

    if not firsts:
        firsts = compute_firsts(G)
    firsts[G.EOF] = ContainerSet(G.EOF)

    start_production = G.start_symbol.productions[0]
    start_item = Item(start_production, 0, lookaheads=(G.EOF,))
    start = frozenset([start_item])

    closure = closure_lr1(start, firsts)
    automaton = State(frozenset(closure), True)

    pending = [start]
    visited = {start: automaton}

    symbols = G.terminals + G.non_terminals
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
            if item.IsReduceItem:
                continue

            next_state = state.get(item.NextSymbol.name)
            next_item = item.NextItem().Center()
            if '#' in item.lookaheads:
                propagate[state, item_kernel].append((next_state, next_item))
            table[next_state, next_item].extend(item.lookaheads - {'#'})


def build_lalr1_automaton(G, firsts=None):
    automaton = build_lr0_automaton(G, just_kernel=True)

    if not firsts:
        firsts = compute_firsts(G)
    firsts['#'] = ContainerSet('#')
    firsts[G.EOF] = ContainerSet(G.EOF)

    table = {(state, item): ContainerSet() for state in automaton for item in state.state}
    propagate = {(state, item): [] for state in automaton for item in state.state}

    for state in automaton:
        determining_lookaheads(state, propagate, table, firsts)
    del firsts['#']

    start_item = list(automaton.state).pop()
    table[automaton, start_item] = ContainerSet(G.EOF)

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

    def __init__(self, G, verbose=False):
        self.G = G
        self.augmented_G = G.augmented_grammar(True)
        self.firsts = compute_firsts(self.augmented_G)
        self.follows = compute_follows(self.augmented_G, self.firsts)
        self.automaton = self._build_automaton()
        self.conflicts = []
        self.verbose = verbose

        self.action = {}
        self.goto = {}
        self.sr = 0
        self.rr = 0
        self._errors = []
        self._build_parsing_table()

        if self.conflicts:
            sys.stderr.write(f"Warning: {self.sr} Shift-Reduce Conflicts\n")
            sys.stderr.write(f"Warning: {self.rr} Reduce-Reduce Conflicts\n")

    ##############
    # Errors API #
    ##############
    @property
    def errors(self):
        return [m for _, _, m in sorted(self._errors)]

    def set_error(self, line, column, message):
        self._errors.append((line, column, message))
    #############
    #    End    #
    #############

    def _build_parsing_table(self):
        G = self.augmented_G
        automaton = self.automaton

        for i, node in enumerate(automaton):
            node.id = i

        for node in automaton:
            for item in node.state:
                if item.IsReduceItem:
                    if item.production.left == G.start_symbol:
                        self._register(self.action, (node.id, G.EOF), (self.OK, None))
                    else:
                        for lookahead in self._lookaheads(item):
                            self._register(self.action, (node.id, lookahead), (self.REDUCE, item.production))
                else:
                    symbol = item.NextSymbol
                    if symbol.IsTerminal:
                        self._register(self.action, (node.id, symbol), (self.SHIFT, node.get(symbol.name).id))
                    else:
                        self._register(self.goto, (node.id, symbol), node.get(symbol.name).id)

    def _register(self, table, key, value):
        if key in table and table[key] != value:
            action, tag = table[key]
            if action != value[0]:
                if action == self.SHIFT:
                    table[key] = value  # By default shifting if exists a Shift-Reduce Conflict
                self.sr += 1
                self.conflicts.append(('SR', value[1], tag))
            else:
                self.rr += 1
                self.conflicts.append(('RR', value[1], tag))
        else:
            table[key] = value

    def _build_automaton(self):
        raise NotImplementedError()

    def _lookaheads(self, item):
        raise NotImplementedError()

    def __call__(self, tokens):
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

                if (state, self.G.ERROR) in self.action:
                    if self.verbose:
                        print(f'Inserted error token {lookahead,}')

                    inserted_error = True
                    lookahead = Token(lookahead.lex, self.G.ERROR, lookahead.line, lookahead.column)
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
        return build_lr0_automaton(self.augmented_G)

    def _lookaheads(self, item):
        return self.follows[item.production.left]


class LR1Parser(ShiftReduceParser):
    def _build_automaton(self):
        return build_lr1_automaton(self.augmented_G, firsts=self.firsts)

    def _lookaheads(self, item):
        return item.lookaheads


class LALR1Parser(LR1Parser):
    def _build_automaton(self):
        return build_lalr1_automaton(self.augmented_G, firsts=self.firsts)
