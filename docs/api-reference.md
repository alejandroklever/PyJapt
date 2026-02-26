# API Reference

This page lists every public class and method exported by PyJapt.

---

## Top-Level Exports

```python
from pyjapt import (
    Grammar,
    Lexer,
    Token,
    ShiftReduceParser,
    SLRParser,
    LR1Parser,
    LALR1Parser,
)
```

---

## `Grammar`

The central object for defining a language.

```python
from pyjapt import Grammar
g = Grammar()
```

### Terminals

---

#### `Grammar.add_terminal(name, regex=None, rule=None) -> Terminal`

Create and register a terminal symbol.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique terminal name. Must be a valid string. |
| `regex` | `str \| None` | Regular expression. If `None`, the regex is `re.escape(name)` (literal match). |
| `rule` | `Callable[[Lexer], Optional[Token]] \| None` | Rule function invoked when this token is matched. |

Returns the new `Terminal` object.

Raises `AssertionError` if `name` is already defined.

---

#### `Grammar.add_terminals(names) -> Tuple[Terminal, ...]`

Convenience wrapper. Splits `names` on whitespace and calls `add_terminal` for each.

```python
plus, minus, star = g.add_terminals('+ - *')
```

---

#### `Grammar.terminal(name, regex) -> Callable`

Decorator factory. Creates the terminal **and** registers the decorated function as its rule.

```python
@g.terminal('int', r'\d+')
def int_rule(lexer):
    lexer.position += len(lexer.token.lex)
    lexer.column   += len(lexer.token.lex)
    return lexer.token
```

---

#### `Grammar.add_terminal_error()`

Registers the built-in `error` terminal for use in error-recovery productions. Call this before writing any production that contains `error`.

---

### Non-Terminals

---

#### `Grammar.add_non_terminal(name, start_symbol=False) -> NonTerminal`

Create and register a non-terminal symbol.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique non-terminal name. |
| `start_symbol` | `bool` | Mark as the start symbol. Only one allowed per grammar. |

Raises `Exception` if a second `start_symbol=True` is provided.

---

#### `Grammar.add_non_terminals(names) -> Tuple[NonTerminal, ...]`

Splits `names` on whitespace and calls `add_non_terminal` for each.

```python
stmt, expr, term = g.add_non_terminals('stmt expr term')
```

---

### Productions

---

#### `Grammar.production(*production_strings) -> Callable`

Decorator factory that registers the decorated function as the semantic action for one or more productions.

The production string format is `'head -> body'` where `body` is a space-separated list of symbol names.

```python
@g.production('expr -> expr + term', 'expr -> expr - term')
def additive(s):
    return s[1] + s[3] if s[2] == '+' else s[1] - s[3]
```

---

#### `NonTerminal.__imod__(other) -> NonTerminal`

Operator `%=` overload for adding productions to a non-terminal.

```python
# Unattributed
expr %= 'expr + term'

# With semantic action
expr %= 'expr + term', lambda s: s[1] + s[3]

# Epsilon
expr %= ''
```

`other` can be:
- A `str` (space-separated symbol names)
- A `Symbol` or `Sentence` (built from Symbol objects with `+`)
- A `tuple` of `(str | Sentence, callable)` for attributed productions
- A `SentenceList` (built with `|`) for multiple alternatives

---

### Error Handlers

---

#### `Grammar.lexical_error(handler) -> handler`

Decorator. Registers a custom lexical error handler.

```python
@g.lexical_error
def lex_error(lexer):
    lexer.add_error(lexer.lineno, lexer.column,
                    f'unexpected "{lexer.token.lex}"')
    lexer.position += 1
    lexer.column   += 1
```

---

#### `Grammar.parsing_error(handler) -> handler`

Decorator. Registers a custom syntactic error handler.

```python
@g.parsing_error
def parse_error(parser):
    tok = parser.current_token
    parser.add_error(tok.line, tok.column, f'unexpected "{tok.lex}"')
```

---

### Generating the Lexer and Parser

---

#### `Grammar.get_lexer() -> Lexer`

Build and return a `Lexer` for this grammar.

---

#### `Grammar.get_parser(name, verbose=False) -> ShiftReduceParser`

Build and return a parser.

| `name` | Parser type |
|--------|-------------|
| `'slr'` | Simple LR |
| `'lalr1'` | LALR(1) |
| `'lr1'` | Canonical LR(1) |

Raises `ValueError` for unknown names.

---

### Serialization

---

#### `Grammar.serialize_lexer(class_name, grammar_module_name, grammar_variable_name='G')`

Generate `lexertab.py` in the current working directory.

---

#### `Grammar.serialize_parser(parser_type, class_name, grammar_module_name, grammar_variable_name='G')`

Generate `parsertab.py` in the current working directory.

---

### Utility

---

#### `Grammar.to_json() -> str`

Serialize the grammar structure (terminals, non-terminals, productions) to a JSON string. Semantic actions and regexes are **not** included.

---

#### `Grammar.from_json(data) -> Grammar`

Class method. Reconstruct a grammar from the JSON string produced by `to_json()`.

---

#### `Grammar.__getitem__(item) -> Symbol | Production | None`

Look up a symbol or production by name/repr-string.

```python
plus_symbol = g['+']
production  = g['expr -> expr + term']
```

---

## `Token`

```python
class Token:
    lex:        str   # lexeme string
    token_type: Any   # terminal name (str) or Terminal object
    line:       int   # 1-based line number
    column:     int   # 1-based column number
```

### Class methods

#### `Token.empty() -> Token`

Return an empty sentinel token `Token('', '', 0, 0)`.

### Properties

#### `Token.is_valid -> bool`

Always `True` for a regular token. (Subclasses may override for error tokens.)

---

## `Lexer`

```python
class Lexer:
    lineno:         int   # current line (1-based)
    column:         int   # current column (1-based)
    position:       int   # byte offset in input
    text:           str   # full input string
    token:          Token # token being processed
    contain_errors: bool  # True after first error
```

### `Lexer.__call__(text) -> List[Token]`

Tokenise `text`. Resets all internal state before each call. Appends an EOF token at the end.

### `Lexer.tokenize(text) -> Generator[Token, None, None]`

Low-level generator. Does **not** reset state. Prefer `__call__` for normal use.

### `Lexer.errors -> List[str]`

Sorted list of error message strings accumulated during the last call.

### `Lexer.add_error(line, col, message)`

Append an error entry. Intended for use inside custom terminal rules and error handlers.

---

## `ShiftReduceParser`

Base class for all three parser variants. Do not instantiate directly; use `Grammar.get_parser`.

```python
class ShiftReduceParser:
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'
    OK = 'OK'
```

### `ShiftReduceParser.__call__(tokens) -> Any`

Parse a list of `Token` objects and return the semantic value of the start symbol, or `None` if parsing failed.

### `ShiftReduceParser.errors -> List[str]`

Sorted list of syntactic error messages.

### `ShiftReduceParser.add_error(line, column, message)`

Append an error entry from inside a semantic action or error handler.

### `ShiftReduceParser.contains_errors -> bool`

`True` if any parsing error has been detected.

### `ShiftReduceParser.current_token -> Token`

The token being processed at the time the most recent error occurred.

### `ShiftReduceParser.conflicts -> List[Tuple]`

List of detected conflicts, each a `('SR' | 'RR', prod_a, prod_b)` tuple.

### `ShiftReduceParser.shift_reduce_count -> int`

Number of shift-reduce conflicts.

### `ShiftReduceParser.reduce_reduce_count -> int`

Number of reduce-reduce conflicts.

---

## `SLRParser`

```python
class SLRParser(ShiftReduceParser): ...
```

Uses the LR(0) automaton and Follow sets for lookaheads.

---

## `LR1Parser`

```python
class LR1Parser(ShiftReduceParser): ...
```

Uses the canonical LR(1) automaton with per-item lookaheads.

---

## `LALR1Parser`

```python
class LALR1Parser(LR1Parser): ...
```

Uses the merged LALR(1) automaton. Same states as SLR, same power as LR(1) for most grammars.

---

## `RuleList`

Passed to every semantic action as `s`. 1-indexed over the production body.

### `RuleList.__getitem__(index) -> Any`

`s[0]` — head value (output).
`s[1]` … `s[n]` — body symbol values.

### `RuleList.add_error(index, message)`

Report an error at the position of `s[index]` (int) or at an explicit `(line, column)` tuple.

### `RuleList.force_parsing_error()`

Mark the parse as failed without adding an error message.

---

## `NonTerminal`

Represents a grammar non-terminal.

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Symbol name |
| `productions` | `List[Production]` | Productions where this symbol is the head |

---

## `Terminal`

Represents a grammar terminal.

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Symbol name |

---

## `Production`

| Attribute | Type | Description |
|-----------|------|-------------|
| `left` | `NonTerminal` | Production head |
| `right` | `Sentence` | Production body |
| `rule` | `Callable \| None` | Semantic action |
