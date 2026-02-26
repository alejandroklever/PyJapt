# Defining a Grammar

A `Grammar` object is the single source of truth for your language. This page covers all the ways to build one.

---

## Creating the Grammar

```python
from pyjapt import Grammar

g = Grammar()
```

---

## Non-Terminals

Non-terminals are the syntactic categories of your language (e.g. `expr`, `statement`, `program`).

### `add_non_terminal(name, start_symbol=False)`

```python
program = g.add_non_terminal('program', start_symbol=True)
stmt    = g.add_non_terminal('stmt')
expr    = g.add_non_terminal('expr')
```

- `name` — must be a unique, non-empty string.
- `start_symbol=True` — marks this as the grammar's start symbol. Only one non-terminal can carry this flag.

Returns a `NonTerminal` object that you use to write productions.

### `add_non_terminals(names)`

Convenience method: accepts a space-separated string and returns a tuple of `NonTerminal` objects in the same order.

```python
stmt, expr, term, fact = g.add_non_terminals('stmt expr term fact')
```

---

## Terminals

Terminals are the atomic tokens produced by the lexer.

### `add_terminal(name, regex=None, rule=None)`

```python
# Literal terminal — the regex is the escaped name
plus  = g.add_terminal('+')
minus = g.add_terminal('-')

# Terminal with a custom regex
num = g.add_terminal('int', regex=r'\d+')

# Terminal with a custom regex AND a lexer rule
num = g.add_terminal('int', regex=r'\d+', rule=lambda lexer: ...)
```

- When `regex` is `None`, the regular expression used is `re.escape(name)`, so `+` matches the literal character `+`.
- `rule` is a function `(Lexer) -> Optional[Token]`. If it returns `None`, the token is discarded.

### `add_terminals(names)`

Accepts a space-separated string and returns a tuple. All created terminals use their name as the literal regex.

```python
plus, minus, star, div, lpar, rpar = g.add_terminals('+ - * / ( )')
```

### `@g.terminal(name, regex)`

A decorator that creates the terminal **and** registers the rule in one step.

```python
@g.terminal('int', r'\d+')
def int_terminal(lexer):
    lexer.column   += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    lexer.token.lex = int(lexer.token.lex)
    return lexer.token
```

The decorated function receives the `Lexer` instance and must either return the `Token` (possibly modified) or return `None`/nothing to discard it.

---

## Productions

Productions define how non-terminals are composed from sequences of terminals and non-terminals.

### Using `%=` with a string (recommended)

```python
expr %= 'expr + term'    # unattributed
expr %= 'expr + term', lambda s: s[1] + s[3]  # with semantic action
```

The string on the right-hand side is a space-separated list of symbol names. Each name must already be declared in the grammar.

Inside the semantic action, `s` is a `RuleList`:

| Index | Meaning |
|-------|---------|
| `s[0]` | The head non-terminal's value (set by returning from the action) |
| `s[1]` | Value of the 1st body symbol |
| `s[2]` | Value of the 2nd body symbol |
| `s[n]` | Value of the nth body symbol |

For a terminal, the value is the token's lexeme (`str`).
For a non-terminal, the value is whatever its production's semantic action returned.

### Using `%=` with `Symbol` objects

```python
expr %= expr + plus + term
expr %= expr + plus + term, lambda s: s[1] + s[3]
```

`Symbol` objects support `+` to build `Sentence` objects, so you can construct productions with the original variable references.

### Epsilon productions

```python
expr %= ''           # empty string → epsilon production
expr %= g.EPSILON    # same thing using the EPSILON symbol directly
```

### `@g.production(*production_strings)`

A decorator alternative to `%=`. It binds the decorated function to one or more production strings.

```python
@g.production('expr -> expr + term')
def expr_add(s):
    return s[1] + s[3]
```

The string format is `'head -> body'` where `->` separates the head non-terminal from the body symbols.

You can attach the same function to multiple productions:

```python
@g.production(
    'expr -> expr + expr',
    'expr -> expr - expr',
    'expr -> expr * expr',
    'expr -> expr / expr',
)
def binary_op(s):
    if s[2] == '+': return s[1] + s[3]
    if s[2] == '-': return s[1] - s[3]
    if s[2] == '*': return s[1] * s[3]
    if s[2] == '/': return s[1] // s[3]
```

---

## Special Terminals

### `g.EOF`

The end-of-file terminal (`$`). It is added automatically; you should not declare it yourself.

### `g.EPSILON`

Represents the empty word. Use it to write nullable productions.

### `g.ERROR`

A special terminal used for error recovery productions. You must register it explicitly before use:

```python
g.add_terminal_error()
```

See [Error Handling](error-handling.md) for full details.

---

## Inspecting the Grammar

```python
# All non-terminals
print(g.non_terminals)

# All terminals
print(g.terminals)

# All productions
print(g.productions)

# Look up any symbol by name
sym = g['expr']

# Look up a production by repr-string
prod = g['expr -> expr + term']
```

### `Grammar.__str__`

```python
print(g)
# Non-Terminals:
#     expr, term, fact
# Terminals:
#     +, -, *, /, (, ), int, whitespace
# Productions:
#     [expr -> expr + term, ...]
```

---

## JSON Import / Export

PyJapt supports a basic JSON representation of the grammar (without semantic actions):

```python
json_str = g.to_json()

g2 = Grammar.from_json(json_str)
```

!!! note
    JSON serialisation does not preserve terminal regexes, terminal rules, or semantic actions. It is useful for inspecting grammar structure, not for production use. Use [Python file serialisation](serialization.md) for production scenarios.
