# Getting Started

This guide walks you through installing PyJapt and building your first working lexer and parser.

---

## Prerequisites

- Python **3.10** or later
- `pip` (any recent version)

---

## Installation

```sh
pip install pyjapt
```

Verify the installation:

```python
import pyjapt
print(pyjapt.__version__)  # e.g. 0.4.1
```

---

## Your First Grammar — Arithmetic Expressions

We will build a complete interpreter for arithmetic expressions that supports `+`, `-`, `*`, `/`, integer literals, and parentheses.

### Step 1 — Create the Grammar object

```python
from pyjapt import Grammar

g = Grammar()
```

`Grammar` is the central object. Everything — terminals, non-terminals, productions, and the resulting lexer and parser — comes from this one instance.

---

### Step 2 — Declare non-terminals

```python
expr = g.add_non_terminal('expr', start_symbol=True)
term, fact = g.add_non_terminals('term fact')
```

`add_non_terminal` creates a single non-terminal and returns a `NonTerminal` object.
Pass `start_symbol=True` to mark it as the grammar's start symbol (only one is allowed).

`add_non_terminals` accepts a space-separated string and returns a tuple.

---

### Step 3 — Declare terminals

```python
g.add_terminals('+ - / * ( )')   # literal terminals
g.add_terminal('int', regex=r'\d+')  # terminal with a custom regex
```

Terminals declared with `add_terminals` use their name as the regex literally.
`add_terminal` lets you provide a custom regular expression.

---

### Step 4 — Handle whitespace

Whitespace is not a meaningful token in this grammar, so we skip it by not returning anything from the rule function.

```python
@g.terminal('whitespace', r' +')
def whitespace(lexer):
    lexer.column += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    # no return → token is discarded
```

---

### Step 5 — Write productions with semantic actions

Productions are attached to non-terminal objects using the `%=` operator.
The second element of the tuple is a *semantic action* — a function (or lambda) that receives the `RuleList` for that production and returns the production's semantic value.

```python
# expr → expr + term | expr - term | term
expr %= 'expr + term', lambda s: s[1] + s[3]
expr %= 'expr - term', lambda s: s[1] - s[3]
expr %= 'term',        lambda s: s[1]

# term → term * fact | term / fact | fact
term %= 'term * fact', lambda s: s[1] * s[3]
term %= 'term / fact', lambda s: s[1] // s[3]
term %= 'fact',        lambda s: s[1]

# fact → ( expr ) | int
fact %= '( expr )',    lambda s: s[2]
fact %= 'int',         lambda s: int(s[1])
```

Inside a semantic action `s` is a `RuleList`.
`s[0]` is the synthesised value of the production's *head* (i.e. what you return).
`s[1]`, `s[2]`, … are the values of each symbol in the production's *body* (1-indexed).

---

### Step 6 — Generate the lexer and parser

```python
lexer  = g.get_lexer()
parser = g.get_parser('slr')  # 'slr', 'lr1', or 'lalr1'
```

The lexer is a callable that turns a string into a list of `Token` objects.
The parser is a callable that takes that list and applies the grammar rules, returning the final semantic value.

---

### Step 7 — Parse an expression

```python
tokens = lexer('(2 + 2) * 2 + 2')
result = parser(tokens)
print(result)  # 10
```

Or more concisely:

```python
print(parser(lexer('(2 + 2) * 2 + 2')))  # 10
```

---

## Full Source

```python
from pyjapt import Grammar

g = Grammar()
expr = g.add_non_terminal('expr', start_symbol=True)
term, fact = g.add_non_terminals('term fact')
g.add_terminals('+ - / * ( )')
g.add_terminal('int', regex=r'\d+')

@g.terminal('whitespace', r' +')
def whitespace(lexer):
    lexer.column += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)

expr %= 'expr + term', lambda s: s[1] + s[3]
expr %= 'expr - term', lambda s: s[1] - s[3]
expr %= 'term',        lambda s: s[1]

term %= 'term * fact', lambda s: s[1] * s[3]
term %= 'term / fact', lambda s: s[1] // s[3]
term %= 'fact',        lambda s: s[1]

fact %= '( expr )',    lambda s: s[2]
fact %= 'int',         lambda s: int(s[1])

lexer  = g.get_lexer()
parser = g.get_parser('slr')

print(parser(lexer('(2 + 2) * 2 + 2')))   # 10
print(parser(lexer('1 + 2 * 5 - 4')))      # 7
print(parser(lexer('((3 + 4) * 5) - 6 / 2')))  # 32
```

---

## Next Steps

- [Defining a Grammar](defining-grammar.md) — all grammar construction options in detail.
- [Configuring the Lexer](lexer.md) — terminal priority, token rules, and ignored tokens.
- [Building a Parser](parser.md) — SLR vs LR(1) vs LALR(1) and how to pick one.
- [Error Handling](error-handling.md) — how to report lexical and syntactic errors.
