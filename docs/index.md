# PyJapt

**PyJapt** — *Just Another Parsing Tool Written in Python* — is a lexer and LR parser generator that lets you define a language grammar in pure Python and immediately produce a working tokeniser and parser from it.

<p align="center">
  <img width="800" alt="PyJapt Logo Banner" src="https://github.com/alejandroklever/PyJapt/assets/45394625/ce9fd982-8f08-41ba-aa9e-54c2de24212b">
</p>

---

## Why PyJapt?

| Feature | Description |
|---------|-------------|
| **Pure Python** | No C extensions, no generated files to check in, no build step. |
| **Three LR parser types** | SLR, LR(1), and LALR(1) — choose the power level you need. |
| **Custom error handling** | Lexical and syntactic error handlers are first-class citizens. |
| **Semantic actions** | Attach a lambda or a decorated function to any production rule. |
| **Serialisation** | Pre-build the parsing tables and serialise them to a Python module for faster startup. |
| **Decorator-based API** | Define terminals and production rules without leaving Python. |

---

## Quick Example

A complete arithmetic expression parser in under 25 lines:

```python
from pyjapt import Grammar

g = Grammar()
expr = g.add_non_terminal('expr', True)
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

print(parser(lexer('(2 + 2) * 2 + 2')))  # 10
```

---

## Installation

```sh
pip install pyjapt
```

PyJapt requires **Python 3.10** or later and has no runtime dependencies.

---

## How It Works

PyJapt revolves around the `Grammar` class. You describe your language by:

1. **Declaring non-terminals** — the syntactic categories of your language.
2. **Declaring terminals** — the tokens produced by the lexer.
3. **Writing productions** — rules that describe how non-terminals are composed, with optional semantic actions.
4. **Generating the lexer and parser** — call `get_lexer()` and `get_parser(type)`.

```
Grammar definition
      │
      ├─► get_lexer()   → Lexer  (regex-based tokeniser)
      │
      └─► get_parser()  → ShiftReduceParser  (SLR / LR1 / LALR1)
```

---

## Next Steps

- Follow the [Getting Started](getting-started.md) guide to build your first language.
- Learn how to [define a grammar](defining-grammar.md) in detail.
- Read about [error handling](error-handling.md) to build robust parsers.
- Check the [API Reference](api-reference.md) for the complete public API.
