# Configuring the Lexer

The lexer produced by `g.get_lexer()` is a regex-based tokeniser. Understanding how it orders and applies patterns is essential for writing grammars with keywords, identifiers, and complex token types.

---

## How the Lexer Works

When called with a string, the lexer scans from left to right trying to match the current position against a single combined regex. The first alternative in that regex that matches wins.

The alternatives are ordered as follows:

1. **Ruled terminals** — terminals declared with `@g.terminal(...)` or via `add_terminal(..., rule=...)`, in the order they were declared.
2. **Non-literal terminals** — terminals with a custom `regex` argument but no rule, sorted longest-regex-first.
3. **Literal terminals** — terminals whose regex is their escaped name (declared via `add_terminal(name)` or `add_terminals(...)`), sorted longest-first.

This ordering means that custom rule functions are checked before pattern-only terminals, and longer patterns take priority over shorter ones within each group.

---

## The Token Class

```python
class Token:
    lex:        str   # the matched lexeme string
    token_type: Any   # the terminal's name (str) or Symbol object
    line:       int   # 1-based line number
    column:     int   # 1-based column number
```

---

## The Lexer Object Inside a Rule

When a terminal rule function is called, it receives the `Lexer` instance with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `lexer.token` | `Token` | The token that was just matched |
| `lexer.position` | `int` | Current byte offset in the input string |
| `lexer.lineno` | `int` | Current line number (1-based) |
| `lexer.column` | `int` | Current column number (1-based) |
| `lexer.text` | `str` | The full input string |
| `lexer.contain_errors` | `bool` | Set to `True` if any error has occurred |

**Important:** you are responsible for advancing `lexer.position` and `lexer.column` inside a rule. If you forget, the lexer will match the same input repeatedly.

---

## Common Terminal Patterns

### Discarding whitespace

```python
@g.terminal('whitespace', r' +')
def whitespace(lexer):
    lexer.column   += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    # return nothing → token is ignored
```

### Tracking newlines

```python
@g.terminal('newline', r'\n+')
def newline(lexer):
    lexer.lineno   += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    lexer.column    = 1
```

### Discarding tabs

```python
@g.terminal('tabulation', r'\t+')
def tab(lexer):
    lexer.column   += 4 * len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
```

### Modifying the lexeme

```python
@g.terminal('int', r'\d+')
def int_terminal(lexer):
    lexer.column   += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    lexer.token.lex = int(lexer.token.lex)  # convert to Python int
    return lexer.token
```

### Single-line comments

```python
@g.terminal('comment', r'//[^\n]*')
def line_comment(lexer):
    lexer.column   += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    # discard — no return
```

### Block comments

```python
@g.terminal('block_comment', r'/\*(.|\n)*?\*/')
def block_comment(lexer):
    lex = lexer.token.lex
    for ch in lex:
        if ch == '\n':
            lexer.lineno += 1
            lexer.column  = 1
        else:
            lexer.column += 1
    lexer.position += len(lex)
```

---

## Keywords vs Identifiers

Suppose your language has keywords (`if`, `else`, `while`) and identifiers (`[a-zA-Z_][a-zA-Z0-9_]*`). A naïve approach would match `if` as an identifier because the identifier regex is broader.

The correct solution is to declare keywords as literal terminals and write a single rule for identifiers that checks whether the matched text is a keyword:

```python
from pyjapt import Grammar

g = Grammar()
keywords = g.add_terminals('if else while return true false')
keyword_names = {t.name for t in keywords}

@g.terminal('id', r'[a-zA-Z_][a-zA-Z0-9_]*')
def id_terminal(lexer):
    lexer.column   += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    if lexer.token.lex in keyword_names:
        lexer.token.token_type = lexer.token.lex  # reclassify as keyword
    return lexer.token
```

Because `id_terminal` is a *ruled* terminal it runs first. If the lexeme is a keyword name, the token type is changed to the keyword name, so the parser sees the keyword terminal instead of an identifier.

---

## Calling the Lexer

```python
lexer = g.get_lexer()

# tokenise a string
tokens = lexer('x + 42')

for tok in tokens:
    print(tok)
# id: x
# +: +
# int: 42
# $: $   ← EOF token appended automatically
```

`Lexer.__call__` resets all internal state (position, line number, column, error list) before each run, so the same instance can be reused safely.

---

## Checking for Lexical Errors

After tokenisation, check `lexer.contain_errors` and read `lexer.errors`:

```python
tokens = lexer(source_code)

if lexer.contain_errors:
    for msg in lexer.errors:
        print(msg)
```

See [Error Handling](error-handling.md) for custom lexical error handlers.
