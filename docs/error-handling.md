# Error Handling

Good error handling is one of PyJapt's core design goals. This page describes how to report and recover from both lexical and syntactic errors.

---

## Lexical Error Handling

### Default behavior

When the lexer encounters a character that matches no terminal pattern, it calls the *lexical error handler*. By default, this adds an error message to the internal errors list and advances past the bad character.

### Custom handler — `@g.lexical_error`

Decorate a function with `@g.lexical_error` to replace the default handler:

```python
@g.lexical_error
def on_lex_error(lexer):
    line, col = lexer.lineno, lexer.column
    bad_char  = lexer.token.lex

    lexer.add_error(line, col,
        f'({line}, {col}) - LexicographicError: unexpected character "{bad_char}"')

    # Always advance to avoid an infinite loop
    lexer.position += 1
    lexer.column   += 1
```

!!! warning "Always advance `lexer.position`"
    If your handler does not advance `lexer.position`, the lexer will match the same bad character indefinitely.

### Reporting errors from a terminal rule

You can also detect and report errors from inside a terminal rule:

```python
@g.terminal('comment_error', r'/\*(.|\n)*$')
def eof_in_comment(lexer):
    """Match a /* comment that reaches EOF without a closing */"""
    lexer.contain_errors = True
    lex = lexer.token.lex
    for ch in lex:
        if ch == '\n':
            lexer.lineno += 1
            lexer.column  = 1
        else:
            lexer.column += 1
    lexer.position += len(lex)
    lexer.add_error(
        lexer.lineno, lexer.column,
        f'({lexer.lineno}, {lexer.column}) - LexicographicError: EOF in comment'
    )
```

### Checking lexical errors

```python
tokens = lexer(source_code)

if lexer.contain_errors:
    for message in lexer.errors:
        print(message)
```

`lexer.errors` returns a list of error message strings, sorted by position.

---

## Syntactic Error Handling

### Default behavior

When the parser cannot find an action for the current `(state, token)` pair it enters *panic-mode recovery*: it calls the error handler and then skips input tokens until it finds one that fits the current state.

### Custom handler — `@g.parsing_error`

```python
@g.parsing_error
def on_parse_error(parser):
    tok = parser.current_token
    parser.add_error(
        tok.line, tok.column,
        f'({tok.line}, {tok.column}) - SyntacticError: unexpected "{tok.lex}"'
    )
```

The handler receives the `ShiftReduceParser` instance. After it returns, the parser automatically skips tokens until it can continue.

### Error productions

An *error production* lets you match known error patterns and keep parsing with a valid (possibly incomplete) AST node. This is the most precise error-recovery mechanism.

**Setup — register the error terminal:**

```python
g.add_terminal_error()
```

**Usage — write productions that include `error`:**

```python
@g.production('stmt -> let id = expr error')
def missing_semicolon(s):
    # s[5] is the Token that triggered the error
    s.add_error(5, f'({s[5].line}, {s[5].column}) - SyntacticError: '
                   f"expected ';' instead of '{s[5].lex}'")
    return LetStatement(s[2], s[4])
```

`s.add_error(index, message)`:

- If `index` is an `int`, it refers to the position in the rule list — `s[5]` is the token at position 5.
- If `index` is a `(line, column)` tuple, it is used directly as the location.

When the parser encounters a token that cannot be shifted, and the current state has a transition on the `error` terminal, it replaces the bad token with an `error` token and continues. The `error` token's semantic value is the original `Token` object, so you still have access to `lex`, `line`, and `column`.

### Forcing a parsing error from a semantic action

Sometimes you want to mark an input as invalid from inside a semantic action — for example, to reject an empty expression:

```python
@g.production('expr -> ')
def empty_expr(s):
    s.force_parsing_error()
    # return nothing or an error sentinel
```

`force_parsing_error()` sets `parser.contains_errors = True` without adding an error message. Add an explicit message via `s.add_error(...)` if needed.

### Checking syntactic errors

```python
result = parser(tokens)

if parser.contains_errors:
    for message in parser.errors:
        print(message)
```

---

## Combining Both Error Handlers

A typical setup collects all errors from both the lexer and the parser and prints them sorted by line:

```python
lexer  = g.get_lexer()
parser = g.get_parser('lalr1')

tokens = lexer(source_code)
result = parser(tokens)

all_errors = lexer.errors + parser.errors

if all_errors:
    for msg in all_errors:
        print(msg)
```

---

## Error Message Conventions

PyJapt does not impose a specific error format. A common convention used in compilers is:

```
(line, column) - ErrorType: description
```

For example:

```
(3, 12) - LexicographicError: unexpected character "@"
(5, 1)  - SyntacticError: expected ';' instead of '}'
```
