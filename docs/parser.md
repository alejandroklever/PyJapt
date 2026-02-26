# Building a Parser

PyJapt provides three LR parser variants. This page explains how they differ, when to use each, and how to work with the parser object.

---

## Choosing a Parser Type

```python
parser = g.get_parser('slr')    # Simple LR
parser = g.get_parser('lr1')    # Canonical LR(1)
parser = g.get_parser('lalr1')  # LALR(1)
```

| Parser | Power | States | Speed | Best For |
|--------|-------|--------|-------|----------|
| `slr`  | Weakest | Fewest | Fastest to build | Simple grammars, prototyping |
| `lalr1` | Middle | Fewest (same as SLR) | Fast to build | Most real-world grammars (e.g. C, Python) |
| `lr1`  | Strongest | Most | Slowest to build | Grammars that LALR(1) cannot handle |

**Rule of thumb:** start with `slr`. If you see shift-reduce or reduce-reduce conflicts that your grammar should not have, try `lalr1`. Use `lr1` only when necessary.

---

## How LR Parsing Works

LR parsers are bottom-up. They maintain a *stack* and follow one of three actions at each step:

- **Shift** — push the current input token onto the stack.
- **Reduce** — pop symbols matching a production's body, run the semantic action, push the head non-terminal.
- **Accept** — the start symbol covers the entire input; return the top semantic value.

The parsing tables (ACTION and GOTO) encode which action to take for every (state, token) pair.

---

## Conflicts

When two actions are valid for the same (state, lookahead) pair, a conflict arises:

- **Shift-reduce (SR)** — the parser can either shift or reduce. PyJapt resolves SR conflicts in favour of **shift** (same as most tools, because it handles `if-else` correctly).
- **Reduce-reduce (RR)** — two different reductions are possible. PyJapt keeps whichever was registered first.

Conflicts are printed to `stderr` and stored in `parser.conflicts`:

```python
parser = g.get_parser('slr')
# Warning: 1 Shift-Reduce Conflicts
# Warning: 0 Reduce-Reduce Conflicts

print(parser.shift_reduce_count)   # 1
print(parser.reduce_reduce_count)  # 0
print(parser.conflicts)            # [('SR', prod_a, prod_b)]
```

---

## Semantic Actions

A semantic action is a callable `(RuleList) -> Any` attached to a production.

```python
fact %= 'int', lambda s: int(s[1])
```

For longer actions, use `@g.production`:

```python
@g.production('stmt -> let id = expr ;')
def let_stmt(s):
    name  = s[2]    # id lexeme
    value = s[4]    # expr semantic value
    return LetStatement(name, value)
```

The `RuleList` `s` is 1-indexed over the body symbols:

```
stmt  ->  let  id  =  expr  ;
s[0]      s[1] s[2] s[3] s[4] s[5]
(head)
```

`s[0]` is set to whatever your action returns.

---

## Calling the Parser

The parser is callable:

```python
result = parser(tokens)  # tokens: List[Token]
```

`tokens` is the list returned by `lexer(text)`. If you use a different tokeniser, ensure each token has `.token_type` set to the terminal name string.

The return value is the semantic value of the start symbol, or `None` if parsing failed without a recovery path.

---

## Checking for Parsing Errors

```python
result = parser(tokens)

if parser.contains_errors:
    for msg in parser.errors:
        print(msg)
```

---

## The `verbose` Flag

Pass `verbose=True` to `get_parser` to print every shift and reduce operation during parsing. Useful for debugging grammars.

```python
parser = g.get_parser('slr', verbose=True)
parser(lexer('1 + 2'))
# expr <-> 1 + 2 $
#
# Shift: ('1', 3)
# ...
```

---

## Parser Internals

You can inspect the generated tables directly:

```python
# ACTION table: {(state_id, Terminal): ('SHIFT', next_state) | ('REDUCE', Production) | ('OK', None)}
print(parser.action)

# GOTO table: {(state_id, NonTerminal): next_state}
print(parser.goto)

# The augmented grammar used internally
print(parser.augmented_grammar)
```

These are Python dicts and can be serialised — see [Serialisation](serialization.md).
