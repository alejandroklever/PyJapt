# PyJapt v1.0 — Action Plan

> **Current version:** 0.4.1
> **Target version:** 1.0.0
> **Status:** Planning phase

This document is the authoritative roadmap for bringing PyJapt to a stable v1.0 release. Items are grouped by category and ordered by priority within each category.

---

## 1. Critical Bug Fixes

These bugs affect correctness and must be resolved before v1.

### 1.1 Lexer state not reset on repeated calls

**File:** `pyjapt/lexing.py` — `Lexer.__call__`

`Lexer.__call__` resets `lineno`, `column`, `position`, `text`, and `token`, but it does **not** reset:
- `self._errors` — errors from a previous run accumulate
- `self.contain_errors` — flag is stale after the first run with errors

**Fix:** Reset `_errors = []` and `contain_errors = False` at the start of `__call__`.

---

### 1.2 `errors` property signature is broken

**Files:** `pyjapt/lexing.py:102`, `pyjapt/parsing.py:1072`

Both `Lexer.errors` and `ShiftReduceParser.errors` are decorated with `@property` but carry a `clean: bool = True` parameter:

```python
@property
def errors(self, clean: bool = True):   # ← wrong: properties don't accept arguments
```

Python silently ignores the `clean` parameter and always calls it as a property. The `clean` branch (returning tuples with row/column) is therefore unreachable.

**Fix:** Replace with two separate accessors:
```python
@property
def errors(self) -> List[str]:
    return [m for _, _, m in sorted(self._errors)]

@property
def errors_with_location(self) -> List[Tuple[int, int, str]]:
    return sorted(self._errors)
```

---

### 1.3 `ShiftReduceParser` class-level mutable state

**File:** `pyjapt/parsing.py:1032-1033`

```python
class ShiftReduceParser:
    contains_errors: bool = False      # ← shared across all instances
    current_token: Optional[Token] = None  # ← shared across all instances
```

Class-level attributes are shared across all instances of a class. Two parsers used in the same program would corrupt each other's state.

**Fix:** Move both to `__init__`.

---

### 1.4 LALR(1) lookahead algorithm mixes strings and Symbol objects

**File:** `pyjapt/parsing.py` — `determining_lookaheads` and `build_lalr1_automaton`

The propagation sentinel `"#"` is a plain string that is mixed into lookahead sets normally occupied by `Symbol` objects. This works coincidentally because `"#"` doesn't collide with any symbol name, but it is fragile and will silently break if a grammar ever names a symbol `#`.

**Fix:** Use a dedicated `PropagationTerminal` singleton (already defined in the file as `PropagationTerminal`) instead of the magic string, or use a `None` sentinel that is excluded from lookahead propagation explicitly.

---

### 1.5 `Grammar.augmented_grammar` semantic rule is wrong

**File:** `pyjapt/parsing.py:462`

```python
new_start_symbol %= start_symbol + grammar.EPSILON, lambda x: x
```

The lambda receives a `RuleList`, not the symbol value directly. The production `S' -> start_symbol` (with epsilon swallowed) should return `s[1]`, not the `RuleList` itself:

```python
new_start_symbol %= start_symbol + grammar.EPSILON, lambda s: s[1]
```

---

### 1.6 `Grammar.__getitem__` returns `None` for missing keys

**File:** `pyjapt/parsing.py:592-599`

When a production string references a symbol that doesn't exist, `Grammar.__getitem__` silently returns `None`. This produces a confusing `AttributeError` deep in the call chain rather than a clear `GrammarError`.

**Fix:** Raise `GrammarError` with the unknown symbol name.

---

## 2. Code Quality & Modernisation

### 2.1 Move `flake8` to dev dependencies

**File:** `pyproject.toml`

`flake8` is listed under `[tool.poetry.dependencies]` (runtime). It is a linter and must move to `[tool.poetry.dev-dependencies]`.

---

### 2.2 Update deprecated Poetry build backend

**File:** `pyproject.toml`

```toml
# current (deprecated)
build-backend = "poetry.masonry.api"

# correct
build-backend = "poetry.core.masonry.api"
```

---

### 2.3 Add `mkdocs` and `mkdocs-material` as dev dependencies

**File:** `pyproject.toml`

Documentation builds are part of the development workflow.

---

### 2.4 Rename `pyjapt/typing.py`

The module name `typing` shadows Python's standard-library `typing` module inside the package. Rename it to `pyjapt/types.py` or `pyjapt/_types.py` and update the import in `tests/test_arithmetic_grammar.py`.

---

### 2.5 Export `RuleList` and parsers from the top-level `__init__.py`

**File:** `pyjapt/__init__.py`

`RuleList` and individual parser classes (`SLRParser`, `LR1Parser`, `LALR1Parser`) are not exported from the package root. Users must import from internal submodules. Add them to `__init__.py`:

```python
from pyjapt.parsing import (
    ShiftReduceParser, SLRParser, LR1Parser, LALR1Parser, Grammar, RuleList
)
```

---

### 2.6 Add type annotations to public API

Currently many method signatures lack return type annotations. Add full annotations to:
- `Grammar.get_lexer`, `Grammar.get_parser`, `Grammar.serialize_*`
- `Lexer.__call__`, `Lexer.tokenize`
- `ShiftReduceParser.__call__`
- All `add_*` methods on `Grammar`

---

### 2.7 Replace bare `assert` with proper exceptions

Bare `assert` statements are silently disabled when Python runs with the `-O` (optimise) flag:

- `pyjapt/parsing.py:835` — `assert len(grammar.start_symbol.productions) == 1`
- `pyjapt/parsing.py:906` — `assert not lookaheads.contains_epsilon`
- `pyjapt/lexing.py` — several in `Grammar.add_terminal`

Replace with `if not ...: raise GrammarError(...)`.

---

### 2.8 Serialised parser resets `augmented_grammar` fields

**File:** `pyjapt/serialization.py`

The serialised parser template does not call `_build_automaton` or compute `firsts`/`follows`, which is correct. But it also doesn't set `augmented_grammar`, `firsts`, or `follows`, which means the serialised parser cannot be safely extended. Document this limitation and add a guard.

---

### 2.9 CI: test against Python 3.10, 3.11, and 3.12

**File:** `.github/workflows/python-test-app.yml`

Add a matrix strategy to test against all supported Python versions.

---

## 3. Testing Improvements

### 3.1 Add tests for LR(1) and LALR(1) parsers

`tests/test_arithmetic_grammar.py` only tests the SLR parser. Add `test_lr1` and `test_lalr1` parameterised over the same set of inputs.

### 3.2 Add tests for lexer error handling

- Unknown character → default error handler → `errors` list populated
- Custom `lexical_error` decorator
- `contain_errors` flag is `True` after a failed tokenisation
- Errors reset correctly across multiple calls

### 3.3 Add tests for parser error handling

- Syntactic error → `errors` list populated
- `contains_errors` flag is `True`
- Error recovery (`error` terminal / panic mode)
- Custom `parsing_error` decorator

### 3.4 Add tests for serialisation

- Round-trip: build grammar → serialise lexer and parser → import → parse identical inputs → same result

### 3.5 Add edge-case tests

- Empty grammar (no productions) raises `GrammarError`
- Duplicate terminal/non-terminal name raises immediately
- Production referencing undefined symbol raises `GrammarError`
- Epsilon productions
- Grammars with conflicts produce correct conflict counts

### 3.6 Measure and enforce coverage

Add `pytest-cov` and set a minimum coverage threshold (target ≥ 85 %) in CI.

---

## 4. Documentation

The documentation website is built with **MkDocs + Material theme** and lives under `docs/`. See `mkdocs.yml` for the full configuration.

| Page | Status |
|------|--------|
| `docs/index.md` | Done |
| `docs/getting-started.md` | Done |
| `docs/defining-grammar.md` | Done |
| `docs/lexer.md` | Done |
| `docs/parser.md` | Done |
| `docs/error-handling.md` | Done |
| `docs/serialization.md` | Done |
| `docs/api-reference.md` | Done |
| `docs/changelog.md` | Done |

### 4.1 Add `CHANGELOG.md`

Track every version with date and changes, following [Keep a Changelog](https://keepachangelog.com) format.

### 4.2 Add `CONTRIBUTING.md`

Describe:
- How to clone and set up the dev environment
- How to run tests and linting
- Branching and PR conventions
- Code of conduct pointer

---

## 5. Packaging & Release

### 5.1 Update version to `1.0.0`

**Files:** `pyjapt/__init__.py` and `pyproject.toml`

### 5.2 Populate package metadata

**File:** `pyproject.toml`

Add:
```toml
license = "MIT"
keywords = ["lexer", "parser", "LR", "LALR", "compiler", "grammar"]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Topic :: Software Development :: Compilers",
]
repository = "https://github.com/alejandroklever/PyJapt"
documentation = "https://alejandroklever.github.io/PyJapt"
```

### 5.3 Add a GitHub Actions workflow to build and deploy docs

Publish the MkDocs site to GitHub Pages on every push to `main`.

### 5.4 Tag and publish to PyPI

After all items above are resolved:
1. Bump version to `1.0.0` in `__init__.py` (the `build.py` script syncs `pyproject.toml` automatically).
2. Push a `v1.0.0` git tag.
3. Create a GitHub Release — the existing publish workflow triggers on `release: published`.

---

## 6. Future Work (Post-v1)

The following are explicitly out of scope for v1 but should be tracked:

| Feature | Notes |
|---------|-------|
| Operator precedence declarations | Resolves SR conflicts declaratively (like `%left`, `%right` in Yacc) |
| LL(1) parser support | Mentioned in README as future work |
| Grammar visualisation | Export automata as DOT / SVG |
| Incremental parsing | Re-lex only changed regions |
| Better conflict reporting | Show the conflicting items and lookaheads in a human-readable table |
| Unicode identifiers in grammars | Non-ASCII symbol names |
| Async tokenisation | Yield tokens lazily for very large inputs |

---

## Priority Order Summary

| Priority | Item |
|----------|------|
| P0 — Must fix before v1 | 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 |
| P1 — Fix before v1 | 2.1, 2.2, 2.4, 2.5, 3.1, 3.2, 3.3 |
| P2 — Nice-to-have before v1 | 2.3, 2.6, 2.7, 2.8, 2.9, 3.4, 3.5, 3.6, 5.1 – 5.4 |
| P3 — Post-v1 | Section 6 |
