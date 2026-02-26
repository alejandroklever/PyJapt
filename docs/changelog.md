# Changelog

All notable changes to PyJapt are documented here.
This project follows [Semantic Versioning](https://semver.org) and the
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

---

## [Unreleased] — v1.0.0

### Planned — Bug Fixes
- Reset `_errors` and `contain_errors` in `Lexer.__call__` so repeated calls don't accumulate stale errors.
- Fix `errors` property signature on `Lexer` and `ShiftReduceParser` (properties cannot accept arguments).
- Move `contains_errors` and `current_token` from class-level to instance-level in `ShiftReduceParser`.
- Fix `Grammar.augmented_grammar` semantic action (`lambda s: s[1]` instead of `lambda x: x`).
- Fix `s.Name` → `s.name` in `Grammar.to_json()` (case mismatch causes `AttributeError`).
- Raise `GrammarError` in `Grammar.__getitem__` instead of returning `None` for missing symbols.
- Replace bare `assert` statements with proper `GrammarError` exceptions.

### Planned — Improvements
- Move `flake8` from runtime to dev dependencies.
- Update build backend to `poetry.core.masonry.api` (replaces deprecated `poetry.masonry.api`).
- Export `RuleList`, `SLRParser`, `LR1Parser`, `LALR1Parser` from `pyjapt.__init__`.
- Rename `pyjapt/typing.py` to `pyjapt/types.py` to avoid shadowing stdlib `typing`.
- Add full type annotations to the public API.
- Expand CI matrix to Python 3.10, 3.11, and 3.12.

### Planned — Testing
- Add tests for LR(1) and LALR(1) parsers.
- Add tests for lexer and parser error handling.
- Add tests for serialisation round-trips.
- Add edge-case tests (empty grammar, duplicate symbols, epsilon productions).
- Enforce minimum test coverage threshold.

### Planned — Documentation
- Full MkDocs site with Material theme.
- Getting-started guide and user-guide sections.
- Complete API reference.
- Changelog (this file).

---

## [0.4.1] — 2024-03-25

### Fixed
- Updated README with corrected examples and improved prose.

---

## [0.4.0] — 2023-02-17

### Added
- GitHub Actions workflow for publishing to PyPI on release.
- `requirements.txt` for legacy `pip install` support.

---

## [0.3.0] — 2021-03-??

### Added
- Default error report in the shift-reduce parser (panic-mode recovery).
- Improved `RuleList` error API.

### Fixed
- Reset lexer parameters when analysing a new string (`Lexer.__call__`).

---

## [0.2.9] — 2021-??-??

### Fixed
- Minor fix in parsing default error detection.

---

## [0.2.x] — 2020

### Added
- SLR, LR(1), and LALR(1) parsers.
- Serialisation of lexer and parser to Python source files.
- `@g.terminal` decorator for inline rule definition.
- `@g.production` decorator for inline production rules.
- `@g.lexical_error` and `@g.parsing_error` decorators.
- `add_terminal_error()` and error terminal support in productions.
- `Grammar.to_json()` / `Grammar.from_json()`.
- JSON grammar import/export.

---

## [0.1.x] — 2020

- Initial release with basic lexer and SLR parser.
