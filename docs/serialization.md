# Serialisation

For large grammars, building the parsing tables from scratch on every run can take seconds. PyJapt lets you *serialise* the pre-computed tables into plain Python modules so that subsequent runs skip the construction step entirely.

---

## How It Works

Calling `serialize_lexer` or `serialize_parser` writes a Python source file (`lexertab.py` / `parsertab.py`) that contains the pre-computed tables as dictionaries. On subsequent runs you import the generated module instead of rebuilding.

The generated classes extend `Lexer` and `ShiftReduceParser` respectively, so they have the full API of their base classes.

---

## Serialising the Lexer

```python
import inspect
from pyjapt import Grammar

g = Grammar()
# ... define grammar ...

if __name__ == '__main__':
    module_name = inspect.getmodulename(__file__)
    g.serialize_lexer(
        class_name='MyLexer',
        grammar_module_name=module_name,
        grammar_variable_name='g',
    )
```

This writes `lexertab.py` in the current working directory. The generated class looks like:

```python
# lexertab.py  (generated — do not edit by hand)
import re
from pyjapt import Token, Lexer
from my_grammar import g

class MyLexer(Lexer):
    def __init__(self):
        self.pattern     = re.compile(r'...')
        self.token_rules = {key: rule for ...}
        self.error_handler = g.lexical_error_handler or self.error
        ...
```

---

## Serialising the Parser

```python
if __name__ == '__main__':
    module_name = inspect.getmodulename(__file__)
    g.serialize_parser(
        parser_type='lalr1',          # 'slr', 'lr1', or 'lalr1'
        class_name='MyParser',
        grammar_module_name=module_name,
        grammar_variable_name='g',
    )
```

This writes `parsertab.py`:

```python
# parsertab.py  (generated — do not edit by hand)
from abc import ABC
from pyjapt import ShiftReduceParser
from my_grammar import g

class MyParser(ShiftReduceParser, ABC):
    def __init__(self, verbose=False):
        self.grammar      = g
        self.action       = self.__action_table()
        self.goto         = self.__goto_table()
        self.error_handler = g.parsing_error_handler or self.error
        ...
```

---

## Using the Generated Classes

```python
from lexertab  import MyLexer
from parsertab import MyParser

lexer  = MyLexer()
parser = MyParser()

result = parser(lexer(source_code))
```

---

## Full Example

**`grammar.py`** — define the grammar and conditionally serialise:

```python
import inspect
from pyjapt import Grammar

g = Grammar()
expr = g.add_non_terminal('expr', start_symbol=True)
term, fact = g.add_non_terminals('term fact')
g.add_terminals('+ - * / ( )')
g.add_terminal('int', regex=r'\d+')

@g.terminal('whitespace', r' +')
def ws(lexer):
    lexer.column   += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)

expr %= 'expr + term', lambda s: s[1] + s[3]
expr %= 'expr - term', lambda s: s[1] - s[3]
expr %= 'term',        lambda s: s[1]
term %= 'term * fact', lambda s: s[1] * s[3]
term %= 'term / fact', lambda s: s[1] // s[3]
term %= 'fact',        lambda s: s[1]
fact %= '( expr )',    lambda s: s[2]
fact %= 'int',         lambda s: int(s[1])

if __name__ == '__main__':
    module = inspect.getmodulename(__file__)
    g.serialize_lexer(class_name='ArithLexer',   grammar_module_name=module, grammar_variable_name='g')
    g.serialize_parser(parser_type='lalr1',
                       class_name='ArithParser',  grammar_module_name=module, grammar_variable_name='g')
```

Run once to generate the tables:

```sh
python grammar.py
```

**`main.py`** — import and use:

```python
from lexertab  import ArithLexer
from parsertab import ArithParser

lexer  = ArithLexer()
parser = ArithParser()

while True:
    line = input('> ')
    print(parser(lexer(line)))
```

---

## Regenerating the Tables

The generated files must be regenerated whenever the grammar changes. A simple convention is to commit the grammar file (`grammar.py`) but add `lexertab.py` and `parsertab.py` to `.gitignore` and generate them as a build step.

```gitignore
# .gitignore
lexertab.py
parsertab.py
```

---

## Caveats

- **Semantic actions are not serialised.** The generated parser still imports the original grammar module (`grammar_module_name`) at runtime to access production rules and semantic actions.
- **The grammar module must be importable.** Make sure `grammar.py` (or whatever you named it) is on the Python path when running the generated classes.
- **Files are written to the current working directory.** Run the serialisation script from the directory where you want the files to be created.
