# Lexer and LR parser generator  "PyJapt"

## Installation

```
pip install pyjapt
```

## PyJapt

PyJapt is a lexer and parser generator developed to provide a solution not only to the creation of these pieces of the compilation process, but also to allow a custom syntactic and lexicographic error handling interface. For its construction we have been inspired by other parser generators such as yacc, bison, ply and antlr, for example.

PyJapt revolves around the concept of grammar.

To define the nonterminals of the grammar we use the `add_non_terminal ()` method of the `Grammar` class.

```python
from pyjapt import Grammar

g = Grammar()

expr = g.add_non_terminal('expr', start_symbol=True)
term = g.add_non_terminal('term')
fact = g.add_non_terminal('fact')
```

To define the terminals of our grammar we will use the `add_terminal ()` method of the `Grammar` class. This method receives the name of the non-terminal as the first parameter and a regular expression for the lexicographic analyzer as an optional parameter. In case the second parameter is not provided, the regular expression will be the literal name of the terminal.

```python
plus = g.add_terminals('+')
minus = g.add_terminals('-')
star = g.add_terminals('*')
div = g.add_terminals('/')

num = g.add_terminal('int', regex=r'\d+')
```

If we have a set of terminals whose regular expression matches their own name we can encapsulate them with the `add_terminals()` function of the `Grammar` class.

```python
plus, minus, star, div = g.add_terminals('+ - * /')
num = g.add_terminal('int', regex=r'\d+')
```

It may also be the case that we want to apply a rule when a specific terminal is found, for this PyJapt gives us the `terminal()` function decorator of the `Grammar` class that receives the terminal name and regular expression. The decorated function must receive as a parameter a reference to the lexer to be able to modify parameters such as the row and column of the terminals or the parser reading position and return a `Token`, if this token is not returned it will be ignored.

```python
@g.terminal('int', r'\d+')
def id_terminal(lexer):
    lexer.column += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    lexer.token.lex = int(lexer.token.lex)
    return lexer.token
```

We can also use this form of terminal definition to skip certain characters or tokens, we just need to ignore the return in the method.

```python
##################
# Ignored Tokens #
##################
@g.terminal('newline', r'\n+')
def newline(lexer):
    lexer.lineno += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    lexer.column = 1


@g.terminal('whitespace', r' +')
def whitespace(lexer):
    lexer.column += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)


@g.terminal('tabulation', r'\t+')
def tab(lexer):
    lexer.column += 4 * len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
```

To define the productions of our grammar we can use an attributed or not attributed form:

```python
# This is an unattributed grammar using previously declared variables
expr %= expr + plus + term
expr %= expr + minus + term
expr %= term

term %= term + star + fact
term %= term + div + fact
term %= fact

fact %= num

# A little easier to read...
# Each symbol in the production string must be separated by a blank space
expr %= 'expr + term'
expr %= 'expr - term'
expr %= 'term'

term %= 'term * factor'
term %= 'term / factor'
term %= 'fact'

fact %= 'int'

# This is an attributed grammar
expr %= 'expr + term', lambda s: s[1] + s[3]
expr %= 'expr - term', lambda s: s[1] + s[3]
expr %= 'term', lambda s: s[1]

term %= 'term * factor', lambda s: s[1] + s[3]
term %= 'term / factor', lambda s: s[1] + s[3]
term %= 'fact', lambda s: s[1]

fact %= 'int', lambda s: int(s[1])

# We can also attribute a function to define a semantic rule
# This function should receive as parameter `s` which is a reference to a
# special list with the semantic rules of each symbol of the production.
# To separate the symbol from the head of the body of the expression
# use the second symbol `->`
@g.production('expr -> expr + term')
def expr_prod(s):
    print('Adding an expression and a term ;)')
    return s[1] + s[3]

# We can also assign a rule to many productions with this decorator
# Please ignore the fact this grammar is ambiguous and pyjapt doesn't have support for it ... yet ;)
@g.production('expr -> expr + expr', 
              'expr -> expr - expr', 
              'expr -> expr * expr', 
              'expr -> expr / expr',
              'expr -> ( expr )' 
              'expr -> int')
def expr_prod(s):
    if len(s) != 2:
        if s[2] == '+':
            return s[1] + s[2]
    
        if s[2] == '-':
            return s[1] + s[2]
        
        if s[2] == '*':
            return s[1] + s[2]
        
        if s[2] == '/':
            return s[1] + s[2]
    
        return s[2]
    return int(s[1])
```

To generate the lexer and the parser of the grammar we will use the get_lexer and get_parser methods respectively. In the case of get_parser it receives a string as an argument to find out what type of parser to use. Valid strings are `'slr'`,`' lr1'`, and `'lalr1'`.

```python
g.get_lexer()
g.get_parser('slr')
```

Finally an example of the entire pipeline in the way that we consider the most readable and comfortable to describe a grammar for the language of arithmetic expressions

```python
from pyjapt import Grammar

g = Grammar()
expr = g.add_non_terminal('expr', True)
term, fact = g.add_non_terminals('term fact')
g.add_terminals('+ - / * ( )')
g.add_terminal('int', regex=r'\d+')

@g.terminal('whitespace', r' +')
def whitespace(_lexer):
    _lexer.column += len(_lexer.token.lex)
    _lexer.position += len(_lexer.token.lex)

expr %= 'expr + term', lambda s: s[1] + s[3]
expr %= 'expr - term', lambda s: s[1] - s[3]
expr %= 'term', lambda s: s[1]

term %= 'term * fact', lambda s: s[1] * s[3]
term %= 'term / fact', lambda s: s[1] // s[3]
term %= 'fact', lambda s: s[1]

fact %= '( expr )', lambda s: s[2]
fact %= 'int', lambda s: int(s[1])

lexer = g.get_lexer()

parser = g.get_parser('slr')

print(parser(lexer('(2 + 2) * 2 + 2'))) # prints 10
```
## Serialization

When the grammar is large enough the parser construction process can be a bottleneck. To solve this problem PyJapt offers a solution by serializing the parser and lexer generated from the grammar into two files `parsetab.py` and` lexertab.py`. These files must refer to the instance of the original grammar.

To serialize, the name of the variable that contains the instance of Grammar is required, the name of the module where the grammar was written, and the name of the class that the Lexer and Parser will have serialized.

```python
import inspect
from pyjapt import Grammar

g = Grammar()

# ...

if __name__ == '__main__':
    module_name = inspect.getmodulename(__file__)
    g.serialize_lexer(class_name='MyLexer', grammar_module_name=module_name, grammar_variable_name='g')
    g.serialize_parser(parser_type='slr', class_name='MyParser', grammar_module_name=module_name, grammar_variable_name='g') 
```

## Lexer Construction

Although it seems simple, there are some things to keep in mind when defining the terminals of our grammar. When detecting a token, the lexer will first recognize those that have been marked with the `terminal` decorator of the` Grammar` class ( or have a rule assigned ) according to their order of appearance, then those whose regular expression does not match their identifier will follow, and finally the literal terminals (those that do match their identifier with their regular expression). In the last two cases, the order of both will be given by the size of their regular expression ( largest first ).

If your language has keywords and identifiers a great idea to avoid the rule of the largest regular expression is set a rule to our identifier terminal

```python
from pyjapt import Grammar


g = Grammar()

keywords = g.add_terminals(
    'class inherits if then else fi while loop pool let in case of esac new isvoid true false not')
keywords_names = {x.name for x in keywords}

@g.terminal('id', r'[a-zA-Z_][a-zA-Z0-9_]*')
def id_terminal(lexer):
    lexer.column += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    if lexer.token.lex in keywords_names:
        # modify the token type ;) 
        lexer.token.token_type = lexer.token.lex
    return lexer.token
```

## Handling of lexicographic and syntactic errors

An important part of the parsing process is handling errors. For this we can do the parser by hand and insert the error report, since techniques such as `Panic Recovery Mode` which implements `PyJapt` only allow the execution of our parser not to stop, to give specific error reports `PyJapt` offers the creation of erroneous productions to report common errors in a programming language such as the lack of a `;`, an unknown operator, etc. For this our grammar must activate the terminal error flag.

```python
g.add_terminal_error() # Add the error terminal to the grammar.

# Example of a possible error production
@g.production("instruction -> let id = expr error")
def attribute_error(s):
    # With this line we report the error message
    # As the semantic rule of s[5] is the token itself (because it is a terminal error), so we have access
    # to your their, token type, line and column.
    s.add_error(5, f"{s[5].line, s[5].column} - SyntacticError: Expected ';' instead of '{s[5].lex}'")

    # With this line we allow to continue creating a node of the ast to
    # be able to detect semantic errors despite syntactic errors
    return LetInstruction(s[2], s[4])
```

To report lexicographical errors the procedure is quite similar we only define a token that contains an error, in this example a multi-line comment that contains an end of string.

```python
@g.terminal('comment_error', r'\(\*(.|\n)*$')
def comment_eof_error(lexer):
    lexer.contain_errors = True
    lex = lexer.token.lex
    for s in lex:
        if s == '\n':
            lexer.lineno += 1
            lexer.column = 0
        lexer.column = 1
    lexer.position += len(lex)
    lexer.add_error(f'{lexer.lineno, lexer.column} -LexicographicError: EOF in comment')
```

And to report general errors during the tokenization process we can use the `lexical_error` decorator.

```python
@g.lexical_error
def lexical_error(lexer):
    lexer.add_error(lexer.line, lexer.column, f'{lexer.lineno, lexer.column} -LexicographicError: ERROR "{lexer.token.lex}"')
    lexer.column += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
```

## Credits 

For each recommendation or bug please write to alejandroklever.workon@gmail.com.
