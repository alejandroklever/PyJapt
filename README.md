# Lexer and LR parser generator  "PyJapt"

PyJapt is a lexer and parser generator developed to provide a solution not only to the creation of these pieces of the compilation process, but also to allow a custom syntactic and lexicographic error handling interface. For its construction we have been inspired by other parser generators such as yacc, bison, ply and antlr, for example.

PyJapt revolves around the concept of grammar.

To define the nonterminals of the grammar we use the `add_non_terminal ()` method of the `Grammar` class.

```python
from pyjapt.grammar import Grammar

G = Grammar()

expr = G.add_non_terminal('expr', start_symbol=True)
term = G.add_non_terminal('term')
fact = G.add_non_terminal('fact')
```

To define the terminals of our grammar we will use the `add_terminal ()` method of the `Grammar` class. This method receives the name of the non-terminal as the first parameter and a regular expression for the lexicographic analyzer as an optional parameter. In case the second parameter is not provided, the regular expression will be the literal name of the terminal.

```python
plus = G.add_terminals('+')
minus = G.add_terminals('-')
star = G.add_terminals('*')
div = G.add_terminals('/')

num = G.add_terminal('int', regex=r'\d+')
```

If we have a set of terminals whose regular expression matches their own name we can encapsulate them with the `add_terminals ()` function of the `Grammar` class.

```python
plus, minus, star, div = G.add_terminals('+ - * /')
num = G.add_terminal('int', regex=r'\d+')
```

It may also be the case that we want to apply a rule when a specific terminal is found, for this PyJapt gives us the `terminal ()` function decorator of the `Grammar` class that receives the terminal name and regular expression. The decorated function must receive as a parameter a reference to the lexer to be able to modify parameters such as the row and column of the terminals or the parser reading position and return a `Token`, if this token is not returned it will be ignored.

```python
@G.terminal('int', r'\d+')
def id_terminal(lexer):
    lexer.column += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    lexer.token.lex = int(lexer.token.lex)
    return lexer.token
```

We can also use this form of terminal definition to skip certain characters or tokens.

```python
##################
# Ignored Tokens #
##################
@G.terminal('newline', r'\n+')
def newline(lexer):
    lexer.lineno += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)
    lexer.column = 1


@G.terminal('whitespace', r' +')
def whitespace(lexer):
    lexer.column += len(lexer.token.lex)
    lexer.position += len(lexer.token.lex)


@G.terminal('tabulation', r'\t+')
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

# A little easier ...
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
# list with the semantic rules of each symbol of the production.
# To separate the symbol from the head of the body of the expression
# use the second symbol `->`
@G.production('expr -> expr + term')
def expr_prod(s):
    print('Adding an expression and a term ;)')
    return s[1] + s[3]
```

## Handling of lexicographic and syntactic errors

An important part of the parsing process is handling errors. For this we can do the parser by hand and insert the error report, since techniques such as `Panic Recovery Mode` which implements `PyJapt` only allow the execution of our parser not to stop, to give specific error reports `PyJapt` offers the creation of erroneous productions to report common errors in a programming language such as the lack of a `;`, an unknown operator, etc. For this our grammar must activate the terminal error flag.

```python
G.add_terminal_error() # Add the error terminal to the grammar.

# Example of a possible error production
@G.production("instruction -> let id = expr error")
def attribute_error(s):
    # With this line we report the error message
    # As the semantic rule of s [5] is the token itself (because it is a terminal), so we have access
    # to your lexeme, token type, line and column.
    s.error(5, f"{s[5].line, s[5].column} - SyntacticError: Expected ';' instead of '{s[5].lex}'")

    # With this line we allow to continue creating a node of the ast to
    # be able to detect semantic errors despite syntactic errors
    return LetInstruction(s[2], s[4])
```

To report lexicographical errors the procedure is quite similar we only define a token that contains an error, in this example a multi-line comment that contains an end of string.

```python
@G.terminal('comment_error', r'\(\*(.|\n)*$')
def comment_eof_error(lexer):
    lexer.contain_errors = True
    lex = lexer.token.lex
    for s in lex:
        if s == '\n':
            lexer.lineno += 1
            lexer.column = 0
        lexer.column = 1
    lexer.position += len(lex)
    lexer.print_error(f'{lexer.lineno, lexer.column} -LexicographicError: EOF in comment')
```

And to report general errors during the tokenization process we can use the `lexical_error` decorator.

```python
@G.lexical_error
def lexical_error(lexer):
    lexer.print_error(f'{lexer.lineno, lexer.column} -LexicographicError: ERROR "{lexer.token.lex}"')
    lexer.column += 1
    lexer.position += 1
```
