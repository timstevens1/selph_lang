"""S-expression parser for SELPH.

Tokenizes input text, then parses into AST nodes.
"""

from __future__ import annotations
from dataclasses import dataclass
from .ast import (
    Node, Symbol, Number, String, TensorLiteral, Bool, List,
    Spec, GoalExamples, GoalPattern, GoalSatisfy, GoalTransform,
    GoalIntent, GoalAll, TypeArrow, TypeList, Shape,
)


# --- Tokenizer ---

@dataclass
class Token:
    kind: str   # 'lparen', 'rparen', 'lbracket', 'rbracket', 'symbol',
                # 'number', 'string', 'keyword', 'arrow', 'comma', 'tensor_mark'
    value: str
    line: int
    col: int


class ParseError(Exception):
    def __init__(self, msg: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        super().__init__(f"line {line}, col {col}: {msg}")


def tokenize(source: str) -> list[Token]:
    """Tokenize SELPH source into a flat list of tokens."""
    tokens = []
    i = 0
    line = 1
    col = 1

    while i < len(source):
        c = source[i]

        # Whitespace
        if c in ' \t\r':
            i += 1
            col += 1
            continue
        if c == '\n':
            i += 1
            line += 1
            col = 1
            continue

        # Comments: ; to end of line
        if c == ';':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Parens and brackets
        if c == '(':
            tokens.append(Token('lparen', '(', line, col))
            i += 1
            col += 1
            continue
        if c == ')':
            tokens.append(Token('rparen', ')', line, col))
            i += 1
            col += 1
            continue
        if c == '[':
            tokens.append(Token('lbracket', '[', line, col))
            i += 1
            col += 1
            continue
        if c == ']':
            tokens.append(Token('rbracket', ']', line, col))
            i += 1
            col += 1
            continue
        if c == ',':
            tokens.append(Token('comma', ',', line, col))
            i += 1
            col += 1
            continue

        # String literal
        if c == '"':
            start_col = col
            i += 1
            col += 1
            buf = []
            while i < len(source) and source[i] != '"':
                if source[i] == '\\' and i + 1 < len(source):
                    i += 1
                    col += 1
                    esc = source[i]
                    buf.append({'n': '\n', 't': '\t', '\\': '\\', '"': '"'}.get(esc, esc))
                else:
                    if source[i] == '\n':
                        line += 1
                        col = 0
                    buf.append(source[i])
                i += 1
                col += 1
            if i >= len(source):
                raise ParseError("unterminated string", line, start_col)
            i += 1  # closing "
            col += 1
            tokens.append(Token('string', ''.join(buf), line, start_col))
            continue

        # Tensor literal marker: #T
        if c == '#' and i + 1 < len(source) and source[i + 1] == 'T':
            tokens.append(Token('tensor_mark', '#T', line, col))
            i += 2
            col += 2
            continue

        # Arrow: ->
        if c == '-' and i + 1 < len(source) and source[i + 1] == '>':
            tokens.append(Token('arrow', '->', line, col))
            i += 2
            col += 2
            continue

        # Keywords: :keyword
        if c == ':':
            start_col = col
            i += 1
            col += 1
            buf = [':']
            while i < len(source) and source[i] not in ' \t\r\n()[],"':
                buf.append(source[i])
                i += 1
                col += 1
            tokens.append(Token('keyword', ''.join(buf), line, start_col))
            continue

        # Numbers (including negative)
        if c.isdigit() or (c == '-' and i + 1 < len(source) and source[i + 1].isdigit()):
            start_col = col
            buf = []
            if c == '-':
                buf.append('-')
                i += 1
                col += 1
            while i < len(source) and source[i].isdigit():
                buf.append(source[i])
                i += 1
                col += 1
            if i < len(source) and source[i] == '.':
                buf.append('.')
                i += 1
                col += 1
                while i < len(source) and source[i].isdigit():
                    buf.append(source[i])
                    i += 1
                    col += 1
            tokens.append(Token('number', ''.join(buf), line, start_col))
            continue

        # Symbols (identifiers) — start with alpha, _, ?, or operator chars like = < > ! + * / -
        SYMBOL_START = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_?=<>!+*/&|~^%-')
        SYMBOL_CONT = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-?.=<>!+*/&|~^%')
        if c in SYMBOL_START:
            start_col = col
            buf = []
            while i < len(source) and source[i] in SYMBOL_CONT:
                buf.append(source[i])
                i += 1
                col += 1
            word = ''.join(buf)
            tokens.append(Token('symbol', word, line, start_col))
            continue

        raise ParseError(f"unexpected character: {c!r}", line, col)

    return tokens


# --- Parser ---

class Parser:
    """Recursive descent parser from token list to SELPH AST."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str, value: str | None = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise ParseError(f"expected {kind} but got EOF")
        if tok.kind != kind:
            raise ParseError(f"expected {kind} but got {tok.kind} ({tok.value!r})", tok.line, tok.col)
        if value is not None and tok.value != value:
            raise ParseError(f"expected {value!r} but got {tok.value!r}", tok.line, tok.col)
        return self.advance()

    def parse(self) -> Node:
        """Parse a single top-level expression."""
        node = self.parse_expr()
        if self.peek() is not None:
            tok = self.peek()
            raise ParseError(f"unexpected token after expression: {tok.value!r}", tok.line, tok.col)
        return node

    def parse_many(self) -> list[Node]:
        """Parse all top-level expressions (for files with multiple forms)."""
        nodes = []
        while self.peek() is not None:
            nodes.append(self.parse_expr())
        return nodes

    def parse_expr(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected EOF")

        if tok.kind == 'lparen':
            return self.parse_list()
        if tok.kind == 'lbracket':
            return self.parse_shape()
        if tok.kind == 'tensor_mark':
            return self.parse_tensor_literal()
        if tok.kind == 'number':
            return self.parse_number()
        if tok.kind == 'string':
            return self.parse_string()
        if tok.kind == 'symbol':
            return self.parse_symbol()
        if tok.kind == 'keyword':
            # Keywords appearing as standalone expressions (e.g., :type inside spec)
            # are treated as symbols
            return self.parse_keyword_as_symbol()

        raise ParseError(f"unexpected token: {tok.kind} ({tok.value!r})", tok.line, tok.col)

    def parse_number(self) -> Number:
        tok = self.advance()
        v = float(tok.value)
        return Number(v)

    def parse_string(self) -> String:
        tok = self.advance()
        return String(tok.value)

    def parse_symbol(self) -> Symbol | Bool:
        tok = self.advance()
        if tok.value == 'true':
            return Bool(True)
        if tok.value == 'false':
            return Bool(False)
        return Symbol(tok.value)

    def parse_keyword_as_symbol(self) -> Symbol:
        tok = self.advance()
        return Symbol(tok.value)

    def parse_shape(self) -> Shape:
        """Parse [dim1, dim2, ...]."""
        self.expect('lbracket')
        dims = []
        while self.peek() and self.peek().kind != 'rbracket':
            tok = self.peek()
            if tok.kind == 'number':
                self.advance()
                dims.append(int(float(tok.value)))
            elif tok.kind == 'symbol':
                self.advance()
                if tok.value == '?':
                    dims.append('?')
                else:
                    dims.append(tok.value)
            elif tok.kind == 'comma':
                self.advance()
                continue
            else:
                raise ParseError(f"unexpected token in shape: {tok.value!r}", tok.line, tok.col)
        self.expect('rbracket')
        return Shape(tuple(dims))

    def parse_tensor_literal(self) -> TensorLiteral:
        """Parse #T[dims] dtype."""
        self.expect('tensor_mark')
        shape_node = self.parse_shape()
        dtype_tok = self.expect('symbol')
        return TensorLiteral(list(shape_node.dims), dtype_tok.value)

    def parse_list(self) -> Node:
        """Parse a parenthesized list, dispatching to special forms."""
        self.expect('lparen')

        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected EOF after (")

        # Special form: (:spec ...)
        if tok.kind == 'keyword' and tok.value == ':spec':
            return self.parse_spec()

        # Special form: (:examples ...)
        if tok.kind == 'keyword' and tok.value == ':examples':
            return self.parse_goal_examples()

        # Special form: (:pattern ...)
        if tok.kind == 'keyword' and tok.value == ':pattern':
            return self.parse_goal_pattern()

        # Special form: (:satisfy ...)
        if tok.kind == 'keyword' and tok.value == ':satisfy':
            return self.parse_goal_satisfy()

        # Special form: (:transform ...)
        if tok.kind == 'keyword' and tok.value == ':transform':
            return self.parse_goal_transform()

        # Special form: (:intent ...)
        if tok.kind == 'keyword' and tok.value == ':intent':
            return self.parse_goal_intent()

        # Special form: (:minimize ...)
        if tok.kind == 'keyword' and tok.value == ':minimize':
            return self.parse_goal_minimize()

        # Special form: (:maximize ...)
        if tok.kind == 'keyword' and tok.value == ':maximize':
            return self.parse_goal_maximize()

        # Special form: (:all ...)
        if tok.kind == 'keyword' and tok.value == ':all':
            return self.parse_goal_all()

        # Type arrow: (-> T1 T2 ... Tresult)
        if tok.kind == 'arrow':
            return self.parse_type_arrow()

        # Generic list
        elements = []
        while self.peek() and self.peek().kind != 'rparen':
            elements.append(self.parse_expr())
        self.expect('rparen')
        return List(tuple(elements))

    def parse_spec(self) -> Spec:
        """Parse (:spec :type T :shape S :goal G :constraints C :input I)."""
        self.advance()  # consume :spec keyword
        type_expr = None
        shape_expr = None
        goal = None
        constraints = None
        input_expr = None

        while self.peek() and self.peek().kind != 'rparen':
            tok = self.peek()
            if tok.kind != 'keyword':
                raise ParseError(f"expected keyword in spec, got {tok.value!r}", tok.line, tok.col)

            kw = self.advance().value
            if kw == ':type':
                type_expr = self.parse_expr()
            elif kw == ':shape':
                shape_expr = self.parse_expr()
            elif kw == ':goal':
                goal = self.parse_expr()
            elif kw == ':constraints':
                constraints = self.parse_expr()
            elif kw == ':input':
                input_expr = self.parse_expr()
            else:
                raise ParseError(f"unknown spec field: {kw}", tok.line, tok.col)

        self.expect('rparen')
        return Spec(type_expr, shape_expr, goal, constraints, input_expr)

    def parse_goal_examples(self) -> GoalExamples:
        """Parse (:examples ((a -> b) (c -> d) ...))."""
        self.advance()  # consume :examples
        self.expect('lparen')
        pairs = []
        while self.peek() and self.peek().kind != 'rparen':
            self.expect('lparen')
            input_expr = self.parse_expr()
            self.expect('arrow')
            output_expr = self.parse_expr()
            self.expect('rparen')
            pairs.append((input_expr, output_expr))
        self.expect('rparen')  # close the examples list
        self.expect('rparen')  # close the (:examples ...)
        return GoalExamples(tuple(pairs))

    def parse_goal_pattern(self) -> GoalPattern:
        """Parse (:pattern expr)."""
        self.advance()  # consume :pattern
        pattern = self.parse_expr()
        self.expect('rparen')
        return GoalPattern(pattern)

    def parse_goal_satisfy(self) -> GoalSatisfy:
        """Parse (:satisfy predicate)."""
        self.advance()  # consume :satisfy
        predicate = self.parse_expr()
        self.expect('rparen')
        return GoalSatisfy(predicate)

    def parse_goal_transform(self) -> GoalTransform:
        """Parse (:transform expr :by expr)."""
        self.advance()  # consume :transform
        input_expr = self.parse_expr()
        self.expect('keyword', ':by')
        by_expr = self.parse_expr()
        self.expect('rparen')
        return GoalTransform(input_expr, by_expr)

    def parse_goal_intent(self) -> GoalIntent:
        """Parse (:intent "string")."""
        self.advance()  # consume :intent
        s = self.expect('string')
        self.expect('rparen')
        return GoalIntent(s.value)

    def parse_goal_all(self) -> GoalAll:
        """Parse (:all goal1 goal2 ...)."""
        self.advance()  # consume :all
        goals = []
        while self.peek() and self.peek().kind != 'rparen':
            goals.append(self.parse_expr())
        self.expect('rparen')
        return GoalAll(tuple(goals))

    def parse_goal_minimize(self):
        """Parse (:minimize fitness-fn)."""
        from .ast import GoalMinimize
        self.advance()  # consume :minimize
        fitness = self.parse_expr()
        self.expect('rparen')
        return GoalMinimize(fitness)

    def parse_goal_maximize(self):
        """Parse (:maximize fitness-fn)."""
        from .ast import GoalMaximize
        self.advance()  # consume :maximize
        fitness = self.parse_expr()
        self.expect('rparen')
        return GoalMaximize(fitness)

    def parse_type_arrow(self) -> TypeArrow:
        """Parse (-> T1 T2 ... Tresult)."""
        self.advance()  # consume ->
        types = []
        while self.peek() and self.peek().kind != 'rparen':
            types.append(self.parse_expr())
        self.expect('rparen')
        if len(types) < 2:
            raise ParseError("arrow type needs at least one param type and a return type")
        return TypeArrow(tuple(types[:-1]), types[-1])

    def parse_type_list(self) -> TypeList:
        """Parse (list T)."""
        self.advance()  # consume 'list'
        elem = self.parse_expr()
        self.expect('rparen')
        return TypeList(elem)


# --- Public API ---

def parse(source: str) -> Node:
    """Parse a single SELPH expression from source text."""
    tokens = tokenize(source)
    if not tokens:
        raise ParseError("empty input")
    return Parser(tokens).parse()


def parse_file(source: str) -> list[Node]:
    """Parse multiple top-level SELPH expressions from source text."""
    tokens = tokenize(source)
    if not tokens:
        return []
    return Parser(tokens).parse_many()
