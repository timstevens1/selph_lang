"""Tests for the SELPH parser, using examples from the architecture spec."""

import pytest
from selph.parser import parse, parse_file, ParseError
from selph.ast import (
    Symbol, Number, String, Bool, TensorLiteral, List,
    Spec, GoalExamples, GoalPattern, GoalSatisfy, GoalTransform,
    GoalIntent, GoalAll, TypeArrow, TypeList, Shape,
)


# --- Atoms ---

class TestAtoms:
    def test_symbol(self):
        assert parse("hello") == Symbol("hello")

    def test_symbol_with_hyphens(self):
        assert parse("outer-product") == Symbol("outer-product")

    def test_symbol_with_dots(self):
        assert parse("ops.tensor.matmul") == Symbol("ops.tensor.matmul")

    def test_integer(self):
        assert parse("42") == Number(42.0)

    def test_float(self):
        assert parse("3.14") == Number(3.14)

    def test_negative_number(self):
        assert parse("-5") == Number(-5.0)

    def test_string(self):
        assert parse('"hello world"') == String("hello world")

    def test_string_escapes(self):
        assert parse(r'"line\none"') == String("line\none")

    def test_bool_true(self):
        assert parse("true") == Bool(True)

    def test_bool_false(self):
        assert parse("false") == Bool(False)

    def test_tensor_literal(self):
        result = parse("#T[3, 4] float32")
        assert isinstance(result, TensorLiteral)
        assert result.shape == [3, 4]
        assert result.dtype == "float32"

    def test_tensor_literal_symbolic_dims(self):
        result = parse("#T[batch, seq, dim] float16")
        assert isinstance(result, TensorLiteral)
        assert result.shape == ["batch", "seq", "dim"]


# --- Lists ---

class TestLists:
    def test_empty_list(self):
        assert parse("()") == List(())

    def test_simple_list(self):
        result = parse("(add 1 2)")
        assert result == List((Symbol("add"), Number(1.0), Number(2.0)))

    def test_nested_list(self):
        result = parse("(add (mul 2 3) 4)")
        assert result == List((
            Symbol("add"),
            List((Symbol("mul"), Number(2.0), Number(3.0))),
            Number(4.0),
        ))

    def test_spec_example_neural_layer(self):
        """From §2.4: (let ((h (relu (add (matmul W x) b)))) (dropout h 0.1))"""
        source = '(let ((h (relu (add (matmul W x) b)))) (dropout h 0.1))'
        result = parse(source)
        assert isinstance(result, List)
        assert result.elements[0] == Symbol("let")

    def test_spec_example_attention(self):
        """From §2.4: (defmacro attention (Q K V) ...)"""
        source = '(defmacro attention (Q K V) (matmul (softmax (matmul Q (transpose K))) V))'
        result = parse(source)
        assert isinstance(result, List)
        assert result.elements[0] == Symbol("defmacro")
        assert result.elements[1] == Symbol("attention")


# --- Specs ---

class TestSpecs:
    def test_simple_spec(self):
        source = '(:spec :type string)'
        result = parse(source)
        assert isinstance(result, Spec)
        assert result.type_expr == Symbol("string")

    def test_spec_with_shape(self):
        source = '(:spec :type tensor :shape [batch, 512])'
        result = parse(source)
        assert isinstance(result, Spec)
        assert result.type_expr == Symbol("tensor")
        assert isinstance(result.shape_expr, Shape)
        assert result.shape_expr.dims == ("batch", 512)

    def test_spec_with_goal_and_input(self):
        source = '(:spec :type string :goal (:intent "find the capital of") :input country)'
        result = parse(source)
        assert isinstance(result, Spec)
        assert result.type_expr == Symbol("string")
        assert isinstance(result.goal, GoalIntent)
        assert result.goal.intent == "find the capital of"
        assert result.input_expr == Symbol("country")


# --- Goals ---

class TestGoals:
    def test_goal_examples(self):
        source = '(:examples (("a" -> "b") ("y" -> "z")))'
        result = parse(source)
        assert isinstance(result, GoalExamples)
        assert len(result.pairs) == 2
        assert result.pairs[0] == (String("a"), String("b"))
        assert result.pairs[1] == (String("y"), String("z"))

    def test_goal_pattern(self):
        source = '(:pattern (word :length 5 :starts-with "h"))'
        result = parse(source)
        assert isinstance(result, GoalPattern)

    def test_goal_satisfy(self):
        source = '(:satisfy (lambda (out) (even out)))'
        result = parse(source)
        assert isinstance(result, GoalSatisfy)
        assert isinstance(result.predicate, List)

    def test_goal_transform(self):
        source = '(:transform input :by reverse-word-order)'
        result = parse(source)
        assert isinstance(result, GoalTransform)
        assert result.input_expr == Symbol("input")
        assert result.by_expr == Symbol("reverse-word-order")

    def test_goal_intent(self):
        source = '(:intent "summarize the document focusing on financial risks")'
        result = parse(source)
        assert isinstance(result, GoalIntent)
        assert result.intent == "summarize the document focusing on financial risks"

    def test_goal_all(self):
        source = '(:all (:intent "be concise") (:intent "be accurate"))'
        result = parse(source)
        assert isinstance(result, GoalAll)
        assert len(result.goals) == 2


# --- Types ---

class TestTypes:
    def test_arrow_type(self):
        source = '(-> matrix matrix matrix)'
        result = parse(source)
        assert isinstance(result, TypeArrow)
        assert result.param_types == (Symbol("matrix"), Symbol("matrix"))
        assert result.return_type == Symbol("matrix")

    def test_list_type(self):
        # (list string) now parses as a generic List, not TypeList.
        # TypeList will be handled in explicit type annotation contexts later.
        source = '(list string)'
        result = parse(source)
        assert isinstance(result, List)
        assert result.elements[0] == Symbol("list")
        assert result.elements[1] == Symbol("string")

    def test_arrow_with_list_param(self):
        source = '(-> (list string) string)'
        result = parse(source)
        assert isinstance(result, TypeArrow)
        assert isinstance(result.param_types[0], List)


# --- Shapes ---

class TestShapes:
    def test_concrete_shape(self):
        result = parse("[3, 4, 5]")
        assert isinstance(result, Shape)
        assert result.dims == (3, 4, 5)

    def test_symbolic_shape(self):
        result = parse("[batch, seq]")
        assert isinstance(result, Shape)
        assert result.dims == ("batch", "seq")

    def test_wildcard_dim(self):
        # '?' is a valid symbol character, parse as symbol dim
        result = parse("[batch, ?]")
        assert isinstance(result, Shape)
        assert result.dims[1] == "?"


# --- Comments ---

class TestComments:
    def test_line_comment(self):
        source = '; this is a comment\n(add 1 2)'
        result = parse(source)
        assert result == List((Symbol("add"), Number(1.0), Number(2.0)))

    def test_inline_comment(self):
        # comment after the expression won't be reached since parse() stops after first expr
        source = '(add 1 2) ; result'
        # parse_file handles multiple exprs; parse will error on trailing tokens
        results = parse_file(source)
        assert len(results) == 1


# --- Roundtrip ---

class TestRoundtrip:
    def test_repr_atom(self):
        assert repr(parse("hello")) == "hello"
        assert repr(parse("42")) == "42"
        assert repr(parse("3.14")) == "3.14"
        assert repr(parse('"hi"')) == '"hi"'
        assert repr(parse("true")) == "true"

    def test_repr_list(self):
        assert repr(parse("(add 1 2)")) == "(add 1 2)"

    def test_repr_nested(self):
        source = "(let ((x 1)) (add x 2))"
        assert repr(parse(source)) == source


# --- Full spec examples ---

class TestSpecExamples:
    def test_agentic_decomposition(self):
        """From §2.4: agentic decomposition with subagent."""
        source = """
        (let ((country (subagent
                (:spec :type string
                       :goal (:intent "identify the largest country by area"))))
              (capital (subagent
                (:spec :type string
                       :goal (:intent "find the capital of")
                       :input country))))
          capital)
        """
        result = parse(source)
        assert isinstance(result, List)
        assert result.elements[0] == Symbol("let")

    def test_stage2_satisfy_example(self):
        """From §2.4: Stage 2 with executable predicates."""
        source = """
        (let ((country (subagent
                (:spec :type string
                       :goal (:satisfy (lambda (out)
                         (= out (max-by area countries)))))))
              (capital (subagent
                (:spec :type string
                       :goal (:satisfy (lambda (out)
                         (= out (lookup capital-of country))))))))
          capital)
        """
        result = parse(source)
        assert isinstance(result, List)

    def test_submodel_invocation(self):
        """From §7.1."""
        source = """
        (submodel :id "vision-encoder"
          (:spec :type tensor :shape [batch, 512] :goal (:satisfy (lambda (out) (unit-norm out))))
          input-image)
        """
        # :id will be parsed as a keyword->symbol, rest is a generic list
        result = parse(source)
        assert isinstance(result, List)

    def test_spec_with_examples_and_constraints(self):
        """From §12.10: spec as training data."""
        source = """
        (:spec :type string
               :goal (:examples (("hello" -> "HELLO") ("world" -> "WORLD"))))
        """
        result = parse(source)
        assert isinstance(result, Spec)
        assert isinstance(result.goal, GoalExamples)
        assert len(result.goal.pairs) == 2

    def test_subagent_with_constraints(self):
        """From §7.2."""
        source = """
        (subagent
          (:spec :type string
                 :goal (:intent "summarize the document")
                 :constraints (max-length 200)))
        """
        result = parse(source)
        assert isinstance(result, List)
        spec = result.elements[1]
        assert isinstance(spec, Spec)
        assert isinstance(spec.constraints, List)


# --- Error handling ---

class TestErrors:
    def test_unterminated_string(self):
        with pytest.raises(ParseError, match="unterminated string"):
            parse('"hello')

    def test_unmatched_paren(self):
        with pytest.raises(ParseError):
            parse("(add 1 2")

    def test_empty_input(self):
        with pytest.raises(ParseError, match="empty input"):
            parse("")

    def test_extra_tokens(self):
        with pytest.raises(ParseError, match="unexpected token after expression"):
            parse("1 2")
