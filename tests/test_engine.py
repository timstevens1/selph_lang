"""Tests for the unified SynthesisEngine."""

import pytest
import tempfile
import os
from selph.engine import SynthesisEngine, SolveResult


class TestBasicSolve:
    def test_solve_add1(self):
        engine = SynthesisEngine()
        engine.add_constant("x", ret_type=0, priority=100.0)
        engine.add_constant("0", ret_type=0)
        engine.add_constant("1", ret_type=0)
        engine.add_component("add", arity=2, ret_type=0, param_types=[0, 0])
        result = engine.solve(
            [0.0, 1.0, 5.0, -1.0],
            [1.0, 2.0, 6.0, 0.0],
            task_name="add1",
        )
        assert result.found
        assert "add" in result.source

    def test_solve_abs(self):
        engine = SynthesisEngine()
        engine.add_constant("x", ret_type=0, priority=100.0)
        engine.add_component("abs", arity=1, ret_type=0, param_types=[0])
        result = engine.solve(
            [-5.0, -3.0, 0.0, 2.0, 7.0],
            [5.0, 3.0, 0.0, 2.0, 7.0],
            task_name="abs",
        )
        assert result.found


class TestMacros:
    def test_macro_solve(self):
        engine = SynthesisEngine()
        engine.add_constant("x", ret_type=0, priority=100.0)
        engine.add_macro("double", ["x"], "(add x x)")
        engine.add_component("double", arity=1, ret_type=0, param_types=[0])
        result = engine.solve(
            [1.0, 3.0, 5.0],
            [2.0, 6.0, 10.0],
        )
        assert result.found


class TestPromotion:
    def test_promote_and_reuse(self):
        engine = SynthesisEngine()
        engine.add_constant("x", ret_type=0, priority=100.0)
        engine.add_constant("0", ret_type=0)
        engine.add_constant("1", ret_type=0)
        engine.add_component("add", arity=2, ret_type=0, param_types=[0, 0])

        # Solve x+1
        r1 = engine.solve([0.0, 5.0], [1.0, 6.0], task_name="add1")
        assert r1.found

        # Promote
        engine.promote(r1, "my_add1", ret_type=0, param_types=[0])

        # Now solve x+2 using my_add1
        r2 = engine.solve([0.0, 5.0], [2.0, 7.0], task_name="add2")
        assert r2.found


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            engine = SynthesisEngine(tmpdir)
            engine.add_constant("x", ret_type=0, priority=100.0)
            engine.add_macro("double", ["x"], "(add x x)")
            engine.add_component("double", arity=1, ret_type=0, param_types=[0])
            engine.solve([1.0, 3.0], [2.0, 6.0], task_name="test")
            engine.save()

            # Load
            engine2 = SynthesisEngine.load(tmpdir)
            assert len(engine2.components) >= 1
            assert len(engine2.macros) >= 1
            assert len(engine2.logger.logs) >= 1


class TestSolveSuite:
    def test_suite(self):
        engine = SynthesisEngine()
        engine.add_constant("x", ret_type=0, priority=100.0)
        engine.add_constant("0", ret_type=0)
        engine.add_constant("1", ret_type=0)
        engine.add_constant("2", ret_type=0)
        engine.add_component("add", arity=2, ret_type=0, param_types=[0, 0])
        engine.add_component("subtract", arity=2, ret_type=0, param_types=[0, 0])

        tasks = [
            ("add1", [0.0, 5.0, -1.0], [1.0, 6.0, 0.0]),
            ("add2", [0.0, 5.0, -1.0], [2.0, 7.0, 1.0]),
        ]
        results = engine.solve_suite(tasks, promote_solutions=True)
        assert all(r.found for r in results)
        assert engine.library_size > 2  # promoted solutions added


class TestSummary:
    def test_summary(self):
        engine = SynthesisEngine()
        s = engine.summary()
        assert "SynthesisEngine" in s
        assert "Components" in s
