"""Tests for library pruning."""

import pytest
from selph.library import (
    prune_library, track_usage, Abstraction,
    register_abstractions,
)
from selph.eval import standard_env
from selph.ast import Symbol, List, Number
from selph.types import TStr, TNum, TFn
from selph.parser import parse


def _make_abs(name, body_source, param_types=None, return_type=None):
    """Helper to create an Abstraction from a body source string."""
    body = parse(body_source)
    if param_types is None:
        param_types = [TStr()]
    if return_type is None:
        return_type = TStr()
    return Abstraction(
        name=name,
        params=["x"],
        body=body,
        param_types=param_types,
        return_type=return_type,
        frequency=1,
        compression=0.0,
    )


class TestObservationalEquivalence:
    def test_remove_duplicate(self):
        """Two abstractions that do the same thing — keep the simpler one."""
        abs1 = _make_abs("simple_upper", "(string-upper x)")
        abs2 = _make_abs("verbose_upper", "(string-upper (identity x))")

        pruned, stats = prune_library([abs1, abs2])
        assert stats["obs_equiv_removed"] >= 1
        assert len(pruned) < 2

    def test_keep_distinct(self):
        """Two abstractions that do different things — keep both."""
        abs1 = _make_abs("trim_upper", "(string-upper (string-trim x))")
        abs2 = _make_abs("trim_reverse", "(string-reverse (string-trim x))")

        pruned, stats = prune_library([abs1, abs2])
        # "  hello  " -> "HELLO" vs "olleh" — clearly different
        assert len(pruned) == 2
        assert stats["obs_equiv_removed"] == 0

    def test_multiple_duplicates(self):
        """Several equivalent abstractions — keep one."""
        abs1 = _make_abs("upper_v1", "(string-upper x)")
        abs2 = _make_abs("upper_v2", "(string-upper (identity x))")
        abs3 = _make_abs("upper_v3", "(string-upper x)")

        pruned, stats = prune_library([abs1, abs2, abs3])
        assert len(pruned) <= 2  # at most one upper + obs equiv removes rest
        assert stats["obs_equiv_removed"] >= 1


class TestBuiltinEquivalence:
    def test_remove_builtin_alias(self):
        """A promoted primitive that's just string-upper gets removed."""
        abs1 = _make_abs("promoted_upper", "(string-upper x)")

        pruned, stats = prune_library([abs1])
        assert stats["builtin_equiv_removed"] == 1
        assert len(pruned) == 0

    def test_keep_composition(self):
        """A composition of builtins is NOT equivalent to any single builtin."""
        abs1 = _make_abs("upper_reverse", "(string-upper (string-reverse x))")

        pruned, stats = prune_library([abs1])
        assert stats["builtin_equiv_removed"] == 0
        assert len(pruned) == 1

    def test_identity_removed(self):
        """A primitive equivalent to identity gets removed."""
        abs1 = _make_abs("my_identity", "x")

        pruned, stats = prune_library([abs1])
        assert stats["builtin_equiv_removed"] == 1
        assert len(pruned) == 0


class TestUsagePruning:
    def test_remove_unused(self):
        """Primitives with zero usage get removed."""
        # Use compositions so they survive builtin-equivalence check
        abs1 = _make_abs("used_one", "(string-upper (string-trim x))")
        abs2 = _make_abs("unused_one", "(string-lower (string-trim x))")

        usage = {"used_one": 3, "unused_one": 0}
        pruned, stats = prune_library(
            [abs1, abs2], usage_counts=usage, min_uses=1)
        assert stats["unused_removed"] >= 1
        names = [a.name for a in pruned]
        assert "unused_one" not in names

    def test_keep_used(self):
        """Primitives with sufficient usage survive."""
        abs1 = _make_abs("popular", "(string-upper (string-reverse x))")
        abs2 = _make_abs("also_popular", "(string-lower (string-trim x))")

        usage = {"popular": 5, "also_popular": 3}
        pruned, stats = prune_library(
            [abs1, abs2], usage_counts=usage, min_uses=1)
        assert stats["unused_removed"] == 0
        assert len(pruned) == 2


class TestTrackUsage:
    def test_count_usage(self):
        abs1 = _make_abs("my_fn", "(string-upper x)")
        abs2 = _make_abs("other_fn", "(string-reverse x)")

        solutions = [
            parse("(my_fn (my_fn x))"),
            parse("(other_fn x)"),
            parse("(add 1 2)"),
        ]

        counts = track_usage(solutions, [abs1, abs2])
        assert counts["my_fn"] == 2
        assert counts["other_fn"] == 1

    def test_zero_usage(self):
        abs1 = _make_abs("never_used", "(string-upper x)")

        solutions = [parse("(add 1 2)")]
        counts = track_usage(solutions, [abs1])
        assert counts["never_used"] == 0


class TestCombinedPruning:
    def test_full_pipeline(self):
        """Run all pruning strategies together."""
        abstractions = [
            _make_abs("just_upper", "(string-upper x)"),  # builtin equiv
            _make_abs("upper_rev", "(string-upper (string-reverse x))"),  # useful
            _make_abs("upper_rev_v2", "(string-upper (string-reverse (identity x)))"),  # obs equiv to above
            _make_abs("unused_thing", "(string-lower (string-reverse x))"),  # unused
        ]

        usage = {
            "just_upper": 0,
            "upper_rev": 5,
            "upper_rev_v2": 0,
            "unused_thing": 0,
        }

        pruned, stats = prune_library(
            abstractions, usage_counts=usage, min_uses=1)

        # just_upper: removed (builtin equiv)
        # upper_rev: kept (useful, used)
        # upper_rev_v2: removed (obs equiv to upper_rev)
        # unused_thing: removed (unused)
        names = [a.name for a in pruned]
        assert "upper_rev" in names
        assert "just_upper" not in names
        assert len(pruned) <= 2  # at most upper_rev and maybe one other

        assert stats["input_count"] == 4
        assert stats["output_count"] <= 2
        print(f"Stats: {stats}")
        print(f"Survivors: {names}")
