"""SELPH: Symbolic Evaluation Language for Programmable Hierarchies.

A homoiconic language for program synthesis with curriculum-driven
library learning.

Quick start:
    from selph.engine import SynthesisEngine

    engine = SynthesisEngine()
    result = engine.solve([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    print(result.source)  # (lambda (x) (add x x))
"""

from .parser import parse, parse_file
from .eval import eval_string, eval_program, standard_env, Namespace
from .engine import SynthesisEngine, SolveResult
