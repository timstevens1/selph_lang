"""
SELPH optimization — Phase 2 of the optimization extension.

Provides `minimize()` which optimizes SELPH Param values to minimize a
loss expression. The SELPH evaluator computes the loss; Python drives
the optimization loop.

Usage:
    import selph_fast

    env = selph_fast.Env()
    env.eval_expr('''
        (define w (param 0.0))
        (define b (param 0.0))
        (define data (dataset (list (list 1 3) (list 2 5) (list 3 7))))
        (define model (lambda (x) (add (multiply w x) b)))
    ''')

    result = selph_fast.minimize(
        env,
        params=["w", "b"],
        loss="(model-loss model data mse)",
    )
    print(result)  # {"params": {"w": 2.0, "b": 1.0}, "loss": 0.0, ...}
"""

from __future__ import annotations
from dataclasses import dataclass, field
from .selph_fast import Env, Param
import math


@dataclass
class Solution:
    """Result of an optimization run."""
    params: dict[str, float]
    loss: float
    steps: int
    converged: bool
    history: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v:.6g}" for k, v in self.params.items())
        return f"Solution({p}, loss={self.loss:.6g}, steps={self.steps}, converged={self.converged})"


def minimize(
    env: Env,
    params: list[str],
    loss: str,
    *,
    method: str = "nelder-mead",
    steps: int = 1000,
    lr: float = 0.01,
    tol: float = 1e-8,
    verbose: bool = False,
) -> Solution:
    """Optimize SELPH Param values to minimize a loss expression.

    Args:
        env: SELPH environment containing param definitions and the model.
        params: Names of SELPH Param bindings to optimize.
        loss: SELPH expression string that evaluates to a scalar loss.
        method: Optimization method — "nelder-mead" or "gd" (finite-difference gradient descent).
        steps: Maximum number of optimization steps.
        lr: Learning rate (for gradient descent only).
        tol: Convergence tolerance on loss change.
        verbose: Print progress every 100 steps.

    Returns:
        Solution with optimized param values, final loss, and convergence info.
    """
    # Validate params exist and are Param type
    param_meta: dict[str, Param] = {}
    for name in params:
        p = env.lookup(name)
        if not isinstance(p, Param):
            raise ValueError(f"{name} is not a Param (got {type(p).__name__})")
        param_meta[name] = p

    if method == "nelder-mead":
        return _nelder_mead(env, params, param_meta, loss, steps, tol, verbose)
    elif method == "gd":
        return _gradient_descent(env, params, param_meta, loss, steps, lr, tol, verbose)
    else:
        raise ValueError(f"unknown method: {method}. Use 'nelder-mead' or 'gd'.")


def _eval_loss(env: Env, params: list[str], values: list[float],
               meta: dict[str, Param], loss_expr: str) -> float:
    """Set param values in env and evaluate the loss expression."""
    for name, val in zip(params, values):
        m = meta[name]
        # Clamp to bounds if present
        if m.min is not None and val < m.min:
            val = m.min
        if m.max is not None and val > m.max:
            val = m.max
        env.define(name, Param(val, m.min, m.max))
    result = env.eval_expr(loss_expr)
    if not isinstance(result, (int, float)):
        raise ValueError(f"loss expression returned {type(result).__name__}, expected number")
    return float(result)


def _gradient_descent(
    env: Env, params: list[str], meta: dict[str, Param],
    loss_expr: str, steps: int, lr: float, tol: float, verbose: bool,
) -> Solution:
    """Finite-difference gradient descent."""
    n = len(params)
    x = [meta[name].value for name in params]
    eps = 1e-6
    history = []

    best_loss = _eval_loss(env, params, x, meta, loss_expr)
    history.append(best_loss)

    for step in range(steps):
        # Compute finite-difference gradient
        grad = [0.0] * n
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            loss_plus = _eval_loss(env, params, x_plus, meta, loss_expr)
            grad[i] = (loss_plus - best_loss) / eps

        # Update
        for i in range(n):
            x[i] -= lr * grad[i]
            # Clamp to bounds
            m = meta[params[i]]
            if m.min is not None:
                x[i] = max(x[i], m.min)
            if m.max is not None:
                x[i] = min(x[i], m.max)

        new_loss = _eval_loss(env, params, x, meta, loss_expr)
        history.append(new_loss)

        if verbose and (step + 1) % 100 == 0:
            print(f"  step {step + 1}: loss={new_loss:.6g}")

        if abs(new_loss - best_loss) < tol:
            # Set final values
            _eval_loss(env, params, x, meta, loss_expr)
            return Solution(
                params=dict(zip(params, x)),
                loss=new_loss,
                steps=step + 1,
                converged=True,
                history=history,
            )
        best_loss = new_loss

    _eval_loss(env, params, x, meta, loss_expr)
    return Solution(
        params=dict(zip(params, x)),
        loss=best_loss,
        steps=steps,
        converged=False,
        history=history,
    )


def _nelder_mead(
    env: Env, params: list[str], meta: dict[str, Param],
    loss_expr: str, steps: int, tol: float, verbose: bool,
) -> Solution:
    """Nelder-Mead simplex optimization.

    Gradient-free, works on any loss function, handles bounded params
    via clamping. Good for 2-20 parameters.
    """
    n = len(params)
    x0 = [meta[name].value for name in params]

    # Initialize simplex: x0 + n unit perturbations
    simplex = [x0[:]]
    for i in range(n):
        point = x0[:]
        point[i] += 0.5 if x0[i] == 0.0 else 0.05 * abs(x0[i])
        simplex.append(point)

    # Evaluate all vertices
    f_values = [_eval_loss(env, params, v, meta, loss_expr) for v in simplex]
    history = [min(f_values)]

    # Standard Nelder-Mead coefficients
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    for step in range(steps):
        # Sort vertices by function value
        order = sorted(range(n + 1), key=lambda i: f_values[i])
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]

        best = f_values[0]
        worst = f_values[-1]
        second_worst = f_values[-2]

        history.append(best)

        if verbose and (step + 1) % 100 == 0:
            print(f"  step {step + 1}: loss={best:.6g}")

        # Check convergence: spread of function values
        if worst - best < tol:
            _eval_loss(env, params, simplex[0], meta, loss_expr)
            return Solution(
                params=dict(zip(params, simplex[0])),
                loss=best,
                steps=step + 1,
                converged=True,
                history=history,
            )

        # Centroid of all vertices except the worst
        centroid = [0.0] * n
        for i in range(n):
            for v in simplex[:-1]:
                centroid[i] += v[i]
            centroid[i] /= n

        # Reflection
        xr = [centroid[i] + alpha * (centroid[i] - simplex[-1][i]) for i in range(n)]
        fr = _eval_loss(env, params, xr, meta, loss_expr)

        if best <= fr < second_worst:
            simplex[-1] = xr
            f_values[-1] = fr
            continue

        # Expansion
        if fr < best:
            xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(n)]
            fe = _eval_loss(env, params, xe, meta, loss_expr)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
            continue

        # Contraction
        xc = [centroid[i] + rho * (simplex[-1][i] - centroid[i]) for i in range(n)]
        fc = _eval_loss(env, params, xc, meta, loss_expr)
        if fc < worst:
            simplex[-1] = xc
            f_values[-1] = fc
            continue

        # Shrink — contract all vertices toward the best
        for i in range(1, n + 1):
            simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
            f_values[i] = _eval_loss(env, params, simplex[i], meta, loss_expr)

    _eval_loss(env, params, simplex[0], meta, loss_expr)
    return Solution(
        params=dict(zip(params, simplex[0])),
        loss=f_values[0],
        steps=steps,
        converged=False,
        history=history,
    )
