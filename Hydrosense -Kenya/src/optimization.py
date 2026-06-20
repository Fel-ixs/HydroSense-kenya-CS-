"""

Irrigation schedule optimization: minimize total irrigation water use
subject to a minimum soil-moisture (crop-stress) constraint.
"""

import numpy as np
import sys
from pathlib import Path


PROJECT_ROOT = Path.cwd()

while not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent


sys.path.insert(0,str(PROJECT_ROOT))


from src.simulation import soil_moisture_derivative


def simulate_with_irrigation(S0, R_series, I_series, ET_series, field_capacity, k, dt=1.0):
    """Euler-style forward simulation used internally by the optimizer."""
    n = len(R_series)
    S = np.zeros(n + 1)
    S[0] = S0
    for t in range(n):
        dS = soil_moisture_derivative(S[t], R_series[t], I_series[t], ET_series[t], field_capacity, k)
        S[t + 1] = S[t] + dt * dS
    return S


def schedule_cost(I_series, S0, R_series, ET_series, field_capacity, k, min_threshold, penalty_weight=1000.0):
    """
    Objective function for the optimizer:
       cost = total irrigation water used + penalty for any moisture below min_threshold
    A heavy penalty enforces the crop-stress constraint without a hard
    constrained solver, which keeps the optimizer simple and dependency-free.
    """
    S = simulate_with_irrigation(S0, R_series, I_series, ET_series, field_capacity, k)
    total_irrigation = np.sum(I_series)
    deficit = np.maximum(0.0, min_threshold - S)
    penalty = penalty_weight * np.sum(deficit ** 2)
    return total_irrigation + penalty


def numerical_gradient(func, x, h=1e-4):
    """Central-difference numerical gradient of a scalar function func(x)."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += h
        x_bwd[i] -= h
        grad[i] = (func(x_fwd) - func(x_bwd)) / (2 * h)
    return grad


def gradient_descent_irrigation(S0, R_series, ET_series, field_capacity, k, min_threshold,
                                 n_days, lr=0.05, n_iter=500, penalty_weight=1000.0):
    """
    Optimize a daily irrigation schedule I_series (length n_days) using
    simple gradient descent on the penalized cost function.
    Irrigation is clipped to be non-negative after every update.
    Returns (I_optimal, cost_history).
    """
    I = np.zeros(n_days)
    cost_history = []

    def cost_fn(I_vec):
        return schedule_cost(I_vec, S0, R_series, ET_series, field_capacity, k, min_threshold, penalty_weight)

    for _ in range(n_iter):
        grad = numerical_gradient(cost_fn, I)
        I = I - lr * grad
        I = np.clip(I, 0, None)
        cost_history.append(cost_fn(I))

    return I, cost_history