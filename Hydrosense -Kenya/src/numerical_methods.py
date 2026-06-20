"""
numerical_methods.py
Manual implementations of root finding, finite differences, numerical
integration, and linear-system solvers used throughout HydroSense-Kenya.
"""

import numpy as np


# ---------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------

def bisection(f, a, b, tol=1e-6, max_iter=200):
    """
    Bisection root-finding method.
    Returns (root, n_iterations, converged, history) where history is a
    list of (iteration, x_mid, f(x_mid)) tuples.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    history = []
    for i in range(1, max_iter + 1):
        mid = (a + b) / 2.0
        fm = f(mid)
        history.append((i, mid, fm))
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return mid, i, True, history
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return mid, max_iter, False, history


def newton_raphson(f, fprime, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson root finding.
    Returns (root, n_iterations, converged, history).
    """
    x = x0
    history = []
    for i in range(1, max_iter + 1):
        fx = f(x)
        history.append((i, x, fx))
        if abs(fx) < tol:
            return x, i, True, history
        dfx = fprime(x)
        if dfx == 0:
            return x, i, False, history
        x = x - fx / dfx
    return x, max_iter, False, history


def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant method root finding (no derivative required).
    Returns (root, n_iterations, converged, history).
    """
    history = []
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        history.append((i, x1, f1))
        if abs(f1) < tol:
            return x1, i, True, history
        if f1 - f0 == 0:
            return x1, i, False, history
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x2
    return x1, max_iter, False, history


# ---------------------------------------------------------------------
# Numerical differentiation
# ---------------------------------------------------------------------

def forward_difference(f_values, h, i):
    """Forward difference approximation of f'(x_i) using a sampled series."""
    return (f_values[i + 1] - f_values[i]) / h


def backward_difference(f_values, h, i):
    """Backward difference approximation of f'(x_i)."""
    return (f_values[i] - f_values[i - 1]) / h


def central_difference(f_values, h, i):
    """Central difference approximation of f'(x_i) (second-order accurate)."""
    return (f_values[i + 1] - f_values[i - 1]) / (2 * h)


# ---------------------------------------------------------------------
# Numerical integration
# ---------------------------------------------------------------------

def trapezoidal_rule(y_values, h):
    """Composite trapezoidal rule for equally spaced samples y with spacing h."""
    y = np.asarray(y_values, dtype=float)
    return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))


def simpsons_rule(y_values, h):
    """
    Composite Simpson's rule. Requires an even number of intervals
    (odd number of points). If the number of points is even, the last
    interval is handled with the trapezoidal rule as a fallback.
    """
    y = np.asarray(y_values, dtype=float)
    n = len(y) - 1  # number of intervals
    if n % 2 == 1:
        # odd number of intervals: Simpson on all but the last, trapezoid for the last
        simpson_part = simpsons_rule(y[:-1], h)
        trap_part = h * 0.5 * (y[-2] + y[-1])
        return simpson_part + trap_part
    s = y[0] + y[-1]
    s += 4 * np.sum(y[1:-1:2])
    s += 2 * np.sum(y[2:-2:2])
    return h * s / 3.0


# ---------------------------------------------------------------------
# Linear systems
# ---------------------------------------------------------------------

def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    A: (n,n) array-like, b: (n,) array-like.
    Returns the solution vector x as a numpy array.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])

    for col in range(n):
        # partial pivoting
        pivot_row = np.argmax(np.abs(Ab[col:, col])) + col
        if Ab[pivot_row, col] == 0:
            raise ValueError("Matrix is singular.")
        if pivot_row != col:
            Ab[[col, pivot_row]] = Ab[[pivot_row, col]]

        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]

    # back substitution
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        x[row] = (Ab[row, -1] - np.dot(Ab[row, row + 1:n], x[row + 1:n])) / Ab[row, row]
    return x
