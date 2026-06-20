"""
test_integration.py
---------------------
Pytest test suite for the numerical integration and finite-difference
implementations in src/numerical_methods.py: trapezoidal rule, Simpson's
rule, and forward/backward/central differences.

Requires pytest . Run with:
    pytest tests/test_integration.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from numerical_methods import (
    trapezoidal_rule,
    simpsons_rule,
    central_difference,
    forward_difference,
    backward_difference,
)


def test_trapezoidal_constant_function():
    """
    Integrating a constant function f(x)=5 over [0, 10] should give exactly
    5 * 10 = 50, regardless of the number of sample points, since the
    trapezoidal rule is exact for any piecewise-linear (here: flat) function.
    """
    y = [5.0] * 11  # f(x) = 5 sampled at 11 equally spaced points, h = 1.0
    result = trapezoidal_rule(y, h=1.0)
    assert result == pytest.approx(50.0, abs=1e-8)


def test_simpsons_quadratic_exact():
    """
    Simpson's rule is exact for polynomials up to cubic degree, so
    integrating f(x)=x^2 over [0, 4] should match the analytic result
    integral(x^2, 0, 4) = 4^3 / 3 = 21.333... to high precision.
    """
    x = np.linspace(0, 4, 9)  # 8 intervals (even number, required by classic Simpson's rule)
    y = x ** 2
    result = simpsons_rule(y, h=x[1] - x[0])
    expected = 4 ** 3 / 3.0
    assert result == pytest.approx(expected, abs=1e-6)


def test_central_difference_linear_function():
    """
    For a linear function f(x)=3x+2, the derivative is constant (=3)
    everywhere, so the central difference approximation should recover it
    essentially exactly (central differences are exact for linear functions).
    """
    x = np.linspace(0, 10, 11)
    y = 3 * x + 2
    h = x[1] - x[0]
    deriv = central_difference(y, h, 5)
    assert deriv == pytest.approx(3.0, abs=1e-8)


def test_central_difference_more_accurate_than_forward_on_curved_function():
    """
    On a curved function like f(x)=x^2, forward/backward differences carry
    O(h) truncation error while central differences carry O(h^2) error.
    At x=5 the exact derivative of x^2 is 10; central difference should be
    noticeably closer to 10 than the forward/backward estimates.
    """
    x = np.linspace(0, 10, 11)
    y = x ** 2
    h = x[1] - x[0]
    i = 5
    fd_error = abs(forward_difference(y, h, i) - 10.0)
    cd_error = abs(central_difference(y, h, i) - 10.0)
    assert cd_error < fd_error


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    tests = [
        test_trapezoidal_constant_function,
        test_simpsons_quadratic_exact,
        test_central_difference_linear_function,
        test_central_difference_more_accurate_than_forward_on_curved_function,
    ]
    passed = 0
    print("=== Running assertions ===")
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{passed}/{len(tests)} tests passed\n")

    # ---- Print the actual numerical results for inspection ----
    print("=== Trapezoidal rule on a constant function f(x)=5, x in [0,10] ===")
    y = [5.0] * 11
    result = trapezoidal_rule(y, h=1.0)
    print(f"y = {y}")
    print(f"Trapezoidal integral estimate: {result}  (exact = 50.0)")

    print("\n=== Simpson's rule on f(x)=x^2, x in [0,4] (8 intervals) ===")
    x = np.linspace(0, 4, 9)
    y = x ** 2
    result = simpsons_rule(y, h=x[1] - x[0])
    exact = 4 ** 3 / 3.0
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"Simpson's integral estimate: {result:.6f}  (exact = {exact:.6f})")
    print(f"Absolute error: {abs(result - exact):.2e}")

    print("\n=== Simpson's rule with odd-interval fallback (7 intervals) ===")
    x_odd = np.linspace(0, 4, 8)
    y_odd = x_odd ** 2
    result_odd = simpsons_rule(y_odd, h=x_odd[1] - x_odd[0])
    print(f"x = {x_odd}")
    print(f"Simpson's (with trapezoidal fallback) estimate: {result_odd:.6f}  (exact = {exact:.6f})")

    print("\n=== Finite differences on f(x)=3x+2 (exact derivative = 3.0) ===")
    x = np.linspace(0, 10, 11)
    y = 3 * x + 2
    h = x[1] - x[0]
    i = 5
    print(f"At x[{i}]={x[i]}:")
    print(f"  Forward difference:  {forward_difference(y, h, i)}")
    print(f"  Backward difference: {backward_difference(y, h, i)}")
    print(f"  Central difference:  {central_difference(y, h, i)}")

    print("\n=== Finite differences on a curved function f(x)=x^2 (exact derivative at x=5 is 10.0) ===")
    x2 = np.linspace(0, 10, 11)
    y2 = x2 ** 2
    fd2 = forward_difference(y2, h, i)
    bd2 = backward_difference(y2, h, i)
    cd2 = central_difference(y2, h, i)
    print(f"At x[{i}]={x2[i]}:")
    print(f"  Forward difference:  {fd2}  (error = {abs(fd2-10):.4f})")
    print(f"  Backward difference: {bd2}  (error = {abs(bd2-10):.4f})")
    print(f"  Central difference:  {cd2}  (error = {abs(cd2-10):.4f})  <- most accurate, O(h^2)")