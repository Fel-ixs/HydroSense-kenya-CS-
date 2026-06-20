"""
test_root_finding.py
---------------------
Pytest test suite for the manual root-finding implementations in
src/numerical_methods.py: bisection, Newton-Raphson, and secant.

Requires pytest . Run with:
    pytest tests/test_root_finding.py -v
"""

import sys, os
# Make src/ importable regardless of the current working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from numerical_methods import bisection, newton_raphson, secant


# ---------------------------------------------------------------------
# Test function: f(x) = x^2 - 4 has known analytic roots at x = 2 and x = -2.
# Using a function with a known closed-form root lets us validate the
# numerical methods against ground truth rather than just checking that
# "some number" came out.
# ---------------------------------------------------------------------

def f(x):
    """f(x) = x^2 - 4, roots at x = +2 and x = -2."""
    return x ** 2 - 4


def fprime(x):
    """Analytic derivative of f, required only by Newton-Raphson."""
    return 2 * x


def test_bisection_finds_root():
    """
    Bisection should converge to the root x=2 when bracketed by [0, 3]
    (f(0) = -4 < 0, f(3) = 5 > 0, so a sign change is guaranteed).
    """
    root, n_iter, converged, history = bisection(f, 0, 3, tol=1e-6)
    assert converged, "Bisection should report convergence within max_iter"
    # pytest.approx allows a tolerance-based equality check instead of
    # manually computing abs(root - 2.0) < tol every time.
    assert root == pytest.approx(2.0, abs=1e-4)


def test_newton_raphson_finds_root():
    """
    Newton-Raphson should converge quadratically to x=2 starting from x0=3.0.
    Quadratic convergence means far fewer iterations than bisection are
    needed to reach the same tolerance.
    """
    root, n_iter, converged, history = newton_raphson(f, fprime, x0=3.0, tol=1e-6)
    assert converged, "Newton-Raphson should report convergence within max_iter"
    assert root == pytest.approx(2.0, abs=1e-4)
    # Sanity check on convergence speed: Newton-Raphson should need far
    # fewer iterations than bisection's ~20 for the same tolerance.
    assert n_iter < 10


def test_secant_finds_root():
    """
    Secant method should converge to x=2 from two starting guesses (1.0, 3.0)
    that bracket the root, without requiring an analytic derivative.
    """
    root, n_iter, converged, history = secant(f, 1.0, 3.0, tol=1e-6)
    assert converged, "Secant method should report convergence within max_iter"
    assert root == pytest.approx(2.0, abs=1e-4)


def test_bisection_raises_on_same_sign():
    """
    Bisection requires f(a) and f(b) to have opposite signs (Intermediate
    Value Theorem). On [3, 4], f(3)=5 and f(4)=12 are both positive, so the
    function should raise ValueError instead of silently returning a wrong
    answer.
    """
    with pytest.raises(ValueError):
        bisection(f, 3, 4, tol=1e-6)


if __name__ == "__main__":
    # Lets you run `python test_root_finding.py` directly to see the actual
    # pass/fail results AND the real computed root-finding values, without
    # needing to invoke pytest. (pytest itself ignores this block when it
    # collects and runs tests via `pytest tests/ -v`.)
    tests = [
        test_bisection_finds_root,
        test_newton_raphson_finds_root,
        test_secant_finds_root,
        test_bisection_raises_on_same_sign,
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

    # ---- Print the actual root-finding results for inspection ----
    print("=== Root-finding results on f(x) = x^2 - 4 ===")
    root_b, n_b, conv_b, hist_b = bisection(f, 0, 3, tol=1e-6)
    print(f"Bisection:       root={root_b:.6f}  iterations={n_b}  converged={conv_b}")

    root_n, n_n, conv_n, hist_n = newton_raphson(f, fprime, x0=3.0, tol=1e-6)
    print(f"Newton-Raphson:  root={root_n:.6f}  iterations={n_n}  converged={conv_n}")

    root_s, n_s, conv_s, hist_s = secant(f, 1.0, 3.0, tol=1e-6)
    print(f"Secant:          root={root_s:.6f}  iterations={n_s}  converged={conv_s}")