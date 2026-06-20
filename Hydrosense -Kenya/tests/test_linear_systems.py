"""

Pytest test suite for the manual Gaussian elimination implementation in
src/numerical_methods.py, cross-checked against numpy.linalg.solve


Requires pytest . Run with:
    pytest tests/test_linear_systems.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from numerical_methods import gaussian_elimination


def test_gaussian_elimination_simple_system():
    """
    Solve a generic well-conditioned 3x3 system Ax=b and check the manual
    Gaussian elimination result against numpy.linalg.solve, which acts as
    the trusted reference implementation here.
    """
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    x = gaussian_elimination(A, b)
    expected = np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float))
    # np.allclose is the array equivalent of pytest.approx for vectors.
    assert np.allclose(x, expected, atol=1e-6)


def test_gaussian_elimination_identity():
    """
    Solving Ix=b with the identity matrix should trivially return x=b.
    This is a useful degenerate-case sanity check before trusting the
    solver on harder systems.
    """
    A = np.eye(3)
    b = [1, 2, 3]
    x = gaussian_elimination(A, b)
    assert np.allclose(x, [1, 2, 3], atol=1e-8)


def test_gaussian_elimination_requires_pivoting():
    """
    A system whose first pivot is zero would fail without partial pivoting
    (division by zero). This checks that gaussian_elimination's pivoting
    logic correctly swaps rows and still produces the right answer.
    """
    A = [[0, 2, 1], [1, 1, 1], [2, -1, 1]]
    b = [3, 6, 2]
    x = gaussian_elimination(A, b)
    expected = np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float))
    assert np.allclose(x, expected, atol=1e-6)


def test_gaussian_elimination_singular_matrix_raises():
    """
    A singular matrix (linearly dependent rows) has no unique solution.
    gaussian_elimination should raise ValueError rather than silently
    returning a meaningless result (e.g. via division by a zero pivot).
    """
    A = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]  # row 2 = 2 * row 1
    b = [6, 12, 3]
    with pytest.raises(ValueError):
        gaussian_elimination(A, b)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    tests = [
        test_gaussian_elimination_simple_system,
        test_gaussian_elimination_identity,
        test_gaussian_elimination_requires_pivoting,
        test_gaussian_elimination_singular_matrix_raises,
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

    # ---- Print the actual solved systems for inspection ----
    print("=== System 1: generic 3x3 linear system ===")
    A1 = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b1 = [8, -11, -3]
    x1 = gaussian_elimination(A1, b1)
    x1_check = np.linalg.solve(np.array(A1, dtype=float), np.array(b1, dtype=float))
    print(f"A =\n{np.array(A1)}")
    print(f"b = {b1}")
    print(f"Gaussian elimination solution: x = {x1}")
    print(f"numpy.linalg.solve solution:   x = {x1_check}")
    print(f"Max difference: {np.max(np.abs(x1 - x1_check)):.2e}")

    print("\n=== System 2: 3x3 identity system ===")
    A2 = np.eye(3)
    b2 = [1, 2, 3]
    x2 = gaussian_elimination(A2, b2)
    print(f"A =\n{A2}")
    print(f"b = {b2}")
    print(f"Solution: x = {x2}")

    print("\n=== System 3: requires partial pivoting (zero first pivot) ===")
    A3 = [[0, 2, 1], [1, 1, 1], [2, -1, 1]]
    b3 = [3, 6, 2]
    x3 = gaussian_elimination(A3, b3)
    x3_check = np.linalg.solve(np.array(A3, dtype=float), np.array(b3, dtype=float))
    print(f"A =\n{np.array(A3)}")
    print(f"b = {b3}")
    print(f"Gaussian elimination (with pivoting) solution: x = {x3}")
    print(f"numpy cross-check:                              {x3_check}")

    print("\n=== System 4: HydroSense three-zone water-allocation example (Level 3) ===")
    A4 = [[1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [1.0, 1.0, 1.0]]
    b4 = [12.0, 9.0, 30.0]  # Zone_A target=12mm, Zone_B target=9mm, total release=30mm
    x4 = gaussian_elimination(A4, b4)
    x4_check = np.linalg.solve(np.array(A4, dtype=float), np.array(b4, dtype=float))
    print(f"A =\n{np.array(A4)}")
    print(f"b (target water mm per constraint) = {b4}")
    print(f"Allocation [Zone_A, Zone_B, Zone_C] mm = {x4}")
    print(f"numpy cross-check:                      {x4_check}")

    print("\n=== System 5: singular matrix (should raise ValueError) ===")
    A5 = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]
    b5 = [6, 12, 3]
    try:
        gaussian_elimination(A5, b5)
        print("No error raised - UNEXPECTED")
    except ValueError as e:
        print(f"ValueError correctly raised: {e}")