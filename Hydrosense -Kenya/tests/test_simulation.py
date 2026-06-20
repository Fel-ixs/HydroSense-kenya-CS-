"""
test_simulation.py
--------------------
Pytest test suite for the evapotranspiration / water-balance functions in
src/water_balance_functions.py and the Euler, Runge-Kutta, and Monte Carlo
simulation functions in src/simulation.py.

Requires pytest . Run with:
    pytest tests/test_simulation.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from simulation import euler_simulation, rk4_simulation, monte_carlo_rainfall
from water_balance_functions import evapotranspiration, water_balance_step


def test_evapotranspiration_nonnegative():
    """
    The simplified ET formula is clipped at zero (ET = max(0, ...)), so
    even under inputs that would otherwise produce a negative raw value
    (e.g. very low temperature/wind/solar combined with high humidity),
    the returned ET must never be negative.
    """
    et = evapotranspiration(T=[10, 20, 50], W=[1, 2, 3], Solar=[0.1, 0.5, 0.9], H=[90, 50, 10])
    assert np.all(et >= 0)


def test_water_balance_step_basic():
    """
    Direct check of the discrete water balance equation
    S_(t+1) = S_t + R_t + I_t - ET_t - D_t
    using simple numbers that are easy to verify by hand: 30+5+2-3-1=33.
    """
    S_next = water_balance_step(S=30, R=5, I=2, ET=3, D=1)
    assert S_next == pytest.approx(33)


def test_euler_simulation_length():
    """
    A simulation over n days should return n+1 soil-moisture values
    (the initial condition S0 plus one value per simulated day). With
    zero rainfall, irrigation, and ET, and starting below field capacity
    (so drainage is also zero), moisture should remain perfectly constant.
    """
    n = 10
    R = np.zeros(n)
    I = np.zeros(n)
    ET = np.zeros(n)
    S = euler_simulation(S0=30, R_series=R, I_series=I, ET_series=ET, field_capacity=40, k=0.2)
    assert len(S) == n + 1
    assert np.allclose(S, 30)


def test_rk4_matches_euler_for_zero_dynamics():
    """
    When the right-hand side of the ODE is identically zero (no rainfall,
    irrigation, or ET), every numerical integrator -- Euler or RK4 -- should
    agree exactly, since there is no dynamics to integrate. This isolates
    a true implementation bug from a genuine accuracy difference between
    the two methods (which does appear once real rainfall/irrigation/ET
    dynamics are introduced).
    """
    n = 5
    R = np.zeros(n)
    I = np.zeros(n)
    ET = np.zeros(n)
    S_euler = euler_simulation(30, R, I, ET, 40, 0.2)
    S_rk4 = rk4_simulation(30, R, I, ET, 40, 0.2)
    assert np.allclose(S_euler, S_rk4)


def test_monte_carlo_rainfall_shape_and_nonnegative():
    """
    monte_carlo_rainfall should return an array shaped
    (n_scenarios, n_days) and, because rainfall cannot physically be
    negative, every value must be clipped at zero.
    """
    rainfall = monte_carlo_rainfall(mean_rainfall=5, std_rainfall=2, n_days=30, n_scenarios=1000)
    assert rainfall.shape == (1000, 30)
    assert np.all(rainfall >= 0)


def test_monte_carlo_rainfall_matches_target_distribution():
    """
    With a large enough sample (1000 scenarios x 30 days = 30,000 draws),
    the empirical mean and standard deviation of the generated rainfall
    should be close to the requested target distribution parameters
    (mean=5mm, std=2mm), confirming the random-number generation is wired
    correctly (not just shaped correctly).
    """
    rainfall = monte_carlo_rainfall(mean_rainfall=5, std_rainfall=2, n_days=30, n_scenarios=1000)
    assert rainfall.mean() == pytest.approx(5, abs=0.2)
    assert rainfall.std() == pytest.approx(2, abs=0.2)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    tests = [
        test_evapotranspiration_nonnegative,
        test_water_balance_step_basic,
        test_euler_simulation_length,
        test_rk4_matches_euler_for_zero_dynamics,
        test_monte_carlo_rainfall_shape_and_nonnegative,
        test_monte_carlo_rainfall_matches_target_distribution,
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

    # ---- Print the actual simulation outputs for inspection ----
    print("=== Evapotranspiration values ===")
    et = evapotranspiration(T=[10, 20, 50], W=[1, 2, 3], Solar=[0.1, 0.5, 0.9], H=[90, 50, 10])
    print("ET:", et)

    print("\n=== Single water balance step ===")
    S_next = water_balance_step(S=30, R=5, I=2, ET=3, D=1)
    print(f"S0=30, R=5, I=2, ET=3, D=1 -> S1={S_next}")

    print("\n=== Euler simulation (10 days, zero rainfall/irrigation/ET) ===")
    n = 10
    R = np.zeros(n)
    I = np.zeros(n)
    ET = np.zeros(n)
    S_euler_flat = euler_simulation(S0=30, R_series=R, I_series=I, ET_series=ET, field_capacity=40, k=0.2)
    print("Soil moisture trajectory:", S_euler_flat)

    print("\n=== Euler vs RK4 with mild dynamics (5 days) ===")
    n2 = 5
    R2 = np.array([5.0, 0.0, 10.0, 0.0, 2.0])
    I2 = np.array([0.0, 3.0, 0.0, 4.0, 0.0])
    ET2 = np.array([2.0, 2.5, 2.0, 2.0, 1.5])
    S_euler = euler_simulation(30, R2, I2, ET2, field_capacity=40, k=0.2)
    S_rk4 = rk4_simulation(30, R2, I2, ET2, field_capacity=40, k=0.2)
    print("Rainfall:   ", R2)
    print("Irrigation: ", I2)
    print("ET:         ", ET2)
    print("Euler S:    ", S_euler)
    print("RK4 S:      ", S_rk4)
    print("Max |Euler-RK4| difference:", np.max(np.abs(S_euler - S_rk4)))

    print("\n=== Monte Carlo rainfall scenarios (1000 scenarios x 30 days) ===")
    rainfall = monte_carlo_rainfall(mean_rainfall=5, std_rainfall=2, n_days=30, n_scenarios=1000)
    print("Shape:", rainfall.shape)
    print("First 3 scenarios, first 10 days:")
    print(rainfall[:3, :10])
    print(f"Overall mean rainfall: {rainfall.mean():.3f} mm  (target mean=5)")
    print(f"Overall std rainfall:  {rainfall.std():.3f} mm  (target std=2)")
    print(f"Min value: {rainfall.min():.3f} (should be >= 0 due to clipping)")