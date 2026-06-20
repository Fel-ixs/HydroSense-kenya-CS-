"""

Soil-moisture simulation using Euler and Runge-Kutta methods, plus
Monte Carlo rainfall-uncertainty modelling.
"""
import numpy as np
import sys
from pathlib import Path


PROJECT_ROOT = Path.cwd()

while not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent


sys.path.insert(0,str(PROJECT_ROOT))


from src.water_balance_functions import drainage, evapotranspiration

def soil_moisture_derivative(S, R, I, ET, field_capacity, k):
    """
    dS/dt = R + I - ET - D(S)
    Continuous-time analogue of the discrete water balance equation.
    """
    D = drainage(S, field_capacity, k)
    return R + I - ET - D


def euler_simulation(S0, R_series, I_series, ET_series, field_capacity, k, dt=1.0):
    """
    Simulate soil moisture over len(R_series) days using the explicit
    Euler method.
    Returns an array of soil moisture values of length n+1 (including S0).
    """
    n = len(R_series)
    S = np.zeros(n + 1)
    S[0] = S0
    for t in range(n):
        dS = soil_moisture_derivative(S[t], R_series[t], I_series[t], ET_series[t], field_capacity, k)
        S[t + 1] = S[t] + dt * dS
    return S


def rk4_simulation(S0, R_series, I_series, ET_series, field_capacity, k, dt=1.0):
    """
    Simulate soil moisture using the classic fourth-order Runge-Kutta method.
    Rainfall, irrigation, and ET are held constant within each daily step
    (a standard simplification for daily-resolution agronomic data).
    """
    n = len(R_series)
    S = np.zeros(n + 1)
    S[0] = S0
    for t in range(n):
        R, I, ET = R_series[t], I_series[t], ET_series[t]
        f = lambda s: soil_moisture_derivative(s, R, I, ET, field_capacity, k)
        k1 = f(S[t])
        k2 = f(S[t] + dt / 2 * k1)
        k3 = f(S[t] + dt / 2 * k2)
        k4 = f(S[t] + dt * k3)
        S[t + 1] = S[t] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


def monte_carlo_rainfall(mean_rainfall, std_rainfall, n_days, n_scenarios, rng=None):
    """
    Generate Monte Carlo rainfall scenarios using a (truncated at zero)
    normal distribution per day.
    Returns an array of shape (n_scenarios, n_days).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    rainfall = rng.normal(loc=mean_rainfall, scale=std_rainfall, size=(n_scenarios, n_days))
    return np.clip(rainfall, 0, None)


def monte_carlo_soil_simulation(S0, rainfall_scenarios, I_series, ET_series, field_capacity, k, dt=1.0):
    """
    Run the Euler soil-moisture simulation across many Monte Carlo rainfall
    scenarios. Returns an array of shape (n_scenarios, n_days+1).
    """
    n_scenarios, n_days = rainfall_scenarios.shape
    results = np.zeros((n_scenarios, n_days + 1))
    for s in range(n_scenarios):
        results[s] = euler_simulation(S0, rainfall_scenarios[s], I_series, ET_series, field_capacity, k, dt)
    return results


def shortage_and_overirrigation_probabilities(simulated_moisture, min_threshold, field_capacity):
    """
    Given simulated soil moisture of shape (n_scenarios, n_days+1), estimate:
      - probability of water shortage (moisture drops below min_threshold at least once)
      - probability of over-irrigation (moisture exceeds field_capacity at least once)
      - expected minimum moisture
      - worst-case (5th percentile) minimum moisture
    """
    shortage = np.any(simulated_moisture < min_threshold, axis=1)
    over = np.any(simulated_moisture > field_capacity, axis=1)
    min_per_scenario = simulated_moisture.min(axis=1)
    return {
        "p_shortage": float(np.mean(shortage)),
        "p_overirrigation": float(np.mean(over)),
        "expected_min_moisture": float(np.mean(min_per_scenario)),
        "worst_case_min_moisture": float(np.percentile(min_per_scenario, 5)),
    }