"""

Core scientific functions for the HydroSense-Kenya water balance model.

Functions
---------
evapotranspiration(T, W, Solar, H)         -> scalar or array ET estimate
evapotranspiration_loop(T, W, Solar, H)    -> ET using a pure python loop (for timing comparisons)
water_balance_step(S, R, I, ET, D)         -> next-day soil storage S_(t+1)
drainage(S, field_capacity, k)             -> drainage term D_t
water_deficit(S, target)                   -> water deficit (target - S, clipped at 0)
"""

import numpy as np


def evapotranspiration(T, W, Solar, H):
    """
    Vectorized simplified empirical evapotranspiration estimate.

    ET = max(0, 0.12*T + 0.35*W + 2.4*Solar - 0.025*H)

    Parameters
    ----------
    T : float or array-like, temperature (deg C)
    W : float or array-like, wind speed (m/s)
    Solar : float or array-like, solar index (0-1)
    H : float or array-like, relative humidity (%)

    Returns
    -------
    float or np.ndarray
    """
    T = np.asarray(T, dtype=float)
    W = np.asarray(W, dtype=float)
    Solar = np.asarray(Solar, dtype=float)
    H = np.asarray(H, dtype=float)
    et = 0.12 * T + 0.35 * W + 2.4 * Solar - 0.025 * H
    return np.maximum(0.0, et)


def evapotranspiration_loop(T, W, Solar, H):
    """
    Pure python loop implementation of the same formula, used only to
    demonstrate the performance difference against the vectorized version
    in Level 2 of the project.
    """
    n = len(T)
    result = [0.0] * n
    for i in range(n):
        val = 0.12 * T[i] + 0.35 * W[i] + 2.4 * Solar[i] - 0.025 * H[i]
        result[i] = max(0.0, val)
    return result


def drainage(S, field_capacity, k=0.2):
    """
    Simple drainage model: any soil moisture above field capacity drains
    away at a fraction k per time step.

    D_t = k * max(0, S_t - field_capacity)
    """
    return k * max(0.0, S - field_capacity)


def water_balance_step(S, R, I, ET, D):
    """
    Single discrete water balance update.

    S_(t+1) = S_t + R_t + I_t - ET_t - D_t
    """
    return S + R + I - ET - D


def water_deficit(S, target):
    """
    Water deficit relative to a target soil moisture level.
    Deficit is zero or positive (never negative -- a surplus is not a deficit).
    """
    return max(0.0, target - S)


def irrigation_required(S_current, target, ET_forecast=0.0, R_forecast=0.0):
    """
    Closed-form irrigation amount needed to reach a target moisture level
    after accounting for forecast rainfall and evapotranspiration.
    Used as the analytic check for the root-finding methods in Level 3.
    """
    return max(0.0, target - S_current + ET_forecast - R_forecast)