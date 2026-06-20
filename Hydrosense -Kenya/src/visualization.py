"""

Reusable Matplotlib plotting functions for HydroSense-Kenya scientific figures.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_timeseries(dates, values, ylabel, title, ax=None, color="tab:blue"):
    """Generic labelled time-series line plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, values, color=color, marker="o", markersize=3)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax


def plot_zone_comparison(df, date_col, value_col, zone_col, ylabel, title):
    """Plot a value over time, one line per farm zone."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for zone, sub in df.groupby(zone_col):
        ax.plot(sub[date_col], sub[value_col], marker="o", markersize=3, label=zone)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax


def plot_monte_carlo_fan(simulated_moisture, min_threshold, title="Monte Carlo Soil Moisture Scenarios"):
    """
    Fan chart of Monte Carlo simulated soil moisture: median plus a
    5th-95th percentile band, with the minimum-moisture threshold marked.
    """
    days = np.arange(simulated_moisture.shape[1])
    p5 = np.percentile(simulated_moisture, 5, axis=0)
    p50 = np.percentile(simulated_moisture, 50, axis=0)
    p95 = np.percentile(simulated_moisture, 95, axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(days, p5, p95, color="tab:blue", alpha=0.25, label="5th-95th percentile")
    ax.plot(days, p50, color="tab:blue", label="Median")
    ax.axhline(min_threshold, color="red", linestyle="--", label="Min moisture threshold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Soil moisture (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax


def plot_irrigation_schedule(days, irrigation, soil_moisture, min_threshold, title="Optimized Irrigation Schedule"):
    """Two-panel figure: irrigation bars and resulting soil moisture trajectory."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.bar(days, irrigation, color="tab:green")
    ax1.set_ylabel("Irrigation (mm)")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)

    ax2.plot(range(len(soil_moisture)), soil_moisture, color="tab:blue", marker="o", markersize=3)
    ax2.axhline(min_threshold, color="red", linestyle="--", label="Min moisture threshold")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Soil moisture (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_convergence_comparison(histories, labels, title="Root-Finding Convergence Comparison"):
    """Plot |f(x)| vs iteration number for several root-finding methods."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for hist, label in zip(histories, labels):
        iters = [h[0] for h in hist]
        errs = [abs(h[2]) for h in hist]
        ax.semilogy(iters, errs, marker="o", markersize=3, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|f(x)| (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    return ax