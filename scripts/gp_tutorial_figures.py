#!/usr/bin/env python
"""Generate figures used by the GP tutorial."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "runs" / "gp_surrogate" / "tutorial_figures"


def configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_1d_gp() -> None:
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(2)
    x_train = np.array([0.0, 1.0, 2.4, 3.7, 5.1, 6.0]).reshape(-1, 1)
    y_train = np.sin(1.6 * x_train).ravel() + rng.normal(0.0, 0.12, size=x_train.shape[0])

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = x_scaler.fit_transform(x_train)
    y = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        length_scale=0.8,
        length_scale_bounds="fixed",
        nu=2.5,
    ) + WhiteKernel(noise_level=0.02, noise_level_bounds="fixed")
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, random_state=4)
    gp.fit(X, y)

    x_grid = np.linspace(-0.5, 6.5, 300).reshape(-1, 1)
    X_grid = x_scaler.transform(x_grid)
    mean_s, std_s = gp.predict(X_grid, return_std=True)
    mean = y_scaler.inverse_transform(mean_s.reshape(-1, 1)).ravel()
    std = std_s * y_scaler.scale_[0]

    plt.figure(figsize=(8.0, 4.8))
    plt.fill_between(
        x_grid.ravel(),
        mean - 2.0 * std,
        mean + 2.0 * std,
        color="#90b7df",
        alpha=0.35,
        label="95% predictive band",
    )
    plt.plot(x_grid, mean, color="#225ea8", lw=2.2, label="GP posterior mean")
    plt.scatter(x_train, y_train, color="#111111", s=48, zorder=3, label="observations")
    plt.xlabel("input x")
    plt.ylabel("output y")
    plt.title("1D Gaussian Process Regression")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gp_1d_regression.png", dpi=180)
    plt.close()


def plot_2d_surface() -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(7)
    low = np.array([300.0, 1.0])
    high = np.array([1200.0, 5.0])
    sampler = qmc.LatinHypercube(d=2, seed=7)
    X_design = qmc.scale(sampler.random(n=45), low, high)

    def simulator(x: np.ndarray) -> float:
        temperature, pressure = x
        return (
            np.sin(temperature / 145.0)
            + 0.35 * pressure
            - 0.0000022 * (temperature - 820.0) ** 2
        )

    y = np.array([simulator(x) for x in X_design]) + rng.normal(0.0, 0.03, size=len(X_design))

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X_design)
    ys = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        length_scale=np.array([0.9, 0.9]),
        length_scale_bounds="fixed",
        nu=2.5,
    ) + WhiteKernel(noise_level=0.002, noise_level_bounds="fixed")
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, random_state=8)
    gp.fit(Xs, ys)

    temp = np.linspace(low[0], high[0], 90)
    pressure = np.linspace(low[1], high[1], 80)
    tt, pp = np.meshgrid(temp, pressure)
    X_grid = np.column_stack([tt.ravel(), pp.ravel()])
    mean_s, std_s = gp.predict(x_scaler.transform(X_grid), return_std=True)
    mean = y_scaler.inverse_transform(mean_s.reshape(-1, 1)).reshape(tt.shape)
    std = (std_s * y_scaler.scale_[0]).reshape(tt.shape)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    im0 = axes[0].contourf(tt, pp, mean, levels=18, cmap="viridis")
    axes[0].scatter(X_design[:, 0], X_design[:, 1], s=18, color="white", edgecolor="black", linewidth=0.4)
    axes[0].set_title("GP mean prediction")
    axes[0].set_xlabel("temperature [K]")
    axes[0].set_ylabel("pressure [bar]")
    fig.colorbar(im0, ax=axes[0], label="predicted yield")

    im1 = axes[1].contourf(tt, pp, std, levels=18, cmap="magma")
    axes[1].scatter(X_design[:, 0], X_design[:, 1], s=18, color="white", edgecolor="black", linewidth=0.4)
    axes[1].set_title("GP predictive uncertainty")
    axes[1].set_xlabel("temperature [K]")
    axes[1].set_ylabel("pressure [bar]")
    fig.colorbar(im1, ax=axes[1], label="standard deviation")

    plt.savefig(OUTPUT_DIR / "gp_2d_surface_uncertainty.png", dpi=180)
    plt.close()


def plot_uncertainty_propagation() -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(11)
    low = np.array([300.0, 1.0])
    high = np.array([1200.0, 5.0])
    sampler = qmc.LatinHypercube(d=2, seed=11)
    X_design = qmc.scale(sampler.random(n=50), low, high)

    def simulator(x: np.ndarray) -> float:
        temperature, pressure = x
        return (
            np.sin(temperature / 160.0)
            + 0.28 * pressure
            - 0.0000018 * (temperature - 760.0) ** 2
        )

    y = np.array([simulator(x) for x in X_design])
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X_design)
    ys = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        length_scale=np.array([0.9, 0.9]),
        length_scale_bounds="fixed",
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, random_state=12)
    gp.fit(Xs, ys)

    temp = np.clip(rng.normal(800.0, 55.0, size=6000), low[0], high[0])
    pressure = np.clip(rng.normal(3.0, 0.55, size=6000), low[1], high[1])
    X_mc = np.column_stack([temp, pressure])
    mean_s, std_s = gp.predict(x_scaler.transform(X_mc), return_std=True)
    mean = y_scaler.inverse_transform(mean_s.reshape(-1, 1)).ravel()
    std = std_s * y_scaler.scale_[0]
    draws = rng.normal(mean, std)

    p05, p50, p95 = np.percentile(draws, [5, 50, 95])
    plt.figure(figsize=(8.0, 4.6))
    plt.hist(draws, bins=45, color="#4c78a8", alpha=0.78, edgecolor="white")
    for value, label, color in [(p05, "5%", "#d73027"), (p50, "median", "#111111"), (p95, "95%", "#d73027")]:
        plt.axvline(value, color=color, lw=2.0, linestyle="--" if label != "median" else "-")
        plt.text(value, plt.ylim()[1] * 0.92, label, rotation=90, va="top", ha="right")
    plt.xlabel("predicted yield")
    plt.ylabel("Monte Carlo samples")
    plt.title("Uncertainty Propagation Through A GP Surrogate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gp_uncertainty_propagation.png", dpi=180)
    plt.close()


def main() -> int:
    configure_matplotlib()
    plot_1d_gp()
    plot_2d_surface()
    plot_uncertainty_propagation()
    print(f"Wrote tutorial figures to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
