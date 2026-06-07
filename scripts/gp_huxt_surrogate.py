#!/usr/bin/env python
"""Gaussian-process surrogate workflow for HUXt CME perturbation studies.

Typical command sequence:

    python scripts/gp_huxt_surrogate.py --event all design --n 300
    python scripts/gp_huxt_surrogate.py --event 2017-09-06 plot-design --limit 50
    python scripts/gp_huxt_surrogate.py --event 2017-09-06 run --limit 20
    python scripts/gp_huxt_surrogate.py --event 2017-09-06 visualize-threshold --max-samples 6
    python scripts/gp_huxt_surrogate.py --event 2017-09-06 fit
    python scripts/gp_huxt_surrogate.py --event 2017-09-06 analyze

The `run` step replays HUXt for each sampled Cone-CME parameter vector and
extracts scalar outcomes for GP training. The default arrival detector is
`--detector-method hybrid`, which requires HUXt's ConeCME front-arrival
geometry at Earth and a speed enhancement in the Earth time series. `front`
uses only the geometric front crossing, `enhancement` uses only the speed
signature, and `jump` keeps the older short-lag fractional speed-jump detector.

The `analyze` step regenerates the GP figures under
`runs/gp_surrogate/<event>/figures/`, including hit-probability slices,
local sensitivity, permutation importance, and posterior arrival-time plots.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time


PARAM_NAMES = ["inject_hour", "longitude", "latitude", "width", "v"]
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = BASE_DIR / "data_dir" / "sw"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "runs" / "gp_surrogate"
CACHE_DIR = BASE_DIR / ".cache"

PRIOR_LOW = np.array([0.0, -90.0, -50.0, 0.0, 100.0], dtype=float)
PRIOR_HIGH = np.array([10.0, 90.0, 50.0, 180.0, 2000.0], dtype=float)
DEFAULT_ABS_SPAN = np.array([1.0, 30.0, 20.0, 40.0, np.nan], dtype=float)
DEFAULT_OBS_SIGMA = np.array([0.5, 10.0, 10.0, 15.0, np.nan], dtype=float)

# Physical window for arrival-time figures. Arrival-time color scales and the posterior
# histogram are clamped to this range; values outside saturate to the end colors. This keeps
# the GP's unphysical extrapolations (in regions the CME barely reaches) from dominating.
ARRIVAL_MIN_HR = 24.0
ARRIVAL_MAX_HR = 96.0


class WorkflowStateError(RuntimeError):
    """Raised when a GP workflow command is run before prerequisite files exist."""


def configure_local_caches() -> None:
    """Keep plotting and SunPy caches inside the project workspace."""
    local_sunpy_config = CACHE_DIR / "sunpy"
    local_mpl_config = CACHE_DIR / "matplotlib"
    local_sunpy_config.mkdir(parents=True, exist_ok=True)
    local_mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SUNPY_CONFIGDIR", str(local_sunpy_config))
    os.environ.setdefault("MPLCONFIGDIR", str(local_mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))


def available_events(data_root: Path) -> list[str]:
    data_root = Path(data_root)
    if not data_root.exists():
        return []
    return sorted(path.name for path in data_root.iterdir() if (path / "event_config.yaml").exists())


def event_list(value: str, data_root: Path) -> list[str]:
    if value == "all":
        events = available_events(data_root)
        if not events:
            raise FileNotFoundError(f"No prepared events found under {data_root}")
        return events
    return [value]


def event_input_dir(event: str, data_root: Path) -> Path:
    return data_root / event


def event_output_dir(event: str, output_root: Path) -> Path:
    return output_root / event


def load_event_config(event: str, data_root: Path) -> dict:
    config_path = event_input_dir(event, data_root) / "event_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing event config: {config_path}")
    with config_path.open("r") as stream:
        config = yaml.safe_load(stream)
    config["initial_theta"] = [float(x) for x in config["initial_theta"]]
    config["cr_num"] = float(config["cr_num"])
    return config


def default_bounds(theta0: np.ndarray, span: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if span is None:
        span = DEFAULT_ABS_SPAN.copy()
        span[4] = 0.25 * float(theta0[4])
    low = np.maximum(PRIOR_LOW, theta0 - span)
    high = np.minimum(PRIOR_HIGH, theta0 + span)
    return low, high


def design_span(
    theta0: np.ndarray,
    inject_hour: float,
    longitude: float,
    latitude: float,
    width: float,
    speed_fraction: float,
) -> np.ndarray:
    return np.array(
        [
            float(inject_hour),
            float(longitude),
            float(latitude),
            float(width),
            float(speed_fraction) * float(theta0[4]),
        ],
        dtype=float,
    )


def default_obs_sigma(theta0: np.ndarray) -> np.ndarray:
    sigma = DEFAULT_OBS_SIGMA.copy()
    sigma[4] = 0.10 * float(theta0[4])
    return sigma


def latin_hypercube(n: int, ndim: int, seed: int) -> np.ndarray:
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=ndim, seed=seed)
        return sampler.random(n)
    except Exception:
        rng = np.random.default_rng(seed)
        sample = np.empty((n, ndim), dtype=float)
        for dim in range(ndim):
            sample[:, dim] = (rng.permutation(n) + rng.random(n)) / n
        return sample


def make_design(
    event: str,
    data_root: Path,
    output_root: Path,
    n: int,
    seed: int,
    force: bool,
    span_inject_hour: float,
    span_longitude: float,
    span_latitude: float,
    span_width: float,
    span_speed_fraction: float,
) -> Path:
    config = load_event_config(event, data_root)
    theta0 = np.array(config["initial_theta"], dtype=float)
    span = design_span(theta0, span_inject_hour, span_longitude, span_latitude, span_width, span_speed_fraction)
    low, high = default_bounds(theta0, span)

    outdir = event_output_dir(event, output_root)
    outdir.mkdir(parents=True, exist_ok=True)
    design_path = outdir / "design.csv"
    if design_path.exists() and not force:
        raise FileExistsError(f"{design_path} exists; pass --force to overwrite")

    n_random = max(0, n - 1)
    unit = latin_hypercube(n_random, len(PARAM_NAMES), seed) if n_random else np.empty((0, len(PARAM_NAMES)))
    theta = low + unit * (high - low)
    theta = np.vstack([theta0, theta])

    frame = pd.DataFrame(theta, columns=PARAM_NAMES)
    frame.insert(0, "sample_id", np.arange(len(frame), dtype=int))
    frame.insert(1, "event", event)
    frame["status"] = "pending"
    frame.loc[0, "status"] = "seed"
    frame.to_csv(design_path, index=False)

    metadata = {
        "event": event,
        "seed": int(seed),
        "n": int(len(frame)),
        "theta0": theta0.tolist(),
        "span": span.tolist(),
        "bounds_low": low.tolist(),
        "bounds_high": high.tolist(),
        "param_names": PARAM_NAMES,
    }
    with (outdir / "design_meta.yaml").open("w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return design_path


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average of values using a uniform kernel.
    
    Args:
        values: Input array of values to smooth.
        window: Size of the moving window. If <= 1, returns a copy of input.
    
    Returns:
        Smoothed array with same length as input, using edge padding.
    """
    if window <= 1:
        return values.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values.astype(float), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def time_to_hours(times: object) -> np.ndarray:
    """Convert time array to hours since the first time point.
    
    Args:
        times: Time array (converted to astropy.time.Time).
    
    Returns:
        Array of hours elapsed since the first time point.
    """
    astropy_time = Time(times)
    unix = astropy_time.unix
    return (unix - unix[0]) / 3600.0


def detect_arrival(
    huxt_time: object,
    huxt_speed: object,
    threshold: float = 0.25,
    lag: int = 2,
    smooth_window: int = 5,
    method: str = "enhancement",
    baseline_window: int = 24,
) -> dict:
    """Detect a CME arrival from either short-lag jump or total enhancement.
    
    This function applies one of two detection methods to identify a CME arrival:
    - "jump": Detects a short-lag fractional jump in smoothed speed. Calculates the fractional 
      change between the current smoothed speed value and the speed value `lag` steps back, 
      identifying rapid accelerations that may indicate the leading edge of a CME shock.
    - "enhancement": Detects total fractional enhancement from baseline. Compares the current 
      smoothed speed value against the minimum smoothed speed over the preceding `baseline_window` 
      hours, calculating the fractional increase from that baseline. This method identifies 
      sustained speed increases relative to quiet-time wind speeds.
    
    Args:
        huxt_time: Time array (converted to astropy.time.Time).
        huxt_speed: Solar wind speed array (m/s or km/s).
        threshold: Minimum fractional change to trigger detection (default 0.25).
        lag: Number of time steps for jump calculation (default 2).
        smooth_window: Window size for moving average smoothing (default 5).
        method: Detection method - "jump", "enhancement", "front", or "hybrid" (default "enhancement").
        baseline_window: Time window (hours) for baseline calculation (default 24).
    
    Returns:
        Dictionary containing:
            - hit: Boolean indicating if CME was detected.
            - arrival_idx: Index of detected arrival, or None if not detected.
            - arrival_time_hr: Time in hours from start to arrival.
            - max_speed_jump: Maximum fractional jump value.
            - max_speed_enhancement: Maximum fractional enhancement value.
            - max_detector_value: Maximum detector value used.
            - detector_method: Method used for detection.
            - peak_vsw: Peak solar wind speed in the array.
    """
    speed = np.asarray(huxt_speed, dtype=float)
    hours = time_to_hours(huxt_time)
    if speed.size == 0 or speed.size != hours.size:
        raise ValueError("huxt_time and huxt_speed must be non-empty arrays with matching length")

    smooth = moving_average(speed, smooth_window)
    jump = np.full(speed.shape, np.nan, dtype=float)
    for idx in range(lag, speed.size):
        denom = smooth[idx - lag]
        if np.isfinite(denom) and abs(denom) > 1e-12:
            jump[idx] = (smooth[idx] - denom) / denom

    enhancement = np.full(speed.shape, np.nan, dtype=float)
    for idx in range(1, speed.size):
        start = max(0, idx - baseline_window)
        baseline = np.nanmin(smooth[start:idx])
        if np.isfinite(baseline) and abs(baseline) > 1e-12:
            enhancement[idx] = (smooth[idx] - baseline) / baseline

    if method == "jump":
        detector = jump
    elif method in {"enhancement", "front", "hybrid"}:
        detector = enhancement
    else:
        raise ValueError(f"Unknown detector method: {method}")

    candidate = np.where(detector >= threshold)[0]
    hit = candidate.size > 0
    arrival_idx = int(candidate[0]) if hit else None
    return {
        "hit": bool(hit),
        "arrival_idx": arrival_idx,
        "arrival_time_hr": float(hours[arrival_idx]) if hit else math.nan,
        "max_speed_jump": float(np.nanmax(jump)) if np.isfinite(jump).any() else math.nan,
        "max_speed_enhancement": float(np.nanmax(enhancement)) if np.isfinite(enhancement).any() else math.nan,
        "max_detector_value": float(np.nanmax(detector)) if np.isfinite(detector).any() else math.nan,
        "detector_method": method,
        "peak_vsw": float(np.nanmax(speed)) if np.isfinite(speed).any() else math.nan,
    }


def arrival_diagnostics(
    huxt_time: object,
    huxt_speed: object,
    threshold: float = 0.25,
    lag: int = 2,
    smooth_window: int = 5,
    method: str = "enhancement",
    baseline_window: int = 24,
) -> dict:
    """Return the detector internals used for threshold diagnostics."""
    speed = np.asarray(huxt_speed, dtype=float)
    hours = time_to_hours(huxt_time)
    if speed.size == 0 or speed.size != hours.size:
        raise ValueError("huxt_time and huxt_speed must be non-empty arrays with matching length")

    smooth = moving_average(speed, smooth_window)
    jump = np.full(speed.shape, np.nan, dtype=float)
    for idx in range(lag, speed.size):
        denom = smooth[idx - lag]
        if np.isfinite(denom) and abs(denom) > 1e-12:
            jump[idx] = (smooth[idx] - denom) / denom

    enhancement = np.full(speed.shape, np.nan, dtype=float)
    for idx in range(1, speed.size):
        start = max(0, idx - baseline_window)
        baseline = np.nanmin(smooth[start:idx])
        if np.isfinite(baseline) and abs(baseline) > 1e-12:
            enhancement[idx] = (smooth[idx] - baseline) / baseline

    if method == "jump":
        detector = jump
    elif method in {"enhancement", "front", "hybrid"}:
        detector = enhancement
    else:
        raise ValueError(f"Unknown detector method: {method}")

    finite = np.isfinite(detector)
    max_idx = int(np.nanargmax(detector)) if finite.any() else None
    metrics = detect_arrival(
        huxt_time,
        huxt_speed,
        threshold=threshold,
        lag=lag,
        smooth_window=smooth_window,
        method=method,
        baseline_window=baseline_window,
    )
    metrics.update(
        {
            "hours": hours,
            "speed": speed,
            "smooth_speed": smooth,
            "jump": jump,
            "enhancement": enhancement,
            "detector": detector,
            "max_jump_idx": max_idx,
            "threshold": float(threshold),
        }
    )
    return metrics


def load_huxt_context(event: str, data_root: Path) -> dict:
    import astropy.units as u

    config = load_event_config(event, data_root)
    event_dir = event_input_dir(event, data_root)
    boundary_path = event_dir / f"v_boundary_{event}.npz"
    if not boundary_path.exists():
        raise FileNotFoundError(f"Missing HUXt boundary file: {boundary_path}")
    boundary = np.load(boundary_path)["speed_map"] * (u.km / u.s)
    return {
        "theta0": np.array(config["initial_theta"], dtype=float),
        "cr_num": float(config["cr_num"]),
        "huxt_kwargs": {
            "v_boundary": boundary,
            "latitude": 0 * u.deg,
            "cr_num": float(config["cr_num"]),
            "frame": "sidereal",
            "simtime": 10 * u.day,
            "dt_scale": 4,
        },
    }


def front_arrival_metrics(cme: object, huxt_time: object) -> dict:
    """Return HUXt ConeCME front-arrival metrics at Earth."""
    stats = cme.compute_arrival_at_body("EARTH")
    hit = bool(stats.get("hit", False))
    start = Time(huxt_time[0])
    if hit:
        arrival_time_hr = float(((stats["t_arrive"] - start).to("hour")).value)
        arrival_idx = int(stats["hit_id"])
        arrival_v = float(stats["v"].to("km/s").value)
    else:
        arrival_time_hr = math.nan
        arrival_idx = math.nan
        arrival_v = math.nan
    return {
        "front_hit": hit,
        "front_arrival_idx": arrival_idx,
        "front_arrival_time_hr": arrival_time_hr,
        "front_arrival_v": arrival_v,
    }


def load_completed_results(results_path: Path) -> pd.DataFrame:
    if results_path.exists():
        return pd.read_csv(results_path)
    return pd.DataFrame()


def write_result(results_path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing.loc[existing["sample_id"].astype(int) != int(row["sample_id"])]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.sort_values("sample_id")
    frame.to_csv(results_path, index=False)


def run_design(
    event: str,
    data_root: Path,
    output_root: Path,
    limit: int | None,
    detector_threshold: float,
    detector_lag: int,
    smooth_window: int,
    detector_method: str,
    baseline_window: int,
    rerun_completed: bool,
) -> Path:
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    import run_huxt_functions as rhf

    outdir = event_output_dir(event, output_root)
    design_path = outdir / "design.csv"
    results_path = outdir / "results.csv"
    if not design_path.exists():
        raise FileNotFoundError(f"Missing design file: {design_path}")

    design = pd.read_csv(design_path)
    results = load_completed_results(results_path)
    completed_ids = set()
    if not results.empty and "status" in results:
        completed_ids = set(results.loc[results["status"] == "completed", "sample_id"].astype(int))

    context = load_huxt_context(event, data_root)
    if rerun_completed:
        todo = design.copy()
    else:
        todo = design.loc[~design["sample_id"].astype(int).isin(completed_ids)].copy()
    if limit is not None:
        todo = todo.head(limit)

    for _, design_row in todo.iterrows():
        sample_id = int(design_row["sample_id"])
        theta = np.array([design_row[name] for name in PARAM_NAMES], dtype=float)
        started = time.time()
        result = {
            "sample_id": sample_id,
            "event": event,
            **{name: float(value) for name, value in zip(PARAM_NAMES, theta)},
        }
        try:
            cme = None
            if detector_method in {"front", "hybrid"}:
                _, huxt_time, huxt_speed, cme = run_huxt_model(theta, context["huxt_kwargs"])
                starttime = huxt_time[0]
                endtime = huxt_time[len(huxt_time) - 1]
            else:
                huxt_time, huxt_speed, starttime, endtime = rhf.run_huxt_sim(theta, context["huxt_kwargs"])
            metrics = detect_arrival(
                huxt_time,
                huxt_speed,
                threshold=detector_threshold,
                lag=detector_lag,
                smooth_window=smooth_window,
                method=detector_method,
                baseline_window=baseline_window,
            )
            if cme is not None:
                front_metrics = front_arrival_metrics(cme, huxt_time)
                metrics.update(front_metrics)
                if detector_method == "front":
                    metrics["hit"] = front_metrics["front_hit"]
                    metrics["arrival_idx"] = front_metrics["front_arrival_idx"]
                    metrics["arrival_time_hr"] = front_metrics["front_arrival_time_hr"]
                    metrics["max_detector_value"] = 1.0 if front_metrics["front_hit"] else 0.0
                elif detector_method == "hybrid":
                    hit = bool(front_metrics["front_hit"] and metrics["hit"])
                    metrics["hit"] = hit
                    if hit:
                        metrics["arrival_idx"] = front_metrics["front_arrival_idx"]
                        metrics["arrival_time_hr"] = front_metrics["front_arrival_time_hr"]
                    else:
                        metrics["arrival_idx"] = math.nan
                        metrics["arrival_time_hr"] = math.nan
            result.update(metrics)
            result["start_time"] = str(starttime)
            result["end_time"] = str(endtime)
            result["runtime_s"] = float(time.time() - started)
            result["status"] = "completed"
            result["error"] = ""
            design.loc[design["sample_id"] == sample_id, "status"] = "completed"
        except Exception as exc:
            result["hit"] = False
            result["arrival_time_hr"] = math.nan
            result["arrival_idx"] = math.nan
            result["max_speed_jump"] = math.nan
            result["max_speed_enhancement"] = math.nan
            result["max_detector_value"] = math.nan
            result["detector_method"] = detector_method
            result["front_hit"] = False
            result["front_arrival_idx"] = math.nan
            result["front_arrival_time_hr"] = math.nan
            result["front_arrival_v"] = math.nan
            result["peak_vsw"] = math.nan
            result["runtime_s"] = float(time.time() - started)
            result["status"] = "failed"
            result["error"] = repr(exc)
            design.loc[design["sample_id"] == sample_id, "status"] = "failed"
        write_result(results_path, result)
        design.to_csv(design_path, index=False)
        print(f"{event} sample {sample_id}: {result['status']}")

    return results_path


def parse_sample_ids(value: str | None) -> set[int] | None:
    if value is None or str(value).strip() == "":
        return None
    sample_ids = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        sample_ids.add(int(part))
    return sample_ids


def run_huxt_model(theta: np.ndarray, huxt_kwargs: dict) -> tuple[object, object, object, object]:
    configure_local_caches()

    import astropy.units as u
    import huxt.huxt as H
    import huxt.huxt_analysis as HA

    inject_hour, longitude, latitude, width, v = theta
    model = H.HUXt(**huxt_kwargs)
    cme = H.ConeCME(
        t_launch=float(inject_hour) * u.hour,
        longitude=float(longitude) * u.deg,
        latitude=float(latitude) * u.deg,
        width=float(width) * u.deg,
        v=float(v) * (u.km / u.s),
        thickness=5 * u.solRad,
        cme_expansion=False,
        cme_fixed_duration=False,
    )
    model.solve([cme])
    huxt_ts = HA.get_observer_timeseries(model, observer="Earth")
    tracked_cme = model.cmes[0] if getattr(model, "cmes", None) else cme
    return model, huxt_ts["time"], huxt_ts["vsw"], tracked_cme


def plot_threshold_timeseries(row: pd.Series, diagnostics: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    hours = diagnostics["hours"]
    speed = diagnostics["speed"]
    smooth = diagnostics["smooth_speed"]
    jump = diagnostics["jump"]
    detector = diagnostics["detector"]
    threshold = diagnostics["threshold"]
    detector_method = diagnostics["detector_method"]
    max_idx = diagnostics["max_jump_idx"]
    arrival_idx = diagnostics["arrival_idx"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(hours, speed, color="0.65", linewidth=1.2, label="Earth Vsw")
    axes[0].plot(hours, smooth, color="tab:blue", linewidth=1.8, label="Smoothed Vsw")
    if max_idx is not None:
        axes[0].axvline(hours[max_idx], color="tab:orange", linestyle="--", linewidth=1.4, label="Max jump")
    if arrival_idx is not None:
        axes[0].axvline(hours[arrival_idx], color="tab:green", linestyle="-", linewidth=1.4, label="Detected arrival")
    axes[0].set_ylabel("Vsw [km/s]")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(hours, detector, color="tab:red", linewidth=1.5, label=f"{detector_method} detector")
    axes[1].axhline(threshold, color="k", linestyle="--", linewidth=1.2, label=f"Threshold {threshold:g}")
    if max_idx is not None:
        axes[1].plot(hours[max_idx], detector[max_idx], "o", color="tab:orange")
    axes[1].set_xlabel("Hours from model start")
    axes[1].set_ylabel("Detector value")
    axes[1].legend(loc="best", fontsize=9)

    title = (
        f"sample {int(row['sample_id'])}: hit={diagnostics['hit']}, "
        f"max {detector_method}={diagnostics['max_detector_value']:.3f}, "
        f"lon={row['longitude']:.1f}, lat={row['latitude']:.1f}, "
        f"width={row['width']:.1f}, v={row['v']:.0f}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_heliosphere_snapshot(model: object, row: pd.Series, diagnostics: dict, output_path: Path) -> None:
    import astropy.units as u
    import matplotlib.pyplot as plt
    import huxt.huxt_analysis as HA

    max_idx = diagnostics["max_jump_idx"]
    if max_idx is None:
        snapshot_time = 5 * u.day
        label = "fallback 5.00 days"
    else:
        snapshot_time = float(diagnostics["hours"][max_idx]) * u.hour
        label = f"max jump at {diagnostics['hours'][max_idx]:.2f} hr"

    fig, _ = HA.plot(model, snapshot_time, plotHCS=False, trace_earth_connection=False)
    fig.suptitle(
        f"sample {int(row['sample_id'])}: {label}, "
        f"max {diagnostics['detector_method']}={diagnostics['max_detector_value']:.3f}",
        y=0.96,
        fontsize=11,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def visualize_threshold_cases(
    event: str,
    data_root: Path,
    output_root: Path,
    detector_threshold: float,
    detector_lag: int,
    smooth_window: int,
    detector_method: str,
    baseline_window: int,
    sample_ids: str | None,
    max_samples: int,
    include_hits: bool,
) -> Path:
    outdir = event_output_dir(event, output_root)
    results_path = outdir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    results = valid_completed_results(results_path)
    requested_ids = parse_sample_ids(sample_ids)
    if requested_ids is not None:
        selected = results.loc[results["sample_id"].astype(int).isin(requested_ids)].copy()
    else:
        selected = results.copy()
        metric_col = "max_speed_enhancement" if detector_method == "enhancement" else "max_speed_jump"
        if metric_col in selected:
            selected = selected.loc[selected[metric_col] < detector_threshold]
        if not include_hits and "hit" in selected:
            selected = selected.loc[~selected["hit"].map(parse_bool)]
        sort_col = metric_col if metric_col in selected else "max_speed_jump"
        selected = selected.sort_values(sort_col, ascending=False).head(max_samples)

    if selected.empty:
        raise ValueError("No completed samples matched the visualization filters")

    context = load_huxt_context(event, data_root)
    figdir = outdir / "figures" / "threshold_diagnostics"
    figdir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for _, row in selected.iterrows():
        sample_id = int(row["sample_id"])
        theta = np.array([row[name] for name in PARAM_NAMES], dtype=float)
        model, huxt_time, huxt_speed, cme = run_huxt_model(theta, context["huxt_kwargs"])
        diagnostics = arrival_diagnostics(
            huxt_time,
            huxt_speed,
            threshold=detector_threshold,
            lag=detector_lag,
            smooth_window=smooth_window,
            method=detector_method,
            baseline_window=baseline_window,
        )
        if detector_method in {"front", "hybrid"}:
            front_metrics = front_arrival_metrics(cme, huxt_time)
            diagnostics.update(front_metrics)
            if detector_method == "front":
                diagnostics["hit"] = front_metrics["front_hit"]
                diagnostics["arrival_idx"] = front_metrics["front_arrival_idx"]
                diagnostics["arrival_time_hr"] = front_metrics["front_arrival_time_hr"]
                diagnostics["max_detector_value"] = 1.0 if front_metrics["front_hit"] else 0.0
            elif detector_method == "hybrid":
                hit = bool(front_metrics["front_hit"] and diagnostics["hit"])
                diagnostics["hit"] = hit
                if hit:
                    diagnostics["arrival_idx"] = front_metrics["front_arrival_idx"]
                    diagnostics["arrival_time_hr"] = front_metrics["front_arrival_time_hr"]
                else:
                    diagnostics["arrival_idx"] = None
                    diagnostics["arrival_time_hr"] = math.nan

        prefix = f"{event}_sample_{sample_id:03d}"
        timeseries_path = figdir / f"{prefix}_timeseries.png"
        heliosphere_path = figdir / f"{prefix}_heliosphere.png"
        plot_threshold_timeseries(row, diagnostics, timeseries_path)
        plot_heliosphere_snapshot(model, row, diagnostics, heliosphere_path)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "hit_at_threshold": diagnostics["hit"],
                "arrival_time_hr": diagnostics["arrival_time_hr"],
                "max_speed_jump": diagnostics["max_speed_jump"],
                "max_speed_enhancement": diagnostics["max_speed_enhancement"],
                "max_detector_value": diagnostics["max_detector_value"],
                "detector_method": diagnostics["detector_method"],
                "front_hit": diagnostics.get("front_hit", math.nan),
                "front_arrival_time_hr": diagnostics.get("front_arrival_time_hr", math.nan),
                "front_arrival_v": diagnostics.get("front_arrival_v", math.nan),
                "peak_vsw": diagnostics["peak_vsw"],
                "timeseries_figure": str(timeseries_path),
                "heliosphere_figure": str(heliosphere_path),
            }
        )
        print(f"Wrote diagnostics for {event} sample {sample_id}")

    pd.DataFrame(manifest_rows).to_csv(figdir / "manifest.csv", index=False)
    return figdir


def valid_completed_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        event_dir = results_path.parent
        design_path = event_dir / "design.csv"
        if design_path.exists():
            next_step = (
                f"Run HUXt samples first:\n"
                f"  python scripts/gp_huxt_surrogate.py --event {event_dir.name} run --limit 20\n"
                f"Then retry:\n"
                f"  python scripts/gp_huxt_surrogate.py --event {event_dir.name} fit"
            )
        else:
            next_step = (
                f"Create a design and run HUXt samples first:\n"
                f"  python scripts/gp_huxt_surrogate.py --event {event_dir.name} design --n 300\n"
                f"  python scripts/gp_huxt_surrogate.py --event {event_dir.name} run --limit 20\n"
                f"Then retry:\n"
                f"  python scripts/gp_huxt_surrogate.py --event {event_dir.name} fit"
            )
        raise WorkflowStateError(
            f"Missing results file: {results_path}\n\n"
            f"The GP fit step needs completed HUXt samples in results.csv.\n"
            f"{next_step}"
        )
    results = pd.read_csv(results_path)
    results = results.loc[results["status"] == "completed"].copy()
    if results.empty:
        raise ValueError(f"No completed HUXt samples in {results_path}")
    results["hit"] = results["hit"].map(parse_bool)
    return results


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def unique_warning_messages(messages: list[str]) -> list[str]:
    unique = []
    for message in messages:
        if message not in unique:
            unique.append(message)
    return unique


def train_models(event: str, output_root: Path, test_fraction: float, random_state: int) -> dict:
    from joblib import dump
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import accuracy_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    def fit_with_warnings(model, x, y) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        return [
            str(warning.message)
            for warning in caught
            if issubclass(warning.category, ConvergenceWarning)
        ]

    outdir = event_output_dir(event, output_root)
    results = valid_completed_results(outdir / "results.csv")
    x = results[PARAM_NAMES].to_numpy(dtype=float)
    hit = results["hit"].astype(int).to_numpy()
    convergence_warnings = []

    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)

    hit_kernel = ConstantKernel(1.0, (1e-3, 1e5)) * Matern(
        length_scale=np.ones(len(PARAM_NAMES)),
        length_scale_bounds=(1e-2, 1e5),
        nu=2.5,
    )
    hit_model = GaussianProcessClassifier(kernel=hit_kernel, random_state=random_state, max_iter_predict=100)
    if np.unique(hit).size >= 2:
        convergence_warnings.extend(fit_with_warnings(hit_model, x_scaled, hit))
        hit_pred = hit_model.predict(x_scaled)
        hit_accuracy = float(accuracy_score(hit, hit_pred))
    else:
        hit_model = None
        hit_accuracy = math.nan

    hit_rows = results.loc[results["hit"] & np.isfinite(results["arrival_time_hr"])].copy()
    arrival_warning = None
    arrival_mae = math.nan
    arrival_model = None
    y_scaler = None
    if len(hit_rows) >= 2:
        x_hit = hit_rows[PARAM_NAMES].to_numpy(dtype=float)
        y_hit = hit_rows[["arrival_time_hr"]].to_numpy(dtype=float)
        x_hit_scaled = x_scaler.transform(x_hit)
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y_hit).ravel()

        kernel = ConstantKernel(1.0, (1e-3, 1e5)) * Matern(
            length_scale=np.ones(len(PARAM_NAMES)),
            length_scale_bounds=(1e-2, 1e4),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
        arrival_model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=3,
            random_state=random_state,
        )

        if len(hit_rows) >= 10:
            x_train, x_test, y_train, y_test = train_test_split(
                x_hit_scaled,
                y_scaled,
                test_size=test_fraction,
                random_state=random_state,
            )
            convergence_warnings.extend(fit_with_warnings(arrival_model, x_train, y_train))
            y_pred = arrival_model.predict(x_test)
            y_pred_hr = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_test_hr = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
            arrival_mae = float(mean_absolute_error(y_test_hr, y_pred_hr))

        convergence_warnings.extend(fit_with_warnings(arrival_model, x_hit_scaled, y_scaled))
        if len(hit_rows) < 30:
            arrival_warning = f"Only {len(hit_rows)} hit cases; arrival GP may be unreliable."
    else:
        arrival_warning = "Fewer than two hit cases; arrival GP was not trained."

    unique_warnings = unique_warning_messages(convergence_warnings)
    model_meta = {
        "event": event,
        "param_names": PARAM_NAMES,
        "n_completed": int(len(results)),
        "n_hit": int(hit.sum()),
        "hit_accuracy_training": hit_accuracy,
        "arrival_mae_holdout_hr": arrival_mae,
        "arrival_warning": arrival_warning,
        "convergence_warnings_count": len(convergence_warnings),
        "convergence_warnings_unique": unique_warnings[:10],
        "hit_kernel": str(hit_model.kernel_) if hit_model is not None else None,
        "arrival_kernel": str(arrival_model.kernel_) if arrival_model is not None else None,
    }
    dump({"x_scaler": x_scaler, "model": hit_model, "metadata": model_meta}, outdir / "gp_hit.joblib")
    dump(
        {"x_scaler": x_scaler, "y_scaler": y_scaler, "model": arrival_model, "metadata": model_meta},
        outdir / "gp_arrival.joblib",
    )
    return model_meta


def predict_arrival(bundle: dict, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = bundle["model"]
    if model is None:
        raise ValueError("Arrival GP is not trained")
    x_scaled = bundle["x_scaler"].transform(np.atleast_2d(theta))
    mean_scaled, std_scaled = model.predict(x_scaled, return_std=True)
    y_scaler = bundle["y_scaler"]
    mean = y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
    std = std_scaled * float(y_scaler.scale_[0])
    return mean, std


def predict_hit_probability(bundle: dict, theta: np.ndarray) -> np.ndarray:
    model = bundle["model"]
    if model is None:
        metadata = bundle.get("metadata", {})
        n_hit = metadata.get("n_hit", 0)
        n_completed = max(metadata.get("n_completed", 1), 1)
        return np.full(np.atleast_2d(theta).shape[0], n_hit / n_completed)
    x_scaled = bundle["x_scaler"].transform(np.atleast_2d(theta))
    return model.predict_proba(x_scaled)[:, 1]


def truncated_normal_samples(
    theta0: np.ndarray,
    sigma: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.normal(theta0, sigma, size=(n, len(theta0)))
    for _ in range(20):
        bad = (samples < low) | (samples > high)
        if not bad.any():
            break
        replacement = rng.normal(theta0, sigma, size=samples.shape)
        samples = np.where(bad, replacement, samples)
    return np.clip(samples, low, high)


def finite_difference_gradients(bundle: dict, theta0: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    base, _ = predict_arrival(bundle, theta0)
    gradients = {}
    for idx, name in enumerate(PARAM_NAMES):
        step = max((high[idx] - low[idx]) * 0.01, 1e-6)
        plus = theta0.copy()
        minus = theta0.copy()
        plus[idx] = min(high[idx], plus[idx] + step)
        minus[idx] = max(low[idx], minus[idx] - step)
        if plus[idx] == minus[idx]:
            gradients[name] = math.nan
            continue
        y_plus, _ = predict_arrival(bundle, plus)
        y_minus, _ = predict_arrival(bundle, minus)
        gradients[name] = float((y_plus[0] - y_minus[0]) / (plus[idx] - minus[idx]))
    gradients["baseline_arrival_time_hr"] = float(base[0])
    return gradients


def permutation_importance(bundle: dict, samples: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    baseline, _ = predict_arrival(bundle, samples)
    baseline_var = float(np.var(baseline))
    importance = {}
    for idx, name in enumerate(PARAM_NAMES):
        permuted = samples.copy()
        permuted[:, idx] = rng.permutation(permuted[:, idx])
        pred, _ = predict_arrival(bundle, permuted)
        importance[name] = float(np.mean((baseline - pred) ** 2))
    importance["prediction_variance_hr2"] = baseline_var
    return importance


def select_next_runs(
    arrival_bundle: dict,
    hit_bundle: dict,
    theta0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    n_candidates: int,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    candidates = low + rng.random((n_candidates, len(theta0))) * (high - low)
    _, arrival_std = predict_arrival(arrival_bundle, candidates)
    p_hit = predict_hit_probability(hit_bundle, candidates)
    sigma = default_obs_sigma(theta0)
    z2 = np.sum(((candidates - theta0) / sigma) ** 2, axis=1)
    likelihood_weight = np.exp(-0.5 * z2 / len(theta0))
    boundary_weight = 1.0 - np.abs(p_hit - 0.5) * 2.0
    score = arrival_std * (0.25 + boundary_weight) * (0.25 + likelihood_weight)

    scaled = arrival_bundle["x_scaler"].transform(candidates)
    selected = []
    available = np.arange(len(candidates))
    first = int(np.argmax(score))
    selected.append(first)
    available = available[available != first]
    while len(selected) < batch_size and available.size:
        distance = np.min(
            np.linalg.norm(scaled[available, None, :] - scaled[np.array(selected)][None, :, :], axis=2),
            axis=1,
        )
        combined = score[available] * (1.0 + distance)
        next_idx = int(available[np.argmax(combined)])
        selected.append(next_idx)
        available = available[available != next_idx]

    frame = pd.DataFrame(candidates[selected], columns=PARAM_NAMES)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame["score"] = score[selected]
    frame["p_hit"] = p_hit[selected]
    frame["arrival_std_hr"] = arrival_std[selected]
    return frame


def analyze_event(
    event: str,
    data_root: Path,
    output_root: Path,
    posterior_samples: int,
    candidate_count: int,
    next_batch: int,
    seed: int,
) -> Path:
    from joblib import load

    configure_local_caches()
    outdir = event_output_dir(event, output_root)
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(outdir / ".matplotlib"))
    import matplotlib.pyplot as plt

    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    arrival_bundle = load(outdir / "gp_arrival.joblib")
    hit_bundle = load(outdir / "gp_hit.joblib")
    config = load_event_config(event, data_root)
    theta0 = np.array(config["initial_theta"], dtype=float)
    low, high = default_bounds(theta0)
    sigma = default_obs_sigma(theta0)

    base_mean, base_std = predict_arrival(arrival_bundle, theta0)
    speed_plus = theta0.copy()
    speed_plus[4] = min(high[4], speed_plus[4] * 1.10)
    speed_mean, speed_std = predict_arrival(arrival_bundle, speed_plus)

    posterior_theta = truncated_normal_samples(theta0, sigma, low, high, posterior_samples, seed)
    posterior_mean, posterior_std = predict_arrival(arrival_bundle, posterior_theta)
    rng = np.random.default_rng(seed + 1)
    posterior_draw = rng.normal(posterior_mean, posterior_std)
    p_hit = predict_hit_probability(hit_bundle, posterior_theta)
    next_runs = select_next_runs(
        arrival_bundle,
        hit_bundle,
        theta0,
        low,
        high,
        candidate_count,
        next_batch,
        seed + 2,
    )
    next_runs.to_csv(outdir / "next_runs.csv", index=False)

    gradients = finite_difference_gradients(arrival_bundle, theta0, low, high)
    importance = permutation_importance(arrival_bundle, posterior_theta, seed + 3)

    summary = {
        "event": event,
        "speed_plus_10_percent": {
            "baseline_arrival_hr": float(base_mean[0]),
            "baseline_std_hr": float(base_std[0]),
            "speed_plus_arrival_hr": float(speed_mean[0]),
            "speed_plus_std_hr": float(speed_std[0]),
            "delta_arrival_hr": float(speed_mean[0] - base_mean[0]),
            "delta_std_hr": float(np.sqrt(base_std[0] ** 2 + speed_std[0] ** 2)),
        },
        "local_gradients_hr_per_unit": gradients,
        "permutation_importance": importance,
        "posterior_arrival_hr": {
            "median": float(np.nanmedian(posterior_draw)),
            "p05": float(np.nanpercentile(posterior_draw, 5)),
            "p95": float(np.nanpercentile(posterior_draw, 95)),
            "mean_p_hit": float(np.mean(p_hit)),
        },
        "model_metadata": arrival_bundle.get("metadata", {}),
    }
    with (outdir / "summary.yaml").open("w") as stream:
        yaml.safe_dump(summary, stream, sort_keys=False)

    pd.Series({k: v for k, v in gradients.items() if k in PARAM_NAMES}).plot(kind="bar")
    plt.ylabel("d arrival time / d parameter")
    plt.tight_layout()
    plt.savefig(figures / "local_sensitivity.png")
    plt.close()

    pd.Series({k: v for k, v in importance.items() if k in PARAM_NAMES}).plot(kind="bar")
    plt.ylabel("Permutation importance")
    plt.tight_layout()
    plt.savefig(figures / "permutation_importance.png")
    plt.close()

    finite_draw = posterior_draw[np.isfinite(posterior_draw)]
    plt.hist(finite_draw, bins=40, range=(ARRIVAL_MIN_HR, ARRIVAL_MAX_HR))
    plt.xlim(ARRIVAL_MIN_HR, ARRIVAL_MAX_HR)
    plt.xlabel("Arrival time since model start [hr]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures / "posterior_arrival_time.png")
    plt.close()

    make_all_pair_slices(arrival_bundle, hit_bundle, theta0, low, high, figures)
    return outdir / "summary.yaml"


def plot_design_pairs(event: str, output_root: Path, limit: int | None) -> Path:
    configure_local_caches()
    outdir = event_output_dir(event, output_root)
    design_path = outdir / "design.csv"
    if not design_path.exists():
        raise FileNotFoundError(f"Missing design file: {design_path}")

    design = pd.read_csv(design_path)
    missing = [name for name in PARAM_NAMES if name not in design]
    if missing:
        raise ValueError(f"{design_path} is missing required parameter columns: {', '.join(missing)}")
    if limit is not None:
        design = design.head(limit).copy()
    if design.empty:
        raise ValueError(f"No design rows to plot from {design_path}")

    import matplotlib.pyplot as plt

    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit_{limit}" if limit is not None else ""
    output_path = figures / f"design_parameter_pairs{suffix}.png"

    ndim = len(PARAM_NAMES)
    values = design[PARAM_NAMES].astype(float)
    sample_ids = design["sample_id"].astype(int).to_numpy() if "sample_id" in design else np.arange(len(design))
    colors = np.arange(len(design), dtype=float)
    point_kwargs = {
        "c": colors,
        "cmap": "viridis",
        "s": 18,
        "alpha": 0.8,
        "edgecolors": "none",
    }

    fig, axes = plt.subplots(ndim, ndim, figsize=(11, 11), constrained_layout=True)
    first_scatter = None
    seed_rows = design.index[sample_ids == 0].to_numpy()
    for row_idx, y_name in enumerate(PARAM_NAMES):
        for col_idx, x_name in enumerate(PARAM_NAMES):
            ax = axes[row_idx, col_idx]
            if row_idx == col_idx:
                ax.hist(values[x_name], bins=min(20, max(5, len(values) // 5)), color="0.45", alpha=0.85)
            elif row_idx > col_idx:
                first_scatter = ax.scatter(values[x_name], values[y_name], **point_kwargs)
                if seed_rows.size:
                    seed = seed_rows[0]
                    ax.scatter(
                        values.loc[seed, x_name],
                        values.loc[seed, y_name],
                        color="white",
                        edgecolor="black",
                        s=42,
                        linewidth=0.9,
                        zorder=3,
                    )
            else:
                ax.axis("off")
                continue

            if row_idx == ndim - 1:
                ax.set_xlabel(x_name)
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(y_name)
            else:
                ax.set_yticklabels([])

    title = f"{event} design parameter pairs, n={len(design)}"
    if limit is not None:
        title += f" (limit={limit})"
    fig.suptitle(title, fontsize=13)
    if first_scatter is not None:
        fig.colorbar(first_scatter, ax=axes, shrink=0.7, label="row order in plotted design")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def pair_slug(x_idx: int, y_idx: int) -> str:
    return f"{PARAM_NAMES[x_idx]}_{PARAM_NAMES[y_idx]}"


def pair_title(quantity: str, x_idx: int, y_idx: int) -> str:
    return f"{quantity}: {PARAM_NAMES[x_idx]} vs {PARAM_NAMES[y_idx]}"


def pair_grid(
    theta0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    x_idx: int,
    y_idx: int,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(low[x_idx], high[x_idx], grid_size)
    y = np.linspace(low[y_idx], high[y_idx], grid_size)
    xx, yy = np.meshgrid(x, y)
    theta = np.tile(theta0, (grid_size * grid_size, 1))
    theta[:, x_idx] = xx.ravel()
    theta[:, y_idx] = yy.ravel()
    return xx, yy, theta


def make_all_pair_slices(
    arrival_bundle: dict,
    hit_bundle: dict,
    theta0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    figures: Path,
    grid_size: int = 80,
) -> None:
    for x_idx, y_idx in itertools.combinations(range(len(PARAM_NAMES)), 2):
        slug = pair_slug(x_idx, y_idx)
        make_hit_slice(
            hit_bundle,
            theta0,
            low,
            high,
            x_idx,
            y_idx,
            figures / f"hit_probability_{slug}.png",
            grid_size=grid_size,
        )
        make_arrival_slice(
            arrival_bundle,
            hit_bundle,
            theta0,
            low,
            high,
            x_idx,
            y_idx,
            figures / f"arrival_mean_{slug}.png",
            quantity="mean",
            grid_size=grid_size,
        )
        make_arrival_slice(
            arrival_bundle,
            hit_bundle,
            theta0,
            low,
            high,
            x_idx,
            y_idx,
            figures / f"arrival_std_{slug}.png",
            quantity="std",
            grid_size=grid_size,
        )
    make_hit_slice(
        hit_bundle,
        theta0,
        low,
        high,
        1,
        4,
        figures / "hit_probability_longitude_speed.png",
        grid_size=grid_size,
    )
    make_hit_slice(
        hit_bundle,
        theta0,
        low,
        high,
        1,
        3,
        figures / "hit_probability_longitude_width.png",
        grid_size=grid_size,
    )


def make_hit_slice(
    hit_bundle: dict,
    theta0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    x_idx: int,
    y_idx: int,
    output_path: Path,
    grid_size: int = 80,
) -> None:
    import matplotlib.pyplot as plt

    xx, yy, theta = pair_grid(theta0, low, high, x_idx, y_idx, grid_size)
    prob = predict_hit_probability(hit_bundle, theta).reshape(grid_size, grid_size)
    plt.contourf(xx, yy, prob, levels=np.linspace(0, 1, 21), cmap="viridis")
    plt.colorbar(label="P(hit)")
    plt.contour(xx, yy, prob, levels=[0.5], colors="white", linewidths=1.5)
    plt.xlabel(PARAM_NAMES[x_idx])
    plt.ylabel(PARAM_NAMES[y_idx])
    plt.title(pair_title("Hit probability", x_idx, y_idx))
    plt.scatter(theta0[x_idx], theta0[y_idx], color="black", s=18, zorder=3, label="seed")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def make_arrival_slice(
    arrival_bundle: dict,
    hit_bundle: dict,
    theta0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    x_idx: int,
    y_idx: int,
    output_path: Path,
    quantity: str,
    grid_size: int = 80,
) -> None:
    import matplotlib.pyplot as plt

    if quantity not in {"mean", "std"}:
        raise ValueError(f"Unknown arrival slice quantity: {quantity}")

    xx, yy, theta = pair_grid(theta0, low, high, x_idx, y_idx, grid_size)
    mean, std = predict_arrival(arrival_bundle, theta)
    p_hit = predict_hit_probability(hit_bundle, theta)
    values = mean if quantity == "mean" else std
    values = values.reshape(grid_size, grid_size)
    prob = p_hit.reshape(grid_size, grid_size)

    label = "Arrival time [hr]" if quantity == "mean" else "Arrival GP std [hr]"
    title = "Arrival-time mean" if quantity == "mean" else "Arrival-time uncertainty"
    if quantity == "mean":
        # Clamp the arrival-time color scale to the physical 24-96 hr window; saturate outside.
        levels = np.linspace(ARRIVAL_MIN_HR, ARRIVAL_MAX_HR, 21)
        contour = plt.contourf(xx, yy, values, levels=levels, cmap="magma", extend="both")
    else:
        contour = plt.contourf(xx, yy, values, levels=21, cmap="magma")
    plt.colorbar(contour, label=label)
    if np.nanmin(prob) <= 0.5 <= np.nanmax(prob):
        plt.contour(xx, yy, prob, levels=[0.5], colors="white", linewidths=1.5)
    plt.xlabel(PARAM_NAMES[x_idx])
    plt.ylabel(PARAM_NAMES[y_idx])
    plt.title(pair_title(title, x_idx, y_idx))
    plt.scatter(theta0[x_idx], theta0[y_idx], color="white", edgecolor="black", s=24, zorder=3, label="seed")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="all", help="Event name under data root, or all.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    design = subparsers.add_parser("design", help="Create Latin-hypercube parameter designs")
    design.add_argument("--n", type=int, default=300)
    design.add_argument("--seed", type=int, default=42)
    design.add_argument("--force", action="store_true")
    design.add_argument("--span-inject-hour", type=float, default=1.0)
    design.add_argument("--span-longitude", type=float, default=30.0)
    design.add_argument("--span-latitude", type=float, default=20.0)
    design.add_argument("--span-width", type=float, default=40.0)
    design.add_argument("--span-speed-fraction", type=float, default=0.25)

    run = subparsers.add_parser("run", help="Run missing HUXt samples from design.csv")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--detector-threshold", type=float, default=0.25)
    run.add_argument("--detector-lag", type=int, default=2)
    run.add_argument("--smooth-window", type=int, default=5)
    run.add_argument("--detector-method", choices=["front", "hybrid", "enhancement", "jump"], default="hybrid")
    run.add_argument("--baseline-window", type=int, default=24)
    run.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Re-run completed sample IDs and replace their rows in results.csv",
    )

    plot_design = subparsers.add_parser("plot-design", help="Plot 2D scatter projections from design.csv")
    plot_design.add_argument("--limit", type=int, default=None, help="Plot only the first N design rows.")

    fit = subparsers.add_parser("fit", help="Fit GP arrival and hit/miss models")
    fit.add_argument("--test-fraction", type=float, default=0.2)
    fit.add_argument("--seed", type=int, default=42)

    visualize = subparsers.add_parser(
        "visualize-threshold",
        help="Replay completed samples and plot threshold diagnostics",
    )
    visualize.add_argument("--detector-threshold", type=float, default=0.25)
    visualize.add_argument("--detector-lag", type=int, default=2)
    visualize.add_argument("--smooth-window", type=int, default=5)
    visualize.add_argument("--detector-method", choices=["front", "hybrid", "enhancement", "jump"], default="hybrid")
    visualize.add_argument("--baseline-window", type=int, default=24)
    visualize.add_argument(
        "--sample-ids",
        default=None,
        help="Comma-separated sample IDs to visualize. Defaults to strongest non-hits below threshold.",
    )
    visualize.add_argument("--max-samples", type=int, default=6)
    visualize.add_argument(
        "--include-hits",
        action="store_true",
        help="When selecting automatically, allow rows already marked hit in results.csv.",
    )

    analyze = subparsers.add_parser("analyze", help="Generate GP analyses and next-run recommendations")
    analyze.add_argument("--posterior-samples", type=int, default=5000)
    analyze.add_argument("--candidate-count", type=int, default=5000)
    analyze.add_argument("--next-batch", type=int, default=20)
    analyze.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    configure_local_caches()
    args = parse_args()
    for event in event_list(args.event, args.data_root):
        if args.command == "design":
            path = make_design(
                event,
                args.data_root,
                args.output_root,
                args.n,
                args.seed,
                args.force,
                args.span_inject_hour,
                args.span_longitude,
                args.span_latitude,
                args.span_width,
                args.span_speed_fraction,
            )
            print(f"Wrote {path}")
        elif args.command == "run":
            path = run_design(
                event,
                args.data_root,
                args.output_root,
                args.limit,
                args.detector_threshold,
                args.detector_lag,
                args.smooth_window,
                args.detector_method,
                args.baseline_window,
                args.rerun_completed,
            )
            print(f"Wrote/updated {path}")
        elif args.command == "plot-design":
            path = plot_design_pairs(event, args.output_root, args.limit)
            print(f"Wrote {path}")
        elif args.command == "fit":
            metadata = train_models(event, args.output_root, args.test_fraction, args.seed)
            print(json.dumps(metadata, indent=2))
        elif args.command == "visualize-threshold":
            path = visualize_threshold_cases(
                event,
                args.data_root,
                args.output_root,
                args.detector_threshold,
                args.detector_lag,
                args.smooth_window,
                args.detector_method,
                args.baseline_window,
                args.sample_ids,
                args.max_samples,
                args.include_hits,
            )
            print(f"Wrote threshold diagnostics to {path}")
        elif args.command == "analyze":
            path = analyze_event(
                event,
                args.data_root,
                args.output_root,
                args.posterior_samples,
                args.candidate_count,
                args.next_batch,
                args.seed,
            )
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowStateError as exc:
        print(f"Workflow setup needed:\n{exc}", file=sys.stderr)
        raise SystemExit(2)
