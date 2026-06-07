#!/usr/bin/env python
"""Additional UQ analyses for trained HUXt GP surrogates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from gp_huxt_surrogate import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    PARAM_NAMES,
    default_bounds,
    default_obs_sigma,
    event_list,
    event_output_dir,
    load_event_config,
    predict_arrival,
    predict_hit_probability,
    parse_bool,
    train_models,
    truncated_normal_samples,
)


def load_bundle(event: str, output_root: Path, name: str) -> dict:
    from joblib import load

    path = event_output_dir(event, output_root) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing GP bundle: {path}. Run `fit` first.")
    return load(path)


def output_dirs(event: str, output_root: Path) -> tuple[Path, Path]:
    outdir = event_output_dir(event, output_root)
    uq_dir = outdir / "uq"
    figure_dir = uq_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return uq_dir, figure_dir


def load_results(event: str, output_root: Path) -> pd.DataFrame:
    path = event_output_dir(event, output_root) / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    data = pd.read_csv(path)
    data = data.loc[data["status"] == "completed"].copy()
    data["hit"] = data["hit"].map(parse_bool)
    return data


def load_theta_context(event: str, data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = load_event_config(event, data_root)
    theta0 = np.array(config["initial_theta"], dtype=float)
    low, high = default_bounds(theta0)
    sigma = default_obs_sigma(theta0)
    return theta0, low, high, sigma


def variance_decomposition(event: str, data_root: Path, output_root: Path, n: int, seed: int) -> dict:
    arrival = load_bundle(event, output_root, "gp_arrival.joblib")
    hit = load_bundle(event, output_root, "gp_hit.joblib")
    theta0, low, high, sigma = load_theta_context(event, data_root)
    samples = truncated_normal_samples(theta0, sigma, low, high, n, seed)
    mean, std = predict_arrival(arrival, samples)
    p_hit = predict_hit_probability(hit, samples)

    input_variance = float(np.nanvar(mean))
    surrogate_variance = float(np.nanmean(std**2))
    total_variance = input_variance + surrogate_variance
    return {
        "event": event,
        "n_samples": int(n),
        "input_variance_hr2": input_variance,
        "surrogate_variance_hr2": surrogate_variance,
        "total_variance_hr2": total_variance,
        "input_fraction": input_variance / total_variance if total_variance else math.nan,
        "surrogate_fraction": surrogate_variance / total_variance if total_variance else math.nan,
        "arrival_std_total_hr": float(np.sqrt(total_variance)),
        "mean_p_hit": float(np.nanmean(p_hit)),
    }


def value_of_information(event: str, data_root: Path, output_root: Path, n: int, seed: int) -> pd.DataFrame:
    arrival = load_bundle(event, output_root, "gp_arrival.joblib")
    hit = load_bundle(event, output_root, "gp_hit.joblib")
    theta0, low, high, sigma = load_theta_context(event, data_root)

    rows = []
    scenarios = [("baseline", None)]
    scenarios.extend((f"halve_{name}_sigma", idx) for idx, name in enumerate(PARAM_NAMES))

    baseline_width = None
    baseline_std = None
    for label, shrink_idx in scenarios:
        scenario_sigma = sigma.copy()
        if shrink_idx is not None:
            scenario_sigma[shrink_idx] *= 0.5
        samples = truncated_normal_samples(theta0, scenario_sigma, low, high, n, seed + (shrink_idx or 0) + 10)
        mean, std = predict_arrival(arrival, samples)
        rng = np.random.default_rng(seed + (shrink_idx or 0) + 100)
        draws = rng.normal(mean, std)
        p_hit = predict_hit_probability(hit, samples)
        width = float(np.nanpercentile(draws, 95) - np.nanpercentile(draws, 5))
        total_std = float(np.nanstd(draws))
        if baseline_width is None:
            baseline_width = width
            baseline_std = total_std
        rows.append(
            {
                "scenario": label,
                "parameter_reduced": PARAM_NAMES[shrink_idx] if shrink_idx is not None else "none",
                "posterior_width_hr": width,
                "posterior_std_hr": total_std,
                "width_reduction_hr": baseline_width - width,
                "width_reduction_fraction": (baseline_width - width) / baseline_width if baseline_width else math.nan,
                "std_reduction_hr": baseline_std - total_std,
                "mean_p_hit": float(np.nanmean(p_hit)),
            }
        )
    return pd.DataFrame(rows)


def detector_threshold_sensitivity(event: str, output_root: Path, thresholds: list[float]) -> pd.DataFrame:
    results = load_results(event, output_root)
    detector_column = "max_detector_value" if "max_detector_value" in results else "max_speed_jump"
    rows = []
    total = len(results)
    for threshold in thresholds:
        hit = results[detector_column] >= threshold
        arrivals = results.loc[hit, "arrival_time_hr"]
        rows.append(
            {
                "threshold": threshold,
                "detector_column": detector_column,
                "n_hit": int(hit.sum()),
                "hit_fraction": float(hit.mean()) if total else math.nan,
                "arrival_median_hr": float(np.nanmedian(arrivals)) if hit.any() else math.nan,
                "arrival_p05_hr": float(np.nanpercentile(arrivals, 5)) if hit.any() else math.nan,
                "arrival_p95_hr": float(np.nanpercentile(arrivals, 95)) if hit.any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def learning_curve(
    event: str,
    output_root: Path,
    sizes: list[int],
    repeats: int,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    results = load_results(event, output_root)
    rng = np.random.default_rng(seed)
    outdir = event_output_dir(event, output_root)
    original = outdir / "results.csv"
    backup = outdir / "results.csv.gp_uq_backup"
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")

    rows = []
    try:
        original.replace(backup)
        for size in sizes:
            if size > len(results):
                continue
            for repeat in range(repeats):
                subset_ids = rng.choice(results.index.to_numpy(), size=size, replace=False)
                subset = results.loc[subset_ids].sort_values("sample_id")
                subset.to_csv(original, index=False)
                metadata = train_models(event, output_root, test_fraction, seed + repeat + size)
                rows.append(
                    {
                        "n_train": int(size),
                        "repeat": int(repeat),
                        "n_hit": metadata["n_hit"],
                        "hit_fraction": metadata["n_hit"] / metadata["n_completed"],
                        "arrival_mae_holdout_hr": metadata["arrival_mae_holdout_hr"],
                    }
                )
    finally:
        if original.exists():
            original.unlink()
        backup.replace(original)
        train_models(event, output_root, test_fraction, seed)

    return pd.DataFrame(rows)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def plot_bar(frame: pd.DataFrame, x: str, y: str, path: Path, ylabel: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent.parent / ".matplotlib"))
    import matplotlib.pyplot as plt

    ax = frame.plot(x=x, y=y, kind="bar", legend=False)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_learning_curve(frame: pd.DataFrame, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent.parent / ".matplotlib"))
    import matplotlib.pyplot as plt

    grouped = frame.groupby("n_train")["arrival_mae_holdout_hr"].agg(["mean", "std"]).reset_index()
    plt.errorbar(grouped["n_train"], grouped["mean"], yerr=grouped["std"], marker="o")
    plt.xlabel("Training samples")
    plt.ylabel("Holdout arrival MAE [hr]")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_uq(event: str, data_root: Path, output_root: Path, n: int, seed: int) -> None:
    uq_dir, figure_dir = output_dirs(event, output_root)
    variance = variance_decomposition(event, data_root, output_root, n, seed)
    with (uq_dir / "variance_decomposition.yaml").open("w") as stream:
        yaml.safe_dump(variance, stream, sort_keys=False)
    voi = value_of_information(event, data_root, output_root, n, seed + 20)
    save_frame(voi, uq_dir / "value_of_information.csv")
    plot_bar(
        voi.loc[voi["parameter_reduced"] != "none"],
        "parameter_reduced",
        "width_reduction_fraction",
        figure_dir / "value_of_information.png",
        "Fractional reduction in arrival window",
    )
    thresholds = detector_threshold_sensitivity(event, output_root, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    save_frame(thresholds, uq_dir / "detector_threshold_sensitivity.csv")
    plot_bar(thresholds, "threshold", "hit_fraction", figure_dir / "detector_threshold_sensitivity.png", "Hit fraction")
    print(json.dumps(variance, indent=2))
    print(f"Wrote {uq_dir}")


def run_learning_curve(
    event: str,
    output_root: Path,
    sizes: list[int],
    repeats: int,
    test_fraction: float,
    seed: int,
) -> None:
    uq_dir, figure_dir = output_dirs(event, output_root)
    frame = learning_curve(event, output_root, sizes, repeats, test_fraction, seed)
    save_frame(frame, uq_dir / "learning_curve.csv")
    plot_learning_curve(frame, figure_dir / "learning_curve.png")
    print(f"Wrote {uq_dir / 'learning_curve.csv'}")


def write_uq_readme(output_root: Path, events: list[str]) -> Path:
    lines = [
        "# GP Surrogate UQ Add-On Analyses",
        "",
        "These analyses use the trained GP surrogates without launching new HUXt simulations.",
        "",
    ]
    for event in events:
        uq_dir = event_output_dir(event, output_root) / "uq"
        lines.extend([f"## {event}", ""])
        variance_path = uq_dir / "variance_decomposition.yaml"
        if variance_path.exists():
            data = yaml.safe_load(variance_path.read_text())
            lines.extend(
                [
                    "### Variance Decomposition",
                    "",
                    f"- Input-parameter uncertainty fraction: {data['input_fraction']:.3f}",
                    f"- GP surrogate uncertainty fraction: {data['surrogate_fraction']:.3f}",
                    f"- Total arrival-time standard deviation: {data['arrival_std_total_hr']:.3f} h",
                    "",
                ]
            )
        voi_path = uq_dir / "value_of_information.csv"
        if voi_path.exists():
            voi = pd.read_csv(voi_path)
            reduced = voi.loc[voi["parameter_reduced"] != "none"].copy()
            if not reduced.empty:
                best = reduced.sort_values("width_reduction_fraction", ascending=False).iloc[0]
                lines.extend(
                    [
                        "### Value Of Additional Observations",
                        "",
                        f"- Most useful uncertainty reduction: `{best['parameter_reduced']}`",
                        f"- Posterior arrival-window reduction: {best['width_reduction_fraction']:.3f}",
                        "",
                        "![Value of information](%s)"
                        % (uq_dir / "figures" / "value_of_information.png").relative_to(output_root).as_posix(),
                        "",
                    ]
                )
        threshold_fig = uq_dir / "figures" / "detector_threshold_sensitivity.png"
        if threshold_fig.exists():
            lines.extend(
                [
                    "### Arrival Detector Sensitivity",
                    "",
                    "![Detector threshold sensitivity](%s)"
                    % threshold_fig.relative_to(output_root).as_posix(),
                    "",
                ]
            )
        learning_fig = uq_dir / "figures" / "learning_curve.png"
        if learning_fig.exists():
            lines.extend(
                [
                    "### Learning Curve",
                    "",
                    "![Learning curve](%s)" % learning_fig.relative_to(output_root).as_posix(),
                    "",
                ]
            )
    output = output_root / "UQ_README.md"
    output.write_text("\n".join(lines))
    return output


def fmt(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    display = frame[columns].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: fmt(value, digits))
    rows = display.astype(str).values.tolist()
    widths = []
    for idx, column in enumerate(columns):
        values = [row[idx] for row in rows]
        widths.append(max(len(str(column)), *(len(value) for value in values)))

    def format_row(values: list[str]) -> str:
        cells = [str(value).ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    header = format_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [format_row(row) for row in rows]
    return "\n".join([header, separator, *body])


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return yaml.safe_load(path.read_text())


def detector_commentary(thresholds: pd.DataFrame) -> str:
    row015 = thresholds.loc[np.isclose(thresholds["threshold"], 0.15)]
    row025 = thresholds.loc[np.isclose(thresholds["threshold"], 0.25)]
    if row015.empty or row025.empty:
        return "The detector-threshold scan should be inspected directly in the table and figure."
    hit015 = row015.iloc[0]
    hit025 = row025.iloc[0]
    return (
        f"At the working threshold of `0.15`, {int(hit015['n_hit'])} of the 300 ensemble members are hits "
        f"(`{hit015['hit_fraction']:.3f}` hit fraction). At the stricter `0.25` threshold this drops to "
        f"{int(hit025['n_hit'])} hits (`{hit025['hit_fraction']:.3f}`), so the arrival definition materially "
        "affects hit/miss counts and should be stated whenever these GP results are reported."
    )


def learning_curve_commentary(curve: pd.DataFrame) -> str:
    if curve.empty:
        return "No learning-curve results were found for this event."
    grouped = curve.groupby("n_train")["arrival_mae_holdout_hr"].agg(["mean", "std"]).reset_index()
    best = grouped.loc[grouped["mean"].idxmin()]
    largest = grouped.loc[grouped["n_train"].idxmax()]
    return (
        f"The subset learning curve reaches its lowest mean held-out MAE at `{int(best['n_train'])}` training samples "
        f"(`{best['mean']:.3f} h`). The largest tested subset has `{int(largest['n_train'])}` samples with mean MAE "
        f"`{largest['mean']:.3f} h`. Because these subset fits reuse random subsets, use this as a practical stability "
        "check rather than a formal convergence proof."
    )


def write_detailed_uq_report(output_root: Path, events: list[str]) -> Path:
    lines = [
        "# Detailed GP Surrogate UQ Report",
        "",
        "This report combines the additional uncertainty-quantification analyses for the two CME-specific HUXt GP surrogates.",
        "The calculations use trained GP models and existing HUXt ensemble results; no new HUXt simulations are launched by this report.",
        "",
        "## Executive Summary",
        "",
    ]

    summary_rows = []
    event_payload = {}
    for event in events:
        uq_dir = event_output_dir(event, output_root) / "uq"
        variance = load_yaml(uq_dir / "variance_decomposition.yaml")
        voi = pd.read_csv(uq_dir / "value_of_information.csv")
        thresholds = pd.read_csv(uq_dir / "detector_threshold_sensitivity.csv")
        learning_path = uq_dir / "learning_curve.csv"
        learning = pd.read_csv(learning_path) if learning_path.exists() else pd.DataFrame()
        best_voi = voi.loc[voi["parameter_reduced"] != "none"].sort_values(
            "width_reduction_fraction", ascending=False
        ).iloc[0]
        summary_rows.append(
            {
                "event": event,
                "input_fraction": variance["input_fraction"],
                "surrogate_fraction": variance["surrogate_fraction"],
                "arrival_std_total_hr": variance["arrival_std_total_hr"],
                "mean_p_hit": variance["mean_p_hit"],
                "best_measurement": best_voi["parameter_reduced"],
                "best_width_reduction_fraction": best_voi["width_reduction_fraction"],
            }
        )
        event_payload[event] = {
            "uq_dir": uq_dir,
            "variance": variance,
            "voi": voi,
            "thresholds": thresholds,
            "learning": learning,
        }

    summary = pd.DataFrame(summary_rows)
    lines.extend(
        [
            markdown_table(
                summary,
                [
                    "event",
                    "input_fraction",
                    "surrogate_fraction",
                    "arrival_std_total_hr",
                    "mean_p_hit",
                    "best_measurement",
                    "best_width_reduction_fraction",
                ],
            ),
            "",
            "Across both CMEs, the arrival-time uncertainty is dominated by uncertain CME inputs rather than GP emulator uncertainty. "
            "This supports using the current GP as an analysis tool, while pointing to improved CME parameter measurements as the main route to narrower forecasts.",
            "",
        ]
    )

    for event, payload in event_payload.items():
        uq_dir = payload["uq_dir"]
        variance = payload["variance"]
        voi = payload["voi"]
        thresholds = payload["thresholds"]
        learning = payload["learning"]
        best_voi = voi.loc[voi["parameter_reduced"] != "none"].sort_values(
            "width_reduction_fraction", ascending=False
        ).iloc[0]
        figure_prefix = uq_dir.relative_to(output_root).as_posix()

        lines.extend(
            [
                f"## {event}",
                "",
                "### Variance Decomposition",
                "",
                f"The total arrival-time standard deviation is `{variance['arrival_std_total_hr']:.3f} h`. "
                f"Input-parameter uncertainty explains `{variance['input_fraction']:.3f}` of the variance, while GP surrogate uncertainty explains "
                f"`{variance['surrogate_fraction']:.3f}`. In other words, the forecast spread is mostly controlled by uncertain CME parameters, "
                "not by emulator uncertainty.",
                "",
                "### Value Of Additional Observations",
                "",
                f"The most valuable single uncertainty reduction is `{best_voi['parameter_reduced']}`. Halving that parameter's assumed uncertainty reduces "
                f"the 5-95 percent posterior arrival window by `{best_voi['width_reduction_fraction']:.3f}` "
                f"(`{best_voi['width_reduction_hr']:.2f} h`).",
                "",
                markdown_table(
                    voi,
                    [
                        "parameter_reduced",
                        "posterior_width_hr",
                        "width_reduction_hr",
                        "width_reduction_fraction",
                        "mean_p_hit",
                    ],
                ),
                "",
                f"![{event} value of information]({figure_prefix}/figures/value_of_information.png)",
                "",
                "### Arrival Detector Sensitivity",
                "",
                detector_commentary(thresholds),
                "",
                markdown_table(
                    thresholds,
                    [
                        "threshold",
                        "n_hit",
                        "hit_fraction",
                        "arrival_median_hr",
                        "arrival_p05_hr",
                        "arrival_p95_hr",
                    ],
                ),
                "",
                f"![{event} detector threshold sensitivity]({figure_prefix}/figures/detector_threshold_sensitivity.png)",
                "",
                "### Learning Curve",
                "",
                learning_curve_commentary(learning),
                "",
            ]
        )
        if not learning.empty:
            grouped = learning.groupby("n_train")["arrival_mae_holdout_hr"].agg(["mean", "std"]).reset_index()
            grouped = grouped.rename(columns={"mean": "mae_mean_hr", "std": "mae_std_hr"})
            lines.extend(
                [
                    markdown_table(grouped, ["n_train", "mae_mean_hr", "mae_std_hr"]),
                    "",
                    f"![{event} learning curve]({figure_prefix}/figures/learning_curve.png)",
                    "",
                ]
            )

    lines.extend(
        [
            "## Reporting Notes",
            "",
            "- The value-of-information calculation halves one parameter uncertainty at a time while leaving the others unchanged.",
            "- The variance decomposition separates uncertainty from CME input sampling and GP predictive uncertainty.",
            "- The detector sensitivity table shows that hit/miss counts depend on the chosen fractional speed-jump threshold.",
            "- Learning curves refit GP models on random subsets of existing HUXt runs; they do not launch new HUXt simulations.",
            "",
        ]
    )
    output = output_root / "report_UQ.md"
    output.write_text("\n".join(lines))
    return output


def report_events(output_root: Path, requested_events: list[str]) -> list[str]:
    available = sorted(
        path.name
        for path in Path(output_root).iterdir()
        if path.is_dir() and ((path / "uq").exists() or (path / "summary.yaml").exists())
    ) if Path(output_root).exists() else []
    if available:
        return available
    return requested_events


def parse_sizes(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="all", help="Event name under data root, or all.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    uq = subparsers.add_parser("uq", help="Variance decomposition, value of information, detector sensitivity")
    uq.add_argument("--samples", type=int, default=5000)
    uq.add_argument("--seed", type=int, default=42)

    lc = subparsers.add_parser("learning-curve", help="Fit subset-size learning curves")
    lc.add_argument("--sizes", type=parse_sizes, default=[25, 50, 100, 200, 300])
    lc.add_argument("--repeats", type=int, default=3)
    lc.add_argument("--test-fraction", type=float, default=0.2)
    lc.add_argument("--seed", type=int, default=42)

    subparsers.add_parser("report", help="Write UQ_README.md from existing UQ outputs")
    subparsers.add_parser("detailed-report", help="Write report_UQ.md with figures and detailed commentary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = event_list(args.event, args.data_root)
    for event in events:
        if args.command == "uq":
            run_uq(event, args.data_root, args.output_root, args.samples, args.seed)
        elif args.command == "learning-curve":
            run_learning_curve(event, args.output_root, args.sizes, args.repeats, args.test_fraction, args.seed)
        elif args.command == "report":
            pass
        elif args.command == "detailed-report":
            pass
    output = write_uq_readme(args.output_root, report_events(args.output_root, events))
    if args.command == "detailed-report":
        detailed = write_detailed_uq_report(args.output_root, report_events(args.output_root, events))
        print(f"Wrote {detailed}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
