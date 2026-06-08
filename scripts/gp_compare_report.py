#!/usr/bin/env python
"""Build a combined Markdown report for GP surrogate HUXt results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GP_ROOT = BASE_DIR / "runs" / "gp_surrogate"
PARAM_NAMES = ["inject_hour", "longitude", "latitude", "width", "v"]
FIGURE_ORDER = [
    ("posterior_arrival_time.png", "Posterior arrival-time distribution"),
    ("permutation_importance.png", "Global parameter importance"),
    ("local_sensitivity.png", "Local sensitivity near the seed CME"),
    ("hit_probability_longitude_v.png", "Hit probability in longitude-speed space"),
    ("hit_probability_longitude_width.png", "Hit probability in longitude-width space"),
    ("arrival_mean_longitude_v.png", "Arrival-time mean in longitude-speed space"),
    ("arrival_std_longitude_v.png", "Arrival-time uncertainty in longitude-speed space"),
]


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    with path.open("r") as stream:
        return yaml.safe_load(stream)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def hours_to_minutes(hours: float | None) -> float | None:
    if hours is None or pd.isna(hours):
        return None
    return float(hours) * 60.0


def strongest_parameter(values: dict, absolute: bool = True) -> tuple[str, float]:
    filtered = {k: float(v) for k, v in values.items() if k in PARAM_NAMES and v is not None}
    if not filtered:
        return "n/a", float("nan")
    if absolute:
        key = max(filtered, key=lambda item: abs(filtered[item]))
    else:
        key = max(filtered, key=filtered.get)
    return key, filtered[key]


def build_row(event: str, summary: dict) -> dict:
    metadata = summary.get("model_metadata", {})
    speed = summary.get("speed_plus_10_percent", {})
    posterior = summary.get("posterior_arrival_hr", {})
    gradients = summary.get("local_gradients_hr_per_unit", {})
    importance = summary.get("permutation_importance", {})
    top_grad, top_grad_value = strongest_parameter(gradients, absolute=True)
    top_importance, top_importance_value = strongest_parameter(importance, absolute=False)
    return {
        "event": event,
        "n_completed": metadata.get("n_completed"),
        "n_hit": metadata.get("n_hit"),
        "hit_fraction": metadata.get("n_hit") / metadata.get("n_completed")
        if metadata.get("n_completed")
        else None,
        "arrival_mae_hr": metadata.get("arrival_mae_holdout_hr"),
        "arrival_mae_min": hours_to_minutes(metadata.get("arrival_mae_holdout_hr")),
        "baseline_arrival_hr": speed.get("baseline_arrival_hr"),
        "speed_plus_delta_hr": speed.get("delta_arrival_hr"),
        "speed_plus_delta_min": hours_to_minutes(speed.get("delta_arrival_hr")),
        "posterior_median_hr": posterior.get("median"),
        "posterior_p05_hr": posterior.get("p05"),
        "posterior_p95_hr": posterior.get("p95"),
        "posterior_width_hr": posterior.get("p95") - posterior.get("p05")
        if posterior.get("p95") is not None and posterior.get("p05") is not None
        else None,
        "mean_p_hit": posterior.get("mean_p_hit"),
        "top_local_sensitivity": top_grad,
        "top_local_sensitivity_value": top_grad_value,
        "top_global_importance": top_importance,
        "top_global_importance_value": top_importance_value,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = frame[columns].astype(str).values.tolist()
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


def rounded_display(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: fmt(value, digits))
    return display


def figure_markdown(gp_root: Path, event: str) -> str:
    blocks = []
    for filename, title in FIGURE_ORDER:
        figure_path = gp_root / event / "figures" / filename
        if figure_path.exists():
            rel = figure_path.relative_to(gp_root)
            blocks.append(f"**{title}**\n\n![{event} {title}]({rel.as_posix()})")
    return "\n\n".join(blocks)


def next_runs_markdown(gp_root: Path, event: str, n: int = 5) -> str:
    path = gp_root / event / "next_runs.csv"
    if not path.exists():
        return "No `next_runs.csv` file found."
    frame = pd.read_csv(path).head(n)
    columns = ["rank", "inject_hour", "longitude", "latitude", "width", "v", "p_hit", "arrival_std_hr"]
    existing = [column for column in columns if column in frame.columns]
    return markdown_table(rounded_display(frame[existing]), existing)


def commentary(row: dict) -> str:
    speed_delta = row["speed_plus_delta_hr"]
    speed_text = (
        f"A 10 percent faster CME is predicted to arrive {abs(speed_delta):.2f} h "
        f"{'earlier' if speed_delta < 0 else 'later'}."
        if speed_delta is not None and not pd.isna(speed_delta)
        else "The 10 percent speed perturbation could not be evaluated."
    )
    posterior_text = (
        f"The posterior arrival window spans {row['posterior_width_hr']:.2f} h "
        f"from the 5th to 95th percentile, with mean hit probability {row['mean_p_hit']:.2f}."
        if row["posterior_width_hr"] is not None and not pd.isna(row["posterior_width_hr"])
        else "The posterior arrival spread could not be evaluated."
    )
    quality_text = (
        f"The arrival-time surrogate holdout MAE is {row['arrival_mae_hr']:.3f} h "
        f"({row['arrival_mae_min']:.1f} min), based on {int(row['n_hit'])} hit cases "
        f"out of {int(row['n_completed'])} HUXt samples."
    )
    sensitivity_text = (
        f"The strongest local sensitivity is `{row['top_local_sensitivity']}`, while the largest "
        f"global contribution to forecast spread is `{row['top_global_importance']}`."
    )
    return "\n\n".join([quality_text, speed_text, posterior_text, sensitivity_text])


def comparison_takeaways(frame: pd.DataFrame) -> str:
    best_quality = frame.loc[frame["arrival_mae_hr"].idxmin()]
    highest_hit = frame.loc[frame["mean_p_hit"].idxmax()]
    widest = frame.loc[frame["posterior_width_hr"].idxmax()]
    speed_abs = frame.assign(speed_response_abs=frame["speed_plus_delta_hr"].abs())
    strongest_speed = speed_abs.loc[speed_abs["speed_response_abs"].idxmax()]
    lines = [
        f"- `{best_quality['event']}` has the lower held-out arrival-time error "
        f"({best_quality['arrival_mae_hr']:.3f} h, {best_quality['arrival_mae_min']:.1f} min).",
        f"- `{highest_hit['event']}` has the larger posterior mean hit probability "
        f"({highest_hit['mean_p_hit']:.3f}).",
        f"- `{widest['event']}` has the wider posterior arrival window "
        f"({widest['posterior_width_hr']:.2f} h between the 5th and 95th percentiles).",
        f"- The stronger 10 percent speed response is for `{strongest_speed['event']}` "
        f"({strongest_speed['speed_plus_delta_hr']:.2f} h).",
    ]
    return "\n".join(lines)


def build_report(gp_root: Path, events: list[str]) -> tuple[pd.DataFrame, str]:
    summaries = {event: load_yaml(gp_root / event / "summary.yaml") for event in events}
    rows = [build_row(event, summaries[event]) for event in events]
    frame = pd.DataFrame(rows)

    display = rounded_display(frame)

    lines = [
        "# GP Surrogate Comparison Report",
        "",
        "This report compares the event-specific Gaussian Process surrogates trained on HUXt perturbation runs.",
        "",
        "## Model Quality And Forecast Summary",
        "",
        markdown_table(
            display,
            [
                "event",
                "n_completed",
                "n_hit",
                "hit_fraction",
                "arrival_mae_hr",
                "arrival_mae_min",
                "mean_p_hit",
                "posterior_width_hr",
            ],
        ),
        "",
        "## Speed Sensitivity And Dominant Parameters",
        "",
        markdown_table(
            display,
            [
                "event",
                "baseline_arrival_hr",
                "speed_plus_delta_hr",
                "speed_plus_delta_min",
                "top_local_sensitivity",
                "top_global_importance",
            ],
        ),
        "",
        "## Cross-Event Takeaways",
        "",
        comparison_takeaways(frame),
        "",
        "## Event Commentary",
        "",
    ]

    for _, row in frame.iterrows():
        event = row["event"]
        lines.extend(
            [
                f"### {event}",
                "",
                commentary(row),
                "",
                "#### Figures",
                "",
                figure_markdown(gp_root, event),
                "",
                "#### Suggested Next HUXt Runs",
                "",
                next_runs_markdown(gp_root, event),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `arrival_mae_hr` is a held-out error estimate for the arrival-time GP on hit cases only.",
            "- `mean_p_hit` is computed by propagating the default observational parameter uncertainty through the hit/miss GP.",
            "- `posterior_width_hr` is the 5th-to-95th percentile width of the surrogate arrival-time posterior.",
            "- `top_local_sensitivity` is based on finite differences around the seed CME parameters.",
            "- `top_global_importance` is based on permutation importance under the default observational uncertainty distribution.",
            "- `next_runs.csv` points target high surrogate uncertainty, likely observational regions, and hit/miss transition zones.",
            "- Additional UQ analyses are available in [UQ_README.md](UQ_README.md).",
            "",
        ]
    )
    return frame, "\n".join(lines)


def available_events(gp_root: Path) -> list[str]:
    if not gp_root.exists():
        return []
    return sorted(path.name for path in gp_root.iterdir() if (path / "summary.yaml").exists())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp-root", type=Path, default=DEFAULT_GP_ROOT)
    parser.add_argument("--events", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.gp_root / "README.md"
    args.gp_root.mkdir(parents=True, exist_ok=True)
    events = args.events or available_events(args.gp_root)
    if not events:
        raise FileNotFoundError(f"No event summary.yaml files found under {args.gp_root}")
    frame, report = build_report(args.gp_root, events)
    frame.to_csv(args.gp_root / "comparison_summary.csv", index=False)
    output.write_text(report)
    print(f"Wrote {output}")
    print(f"Wrote {args.gp_root / 'comparison_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
