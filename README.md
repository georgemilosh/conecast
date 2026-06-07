# conecast

A local, self-contained workflow for emulating **CME arrival at Earth** with
**HUXt** (Heliospheric Upwind eXtrapolation, time-dependent) and Gaussian-process
surrogates.

The first stage prepares solar-wind boundary conditions from GONG magnetograms via
WSA+. The second stage builds per-event GP surrogates for scalar HUXt outcomes
(hit/miss and arrival time) and uses them for sensitivity, uncertainty, and
next-run selection. Teaching notebooks walk every step.

> **Data is not shipped.** This repository contains only source (scripts,
> notebooks, docs) and the small per-event seed configs. The WSA+ checkpoint
> `data_dir/sw/wsaplus.pt`, GONG magnetograms, WSA+ maps, HUXt boundaries, and all
> `runs/` outputs are **downloaded or generated on first run** of
> `scripts/generate_huxt_input.py`.

See [Tutorial.md](Tutorial.md) for the `generate_huxt_input.py` input-preparation
walkthrough, and [notebooks/README.md](notebooks/README.md) for the tutorial
notebooks (GP theory, HUXt input prep, the arrival detector, and the GP surrogate).

---

## Seed CME parameters

Prepared events use a 5-dimensional seed parameter vector `theta`:

| Index | Name          | Units | Prior bounds | Meaning                              |
|-------|---------------|-------|--------------|--------------------------------------|
| 0     | `inject_hour` | h     | (0, 10)      | Time after model start to launch CME |
| 1     | `longitude`   | deg   | (-90, 90)    | Cone-CME central longitude           |
| 2     | `latitude`    | deg   | (-50, 50)    | Cone-CME central latitude            |
| 3     | `width`       | deg   | (0, 180)     | Cone-CME angular half-width          |
| 4     | `v`           | km/s  | (100, 2000)  | Cone-CME initial speed               |

The input-preparation script writes the seed values to `event_config.yaml`
(`initial_theta` + `cr_num`). The GP surrogate workflow reads them back from there.

---

## Directory layout

```
conecast/
├── data_dir/
│   ├── events.csv                    # event seed parameters for generate_huxt_input.py
│   └── sw/<event>/event_config.yaml  # per-event seed (initial_theta + cr_num)
├── scripts/
│   ├── generate_huxt_input.py        # Stage 1: GONG -> WSA+ -> HUXt boundary + seed config
│   ├── run_huxt_functions.py         # shared HUXt helpers
│   ├── gp_huxt_surrogate.py          # GP surrogate: design / run / fit / analyze
│   ├── gp_uq_tools.py                # uncertainty-decomposition + value-of-information
│   ├── gp_compare_report.py          # cross-event comparison report
│   ├── gp_tutorial_figures.py        # tutorial figure generation
│   └── create_gp_notebooks.py        # regenerates the notebooks/
├── notebooks/                        # 01..04 tutorials (generated) + README
├── tests/                            # pytest for the GP/detector code
├── requirements.txt                  # pinned deps (HUXt + WSA+ via git/PyPI)
├── setup.sh                          # local venv bootstrap
├── README.md
└── Tutorial.md                       # generate_huxt_input.py walkthrough
```

At runtime, `data_dir/sw/<event>/` fills with GONG FITS, WSA+ maps, `v_boundary_*.npz`,
and diagnostics, and GP outputs land under `runs/gp_surrogate/<event>/`.

---

## Environment

```bash
bash setup.sh                 # creates .venv and installs requirements.txt
source .venv/bin/activate
```

`requirements.txt` is a pinned freeze; on a local machine a few platform-specific
pins (e.g. `torch`) may need relaxing. HUXt and WSA+ install from their git/PyPI
sources listed there. Obtain the WSA+ checkpoint `data_dir/sw/wsaplus.pt` separately
before running the WSA+ step.

---

## Quick start

```bash
# 1. Prepare an event (GONG/WSA+ products, HUXt boundary, seed config)
python scripts/generate_huxt_input.py --event 2017-09-06

# 2. Reuse existing GONG FITS files if the event directory is already populated
python scripts/generate_huxt_input.py --event 2017-09-06 --no-download

# 3. Skip the optional seed HUXt sanity plot when only boundary files are needed
python scripts/generate_huxt_input.py --event 2017-09-06 --skip-sanity-plot
```

For the full input-preparation workflow and script internals, see
[Tutorial.md](Tutorial.md).

---

## GP surrogate workflow

The GP workflow builds per-event surrogates for scalar HUXt outcomes such as arrival
time and hit/miss status. It uses the prepared event inputs in `data_dir/sw/<event>/`
and writes outputs to `runs/gp_surrogate/<event>/`.

```bash
# Create 300 perturbations for every prepared event
python scripts/gp_huxt_surrogate.py --event all design --n 300

# Run a small batch first, then repeat or raise the limit
python scripts/gp_huxt_surrogate.py --event 2017-09-06 run --limit 20

# Re-run completed rows if you change the hit detector settings
python scripts/gp_huxt_surrogate.py --event 2017-09-06 run --limit 20 --rerun-completed \
  --detector-method hybrid

# Fit GP models after enough HUXt samples complete
python scripts/gp_huxt_surrogate.py --event 2017-09-06 fit

# Generate sensitivity, posterior, hit-boundary, and next-run outputs
python scripts/gp_huxt_surrogate.py --event 2017-09-06 analyze
python scripts/gp_huxt_surrogate.py --event all analyze

# Visualize threshold diagnostics for near-miss or hand-picked samples
python scripts/gp_huxt_surrogate.py --event 2017-09-06 visualize-threshold \
  --detector-method hybrid --sample-ids 6

# Build a combined comparison report after several events are analyzed
python scripts/gp_compare_report.py

# Add uncertainty-decomposition and value-of-information analyses
python scripts/gp_uq_tools.py --event all uq

# Optional: estimate how many HUXt runs are enough
python scripts/gp_uq_tools.py --event all learning-curve
```

The generated files include `design.csv`, `results.csv`, `gp_arrival.joblib`,
`gp_hit.joblib`, `summary.yaml`, `next_runs.csv`, and diagnostic figures. The
`analyze` command writes these per-event figures under
`runs/gp_surrogate/<event>/figures/`:

- `hit_probability_<param1>_<param2>.png` for the two-parameter combinations
- `arrival_mean_<param1>_<param2>.png` and `arrival_std_<param1>_<param2>.png`
- `local_sensitivity.png`, `permutation_importance.png`, `posterior_arrival_time.png`

Each two-parameter slice varies the named parameters across the design bounds and
keeps the remaining Cone-CME parameters fixed at the event seed. Hit probability comes
from the GP classifier; arrival-time mean and uncertainty come from the GP regressor
trained on hit cases only. Arrival-time figures are clamped to a physical 24-96 hr
window so the GP's extrapolation outside the hit region does not dominate the scale.

`fit` requires `results.csv`, created by the `run` subcommand. If `results.csv` is
missing, run `design` first, then a small HUXt batch with `run --limit 20`.

The default hit detector is `--detector-method hybrid`: HUXt's tracked ConeCME front
must reach Earth **and** the smoothed Earth speed must exceed the enhancement
threshold. `front` uses only `compute_arrival_at_body("EARTH")`; `enhancement` uses
only the Earth speed signature; `jump` keeps the older short-lag fractional derivative.

### GP command reference

Global options:

| Option | Meaning |
|--------|---------|
| `--event <name>` / `--event all` | Select one prepared event under `data_dir/sw/`, or every prepared event. |
| `--data-root <path>` | Input root containing `<event>/event_config.yaml` and `v_boundary_<event>.npz`. |
| `--output-root <path>` | Output root for designs, HUXt results, GP models, figures, and summaries. |

`design` creates `design.csv` and `design_meta.yaml`.

| Option | Meaning |
|--------|---------|
| `--n` | Number of parameter samples including the seed row. |
| `--seed` | Latin-hypercube random seed. |
| `--force` | Overwrite an existing design. |
| `--span-inject-hour` | Half-width of the sampled launch-time range in hours. |
| `--span-longitude` | Half-width of sampled longitude range in degrees. |
| `--span-latitude` | Half-width of sampled latitude range in degrees. |
| `--span-width` | Half-width of sampled cone half-width range in degrees. |
| `--span-speed-fraction` | Speed span as a fraction of seed speed. |

`run` evaluates HUXt samples and writes/updates `results.csv`.

| Option | Meaning |
|--------|---------|
| `--limit` | Run only the next N pending/missing samples. |
| `--rerun-completed` | Re-run completed sample IDs and replace their rows. Use after changing detector settings. |
| `--detector-method {hybrid,front,enhancement,jump}` | Arrival/hit definition. `hybrid` is the default. |
| `--detector-threshold` | Speed-signature threshold for `hybrid`, `enhancement`, and `jump`. |
| `--detector-lag` | Lag used by the short-lag `jump` detector. |
| `--smooth-window` | Moving-average window for speed diagnostics. |
| `--baseline-window` | Lookback window for the enhancement baseline. |

`fit` trains `gp_hit.joblib` and `gp_arrival.joblib`.

| Option | Meaning |
|--------|---------|
| `--test-fraction` | Held-out fraction for arrival-time MAE when enough hit rows exist. |
| `--seed` | Random seed for train/test split and GP fitting. |

`analyze` writes `summary.yaml`, figures, and `next_runs.csv`.

| Option | Meaning |
|--------|---------|
| `--posterior-samples` | Number of posterior parameter samples used in summary plots. |
| `--candidate-count` | Number of random candidate points considered for next-run selection. |
| `--next-batch` | Number of suggested follow-up HUXt runs. |
| `--seed` | Random seed for posterior/candidate sampling. |

---

## Event preparation workflow

Event seed parameters live in `data_dir/events.csv`; add new rows there instead of
editing Python code.

Required columns: `event`, `cme_onset`, `cme_0p1_au`, `longitude`, `latitude`,
`width`, `speed`. Optional `enabled` and `notes` are allowed; disabled rows are
skipped when `enabled` is false.

```bash
# Prepare one event / every enabled event from the CSV
python scripts/generate_huxt_input.py --event 2017-09-06
python scripts/generate_huxt_input.py --event all

# Use existing GONG FITS files only (no network)
python scripts/generate_huxt_input.py --event 2017-09-06 --no-download

# Custom CSV / output root / WSA+ checkpoint
python scripts/generate_huxt_input.py \
  --events-file data_dir/events.csv \
  --output-root data_dir/sw \
  --checkpoint-path data_dir/sw/wsaplus.pt
```

Outputs are written to `data_dir/sw/<event>/`. The `gp_huxt_surrogate.py --event all`,
`gp_compare_report.py`, and `gp_uq_tools.py --event all ...` commands discover
prepared/analyzed event directories automatically.

### Event-preparation command reference

| Option | Meaning |
|--------|---------|
| `--events-file <path>` | CSV containing event seed parameters. Defaults to `data_dir/events.csv`. |
| `--output-root <path>` | Per-event output root. Defaults to `data_dir/sw`. |
| `--checkpoint-path <path>` | WSA+ checkpoint. Defaults to `<output-root>/wsaplus.pt`. |
| `--event <name> [<name> ...]` | Process one or more CSV event names, or `all`. |
| `--no-download` | Reuse existing GONG FITS files and fail if none are present. |
| `--force-config` | Overwrite existing `event_config.yaml`. |
| `--skip-sanity-plot` | Skip the seed HUXt run and HUXt sanity plots. |

The preparation script is cache-aware: existing GONG FITS files, WSA+ speed maps, and
boundary files are reused where possible. `--force-config` only controls the YAML
overwrite; remove event files manually to regenerate intermediate products.

---

## Known pitfalls

- **WSA+ checkpoint.** `generate_huxt_input.py` needs `data_dir/sw/wsaplus.pt`; it is
  not shipped. The WSA+ step is skipped/falls back if it is missing.
- **OMNI baseline file naming.** `read_huxt_output()` expects
  `HUXt_CR<cr>_<tag>_earth_timeseries_omni.npz` in the data dir; pre-compute it if you
  compare against OMNI.
- **`v_boundary` units.** Stored as a bare numpy array; loaders re-attach `u.km/u.s`.
- **Few hit cases.** If too few samples are classified as hits, inspect
  `max_detector_value` / `max_speed_enhancement` and the threshold diagnostic plots,
  and consider expanding the design around longitude and width for strong/wide events.

---

## References

- HUXt model: Owens et al. (2020), *Sol. Phys.* 295, 43. DOI: [10.1007/s11207-020-01605-3](https://doi.org/10.1007/s11207-020-01605-3)
- HUXt software: Barnard & Owens (2022), *Front. Phys.* DOI: [10.3389/fphy.2022.1005621](https://doi.org/10.3389/fphy.2022.1005621)
- Gaussian Processes: Rasmussen & Williams, [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/)
