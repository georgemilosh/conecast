#!/usr/bin/env python
"""Create tutorial notebooks for the HUXt GP surrogate work."""

from __future__ import annotations

import json
from textwrap import dedent
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = BASE_DIR / "notebooks"


def md(source: str) -> dict:
    text = dedent(source).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    text = dedent(source).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / filename
    with path.open("w", encoding="utf-8") as stream:
        json.dump(notebook(cells), stream, indent=2)
        stream.write("\n")
    print(f"Wrote {path}")


def common_setup_cell() -> dict:
    return code(
        r"""
        from pathlib import Path
        import sys

        cwd = Path.cwd().resolve()
        if (cwd / "scripts").exists():
            BASE_DIR = cwd
        elif (cwd.parent / "scripts").exists():
            BASE_DIR = cwd.parent
        else:
            # Fallback: assume the notebook is run from inside the repo.
            BASE_DIR = cwd

        SCRIPT_DIR = BASE_DIR / "scripts"
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))

        print("BASE_DIR =", BASE_DIR)
        """
    )


def colab_bootstrap_cell(install_deps: bool = True) -> dict:
    """Optional Google Colab setup. A no-op when running locally.

    `install_deps=False` is used by notebooks that only need NumPy/scikit-learn/matplotlib
    (preinstalled on Colab), so they avoid the pip install that perturbs NumPy.
    """
    if not install_deps:
        return code(
            r"""
            # --- Google Colab note (no-op locally) ---
            import sys
            if "google.colab" in sys.modules:
                print("Colab: this notebook is self-contained - numpy / scikit-learn / matplotlib "
                      "are preinstalled, so no setup or runtime restart is needed.")
            else:
                print("Not in Colab - using the local checkout.")
            """
        )
    return code(
        r"""
        # --- Google Colab bootstrap (no-op locally) ---
        import sys, os

        if "google.colab" in sys.modules:
            REPO_URL = os.environ.get("CONECAST_REPO", "https://github.com/georgemilosh/conecast")
            REPO_DIR = "/content/conecast"
            if not os.path.isdir(REPO_DIR):
                os.system(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
            os.chdir(REPO_DIR)
            # Decide what to do from whether the deps actually import (not from the clone existing),
            # so a failed/partial install self-heals on a re-run.
            try:
                import sunpy, huxt, wsaplus  # noqa: F401
                print("Colab bootstrap complete; cwd =", os.getcwd())
            except ModuleNotFoundError:
                print("Installing sunpy + WSA+ + HUXt (one-time, ~2 min)...")
                # Install in two steps so sunpy/wsaplus land even if the git build of HUXt is slow.
                os.system("pip install -q sunpy wsaplus")
                os.system("pip install -q "
                          "'huxt @ git+https://github.com/University-of-Reading-Space-Science/HUXt'")
                # Those installs pull a newer NumPy/SciPy; restart so it loads cleanly.
                print("Done - restarting the runtime. When it reconnects, RUN THIS CELL AGAIN "
                      "(or Runtime > Run all).")
                os.kill(os.getpid(), 9)
            # The WSA+ checkpoint (~317 MB) is fetched from Zenodo on demand by notebook 02.
        else:
            print("Not in Colab - using the local checkout.")
        """
    )


def gp_tutorial_notebook() -> list[dict]:
    return [
        md(
            r"""
            # 01. Gaussian Processes: A Tutorial Introduction

            **Why this notebook exists.** The rest of this series uses a Gaussian Process (GP) as a cheap stand-in for an expensive HUXt simulation (notebook 04). Before applying GPs to CMEs, it helps to see what a GP *is* on small, familiar problems. This notebook builds that intuition with two standard, non-CME examples; it complements the longer write-up in `runs/gp_surrogate/GP_TUTORIAL.md`.

            **The one idea to take away.** A GP does not just draw a best-fit curve - it returns, at every point, a **prediction *and* an honest uncertainty**. The uncertainty is small near data you have seen and large where you have not. That "I know how much I don't know" is exactly what makes a GP useful for deciding where to spend the next expensive simulation (active learning, notebook 04, Task 4e).

            **Roadmap:**

            1. What a GP is, in words and in one equation.
            2. The regression formula that produces the mean and uncertainty.
            3. **Example 1** - the GP prior vs posterior on 1-D data: sample functions from each and watch the uncertainty collapse at the observations.
            4. **Example 2** - a GP as a 2-input "simulator" emulator (the pattern notebook 04 uses).
            5. A practical checklist for building a surrogate.

            Set up a local environment first (see `setup.sh` / `README.md`):

            ```bash
            python -m venv .venv
            source .venv/bin/activate
            pip install -r requirements.txt
            python -m ipykernel install --user --name conecast --display-name "Python (conecast)"
            ```
            """
        ),
        colab_bootstrap_cell(install_deps=False),
        common_setup_cell(),
        md(
            r"""
            ## What Is A GP?

            A Gaussian Process is a probability distribution over **functions**. Where an ordinary fit gives you one curve, a GP gives you a whole *family* of plausible curves consistent with the data, and summarizes that family by a mean and a spread:

            $$
            f(x) \sim \mathcal{GP}(m(x), k(x, x'))
            $$

            - The **mean function** `m(x)` is the expected function value (often taken as 0 after centering the data).
            - The **kernel** `k(x, x')` is the heart of the method: it says how *correlated* two function values are based on how close their inputs are. Nearby inputs -> highly correlated outputs (the function is smooth); distant inputs -> nearly independent.

            **Kernel intuition.** The kernel carries two knobs you will see throughout: an **output scale** (how far the function swings up and down) and a **length scale** (how far you must move in `x` before the function changes appreciably). A short length scale means a wiggly function; a long one means a slowly-varying function. In the CME work, fitting a separate length scale per parameter is what tells us which parameters the output is sensitive to.

            After observing data, a GP gives:

            - a posterior **mean** prediction,
            - a posterior **uncertainty** (standard deviation),
            - **wider** uncertainty far from training data (extrapolation),
            - **narrower** uncertainty near informative data (interpolation).

            > **Mental model.** Pin a flexible sheet to your data points. Between nearby pins the sheet is well-constrained (low uncertainty); far from any pin it flaps freely (high uncertainty). The kernel sets how stiff the sheet is.
            """
        ),
        md(
            r"""
            ## Regression Formula

            For noisy observations

            $$
            y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_n^2),
            $$

            the predictive distribution at a new point is Gaussian:

            $$
            f(x_*) \mid X, y, x_* \sim \mathcal{N}(\mu_*, \sigma_*^2).
            $$

            With covariance matrix `K` and covariance vector `k_*`,

            $$
            \mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y
            $$

            and

            $$
            \sigma_*^2 = k(x_*, x_*) - k_*^T (K + \sigma_n^2 I)^{-1} k_*.
            $$

            **Reading the formulas in words:**

            - `K` is the kernel evaluated between every pair of *training* points; `k_*` is the kernel between the new point and each training point. Both come straight from your kernel choice.
            - The **mean** $\mu_*$ is a weighted average of the observed `y`, with more weight on training points whose inputs are close to $x_*$ (large `k_*`). Far from all data, `k_*` -> 0 and the mean falls back to the prior mean.
            - The **variance** $\sigma_*^2$ starts at the prior variance `k(x_*, x_*)` and is *reduced* by whatever the data already explain. Near data, the subtracted term is large -> small uncertainty; far away it vanishes -> uncertainty returns to the prior.
            - $\sigma_n^2$ is the assumed observation **noise**; adding it on the diagonal is what lets the mean curve pass *near* (not exactly through) noisy points.

            > **Side note (cost).** That matrix inverse is $O(n^3)$ in the number of training points - fine for the hundreds of HUXt runs here, but the reason GPs are reserved for *expensive* models with modest sample counts rather than big data.
            """
        ),
        md(
            r"""
            ## Example 1: Prior And Posterior On 1-D Data

            This is the canonical picture of a GP (compare Rasmussen & Williams, Fig. 2.2). A GP is first a **prior over functions** - before seeing any data it already says "the function is smooth, with this amplitude and length scale". We can *sample* random functions from that prior. **Conditioning** on a few observations turns the prior into a **posterior**: the subset of those functions that still pass through (or near) the data. We sample from the posterior too.

            We use **noise-free** observations here on purpose. With no observation noise the posterior is forced *exactly* through each data point, so the uncertainty there collapses to zero - the lesson that was hidden when we added a noise term.

            **Left panel (prior):** five functions drawn from `GP(0, k)` with the mean (flat 0) and the +/-2σ band. Every sample is a different but equally-plausible smooth curve, and the uncertainty is the **same everywhere** - we have no data yet.

            **Right panel (posterior):** the same GP after conditioning on five red points. Now every sampled function threads the data, the mean is our best estimate, and the band **pinches to zero at each observation** and balloons between and beyond them.

            > **What to look for:** the band touching zero at every red dot, and the posterior samples fanning out only where there is no data. (To model *noisy* data instead, add a `WhiteKernel` or set `alpha > 0`; the band then stays finite at the points - this is what notebook 04 does.)
            """
        ),
        code(
            r"""
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF

            # A smooth ground truth we reveal at only a few points.
            def truth(x):
                return np.sin(x) + 0.3 * x

            x = np.linspace(0.0, 10.0, 400).reshape(-1, 1)

            # Output-scale * squared-exponential kernel. optimizer=None fixes the hyperparameters
            # (deterministic demo); alpha ~ 0 means noise-free, so the posterior interpolates exactly.
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer=None)

            # --- PRIOR: sample functions before seeing any data ---
            prior_samples = gp.sample_y(x, n_samples=5, random_state=1)
            prior_mean, prior_std = gp.predict(x, return_std=True)

            # --- POSTERIOR: condition on five noise-free observations ---
            x_train = np.array([1.0, 3.0, 5.5, 7.0, 9.0]).reshape(-1, 1)
            y_train = truth(x_train).ravel()
            gp.fit(x_train, y_train)
            post_samples = gp.sample_y(x, n_samples=5, random_state=1)
            post_mean, post_std = gp.predict(x, return_std=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)

            axes[0].fill_between(x.ravel(), prior_mean - 2 * prior_std, prior_mean + 2 * prior_std,
                                 alpha=0.2, color="tab:blue", label="+/-2 std (95%)")
            axes[0].plot(x, prior_samples, lw=1.0, alpha=0.8)
            axes[0].plot(x, prior_mean, "k", lw=2, label="mean")
            axes[0].set_title("Prior: functions before seeing data")
            axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)"); axes[0].legend(loc="upper left")

            axes[1].fill_between(x.ravel(), post_mean - 2 * post_std, post_mean + 2 * post_std,
                                 alpha=0.2, color="tab:blue", label="+/-2 std (95%)")
            axes[1].plot(x, post_samples, lw=1.0, alpha=0.8)
            axes[1].plot(x, post_mean, "k", lw=2, label="mean")
            axes[1].scatter(x_train, y_train, c="red", zorder=5, label="observations")
            axes[1].set_title("Posterior: functions after conditioning on data")
            axes[1].set_xlabel("x"); axes[1].legend(loc="upper left")
            """
        ),
        md(
            r"""
            ## Example 2: Two-Input Simulation

            This is the pattern the CME work uses, in miniature. We pretend a synthetic function is an **expensive simulator** with two inputs - temperature and pressure - and build a GP **surrogate** for it:

            1. **Space-filling design** (`LatinHypercube`): choose 45 input combinations that cover the 2-D box evenly (exactly what notebook 04's `design` step does in 5-D).
            2. **Run the "simulator"** at those points and **scale** inputs/outputs.
            3. **Fit** a 2-D GP (one length scale per input).
            4. **Predict** on a dense grid and plot **two** surfaces: the mean response and the GP's uncertainty.

            > **What to look for:** in the right-hand uncertainty panel, the valleys sit **on the design points** (white markers) and the ridges sit in the gaps between them. That map of "where am I unsure" is precisely what drives next-run selection in notebook 04 (Task 4e) - you add simulations where the surrogate is least certain.
            """
        ),
        code(
            r"""
            from scipy.stats import qmc
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
            from sklearn.preprocessing import StandardScaler

            low = np.array([300.0, 1.0])
            high = np.array([1200.0, 5.0])
            sampler = qmc.LatinHypercube(d=2, seed=7)
            X_design = qmc.scale(sampler.random(n=45), low, high)

            def simulator(x):
                temperature, pressure = x
                return np.sin(temperature / 145.0) + 0.35 * pressure - 0.0000022 * (temperature - 820.0) ** 2

            y_design = np.array([simulator(x) for x in X_design])

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            Xs = x_scaler.fit_transform(X_design)
            ys = y_scaler.fit_transform(y_design.reshape(-1, 1)).ravel()

            kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(2), nu=2.5) + WhiteKernel(noise_level=0.002)
            gp2 = GaussianProcessRegressor(kernel=kernel, optimizer=None, random_state=8)
            gp2.fit(Xs, ys)

            temp = np.linspace(low[0], high[0], 90)
            pressure = np.linspace(low[1], high[1], 80)
            tt, pp = np.meshgrid(temp, pressure)
            X_grid = np.column_stack([tt.ravel(), pp.ravel()])
            mean_s, std_s = gp2.predict(x_scaler.transform(X_grid), return_std=True)
            mean = y_scaler.inverse_transform(mean_s.reshape(-1, 1)).reshape(tt.shape)
            std = (std_s * y_scaler.scale_[0]).reshape(tt.shape)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
            im0 = axes[0].contourf(tt, pp, mean, levels=18)
            axes[0].scatter(X_design[:, 0], X_design[:, 1], s=16, color="white", edgecolor="black")
            axes[0].set_title("GP mean")
            axes[0].set_xlabel("temperature [K]")
            axes[0].set_ylabel("pressure [bar]")
            fig.colorbar(im0, ax=axes[0])

            im1 = axes[1].contourf(tt, pp, std, levels=18)
            axes[1].scatter(X_design[:, 0], X_design[:, 1], s=16, color="white", edgecolor="black")
            axes[1].set_title("GP uncertainty")
            axes[1].set_xlabel("temperature [K]")
            axes[1].set_ylabel("pressure [bar]")
            fig.colorbar(im1, ax=axes[1])
            """
        ),
        md(
            r"""
            ## Practical Checklist

            This is the recipe the rest of the series follows. Map each step to its notebook-04 task as you go:

            1. Define input parameters and output quantities. *(the 5 Cone-CME parameters; hit + arrival time)*
            2. Choose physically meaningful bounds. *(Task 1: priors + spans)*
            3. Create a space-filling design. *(Task 1: Latin hypercube)*
            4. Run the expensive model or experiment. *(Task 2: HUXt + detector)*
            5. Scale inputs and outputs. *(Task 3)*
            6. Fit the GP. *(Task 3: classifier + regressor)*
            7. Validate with held-out data and uncertainty coverage. *(Task 3: holdout MAE)*
            8. Use the GP for sensitivity analysis, uncertainty propagation, or active learning. *(Task 4a-4e)*

            > **Common pitfalls.** Forgetting to scale (length scales become meaningless); too few design points for the dimensionality (the surrogate extrapolates wildly - watch the uncertainty); and trusting the mean where the GP reports large uncertainty. The uncertainty band is not decoration - it is the model telling you when to stop trusting it.

            Useful references:

            - Rasmussen and Williams, [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/)
            - scikit-learn, [Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
            - Duvenaud, [The Kernel Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/index.html)
            """
        ),
    ]


def huxt_runs_notebook() -> list[dict]:
    return [
        md(
            r"""
            # 02. Preparing HUXt Inputs, One Operation At A Time

            **What this notebook is.** HUXt is a fast solar-wind model: given a map of the wind speed at an inner boundary (a few tens of solar radii) plus a Cone-CME description, it advects that wind out to 1 AU and tells you when a CME reaches Earth. But HUXt cannot start from nothing - it needs a *physically grounded* boundary condition. Building that boundary, for a specific real CME event, is the job of `scripts/generate_huxt_input.py`.

            **What we do here.** We take that script apart **one operation at a time**. Each task below reimplements a single step inline on local data, names the script function it mirrors, and explains *why* the step exists in the chain. The end product is a prepared event directory `data_dir/sw/<event>/` containing a HUXt boundary file and a seed-parameter config.

            **The physical chain**, from the Sun's surface to a HUXt-ready input:

            ```text
            photospheric magnetic field   (GONG magnetogram)
                -> coronal + solar-wind speed map   (WSA+)
                -> speed Earth actually sees         (sub-Earth track sampling)
                -> 1-D inner boundary for HUXt       (v_boundary_<event>.npz)
                -> + CME launch time and geometry    (event_config.yaml seed)
            ```

            **Cost.** Two steps are expensive: the GONG download needs the network (kept guarded), and the WSA+ map needs the `wsaplus.pt` checkpoint. WSA+ **runs live** here (with a cached fallback); everything else runs live on the local files.

            Task order follows the script's own call chain:

            1. **`load_events`** - read the seed CME parameters from `events.csv`.
            2. **`download_gong_mag`** - fetch GONG magnetograms near onset (guarded).
            3. **`find_closest_map`** - pick the magnetogram closest to onset and read its Carrington rotation.
            4. **`run_wsaplus`** - build the longitude-latitude WSA+ speed map (live).
            5. **`compute_subearth_track`** - trace Earth through the rotation.
            6. **`sample_interpolated` / `sample_nearest`** - sample the map along that track.
            7. **`map_input_huxt`** - reduce the track to the 1-degree HUXt boundary.
            8. **injection time + `create_config`** - compute `inject_hour` and write the seed config.
            """
        ),
        colab_bootstrap_cell(),
        common_setup_cell(),
        md(
            r"""
            ## Pipeline Overview

            `generate_huxt_input.py` is one call chain, run once per selected event:

            ```text
            main()
              ├─ load_events()
              └─ process_event()                  one call per event
                   ├─ prepare_background()
                   │    ├─ download_gong_mag()         optional, needs network
                   │    ├─ find_closest_map()
                   │    ├─ run_wsaplus()               skipped if WSA+ cache exists
                   │    └─ map_input_huxt()
                   │         ├─ compute_subearth_track()
                   │         ├─ sample_interpolated()
                   │         └─ sample_nearest()
                   ├─ rhf.run_huxt_sim()               optional seed sanity run
                   └─ write_event_config()             writes event_config.yaml
            ```

            The five Cone-CME parameters carried through to the seed config are:

            | index | name | units | meaning |
            | --- | --- | --- | --- |
            | 0 | `inject_hour` | h | launch time after model start |
            | 1 | `longitude` | deg | central longitude |
            | 2 | `latitude` | deg | central latitude |
            | 3 | `width` | deg | angular half-width |
            | 4 | `v` (`speed`) | km/s | initial CME speed |

            Only `inject_hour` is computed (from the magnetogram time and `cme_0p1_au`); the other four come straight from the CSV.

            > **Why a separate "background" step?** A CME does not travel through a vacuum - it ploughs into the ambient solar wind, which can be fast or slow depending on where you look. Getting the *arrival time* right means getting that background right first. Tasks 2-7 build the background; Task 8 adds the CME on top.
            """
        ),
        md(
            r"""
            ## Setup

            Import the shared libraries and choose one event. `event_dir` is the per-event directory the script fills in; every task re-loads what it needs from it, so the tasks can be run independently once this cell has executed.

            > **Prepared data required (Tasks 3-9).** This repo ships only source + the seed config; the GONG magnetograms, WSA+ map, and boundary are *not* included. Tasks 1-2 run anywhere, but Tasks 3 onward need prepared inputs for the chosen event. Generate them once with `python scripts/generate_huxt_input.py --event <event>` (downloads GONG + runs WSA+; needs `data_dir/sw/wsaplus.pt`). Until then, Task 3 stops with a clear message.

            > **Tip.** Change `event` to any prepared event to re-run the whole walkthrough for a different CME.
            """
        ),
        code(
            r"""
            import numpy as np
            import pandas as pd
            import yaml
            import matplotlib.pyplot as plt
            from pathlib import Path
            import astropy.units as u
            from astropy.time import Time
            from IPython.display import display

            DATA_ROOT = BASE_DIR / "data_dir" / "sw"
            EVENTS_CSV = BASE_DIR / "data_dir" / "events.csv"

            event = "2017-09-06"
            event_dir = DATA_ROOT / event
            print("event:", event, "| dir exists:", event_dir.exists())
            for path in sorted(event_dir.glob("*"))[:10]:
                print("  ", path.name)
            """
        ),
        md(
            r"""
            ## Task 1: Read The Event Catalogue

            *Mirrors `load_events()` and `truthy()`.*

            Everything starts from `data_dir/events.csv`, a small catalogue where each row is one observed CME with the numbers needed to seed a simulation. These come from observations (coronagraph fits, CME catalogues) - they are the *first guess* that the downstream GP surrogate work then refines.

            **Columns the script requires:**

            | column | meaning |
            | --- | --- |
            | `event` | event label, also the output directory name |
            | `cme_onset` | time the CME is first seen (used to pick the magnetogram) |
            | `cme_0p1_au` | time the CME front reaches 0.1 AU (sets the launch time) |
            | `longitude`, `latitude` | CME nose direction in degrees |
            | `width` | angular half-width of the cone |
            | `speed` | initial radial speed (km/s) |

            An optional `enabled` column lets you switch rows off without deleting them; `truthy()` decides what counts as "on". This task reads the CSV, checks the required columns exist, drops disabled rows, and pulls out one event.

            > **What to look for:** the selected row's `speed` and `width` are the CME's headline properties; `cme_onset` and `cme_0p1_au` are a few hours apart and together pin down *when* to launch.
            """
        ),
        code(
            r"""
            REQUIRED = {"event", "cme_onset", "cme_0p1_au", "longitude", "latitude", "width", "speed"}

            def truthy(value):
                return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

            events_table = pd.read_csv(EVENTS_CSV)
            print("missing required columns:", (REQUIRED - set(events_table.columns)) or "none")

            if "enabled" in events_table.columns:
                keep = events_table["enabled"].map(lambda v: str(v).strip() == "" or truthy(v))
                events_table = events_table[keep]
            display(events_table.head())

            selected_row = events_table.loc[events_table["event"] == event].iloc[0]
            print("seed row for", event, ":")
            display(selected_row[["event", "cme_onset", "cme_0p1_au", "longitude", "latitude", "width", "speed"]])
            """
        ),
        md(
            r"""
            ## Task 2: Download The GONG Magnetogram (guarded)

            *Mirrors `download_gong_mag()`.*

            **What GONG is.** The Global Oscillation Network Group operates ground-based solar telescopes that, among other things, publish *synoptic magnetograms*: full-Sun maps of the photospheric line-of-sight magnetic field in Carrington coordinates, updated roughly hourly. That surface field is the boundary condition for any coronal model - it is where the solar wind ultimately comes from.

            **Why near the onset.** The corona evolves, so we want the magnetic map that best represents the Sun *at the moment the CME launched*. The script searches a 6-hour window ending at `cme_onset` and takes what GONG has there.

            **How the fetch works.** SunPy's `Fido` is a federated search/download client: `Fido.search(Time(...), Instrument("GONG"))` finds matching files and `Fido.fetch(...)` downloads them. GONG files arrive gzip-compressed (`.fits.gz`); Task 3 decompresses them to `.fits` before reading.

            > **Flaky server.** `gong2.nso.edu` sometimes drops connections (SSL / connect errors) and only part of a batch downloads. That is fine - Task 3 just picks the closest of whatever arrived, so one good map is enough. Re-run this cell to retry the rest.

            This needs network access, so the live fetch is guarded by `RUN_DOWNLOAD`. We instead list the FITS already present - exactly what a real run would have left behind.

            > **Side note.** "Synoptic" means the map is assembled over a full rotation, so a single magnetogram already spans all 360 deg of Carrington longitude - that is what makes it usable as a global boundary later.
            """
        ),
        code(
            r"""
            from sunpy.net import Fido, attrs as a

            t_cme = Time(selected_row["cme_onset"], scale="utc")
            t_start_mag = t_cme - 6 * u.hour
            t_end_mag = t_cme
            print("CME onset:", t_cme.isot)
            print("GONG search window:", t_start_mag.isot, "->", t_end_mag.isot)

            local_fits = sorted(event_dir.glob("*.fits"))
            print("local GONG FITS:", [p.name for p in local_fits])

            RUN_DOWNLOAD = False   # set True to query/fetch over the network
            if RUN_DOWNLOAD:
                res = Fido.search(a.Time(t_start_mag, t_end_mag), a.Instrument("GONG"))
                print(res)
                files = Fido.fetch(res, path=str(event_dir / "{file}"))
                print("downloaded:", files)
            elif local_fits:
                print("Skipping live download; reusing the local FITS above.")
            else:
                print("No local GONG FITS found. The data tasks (3-9) need prepared inputs.")
                print("Generate them with (downloads GONG + runs WSA+; needs data_dir/sw/wsaplus.pt):")
                print(f"  python scripts/generate_huxt_input.py --event {event}")
                print("...or set RUN_DOWNLOAD = True above to fetch just the magnetograms here.")
            """
        ),
        md(
            r"""
            ## Task 3: Select The Closest Magnetogram

            *Mirrors `find_closest_map()`.*

            The download window may have left several FITS files. This step loads each one as a SunPy `Map`, reads its observation time, and keeps the one with the smallest `|obs_time - cme_onset|`.

            **Carrington rotation number.** From the chosen map's date we read its *Carrington rotation* (CR) number - a running count of solar rotations (~27.27 days each) used as the Sun's natural calendar. The CR number, a non-integer here, encodes both which rotation and how far through it we are; Task 5 uses it to work out where Earth sits in Carrington longitude.

            **Reading the magnetogram.** The plot uses a symmetric color range (`vmin = -vmax`) so that zero field is white, red is one magnetic polarity and blue the other. The large bipolar regions are active regions / sunspot groups; the quiet background is weak mixed field.

            > **What to look for:** the printed `dt[hr]` should be small (a fraction of an hour to a few hours) - if it is large, GONG had a gap near onset and the background is less trustworthy.
            """
        ),
        code(
            r"""
            import sunpy.map
            from sunpy.coordinates.sun import carrington_rotation_number
            from astropy.io import fits

            # GONG magnetograms arrive gzipped; decompress any *.fits.gz so they are picked up
            # (mirrors download_gong_mag() in generate_huxt_input.py).
            for gzfile in sorted(event_dir.glob("*.fits.gz")):
                fitsfile = gzfile.with_suffix("")  # strip .gz -> .fits
                with fits.open(gzfile) as hdul:
                    hdul.writeto(fitsfile, overwrite=True)
                gzfile.unlink()
                print("decompressed:", fitsfile.name)

            rows = []
            for fits_path in sorted(event_dir.glob("*.fits")):
                try:
                    gong_map = sunpy.map.Map(fits_path)
                    rows.append({
                        "file": fits_path.name,
                        "obs_time": gong_map.date.isot,
                        "dt_hours": abs((gong_map.date - t_cme).to(u.hour).value),
                    })
                except Exception as exc:
                    print("skip", fits_path.name, "|", exc)

            if not rows:
                raise FileNotFoundError(
                    f"No GONG FITS files in {event_dir}.\n"
                    f"Tasks 3-9 need prepared inputs for this event. Generate them first with:\n"
                    f"  python scripts/generate_huxt_input.py --event {event}\n"
                    f"(downloads GONG magnetograms + runs WSA+; needs data_dir/sw/wsaplus.pt)."
                )

            magnetograms = pd.DataFrame(rows).sort_values("dt_hours").reset_index(drop=True)
            display(magnetograms)

            closest_file = event_dir / magnetograms.iloc[0]["file"]
            closest_map = sunpy.map.Map(closest_file)
            closest_time = closest_map.date
            cr_num = float(carrington_rotation_number(closest_time))
            print("closest:", closest_file.name, "| dt[hr]:", magnetograms.iloc[0]["dt_hours"], "| CR:", cr_num)

            lim = float(np.nanmax(np.abs(closest_map.data)))
            plt.figure(figsize=(8, 4))
            closest_map.plot(cmap="RdBu_r", vmin=-lim, vmax=lim)
            plt.colorbar(label="B [G]")
            plt.title(f"Closest GONG magnetogram: {closest_file.name}")
            plt.tight_layout()
            """
        ),
        md(
            r"""
            ## Task 4: Build The WSA+ Speed Map (live)

            *Mirrors `run_wsaplus()`.*

            **What WSA+ does.** WSA (Wang-Sheeley-Arge) is the classic empirical recipe for turning a photospheric magnetogram into a solar-wind *speed* map: trace the coronal magnetic field (a potential-field source-surface extrapolation), measure how fast flux tubes expand and how far each footpoint sits from the nearest coronal-hole boundary, and feed those two geometric quantities into an empirical speed formula. Open-field regions (coronal holes) give fast wind; field near the streamer belt gives slow wind. The "+" here is a machine-learning-enhanced variant whose weights live in `wsaplus.pt`.

            **Input and output.** Input: the single magnetogram from Task 3. Output: a 2-D map of wind speed on a `(longitude, latitude)` grid - `speed_kms` over `phi_grid_deg` x `theta_grid_deg`. This is still a *full-Sun* map; it is not yet the 1-D HUXt boundary.

            **Checkpoint.** The `wsaplus.pt` weights (~317 MB) are not shipped; the cell **downloads them from Zenodo on first use** (via `scripts/fetch_wsaplus_checkpoint.py`, DOI 10.5281/zenodo.16883042). This is the one genuinely heavy compute step (neural-network inference), so the result is reusable as `wsaplus_speed_map_<event>.npz`.

            > **What to look for:** broad fast-wind (>600 km/s) patches over coronal holes, and a slow-wind band (~300-400 km/s) following the magnetic neutral line. Those structures are what the CME will run into.
            """
        ),
        code(
            r"""
            checkpoint_path = DATA_ROOT / "wsaplus.pt"
            wsaplus_cache = event_dir / f"wsaplus_speed_map_{event}.npz"

            # The WSA+ checkpoint (~317 MB) is not shipped; fetch it from Zenodo on first use.
            if not checkpoint_path.exists() and not wsaplus_cache.exists():
                import fetch_wsaplus_checkpoint as fw
                fw.download(checkpoint_path)
            print("checkpoint exists:", checkpoint_path.exists(), "| cache exists:", wsaplus_cache.exists())

            res = None
            try:
                from wsaplus import generate_wsaplus_map
                res = generate_wsaplus_map(closest_map, mag_type="GONG", checkpoint_path=str(checkpoint_path))
                print("Ran WSA+ live.")
            except Exception as exc:
                if wsaplus_cache.exists():
                    print("WSA+ live run unavailable (", type(exc).__name__, "); loading cached map.")
                    res = np.load(wsaplus_cache, allow_pickle=True)["speed_map"].item()
                else:
                    raise RuntimeError(
                        f"WSA+ could not run ({type(exc).__name__}: {exc}).\n"
                        f"The checkpoint is data_dir/sw/wsaplus.pt; fetch it with "
                        f"`python scripts/fetch_wsaplus_checkpoint.py`."
                    ) from exc

            print("speed_kms:", res.speed_kms.shape, float(np.nanmin(res.speed_kms)), float(np.nanmax(res.speed_kms)))
            print("phi_grid_deg:", res.phi_grid_deg.shape)
            print("theta_grid_deg:", res.theta_grid_deg.shape)

            plt.figure(figsize=(8, 3.5))
            plt.pcolormesh(res.phi_grid_deg, res.theta_grid_deg, res.speed_kms, shading="auto", cmap="viridis")
            plt.colorbar(label="v [km/s]")
            plt.xlabel("Carrington longitude [deg]")
            plt.ylabel("latitude [deg]")
            plt.title(f"{event}: WSA+ speed map")
            plt.tight_layout()
            """
        ),
        md(
            r"""
            ## Task 5: Compute The Sub-Earth Carrington Track

            *Mirrors `compute_subearth_track()`.*

            HUXt is run in the ecliptic plane, so the boundary it needs is the wind speed **along the path Earth occupies** as the Sun rotates beneath it - not the whole 2-D map. The *sub-Earth point* is the spot on the Sun directly below Earth; over one Carrington rotation it sweeps through all 360 deg of longitude and wobbles a little in latitude.

            **What this task computes.** For the selected rotation it steps Earth hour-by-hour from the CR start to CR+1, transforms Earth's position into Heliographic Carrington coordinates, and records `(longitude, latitude)` at each step - the curve we will sample the speed map along.

            **Why latitude matters.** Earth's heliographic latitude (the B0 angle) drifts roughly +/-7.25 deg over the year. That offset means the sub-Earth track is *not* simply the map's equator - sampling the true track is what makes the boundary specific to this event's date.

            > **Side note.** Carrington longitude of the sub-Earth point *decreases* with time, because the Sun rotates eastward under a (more slowly moving) Earth. Don't be surprised that the track runs "backwards" in longitude.
            """
        ),
        code(
            r"""
            import sunpy.coordinates.sun
            from sunpy.coordinates import frames, ephemeris

            t_start = sunpy.coordinates.sun.carrington_rotation_time(cr_num)
            t_end = sunpy.coordinates.sun.carrington_rotation_time(cr_num + 1)
            dt = t_end - t_start
            n_hr = int(dt.value * 24)
            obs_time = t_start + dt * np.linspace(1e-6, 1 - 1e-6, n_hr, endpoint=False)

            # Vectorized: one get_earth + one transform for the whole time array (the per-sample
            # loop is hundreds of calls and is very slow on Colab, where each call hits the network).
            coords = ephemeris.get_earth(time=obs_time).transform_to(
                frames.HeliographicCarrington(observer="earth")
            )
            SBElon = np.asarray(coords.lon.value, dtype=float)
            SBElat = np.asarray(coords.lat.value, dtype=float)

            print(f"CR {cr_num:.2f}: {t_start.isot} -> {t_end.isot} | {n_hr} hourly samples")
            print("lon range:", float(SBElon.min()), float(SBElon.max()))
            print("lat range:", float(SBElat.min()), float(SBElat.max()))
            """
        ),
        md(
            r"""
            ## Task 6: Sample The Map Along The Track

            *Mirrors `sample_interpolated()` and `sample_nearest()`.*

            Now we read the WSA+ speed off the 2-D map at every `(lon, lat)` on the sub-Earth track, using two methods:

            - **Interpolated** (`RegularGridInterpolator`): blends the four surrounding grid cells, giving a smooth speed series.
            - **Nearest grid cell**: just takes the value of the closest cell - what the map literally says there.

            The script keeps both for comparison but writes the **nearest-grid** version as the production boundary, because it preserves the map's native values without interpolation artefacts at sharp coronal-hole edges.

            **The two panels:** left shows the track laid over the speed map (red = exact track, black = the nearest cells actually used); right shows the speed sampled along the track by each method.

            > **What to look for:** the two curves should sit almost on top of each other, separating only where the map has steep gradients (fast/slow wind boundaries). Big separations flag places where the 1-D boundary is sensitive to the sampling choice.
            """
        ),
        code(
            r"""
            from scipy.interpolate import RegularGridInterpolator

            speed_map = np.asarray(res.speed_kms)
            lon_vals = np.asarray(res.phi_grid_deg[:, 0])
            lat_vals = np.asarray(res.theta_grid_deg[0, :])

            interp = RegularGridInterpolator((lon_vals, lat_vals), speed_map, bounds_error=False, fill_value=np.nan)
            speed_interp = interp(np.column_stack([SBElon, SBElat]))

            lon_nearest, lat_nearest, speed_nearest = [], [], []
            for lon, lat in zip(SBElon, SBElat):
                i_lon = int(np.argmin(np.abs(lon_vals - lon)))
                i_lat = int(np.argmin(np.abs(lat_vals - lat)))
                lon_nearest.append(lon_vals[i_lon])
                lat_nearest.append(lat_vals[i_lat])
                speed_nearest.append(speed_map[i_lon, i_lat])
            lon_nearest = np.asarray(lon_nearest)
            lat_nearest = np.asarray(lat_nearest)
            speed_nearest = np.asarray(speed_nearest)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            mesh = axes[0].pcolormesh(lon_vals, lat_vals, speed_map.T, shading="auto")
            axes[0].scatter(SBElon, SBElat, s=8, c="red", label="sub-Earth track")
            axes[0].scatter(lon_nearest, lat_nearest, s=5, c="black", alpha=0.5, label="nearest cells")
            axes[0].set_xlabel("Carrington longitude [deg]")
            axes[0].set_ylabel("latitude [deg]")
            axes[0].set_title("Track on WSA+ map")
            axes[0].legend(loc="upper right")
            fig.colorbar(mesh, ax=axes[0], label="v [km/s]")

            axes[1].plot(speed_interp, label="interpolated")
            axes[1].plot(speed_nearest, label="nearest", alpha=0.8)
            axes[1].set_xlabel("hourly sample index")
            axes[1].set_ylabel("v [km/s]")
            axes[1].set_title("Speed sampled along the track")
            axes[1].legend()
            """
        ),
        md(
            r"""
            ## Task 7: Reduce The Track To The HUXt Boundary

            *Mirrors the tail of `map_input_huxt()`.*

            HUXt's inner boundary is a 1-D array: one wind speed per Carrington longitude degree (360 values). The track samples from Task 6 are in time order, so we **sort them by longitude** and resample onto `np.arange(1, 361)`. That array is exactly what HUXt advects outward.

            **This cell writes the boundary** to `v_boundary_<event>.npz` (the same file and `speed_map` key that `generate_huxt_input.py` produces), so the GP workflow - notebook 04's live HUXt run and `gp_huxt_surrogate.py` - can load it. If a boundary already exists, it is compared before being overwritten (a re-run with the same WSA+ map reproduces it to the last digit).

            > **What to look for:** the interpolated and nearest profiles should track closely, and the cell should report that it wrote a 360-point `v_boundary_<event>.npz`.
            """
        ),
        code(
            r"""
            lon_grid_360 = np.arange(1, 361)

            def reduce_to_boundary(sample_lon, sample_speed):
                order = np.argsort(sample_lon)
                return np.interp(lon_grid_360, np.asarray(sample_lon)[order], np.asarray(sample_speed)[order])

            speed_nearest_360 = reduce_to_boundary(lon_nearest, speed_nearest)
            speed_interp_360 = reduce_to_boundary(SBElon, speed_interp)
            boundary = speed_nearest_360   # the production (nearest-grid) HUXt inner boundary

            plt.figure(figsize=(9, 3.5))
            plt.plot(lon_grid_360, speed_interp_360, label="interpolated -> 360")
            plt.plot(lon_grid_360, speed_nearest_360, label="nearest -> 360 (production)")

            boundary_file = event_dir / f"v_boundary_{event}.npz"
            # If a boundary already exists (from a previous run or generate_huxt_input.py), compare first.
            if boundary_file.exists():
                prev = np.load(boundary_file)["speed_map"]
                plt.plot(lon_grid_360, prev, "--", label="existing v_boundary", alpha=0.8)
                print("existing v_boundary vs freshly computed, max|delta|:",
                      float(np.nanmax(np.abs(boundary - prev))))

            # Persist the boundary (same file/format as generate_huxt_input.py) so the GP workflow -
            # notebook 04's live HUXt run and gp_huxt_surrogate.py - can load it.
            np.savez(boundary_file, speed_map=boundary)
            print(f"wrote {boundary_file} ({boundary.shape[0]} points)")

            plt.xlabel("Carrington longitude [deg]")
            plt.ylabel("v [km/s]")
            plt.title(f"{event}: HUXt inner-boundary speed profile")
            plt.legend()
            plt.tight_layout()
            """
        ),
        md(
            r"""
            ## Task 8: Injection Time And The Seed Config

            *Mirrors the injection-time block of `process_event()` and `write_event_config()`.*

            With the background built, we add the CME. Four of its five parameters - `longitude`, `latitude`, `width`, `speed` - come straight from the CSV row. The fifth, `inject_hour`, is *computed*: it is how long after the model's start epoch the CME should be launched, so that it crosses 0.1 AU at the observed `cme_0p1_au` time:

            ```text
            inject_hour = Time(cme_0p1_au) - closest_magnetogram_time   (in hours)
            ```

            `write_event_config()` then writes `initial_theta = [inject_hour, longitude, latitude, width, speed]` plus the `cr_num` into `event_config.yaml`. That file is the **seed** the GP-surrogate workflow starts from (it is read back in notebook 04's Task 1). This cell rebuilds it, **writes it** (so the GP workflow has the seed even on a fresh checkout), and diffs it against any pre-existing copy.

            > **What to look for:** the reconstructed `initial_theta` and `cr_num` should match the saved config row-for-row. A mismatch in `inject_hour` usually means a different magnetogram was selected in Task 3.
            """
        ),
        code(
            r"""
            inject_time = Time(selected_row["cme_0p1_au"]) - closest_time
            inject_hour = inject_time.to(u.hour).value
            print("cme_0p1_au:", selected_row["cme_0p1_au"], "| closest mag time:", closest_time.isot)
            print("inject_hour:", inject_hour)

            initial_theta = [
                inject_hour,
                selected_row["longitude"],
                selected_row["latitude"],
                selected_row["width"],
                selected_row["speed"],
            ]
            config = {
                "initial_theta": list(map(float, initial_theta)),
                "cr_num": float(cr_num),
            }

            # Write the seed config (same file write_event_config() produces) so the GP workflow
            # (notebook 04, gp_huxt_surrogate.py) has it; compare against any pre-existing one.
            config_file = event_dir / "event_config.yaml"
            if config_file.exists():
                saved_cfg = yaml.safe_load(config_file.open())
                display(pd.DataFrame({
                    "parameter": ["inject_hour", "longitude", "latitude", "width", "speed"],
                    "reconstructed": config["initial_theta"],
                    "saved_config": saved_cfg["initial_theta"],
                }))
                print("cr_num reconstructed:", float(cr_num), "| saved:", saved_cfg["cr_num"])
            else:
                display(pd.DataFrame({
                    "parameter": ["inject_hour", "longitude", "latitude", "width", "speed"],
                    "value": config["initial_theta"],
                }))
                print("No saved event_config.yaml yet (fresh checkout); cr_num =", float(cr_num))

            with config_file.open("w") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            print("wrote", config_file)
            """
        ),
        md(
            r"""
            ## Task 9: Feed The Boundary To HUXt

            *Mirrors the `HUXT_KWARGS` / `rhf.run_huxt_sim()` block of `process_event()`.*

            This is the hand-off. The 360-point boundary becomes HUXt's `v_boundary`; together with `cr_num`, the frame, and the simulation length it defines a runnable model, and the seed `theta` defines the Cone-CME to inject into it. We assemble those kwargs here so the connection between *input preparation* (this notebook) and *running HUXt* (notebook 04) is explicit.

            **The kwargs, annotated:**

            | key | role |
            | --- | --- |
            | `v_boundary` | the 360-point inner-boundary speed profile from Task 7 |
            | `cr_num` | Carrington rotation, sets the rotating-frame phase |
            | `latitude=0 deg` | run in the ecliptic plane |
            | `frame="sidereal"` | rotation frame for the inner boundary |
            | `simtime=10 day` | how long to propagate (long enough to reach 1 AU) |
            | `dt_scale=4` | output cadence (coarser than the internal time step) |

            The actual seed HUXt run + arrival detection is the **live example in notebook 04 (Task 2)**, so we don't repeat it here.
            """
        ),
        code(
            r"""
            vboundary = boundary * (u.km / u.s)   # the live nearest-360 boundary from Task 7
            HUXT_KWARGS = dict(
                v_boundary=vboundary,
                latitude=0 * u.deg,
                cr_num=cr_num,
                frame="sidereal",
                simtime=10 * u.day,
                dt_scale=4,
            )
            print("HUXT_KWARGS ready; boundary length:", len(vboundary))
            print("seed theta:", tuple(config["initial_theta"]))
            print("See notebook 04, Task 2, for a live HUXt run on this boundary.")
            """
        ),
        md(
            r"""
            ## Useful Commands

            Everything above runs as a single script from a terminal. The flags control the two expensive/destructive bits (network download and config overwrite):

            ```bash
            # one event, reusing local GONG FITS (no network), skipping the seed run:
            python scripts/generate_huxt_input.py --event 2017-09-06 --no-download --skip-sanity-plot

            # rewrite only the seed config after editing data_dir/events.csv:
            python scripts/generate_huxt_input.py --event 2017-09-06 --no-download --force-config

            # prepare every enabled event:
            python scripts/generate_huxt_input.py --event all
            ```

            > **Heads-up.** Without `--no-download` the script will hit the network for GONG data, and without `--skip-sanity-plot` it will launch a full seed HUXt run per event. Start with the guarded form above.
            """
        ),
        code(
            r"""
            RUN_EXPENSIVE = False

            if RUN_EXPENSIVE:
                # Launches the full input-generation workflow (downloads + WSA+ + sanity run).
                import subprocess
                subprocess.run(
                    ["python", str(BASE_DIR / "scripts" / "generate_huxt_input.py"),
                     "--event", event, "--no-download", "--skip-sanity-plot"],
                    check=True,
                )
            else:
                print("Skipping full workflow. Set RUN_EXPENSIVE = True to run generate_huxt_input.py end to end.")
            """
        ),
    ]


def gp_huxt_application_notebook() -> list[dict]:
    return [
        md(
            r"""
            # 04. Applying Gaussian Processes To HUXt Runs

            **What this notebook is.** Notebook 02 built a HUXt input for one event. A single HUXt run is fast (seconds to a minute), but the science questions here - *how does CME arrival time depend on launch parameters?*, *which parameters matter most?*, *where should the next run go?* - need **thousands** of evaluations across the 5-D parameter space. That is too many for HUXt directly.

            **The surrogate idea.** Run HUXt at a few hundred carefully chosen parameter vectors, then fit a cheap **Gaussian-process (GP) surrogate** that emulates HUXt: given any parameter vector it returns, instantly, a predicted arrival time *and* an uncertainty, plus a hit/miss probability. `scripts/gp_huxt_surrogate.py` is that pipeline. (GP fundamentals are in notebook 01.)

            **What we do here.** We take the pipeline apart **one operation at a time**. Instead of only reading the finished files under `runs/gp_surrogate/<event>/`, each task **reimplements the underlying step inline** so you can see exactly what the script computes, and names the script function it mirrors.

            The GP does not replace HUXt - it interpolates between HUXt runs. The script first evaluates HUXt at many Cone-CME parameter vectors, then trains the surrogates for hit/miss status and arrival time. The tasks follow that order:

            1. **`design`** - build parameter bounds and a Latin-hypercube sample table.
            2. **`run` / `detect_arrival`** - replay HUXt once for the seed and detect the arrival (the only task that launches a live simulation).
            3. **`fit`** - train the hit classifier and the arrival regressor from scratch.
            4. **`analyze`** - five independent sub-tasks: pairwise slices, local sensitivity, permutation importance, posterior arrival time, and next-run selection.
            5. **Compare** - check the inline reconstructions against the committed CLI figures.

            Each task re-loads what it needs, so they can be run independently once the setup cells have executed.
            """
        ),
        colab_bootstrap_cell(),
        common_setup_cell(),
        md(
            r"""
            ## Workflow Order

            `gp_huxt_surrogate.py` is organized around subcommands:

            ```text
            design
              -> run
              -> visualize-threshold   optional diagnostic replay
              -> fit
              -> analyze
            ```

            | step | script function | input | output |
            | --- | --- | --- | --- |
            | `design` | `make_design()` | `data_dir/sw/<event>/event_config.yaml` | `design.csv`, `design_meta.yaml` |
            | `run` | `run_design()` / `detect_arrival()` | `design.csv`, HUXt boundary/config | `results.csv` |
            | `visualize-threshold` | `visualize_threshold_cases()` | completed rows in `results.csv` | `figures/threshold_diagnostics/` |
            | `fit` | `train_models()` | completed `results.csv` | `gp_hit.joblib`, `gp_arrival.joblib` |
            | `analyze` | `analyze_event()` | trained GP bundles | `summary.yaml`, figures, `next_runs.csv` |

            > **Two surrogates, not one.** A CME that misses Earth has no arrival time, so the script trains a **classifier** for *whether* it hits and a separate **regressor** for *when* - the latter only on the runs that actually hit. Keep that split in mind throughout.
            """
        ),
        md(
            r"""
            ## Setup

            Import the shared libraries and the module-level constants that `gp_huxt_surrogate.py` uses. We copy the priors and default spans here (from `gp_huxt_surrogate.py` lines 42-51) so the reimplemented tasks are self-contained and you can see every number that goes into the workflow.

            > **What Tasks 3-5 need.** They train/analyze the GP on a **completed HUXt batch** (`results.csv` under `runs/gp_surrogate/<event>/`). If it does not exist, Task 3 **generates one inline** (reusing the script's design + run, ~200 HUXt samples, tens of minutes) so the notebook is self-contained - it only needs the event's boundary from notebook 02 (or `generate_huxt_input.py`). Lower `N_BATCH` in that cell for a quick look. The batch is saved, so re-runs are instant.

            > **Why copy the constants?** Re-stating `PRIOR_LOW/HIGH`, the spans, and `DEFAULT_OBS_SIGMA` here (rather than importing the script) keeps each task self-contained and makes every magic number visible. If you change them in the script, change them here too.
            """
        ),
        code(
            r"""
            import numpy as np
            import pandas as pd
            import yaml
            import matplotlib.pyplot as plt
            from IPython.display import display, Image
            from joblib import load

            GP_ROOT = BASE_DIR / "runs" / "gp_surrogate"
            DATA_ROOT = BASE_DIR / "data_dir" / "sw"
            PARAM_NAMES = ["inject_hour", "longitude", "latitude", "width", "v"]

            # Constants mirrored from gp_huxt_surrogate.py (PRIOR_LOW/HIGH, DEFAULT_ABS_SPAN, DEFAULT_OBS_SIGMA).
            PRIOR_LOW = np.array([0.0, -90.0, -50.0, 0.0, 100.0], dtype=float)
            PRIOR_HIGH = np.array([10.0, 90.0, 50.0, 180.0, 2000.0], dtype=float)
            DEFAULT_ABS_SPAN = np.array([1.0, 30.0, 20.0, 40.0, np.nan], dtype=float)
            DEFAULT_OBS_SIGMA = np.array([0.5, 10.0, 10.0, 15.0, np.nan], dtype=float)
            # design subcommand defaults (argparse): inject_hour, longitude, latitude, width spans + speed fraction.
            DEFAULT_SPAN = dict(inject_hour=1.0, longitude=30.0, latitude=20.0, width=40.0, speed_fraction=0.25)

            # Pick an event. Tasks 3-5 need GP outputs under runs/gp_surrogate/<event>/,
            # produced by: python scripts/gp_huxt_surrogate.py --event <event> design/run/fit
            event = "2017-09-06"
            outdir = GP_ROOT / event

            print("Prepared GP event directories:")
            if GP_ROOT.exists():
                for path in sorted(GP_ROOT.iterdir()):
                    if path.is_dir() and (path / "design.csv").exists():
                        print(" ", path.name)
            else:
                print("  (none yet - runs/gp_surrogate/ does not exist)")
            print("\nUsing event:", event)
            print("Output dir exists:", outdir.exists(), "(needs gp_huxt_surrogate.py design/run/fit)")
            """
        ),
        md(
            r"""
            ## Task 1: The `design` Step - bounds and Latin-hypercube sampling

            *Mirrors `make_design()`, `design_span()`, `default_bounds()`, and `latin_hypercube()`.*

            Before training a surrogate you must decide **where** to evaluate HUXt. That is a design-of-experiments problem: with only a few hundred runs to spend across 5 dimensions, you want them spread out, not clustered. The `design` step does three things:

            1. **Bounds** (`default_bounds`/`design_span`): build a box around the seed `initial_theta` - a span on each parameter (the speed span is a *fraction* of the seed speed; the others absolute) clipped to the global physical priors `PRIOR_LOW`/`PRIOR_HIGH`.
            2. **Latin-hypercube sample** (`latin_hypercube`): draw points so each parameter's range is evenly covered. Unlike uniform random sampling, a Latin hypercube guarantees no gaps or clumps in any single dimension - much better coverage for the same number of runs.
            3. **Seed row**: prepend the seed vector as row 0, so the best-guess CME is always evaluated.

            We reconstruct each step and check the result against the saved `design.csv`.

            > **What to look for:** the per-parameter histograms should be roughly flat across each bound (even coverage), with the black seed line inside the range, and the reconstructed table should match the saved design exactly.
            """
        ),
        code(
            r"""
            # 1a. Load the seed parameter vector from the event config.
            config_path = DATA_ROOT / event / "event_config.yaml"
            with config_path.open() as stream:
                config = yaml.safe_load(stream)
            theta0 = np.array([float(x) for x in config["initial_theta"]], dtype=float)

            print("seed theta0:")
            display(pd.Series(theta0, index=PARAM_NAMES))
            """
        ),
        code(
            r"""
            # 1b. Build the span and the bounds box (design_span + default_bounds).
            # The speed span is a fraction of the seed speed; the others are absolute.
            span = np.array([
                DEFAULT_SPAN["inject_hour"],
                DEFAULT_SPAN["longitude"],
                DEFAULT_SPAN["latitude"],
                DEFAULT_SPAN["width"],
                DEFAULT_SPAN["speed_fraction"] * theta0[4],
            ], dtype=float)

            # Clip the box to the global priors so it never leaves the physical range.
            low = np.maximum(PRIOR_LOW, theta0 - span)
            high = np.minimum(PRIOR_HIGH, theta0 + span)

            bounds = pd.DataFrame({"low": low, "theta0": theta0, "high": high}, index=PARAM_NAMES)
            print("Bounds box around the seed:")
            display(bounds)
            """
        ),
        code(
            r"""
            # 1c. Draw a Latin-hypercube sample in the unit cube, then map it into the box.
            # This mirrors latin_hypercube(): prefer scipy's qmc, fall back to a numpy permutation design.
            def latin_hypercube(n, ndim, seed):
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

            n = 300       # design --n default
            seed = 42     # design --seed default
            n_random = max(0, n - 1)
            unit = latin_hypercube(n_random, len(PARAM_NAMES), seed)
            theta = low + unit * (high - low)
            theta = np.vstack([theta0, theta])   # row 0 is the seed

            recon = pd.DataFrame(theta, columns=PARAM_NAMES)
            print("reconstructed design shape:", recon.shape)
            display(recon.head())
            """
        ),
        code(
            r"""
            # 1d. Compare with the committed design.csv if present; else use the reconstruction.
            design_file = outdir / "design.csv"
            if design_file.exists():
                design = pd.read_csv(design_file)
                with (outdir / "design_meta.yaml").open() as stream:
                    design_meta = yaml.safe_load(stream)
                print("saved design shape:", design.shape)
                print("saved metadata bounds match reconstruction:",
                      np.allclose(design_meta["bounds_low"], low) and np.allclose(design_meta["bounds_high"], high))
                print("seed row (row 0) matches:", np.allclose(design.loc[0, PARAM_NAMES].to_numpy(float), theta0))
            else:
                print(f"No saved design.csv yet. Create one with:")
                print(f"  python scripts/gp_huxt_surrogate.py --event {event} design --n 300")
                print("Showing the reconstructed design instead.")
                design = recon

            fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(13, 2.6), constrained_layout=True)
            for ax, name, seed_value in zip(axes, PARAM_NAMES, theta0):
                ax.hist(design[name], bins=24, color="steelblue", alpha=0.75)
                ax.axvline(seed_value, color="black", lw=2, label="seed")
                ax.set_title(name)
            axes[0].legend()
            """
        ),
        md(
            r"""
            ## Task 2: The `run` Step - one live HUXt simulation and the arrival detector

            *Mirrors `run_huxt_model()`, `load_huxt_context()`, `moving_average()`, `detect_arrival()`, and `front_arrival_metrics()`.*

            HUXt outputs a *time series* of solar-wind speed at Earth. To train a surrogate we must reduce that to two scalars: **did the CME arrive**, and **when**. That reduction is the job of the arrival *detector*, and it is subtler than it sounds - the CME signature can be a sharp shock, a gentle speed bump, or (for a near-miss) barely anything.

            This task launches HUXt once for the seed (Task 3 later runs a small batch). We run the seed Cone-CME, pull the Earth solar-wind time series, and then reconstruct each detector from scratch:

            - **`jump`** - short-lag fractional change in smoothed speed (catches a shock-like acceleration; can miss slow risers).
            - **`enhancement`** - fractional rise above the rolling baseline minimum (catches a sustained speed-up; robust to gradual arrivals).
            - **`front`** - geometric ConeCME front crossing Earth (`compute_arrival_at_body`); from CME geometry alone, ignoring the speed trace.
            - **`hybrid`** (script default) - requires both a `front` crossing **and** an `enhancement` trigger, so geometry and the speed signature must agree.

            > **What to look for:** in the two-panel plot, the detected-arrival line should sit at the leading edge of the speed enhancement, where the detector curve crosses the threshold. When `front` and `enhancement` disagree, `hybrid` calls it a miss - that conservatism is deliberate.

            The cell is guarded: if HUXt or the boundary file is unavailable, it prints a message and skips the live run so the rest of the notebook still works. The same detector modes are exercised on controlled synthetic series in notebook 03.
            """
        ),
        code(
            r"""
            # 2a. Try to run one HUXt simulation at the seed parameter vector.
            huxt_ran = False
            huxt_time = huxt_speed = cme = None
            try:
                import astropy.units as u
                import huxt.huxt as H
                import huxt.huxt_analysis as HA

                boundary_path = DATA_ROOT / event / f"v_boundary_{event}.npz"
                if not boundary_path.exists():
                    raise FileNotFoundError(f"Missing HUXt boundary file: {boundary_path}")

                v_boundary = np.load(boundary_path)["speed_map"] * (u.km / u.s)
                huxt_kwargs = dict(
                    v_boundary=v_boundary,
                    latitude=0 * u.deg,
                    cr_num=float(config["cr_num"]),
                    frame="sidereal",
                    simtime=10 * u.day,
                    dt_scale=4,
                )

                inject_hour, longitude, latitude, width, v = theta0
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
                huxt_time, huxt_speed = huxt_ts["time"], huxt_ts["vsw"]
                cme = model.cmes[0] if getattr(model, "cmes", None) else cme
                huxt_ran = True
                print("HUXt run complete:", len(huxt_speed), "time steps")
            except Exception as exc:
                print("Skipping live HUXt run:", repr(exc))
                print("Task 2 plots will be skipped; the rest of the notebook still runs.")
            """
        ),
        code(
            r"""
            # 2b. Reconstruct the detector internals (moving_average + jump/enhancement) on the live series.
            from astropy.time import Time

            def moving_average(values, window):
                if window <= 1:
                    return values.astype(float, copy=True)
                kernel = np.ones(window, dtype=float) / float(window)
                pad_left = window // 2
                pad_right = window - 1 - pad_left
                padded = np.pad(values.astype(float), (pad_left, pad_right), mode="edge")
                return np.convolve(padded, kernel, mode="valid")

            def time_to_hours(times):
                unix = Time(times).unix
                return (unix - unix[0]) / 3600.0

            if huxt_ran:
                threshold, lag, smooth_window, baseline_window = 0.25, 2, 5, 24
                speed = np.asarray(huxt_speed, dtype=float)
                hours = time_to_hours(huxt_time)
                smooth = moving_average(speed, smooth_window)

                # jump detector: fractional change vs `lag` steps back.
                jump = np.full(speed.shape, np.nan)
                for idx in range(lag, speed.size):
                    denom = smooth[idx - lag]
                    if np.isfinite(denom) and abs(denom) > 1e-12:
                        jump[idx] = (smooth[idx] - denom) / denom

                # enhancement detector: fractional rise above the rolling baseline minimum.
                enhancement = np.full(speed.shape, np.nan)
                for idx in range(1, speed.size):
                    baseline = np.nanmin(smooth[max(0, idx - baseline_window):idx])
                    if np.isfinite(baseline) and abs(baseline) > 1e-12:
                        enhancement[idx] = (smooth[idx] - baseline) / baseline

                enh_hit_idx = np.where(enhancement >= threshold)[0]
                enh_hit = enh_hit_idx.size > 0
                enh_arrival_hr = float(hours[enh_hit_idx[0]]) if enh_hit else np.nan
                print(f"enhancement detector: hit={enh_hit}, arrival_hr={enh_arrival_hr:.2f}" if enh_hit
                      else "enhancement detector: no hit")
                print(f"max jump={np.nanmax(jump):.3f}, max enhancement={np.nanmax(enhancement):.3f}")
            else:
                print("No live HUXt series available; skipping detector reconstruction.")
            """
        ),
        code(
            r"""
            # 2c. Front + hybrid combine (front_arrival_metrics) and the diagnostic plot.
            if huxt_ran:
                stats = cme.compute_arrival_at_body("EARTH")
                front_hit = bool(stats.get("hit", False))
                if front_hit:
                    start = Time(huxt_time[0])
                    front_arrival_hr = float(((stats["t_arrive"] - start).to("hour")).value)
                else:
                    front_arrival_hr = np.nan

                # hybrid (script default): front crossing AND enhancement trigger.
                hybrid_hit = bool(front_hit and enh_hit)
                print(f"front_hit={front_hit}, front_arrival_hr={front_arrival_hr:.2f}" if front_hit
                      else "front_hit=False")
                print("hybrid hit (front AND enhancement):", hybrid_hit)

                fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
                axes[0].plot(hours, speed, color="0.65", lw=1.2, label="Earth Vsw")
                axes[0].plot(hours, smooth, color="tab:blue", lw=1.8, label="Smoothed Vsw")
                if enh_hit:
                    axes[0].axvline(enh_arrival_hr, color="tab:green", lw=1.4, label="enhancement arrival")
                if front_hit:
                    axes[0].axvline(front_arrival_hr, color="tab:purple", ls=":", lw=1.6, label="front arrival")
                axes[0].set_ylabel("Vsw [km/s]")
                axes[0].legend(loc="best", fontsize=9)

                axes[1].plot(hours, enhancement, color="tab:red", lw=1.5, label="enhancement detector")
                axes[1].axhline(threshold, color="k", ls="--", lw=1.2, label=f"threshold {threshold:g}")
                axes[1].set_xlabel("Hours from model start")
                axes[1].set_ylabel("Detector value")
                axes[1].legend(loc="best", fontsize=9)
                fig.suptitle(f"{event} seed run: hybrid hit={hybrid_hit}", fontsize=11)
                fig.tight_layout()
            else:
                print("No live HUXt series available; skipping the detector plot.")
            """
        ),
        md(
            r"""
            ### Building the training set

            For every design row, `run_design()` runs the same simulation, calls `detect_arrival()` with the chosen `--detector-method`, and writes one scalar row to `results.csv`. The columns are exactly what Task 3 trains on: `hit` (the classifier target), `arrival_time_hr` (the regressor target, only meaningful when `hit`), and the raw detector diagnostics (`max_speed_*`, `front_*`) that explain *why* each row was called a hit or miss.

            **Self-contained run.** If `results.csv` does not exist yet, the cell below **generates the batch inline** - it reuses the script's `make_design()` + `run_design()` to evaluate HUXt at ~200 samples (tens of minutes) and save `results.csv`. That keeps this notebook runnable on its own (lower `N_BATCH` for a quick look). It needs the event's boundary from notebook 02 (or `generate_huxt_input.py`).

            > **What to look for:** the `hit` counts show how balanced the training set is - an arrival GP trained on very few hits (the script warns under ~30) should be trusted less.
            """
        ),
        code(
            r"""
            import gp_huxt_surrogate as gp

            results_file = outdir / "results.csv"
            boundary_file = DATA_ROOT / event / f"v_boundary_{event}.npz"
            if not results_file.exists():
                if not boundary_file.exists():
                    raise FileNotFoundError(
                        f"No {results_file} and no boundary {boundary_file}.\n"
                        f"Run notebook 02 first (it downloads GONG, runs WSA+, and writes the boundary),\n"
                        f"or: python scripts/generate_huxt_input.py --event {event}. Then re-run this cell."
                    )
                # Self-contained path: build a small HUXt batch inline so the notebook can train a GP
                # without a separate gp_huxt_surrogate.py run. Reuses the script's design + run steps.
                N_BATCH = 200   # runs HUXt N_BATCH times (~tens of minutes); lower it for a quick look
                print(f"No results.csv yet - generating a {N_BATCH}-sample HUXt batch inline (~minutes)...")
                gp.make_design(event, DATA_ROOT, GP_ROOT, n=N_BATCH, seed=42, force=True,
                               span_inject_hour=1.0, span_longitude=30.0, span_latitude=20.0,
                               span_width=40.0, span_speed_fraction=0.25)
                gp.run_design(event, DATA_ROOT, GP_ROOT, limit=N_BATCH, detector_threshold=0.25,
                              detector_lag=2, smooth_window=5, detector_method="hybrid",
                              baseline_window=24, rerun_completed=False)
                print(f"Wrote {results_file} (re-runs reuse it). For a sharper GP, run a larger batch:")
                print(f"  python scripts/gp_huxt_surrogate.py --event {event} design --n 300 / run / fit")

            results = pd.read_csv(results_file)
            detector_cols = [
                "sample_id", "hit", "arrival_time_hr", "max_speed_jump",
                "max_speed_enhancement", "max_detector_value", "front_hit", "front_arrival_time_hr",
            ]
            print("status counts:")
            print(results["status"].value_counts(dropna=False))
            print("\nhit counts:")
            print(results["hit"].value_counts(dropna=False))
            display(results[detector_cols].head(10))
            """
        ),
        md(
            r"""
            ## Task 3: The `fit` Step - training both GPs from scratch

            *Mirrors `train_models()`, `predict_hit_probability()`, and `predict_arrival()`.*

            `fit` trains two models on the completed `results.csv` rows:

            - **`gp_hit`** - a `GaussianProcessClassifier` for `P(hit)` over **all** completed rows.
            - **`gp_arrival`** - a `GaussianProcessRegressor` for `arrival_time_hr`, trained on **hit rows only** (a miss has no arrival time).

            **A few modeling choices worth understanding:**

            - **The kernel** (`ConstantKernel * Matern(nu=2.5)`): the Matern kernel sets how smoothly the prediction varies; one **length scale per parameter** lets the GP learn that, say, speed matters over a different scale than width. `nu=2.5` is a common moderately-smooth choice.
            - **Standardizing** the inputs (and the arrival target) puts every parameter on a comparable scale, so those length scales are interpretable and the optimizer is well-conditioned.
            - **`WhiteKernel`** on the regressor adds a learned noise floor that absorbs the run-to-run jitter in the *detected* arrival time (the detector is not perfectly smooth).
            - **Holdout MAE**: when there are enough hits, a held-out split estimates the typical arrival-time error in hours - the headline accuracy number.

            > **What to look for:** the learned length scales (printed in the kernel) hint at which parameters arrival time is most sensitive to; the holdout MAE tells you whether to trust the surrogate to the hour or only the half-day.
            """
        ),
        code(
            r"""
            from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, mean_absolute_error
            from sklearn.model_selection import train_test_split

            def parse_bool(value):
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
                return str(value).strip().lower() in {"1", "true", "yes", "y"}

            completed = results.loc[results["status"] == "completed"].copy()
            completed["hit"] = completed["hit"].map(parse_bool)
            x = completed[PARAM_NAMES].to_numpy(float)
            hit = completed["hit"].astype(int).to_numpy()

            x_scaler = StandardScaler().fit(x)
            x_scaled = x_scaler.transform(x)
            print(f"completed rows: {len(completed)}, hit rows: {int(hit.sum())}")
            """
        ),
        code(
            r"""
            # 3a. Hit classifier over all completed rows.
            hit_kernel = ConstantKernel(1.0, (1e-3, 1e5)) * Matern(
                length_scale=np.ones(len(PARAM_NAMES)), length_scale_bounds=(1e-2, 1e5), nu=2.5)
            hit_model = GaussianProcessClassifier(kernel=hit_kernel, random_state=42, max_iter_predict=100)
            hit_model.fit(x_scaled, hit)
            print("training accuracy:", accuracy_score(hit, hit_model.predict(x_scaled)))
            print("learned hit kernel:", hit_model.kernel_)
            """
        ),
        code(
            r"""
            # 3b. Arrival regressor on hit rows only, with a holdout MAE when there are enough cases.
            hit_rows = completed.loc[completed["hit"] & np.isfinite(completed["arrival_time_hr"])].copy()
            x_hit = x_scaler.transform(hit_rows[PARAM_NAMES].to_numpy(float))
            y_hit = hit_rows[["arrival_time_hr"]].to_numpy(float)
            y_scaler = StandardScaler().fit(y_hit)
            y_scaled = y_scaler.transform(y_hit).ravel()

            arrival_kernel = ConstantKernel(1.0, (1e-3, 1e5)) * Matern(
                length_scale=np.ones(len(PARAM_NAMES)), length_scale_bounds=(1e-2, 1e4), nu=2.5
            ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
            arrival_model = GaussianProcessRegressor(
                kernel=arrival_kernel, normalize_y=False, n_restarts_optimizer=3, random_state=42)

            if len(hit_rows) >= 10:
                xtr, xte, ytr, yte = train_test_split(x_hit, y_scaled, test_size=0.2, random_state=42)
                arrival_model.fit(xtr, ytr)
                pred_hr = y_scaler.inverse_transform(arrival_model.predict(xte).reshape(-1, 1)).ravel()
                true_hr = y_scaler.inverse_transform(yte.reshape(-1, 1)).ravel()
                print("holdout arrival MAE [hr]:", mean_absolute_error(true_hr, pred_hr))

            arrival_model.fit(x_hit, y_scaled)   # final fit on all hit rows
            print("learned arrival kernel:", arrival_model.kernel_)
            """
        ),
        code(
            r"""
            # 3c. Prediction helpers (mirror predict_hit_probability / predict_arrival) used by Task 4.
            def predict_hit_probability(theta):
                return hit_model.predict_proba(x_scaler.transform(np.atleast_2d(theta)))[:, 1]

            def predict_arrival(theta):
                mean_s, std_s = arrival_model.predict(x_scaler.transform(np.atleast_2d(theta)), return_std=True)
                mean = y_scaler.inverse_transform(mean_s.reshape(-1, 1)).ravel()
                std = std_s * float(y_scaler.scale_[0])
                return mean, std

            mean_hr, std_hr = predict_arrival(theta0)
            print("seed P(hit):", float(predict_hit_probability(theta0)[0]))
            print("seed arrival mean [hr]:", float(mean_hr[0]), "std [hr]:", float(std_hr[0]))

            # These freshly trained models match the committed gp_hit.joblib / gp_arrival.joblib bundles.
            """
        ),
        md(
            r"""
            ## Task 4: The `analyze` Step

            *Mirrors `analyze_event()`.* With a trained surrogate in hand, `analyze` is where it pays off: instead of more HUXt runs, we interrogate the *cheap* GP thousands of times to answer the science questions - how arrival time varies across the parameter space (4a), which parameters drive it locally (4b) and globally (4c), what arrival time to expect given observational uncertainty (4d), and where the next HUXt run would be most informative (4e).

            The single `analyze` subcommand bundles all five. We unpack each as an **independent sub-task** that runs from the Task 3 models (`predict_hit_probability` / `predict_arrival`, `low`, `high`, `theta0`). Each cell stands alone.
            """
        ),
        md(
            r"""
            ### Task 4a: Pairwise slices

            *Mirrors `pair_grid()`, `make_hit_slice()`, `make_arrival_slice()`.*

            You cannot draw a 5-D function, so we take 2-D **slices**: pick two parameters, sweep them across the bounds box on a grid, and hold the other three fixed at the seed. For each grid we render three fields - `P(hit)`, the arrival-time **mean**, and the arrival-time **std** (the GP's own uncertainty) - with the `P(hit)=0.5` decision contour drawn in white. Read the std panel as "where would I *not* trust the mean".

            Arrival time is only meaningful where the CME actually hits. Where `P(hit)` collapses to ~0 the arrival GP extrapolates to unphysical values that dominate the color scale. Rather than punch holes in the map, every cell stays colored and we only adjust the **color limits**: they are computed from the hit region (`P(hit) >= 0.5`) alone, floored at 48 hr and capped at the **90th percentile** of the in-hit arrival times. Cells outside that range - including the no-hit band - saturate to the end colors (`extend="both"`), so the dense bulk keeps its contrast while the map stays continuous.

            > **What to look for:** the white contour separates "arrives" from "misses"; gradients in the mean panel show which direction speeds up or delays arrival; bright bands in the std panel flag thinly-sampled regions where another HUXt run would help (revisited in 4e).
            """
        ),
        code(
            r"""
            ARRIVAL_FLOOR_HR = 48.0   # color floor: arrival times below this are not physical
            HIT_FLOOR = 0.5           # set color limits from the hit region only
            ARRIVAL_CAP_PCT = 90.0    # cap the upper color limit so the grazing-CME tail doesn't dominate

            def pair_grid(x_idx, y_idx, grid_size=60):
                gx = np.linspace(low[x_idx], high[x_idx], grid_size)
                gy = np.linspace(low[y_idx], high[y_idx], grid_size)
                xx, yy = np.meshgrid(gx, gy)
                theta = np.tile(theta0, (grid_size * grid_size, 1))
                theta[:, x_idx] = xx.ravel()
                theta[:, y_idx] = yy.ravel()
                return xx, yy, theta

            def color_limits(field, hit, vmin=None):
                # Limits from the hit region only: lo floored (if given), hi at the 90th percentile.
                vals = field[hit & np.isfinite(field)]
                if vals.size == 0:
                    return None
                lo = float(vmin) if vmin is not None else float(vals.min())
                return lo, float(np.percentile(vals, ARRIVAL_CAP_PCT))

            pairs = [("longitude", "latitude"), ("longitude", "v"), ("longitude", "width")]
            for xn, yn in pairs:
                xi, yi = PARAM_NAMES.index(xn), PARAM_NAMES.index(yn)
                xx, yy, theta = pair_grid(xi, yi)
                gs = xx.shape[0]
                prob = predict_hit_probability(theta).reshape(gs, gs)
                mean, std = predict_arrival(theta)
                mean = mean.reshape(gs, gs)
                std = std.reshape(gs, gs)

                # Keep every cell colored (no holes); use the hit region only to set robust color limits.
                hit = prob >= HIT_FLOOR
                fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
                panels = [
                    (axes[0], prob, "viridis", "P(hit)", (0.0, 1.0)),
                    (axes[1], mean, "magma", "arrival mean [hr]", color_limits(mean, hit, vmin=ARRIVAL_FLOOR_HR)),
                    (axes[2], std, "magma", "arrival std [hr]", color_limits(std, hit)),
                ]
                for ax, field, cmap, label, limits in panels:
                    if limits is None or np.isclose(limits[0], limits[1]):
                        c = ax.contourf(xx, yy, field, levels=21, cmap=cmap)
                    else:
                        levels = np.linspace(limits[0], limits[1], 21)
                        c = ax.contourf(xx, yy, field, levels=levels, cmap=cmap, extend="both")
                    fig.colorbar(c, ax=ax, label=label)
                    if np.nanmin(prob) <= 0.5 <= np.nanmax(prob):
                        ax.contour(xx, yy, prob, levels=[0.5], colors="white", linewidths=1.4)
                    ax.scatter(theta0[xi], theta0[yi], color="black", s=22, zorder=3)
                    ax.set_xlabel(xn)
                    ax.set_ylabel(yn)
                fig.suptitle(f"{xn} vs {yn}", fontsize=11)
                plt.show()
            """
        ),
        md(
            r"""
            ### Task 4b: Local sensitivity (finite-difference gradients)

            *Mirrors `finite_difference_gradients()`.* This answers "right at the best-guess CME, how does arrival time respond to a small change in each parameter?" We nudge each parameter by 1% of its range around the seed and take a central difference of the GP-predicted arrival time, giving **d(arrival)/d(parameter)** in hours per unit (hours per km/s, hours per degree, ...).

            The sign gives direction (a faster CME arrives earlier, so a negative slope on `v`); the magnitude gives leverage. Because the units differ per parameter, compare with care - 4c gives a unit-free global view.

            > **What to look for:** `v` (speed) usually has the strongest, negative slope. This is a **local** measure, valid near the seed only; far away the surface bends (see the slices in 4a).
            """
        ),
        code(
            r"""
            base_mean, _ = predict_arrival(theta0)
            gradients = {}
            for idx, name in enumerate(PARAM_NAMES):
                step = max((high[idx] - low[idx]) * 0.01, 1e-6)
                plus, minus = theta0.copy(), theta0.copy()
                plus[idx] = min(high[idx], plus[idx] + step)
                minus[idx] = max(low[idx], minus[idx] - step)
                if plus[idx] == minus[idx]:
                    gradients[name] = np.nan
                    continue
                y_plus, _ = predict_arrival(plus)
                y_minus, _ = predict_arrival(minus)
                gradients[name] = float((y_plus[0] - y_minus[0]) / (plus[idx] - minus[idx]))

            print("baseline arrival [hr]:", float(base_mean[0]))
            pd.Series(gradients).plot(kind="bar", color="slateblue")
            plt.ylabel("d arrival / d parameter [hr/unit]")
            plt.title("Local sensitivity at the seed")
            plt.tight_layout()
            """
        ),
        md(
            r"""
            ### Task 4c: Permutation importance

            *Mirrors `permutation_importance()`.* Where 4b is a *local* derivative at the seed, this is a *global* importance over the whole plausible region. We draw a cloud of posterior-like samples (Task 4d's distribution), then shuffle one parameter's column at a time and measure how much the predicted arrival time changes (mean squared deviation, in hr^2). If scrambling a parameter barely changes the prediction the surrogate does not rely on it; if it changes a lot, that parameter dominates.

            > **What to look for:** the ranking should broadly agree with 4b's magnitudes (speed near the top) but can differ where the response is non-linear across the region. Unlike 4b these values are directionless and comparable across parameters.
            """
        ),
        code(
            r"""
            def truncated_normal_samples(theta0, sigma, low, high, n, seed):
                rng = np.random.default_rng(seed)
                samples = rng.normal(theta0, sigma, size=(n, len(theta0)))
                for _ in range(20):
                    bad = (samples < low) | (samples > high)
                    if not bad.any():
                        break
                    samples = np.where(bad, rng.normal(theta0, sigma, size=samples.shape), samples)
                return np.clip(samples, low, high)

            sigma = DEFAULT_OBS_SIGMA.copy()
            sigma[4] = 0.10 * theta0[4]
            samples = truncated_normal_samples(theta0, sigma, low, high, 3000, seed=43)

            baseline, _ = predict_arrival(samples)
            rng = np.random.default_rng(45)
            importance = {}
            for idx, name in enumerate(PARAM_NAMES):
                permuted = samples.copy()
                permuted[:, idx] = rng.permutation(permuted[:, idx])
                pred, _ = predict_arrival(permuted)
                importance[name] = float(np.mean((baseline - pred) ** 2))

            pd.Series(importance).plot(kind="bar", color="indianred")
            plt.ylabel("permutation importance [hr^2]")
            plt.title("Global importance for arrival time")
            plt.tight_layout()
            """
        ),
        md(
            r"""
            ### Task 4d: Posterior arrival time

            *Mirrors the posterior block of `analyze_event()` plus `truncated_normal_samples()`.* This turns "we don't know the CME parameters exactly" into "here is the spread of arrival times to expect". We treat the seed as a **noisy observation**, draw parameter vectors around it (a truncated normal with widths `DEFAULT_OBS_SIGMA`), and push each through the arrival GP.

            The result folds together **two** sources of uncertainty: the *input* uncertainty (we are unsure of the parameters) and the *surrogate* uncertainty (the GP's own predictive std). Adding a normal draw of width `post_std` to each predicted mean produces the posterior-predictive distribution.

            > **What to look for:** the histogram's spread is your honest arrival-time error bar; the printed p05/median/p95 form a 90% credible interval. A wide distribution means uncertain inputs, a thinly-trained GP, or both.
            """
        ),
        code(
            r"""
            post_theta = truncated_normal_samples(theta0, sigma, low, high, 5000, seed=42)
            post_mean, post_std = predict_arrival(post_theta)
            draw = np.random.default_rng(43).normal(post_mean, post_std)
            draw = draw[np.isfinite(draw)]

            p05, med, p95 = np.percentile(draw, [5, 50, 95])
            print(f"posterior arrival [hr]  p05={p05:.1f}  median={med:.1f}  p95={p95:.1f}")
            print("mean P(hit) over posterior:", float(np.mean(predict_hit_probability(post_theta))))

            plt.hist(draw, bins=40, color="teal", alpha=0.8)
            for q, c in [(p05, "k"), (med, "tab:orange"), (p95, "k")]:
                plt.axvline(q, color=c, ls="--")
            plt.xlabel("Arrival time since model start [hr]")
            plt.ylabel("Count")
            plt.title("Posterior predictive arrival time")
            plt.tight_layout()
            """
        ),
        md(
            r"""
            ### Task 4e: Next-run selection

            *Mirrors `select_next_runs()`.* HUXt runs are the expensive resource, so where should the next batch go? This is **active learning**: pick parameter vectors that will teach the surrogate the most. The script scores random candidates with three factors, then greedily spreads the picks out in parameter space:

            ```text
            score = arrival_std * (0.25 + boundary_weight) * (0.25 + likelihood_weight)
            ```

            - `arrival_std` favors candidates where the arrival GP is **uncertain** (most to learn).
            - `boundary_weight = 1 - |P(hit) - 0.5| * 2` favors the **hit/miss boundary** (where classification is hardest).
            - `likelihood_weight = exp(-0.5 * z^2 / ndim)` keeps candidates **near the seed** (where it matters for this event).

            After scoring, a distance penalty in scaled space spreads the batch out so we do not propose near-duplicate runs.

            > **What to look for:** the selected points should cluster near the `P(hit)=0.5` boundary and in the high-std bands from 4a, while staying in the seed's neighborhood - exactly the places another HUXt run sharpens the surrogate.
            """
        ),
        code(
            r"""
            n_candidates, batch_size = 4000, 20
            rng = np.random.default_rng(44)
            candidates = low + rng.random((n_candidates, len(theta0))) * (high - low)

            _, arrival_std = predict_arrival(candidates)
            p_hit = predict_hit_probability(candidates)
            z2 = np.sum(((candidates - theta0) / sigma) ** 2, axis=1)
            likelihood_weight = np.exp(-0.5 * z2 / len(theta0))
            boundary_weight = 1.0 - np.abs(p_hit - 0.5) * 2.0
            score = arrival_std * (0.25 + boundary_weight) * (0.25 + likelihood_weight)

            # Greedy space-filling selection in scaled coordinates.
            scaled = x_scaler.transform(candidates)
            selected = [int(np.argmax(score))]
            available = np.delete(np.arange(n_candidates), selected[0])
            while len(selected) < batch_size and available.size:
                dist = np.min(np.linalg.norm(
                    scaled[available, None, :] - scaled[np.array(selected)][None, :, :], axis=2), axis=1)
                nxt = int(available[np.argmax(score[available] * (1.0 + dist))])
                selected.append(nxt)
                available = available[available != nxt]

            next_runs = pd.DataFrame(candidates[selected], columns=PARAM_NAMES)
            next_runs["score"] = score[selected]
            next_runs["p_hit"] = p_hit[selected]
            next_runs["arrival_std_hr"] = arrival_std[selected]
            display(next_runs.head(10))

            plt.figure(figsize=(6, 3.2))
            plt.scatter(next_runs["p_hit"], next_runs["arrival_std_hr"], c=next_runs["score"], cmap="magma", s=35)
            plt.colorbar(label="selection score")
            plt.xlabel("P(hit)")
            plt.ylabel("arrival_std_hr")
            plt.title("Suggested next HUXt runs")
            plt.tight_layout()
            """
        ),
        md(
            r"""
            ## Task 5: Compare against the committed CLI figures

            *Mirrors the figure-writing in `analyze_event()`.* Every analysis above was reimplemented inline; the production script saves its own versions under `figures/`. Displaying them next to your reconstructions is the final check that you understood each operation - if a panel disagrees, the difference is a clue. For example, the committed `arrival_mean_*.png` uses the 24-96 hr window set in `gp_huxt_surrogate.py`, while Task 4a uses the floor-48 / p90-cap recipe, so their color scales differ by design.

            > **What to look for:** shapes and structures should match; exact color scales may differ where the script and notebook use different limit conventions.
            """
        ),
        code(
            r"""
            figure_names = [
                "posterior_arrival_time.png",
                "permutation_importance.png",
                "local_sensitivity.png",
                "hit_probability_longitude_v.png",
                "arrival_mean_longitude_v.png",
                "arrival_std_longitude_v.png",
            ]

            for name in figure_names:
                path = outdir / "figures" / name
                print(name, "exists:", path.exists())
                if path.exists():
                    display(Image(filename=str(path), width=760))
            """
        ),
    ]


def arrival_detector_examples_notebook() -> list[dict]:
    return [
        md(
            r"""
            # 03. A Hit And A Miss: HUXt Arrival Detection

            This notebook runs HUXt for **two Cone-CMEs that differ only in pointing direction** - one aimed at Earth (a **hit**) and the same CME rotated ~130 deg away in longitude (a **miss**) - and shows exactly what the arrival *detector* does: reduce each Earth solar-wind speed series `Vsw(t)` to a hit/miss label plus an arrival time.

            That reduction is the heart of `gp_huxt_surrogate.py`'s `run` step (every design sample is collapsed this way) and of its `visualize-threshold` diagnostic. We reproduce both of `visualize-threshold`'s plots inline: the **time-series detector** panel and a **heliosphere snapshot**.

            **Detector modes** (the `--detector-method` flag): `enhancement` triggers on a sustained speed rise above a recent baseline; `jump` on a short-lag fractional jump; `front` on HUXt's tracked Cone-CME front reaching Earth; `hybrid` (the default) requires *both* a front crossing and an enhancement.

            > **Needs a boundary.** This notebook runs HUXt, so it needs the event's `v_boundary_<event>.npz`. Run notebook 02 first (it writes one) or `python scripts/generate_huxt_input.py --event <event>`.
            """
        ),
        colab_bootstrap_cell(),
        common_setup_cell(),
        code(
            r"""
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from pathlib import Path

            import gp_huxt_surrogate as gp

            DATA_ROOT = BASE_DIR / "data_dir" / "sw"
            event = "2017-09-06"
            PARAM_NAMES = gp.PARAM_NAMES   # ["inject_hour", "longitude", "latitude", "width", "v"]

            boundary_file = DATA_ROOT / event / f"v_boundary_{event}.npz"
            if not boundary_file.exists():
                raise FileNotFoundError(
                    f"No boundary {boundary_file}.\n"
                    f"Run notebook 02 first (it downloads GONG, runs WSA+, writes the boundary),\n"
                    f"or: python scripts/generate_huxt_input.py --event {event}."
                )

            # HUXt context: inner-boundary speed map + Carrington rotation + seed Cone-CME vector.
            ctx = gp.load_huxt_context(event, DATA_ROOT)
            huxt_kwargs, theta0 = ctx["huxt_kwargs"], ctx["theta0"]
            print("event:", event, "| seed theta0:", theta0)
            """
        ),
        md(
            r"""
            ## Two Cone-CMEs: aimed at Earth vs rotated away

            We take the event's seed Cone-CME and make a second copy that is identical **except for its longitude**, rotated ~130 deg so it points away from Earth. Changing only one parameter isolates the cause: the same launch time, latitude, width, and speed - just a different direction - flips the outcome from hit to miss.
            """
        ),
        code(
            r"""
            hit_theta = theta0.copy()                  # the seed: a wide CME aimed near Earth
            miss_theta = theta0.copy()
            miss_theta[1] = theta0[1] + 127.0          # rotate longitude ~130 deg away from Earth

            display(pd.DataFrame(
                {"hit (aimed at Earth)": hit_theta, "miss (rotated away)": miss_theta},
                index=PARAM_NAMES,
            ))
            """
        ),
        md(
            r"""
            ## Run HUXt and detect the arrival

            For each CME we call `run_huxt_model()` (build the model + Cone-CME, solve, pull the Earth time series), then `arrival_diagnostics()` to compute the detector internals (smoothed speed, the enhancement curve, the threshold crossing) and `front_arrival_metrics()` for the geometric front. This runs HUXt twice - a minute or two.
            """
        ),
        code(
            r"""
            def run_case(name, theta):
                model, t, v, cme = gp.run_huxt_model(theta, huxt_kwargs)
                diag = gp.arrival_diagnostics(t, v, threshold=0.25, method="enhancement")
                front = gp.front_arrival_metrics(cme, t)
                print(f"{name}: enhancement hit={diag['hit']}, front hit={front['front_hit']}, "
                      f"peak Vsw={diag['peak_vsw']:.0f} km/s")
                return {"name": name, "theta": theta, "model": model, "cme": cme, "diag": diag, "front": front}

            hit = run_case("hit (aimed at Earth)", hit_theta)
            miss = run_case("miss (rotated away)", miss_theta)
            """
        ),
        md(
            r"""
            ## The threshold diagnostic plot

            This is the **time-series panel** `visualize-threshold` writes. Top: the Earth solar-wind speed (raw + smoothed), with a line at the detected arrival. Bottom: the enhancement detector against the `0.25` threshold. For the hit, the smoothed speed jumps and the detector crosses the threshold; for the miss, the CME never reaches Earth, so the speed stays at the ambient background and the detector never crosses.
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharex=True, sharey="row")
            for col, case in enumerate([hit, miss]):
                d = case["diag"]
                top, bot = axes[0, col], axes[1, col]
                top.plot(d["hours"], d["speed"], color="0.65", lw=1.2, label="Earth Vsw")
                top.plot(d["hours"], d["smooth_speed"], color="tab:blue", lw=1.8, label="smoothed")
                if d["hit"]:
                    top.axvline(d["arrival_time_hr"], color="tab:green", lw=1.5, label="detected arrival")
                top.set_title(f"{case['name']}  -  hit={d['hit']}")
                top.set_ylabel("Vsw [km/s]")
                top.legend(loc="upper left", fontsize=8)

                bot.plot(d["hours"], d["detector"], color="tab:red", lw=1.5, label="enhancement detector")
                bot.axhline(d["threshold"], color="k", ls="--", lw=1.2, label=f"threshold {d['threshold']:g}")
                if d["hit"]:
                    bot.axvline(d["arrival_time_hr"], color="tab:green", lw=1.5)
                bot.set_xlabel("Hours from model start")
                bot.set_ylabel("fractional enhancement")
                bot.legend(loc="upper left", fontsize=8)
            fig.suptitle("Earth solar-wind detector: a hit and a miss", fontsize=12)
            fig.tight_layout()
            """
        ),
        md(
            r"""
            ## Hit vs miss summary

            The scalar outcomes - one row each - are exactly what `run_design()` writes to `results.csv` and what the GP trains on (Task 3 of notebook 04).
            """
        ),
        code(
            r"""
            def summary_row(case):
                d, f = case["diag"], case["front"]
                return {
                    "case": case["name"],
                    "longitude": round(float(case["theta"][1]), 1),
                    "hit": d["hit"],
                    "arrival_time_hr": d["arrival_time_hr"],
                    "max_enhancement": d["max_speed_enhancement"],
                    "front_hit": f["front_hit"],
                    "peak_vsw": d["peak_vsw"],
                }

            display(pd.DataFrame([summary_row(hit), summary_row(miss)]))
            """
        ),
        md(
            r"""
            ## Heliosphere snapshot

            The **second plot** `visualize-threshold` writes is a top-down view of the heliosphere, which makes the geometry obvious: the hit CME sweeps over Earth, the miss CME slides past on the far side. We snapshot the hit at its arrival time and the miss partway through the run.
            """
        ),
        code(
            r"""
            import astropy.units as u
            import huxt.huxt_analysis as HA

            for case in [hit, miss]:
                d = case["diag"]
                snap = (d["arrival_time_hr"] if d["hit"] else 72.0) * u.hour
                try:
                    figh, _ = HA.plot(case["model"], snap, plotHCS=False, trace_earth_connection=False)
                    figh.suptitle(f"{case['name']} - heliosphere at {snap.to('hr').value:.0f} hr", y=0.97)
                except Exception as exc:
                    print(f"heliosphere snapshot for {case['name']} skipped:", exc)
            """
        ),
        md(
            r"""
            ## What `visualize-threshold` Does

            The `run` step reduces *every* design sample to one row of `results.csv` with exactly the detector above. When a case looks like a visual hit but is labelled a miss (or vice-versa), `visualize-threshold` replays selected completed samples and writes the two plots we just reproduced - the time-series detector panel and the heliosphere snapshot - to `runs/gp_surrogate/<event>/figures/threshold_diagnostics/`:

            ```bash
            python scripts/gp_huxt_surrogate.py --event 2017-09-06 visualize-threshold \
              --detector-threshold 0.25 \
              --detector-method enhancement \
              --sample-ids 8
            ```

            `--sample-ids 8` picks sample 8 from `results.csv` (build a batch first via notebook 04 or `gp_huxt_surrogate.py … design/run`). Drop `--sample-ids` to auto-select the strongest near-miss cases instead, and switch `--detector-method` to compare how `enhancement`, `front`, and `hybrid` label the same run.
            """
        ),
    ]


def main() -> int:
    write_notebook("01_gp_tutorial.ipynb", gp_tutorial_notebook())
    write_notebook("02_huxt_runs.ipynb", huxt_runs_notebook())
    write_notebook("03_arrival_detector_examples.ipynb", arrival_detector_examples_notebook())
    write_notebook("04_gp_huxt_application.ipynb", gp_huxt_application_notebook())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
