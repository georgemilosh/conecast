# Conecast exercises — ASAP Summer School (2-hour hands-on session)

These are the hands-on companion to the tutorials in [`../notebooks`](../notebooks), which
remain the **reference solutions** — every `TODO` here is a worked cell in tutorial 03 or 04.

These exercises are best done on a **personal machine**: HUXt runs at ~1.5 s each there, so a
full 300-run batch is only ~7–8 minutes. Colab works too but its CPUs are slower (see Setup).

You pick **one event** in Exercise 1 and reuse it throughout Exercises 2–4.

## The four exercises

| # | Notebook | Builds on | HUXt cost | ~Time |
| - | -------- | --------- | --------- | ----- |
| 1 | [`01_exercise_new_event.ipynb`](01_exercise_new_event.ipynb) | nb 02 + 04 | provisions your event + runs a 300-run batch (~7–8 min locally) | ~25 min |
| 2 | [`02_exercise_how_many_runs.ipynb`](02_exercise_how_many_runs.ipynb) | nb 04 (Task 3) | reuses your Exercise-1 batch — no new runs | 30 min |
| 3 | [`03_exercise_what_is_a_hit.ipynb`](03_exercise_what_is_a_hit.ipynb) | nb 03 + 04 | reuses your Exercise-1 batch — no reruns | 35 min |
| 4 | [`04_exercise_your_event_analysis.ipynb`](04_exercise_your_event_analysis.ipynb) | nb 04 (Tasks 3–4) | none (optionally loads the tutorial 2017-09-06 batch to compare) | 30 min |

## The session arc (why this order)

1. **Run a real batch, analyse while it finishes.** Exercise 1 picks **one** event from
   `data_dir/events.csv`, generates its HUXt boundary (the WSA+ checkpoint `wsaplus.pt` is
   already local, so only the GONG magnetogram is fetched), and runs a **300-run `design → run`
   batch (~7–8 min on a laptop, ~1.5 s/run)**. `run_design` writes `results.csv` incrementally,
   so you can start Exercises 2–3 while it finishes; Exercise 4 then analyses that same event.
2. **Exercises 2 and 3 reuse that event — no *new* HUXt runs.** Since `runs/` is not shipped,
   they need Exercise 1 to have provisioned the event; if its batch is missing but the boundary
   exists they regenerate it (~7–8 min), otherwise they point you back to Exercise 1. Exercise 2
   holds out a fixed test set and trains on growing subsets (≈25 → 250) to find how many runs are
   "enough" and whether the GP's uncertainty is honest. Exercise 3 recomputes the hit label with
   a **compound threshold** (enhancement **and** `peak_vsw`) straight from the CSV columns — a
   genuinely different problem, zero reruns.
3. **Exercise 4 analyses your event** — fit the surrogate, draw a slice, rank parameter
   importance, and **compare against the tutorial event (2017-09-06 from notebook 04)** if you
   have its batch — a clean two-event contrast (e.g. a limb event vs the near-central tutorial one).

## Suggested 120-minute run sheet

| Time | Segment |
| ---- | ------- |
| 0:00–0:10 | **Setup.** Open the `conecast` kernel (or Colab). Recap the take-away from tutorial 01: a GP returns a prediction **and** an honest uncertainty. |
| 0:10–0:30 | **Exercise 1.** Pick your event, generate its boundary, start its 300-run batch (~7–8 min) — let it run and move on. |
| 0:30–1:00 | **Exercise 2.** Subsample your event's batch: scaling curve of MAE + uncertainty vs N; is the GP over-confident? |
| 1:00–1:35 | **Exercise 3.** Compound hit threshold: how the hit map / arrival / importance shift when "hit" also means "fast". |
| 1:35–1:55 | **Exercise 4.** Fit + feature-importance on your event; compare with the tutorial event (2017-09-06). |
| 1:55–2:00 | **Wrap-up discussion.** |

## Setup

**On generated data:** the repo does **not** ship `runs/` — students generate the runs
themselves. Exercise 1 provisions one event (boundary + 300-run batch); Exercises 2–4 reuse it.
If a notebook is opened without Exercise 1 having run for that event, it regenerates the batch
when the boundary is present (~7–8 min) or otherwise points back to Exercise 1. The only shipped
HUXt input is the committed boundary `v_boundary_2017-09-06.npz` — used by tutorial notebook 04
and available to Exercise 4 for the optional comparison.

**Local (recommended):**

```bash
conda activate conecast
python -m ipykernel install --user --name conecast --display-name "Python (conecast)"
jupyter lab exercises/
```

**Colab:** the first cell of each notebook clones the repo and installs HUXt + WSA+. The
bootstrap may restart the runtime once — re-run it when it reconnects. Exercise 1 carries a full
**"Running on Google Colab"** note; the essentials:

- **No Terminal on the free tier.** Run the 300-run batch as a background process from a cell
  (`subprocess.Popen(...)`) rather than a terminal — it returns immediately so the kernel stays
  free to poll `results.csv`. Exercise 1 has ready-made *launch* and *progress* cells.
- **Each notebook is its own VM.** Separate Colab tabs don't share a filesystem, and Exercises
  2–4 all reuse your Exercise-1 event — so **run them in the same tab** as Exercise 1, or write
  `runs/` to a mounted Google Drive.
- **Slower CPUs + disconnects.** Colab CPUs are slower than a laptop (the 300-run batch takes
  longer than the ~7–8 min it needs locally) and idle VMs are recycled (~90 min) — keep the tab
  active, or persist `results.csv` to Drive.
