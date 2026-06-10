# Conecast exercises — ASAP Summer School (2-hour hands-on session)

These are the hands-on companion to the tutorials in [`../notebooks`](../notebooks), which
remain the **reference solutions** — every `TODO` here is a worked cell in tutorial 03 or 04.

## The four exercises

| # | Notebook | Builds on | HUXt cost | ~Time |
| - | -------- | --------- | --------- | ----- |
| 1 | [`01_exercise_new_event.ipynb`](01_exercise_new_event.ipynb) | nb 02 + 04 | prepares a new event + launches a 200-run batch (runs in the background) | 20 min active |
| 2 | [`02_exercise_how_many_runs.ipynb`](02_exercise_how_many_runs.ipynb) | nb 04 (Task 3) | none — subsamples committed 2017-09-06 `results.csv` | 30 min |
| 3 | [`03_exercise_what_is_a_hit.ipynb`](03_exercise_what_is_a_hit.ipynb) | nb 03 + 04 | none — relabels committed `results.csv` | 35 min |
| 4 | [`04_exercise_your_event_analysis.ipynb`](04_exercise_your_event_analysis.ipynb) | nb 04 (Tasks 3–4) | none beyond Exercise 1's batch | 30 min |

## The session arc (why this order)

1. **Launch early, analyse while it runs.** Exercise 1 picks a contrasting event from
   `data_dir/events.csv`, generates its HUXt boundary (the WSA+ checkpoint `wsaplus.pt` is
   already local, so only the GONG magnetogram is fetched), and **launches a 200-run batch in a
   terminal**. A 200-run batch takes much longer than the session (~1–2 min/run), so it runs in
   the background while students work on Exercises 2–3. `run_design` writes `results.csv`
   incrementally, so Exercise 4 analyses however many runs have completed.
2. **Exercises 2 and 3 need no HUXt at all** — they work on the committed 2017-09-06 batch
   (`runs/gp_surrogate/2017-09-06/results.csv`, 200 runs / 155 hits). Exercise 2 subsamples it
   to N = 25…200 to find how many runs are "enough" and whether the GP's uncertainty is honest.
   Exercise 3 recomputes the hit label with a **compound threshold** (enhancement **and**
   `peak_vsw`) straight from the CSV columns — a genuinely different problem, zero reruns.
3. **Exercise 4 returns to the student's event** — fit the surrogate on the completed rows,
   draw a slice, rank parameter importance, and contrast with 2017-09-06.

## Suggested 120-minute run sheet

| Time | Segment |
| ---- | ------- |
| 0:00–0:10 | **Setup.** Open the `conecast` kernel (or Colab). Recap the take-away from tutorial 01: a GP returns a prediction **and** an honest uncertainty. |
| 0:10–0:30 | **Exercise 1.** Pick a contrasting event, generate its boundary, **start the 200-run batch in a terminal** — then leave it running. |
| 0:30–1:00 | **Exercise 2.** Subsample 2017-09-06: scaling curve of MAE + uncertainty vs N; is the GP over-confident? |
| 1:00–1:35 | **Exercise 3.** Compound hit threshold: how the hit map / arrival / importance shift when "hit" also means "fast". |
| 1:35–1:55 | **Exercise 4.** Fit + feature-importance on the student's event (whatever has completed); contrast with 2017-09-06. |
| 1:55–2:00 | **Wrap-up discussion.** |

## Setup

**Colab:** the first cell of each notebook clones the repo and installs HUXt + WSA+; Exercises
2–4 need no HUXt. The bootstrap may restart the runtime once — re-run it when it reconnects.
Exercise 1 carries a full **"Running on Google Colab"** note; the essentials:

- **No Terminal on the free tier.** Launch the 200-run batch as a background process from a cell
  (`subprocess.Popen(...)`) rather than a terminal — it returns immediately so the kernel stays
  free to poll `results.csv`. Exercise 1 has ready-made *launch* and *progress* cells.
- **Each notebook is its own VM.** Separate Colab tabs don't share a filesystem, so Exercise 4
  must run in the **same tab** as Exercise 1 (or write `results.csv` to a mounted Google Drive).
  Exercises 2–3 use only the committed 2017-09-06 data, so they're self-contained.
- **Disconnects + speed.** Colab recycles idle VMs (~90 min) and free CPUs are slow, so a
  background run can die before 200 finishes. Lower the batch to `n=60–100` for a live result
  (200 stays as homework), keep the tab active, or persist to Drive.