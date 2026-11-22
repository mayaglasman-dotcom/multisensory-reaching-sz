# Multisensory modulation of reaching dynamics in neurotypical adults and schizophrenia

Analysis code for the paper. Participants reached to targets in virtual reality with the hand visible or hidden while hand position was recorded at 72 Hz. Each reach is analyzed from movement onset (where hand speed last rises above 10% of the trial's peak speed before the peak) to target contact.

The data are not included in this repository.

## Setup

Install the packages into a new environment, not an existing one such as Anaconda's base, because these versions can conflict with packages already installed there:

```
conda create -n reach python=3.12 -y
conda activate reach
pip install -r requirements.txt
```

Place the data in `data/` (layout in [data/README.md](data/README.md)), or point the `REACH_DATA_DIR` environment variable at another folder. All paths are set in `src/config.py`.

## Run

```
python run_all.py
```

This checks that every participant listed in the trial-duration files has a trajectory file, then runs the steps below in order. Tables and statistics are written to `results/` (one log per step in `results/logs/`), and figures to `figures/`. A full run takes about four minutes.

| Step | Script | Produces |
| --- | --- | --- |
| 0 | `prepare_data.py` | Corrected copies of the trajectory files, and a check that every participant's trials line up (see Data corrections) |
| 1 | `q1_final.py` | Jerk in each quarter of the reach (Table 1, Figure 3) |
| 2 | `jerk_filter_checks.py` | Checks on the 10 Hz filter: share of signal below 6 and 10 Hz, results at 6 Hz, and a second way of filtering the end of each recording |
| 3 | `reach_definitions.py` | Directness and smoothness per trial |
| 4 | `duration_window_tests.py` | Per-person directness and smoothness (Table 2) |
| 5 | `final_measures.py` | Whole-reach jerk; session halves |
| 6 | `carryover57.py` | Carryover from the previous trial (Table 3, Figure 4) |
| 7 | `equivalence.py` | Equivalence tests and Bayes factors (Table S2) |
| 8 | `recompute.py`, `analyze.py` | Main results recomputed with separately written code (same methods; checks the arithmetic) |
| 9 | `stats_all.py` | Every statistic reported in the paper (`results/logs/stats_all.txt`) |
| 10 | `path_speed_check.py` | Whether the link between directness and movement time is only the extra length of a curved path |
| 11 | `extra_checks.py` | Age, diagnosis-only and pre-movement checks |
| 12 | `thresh.py` | Results at 5%, 10% and 20% onset thresholds (Table S3) |
| 13 | `make_figs_onset.py`, `make_fig1_onset.py` | Figures 1–4 and S1–S3 |
| 14 | `table_s1.py` | Participant characteristics (Table S1) |
| 15 | `smoothness_vs_jerk.py` | How path smoothness relates to jerk, across participants and trial by trial within each participant |
| 16 | `dimensionless_jerk.py` | Speed-independent (dimensionless) jerk as a check on the main jerk measure |
| 17 | `sparc_check.py` | Spectral arc length (SPARC) as a second speed-independent smoothness check |
| 18 | `figure2_same_speed.py` | Figure 2 with movement time accounted for |
| 19 | `path_measures_vs_movement_time.py` | Path smoothness and directness vs movement time, across and within participants |
| 20 | `example_reaches.py` | Example reaches with similar path smoothness but different jerk |
| 21 | `curved_reaches.py` | Occasional strongly curved reaches: group and visibility comparisons (exploratory) |

Steps 11 and 14 need the demographics spreadsheet and are skipped if it is missing.

Helper modules: `config.py` (paths), `directness_smoothness.py` (smoothness fit), `carryover_by_visibility_transition.py` and `carryover_lag_analysis.py` (trial pairs for carryover).

## Data corrections

`prepare_data.py` copies the exported trajectory files to `data/corrected_trajectories/`, and every analysis reads those copies. The raw files are never changed.

| Participant | Problem in the exported file | Correction |
| --- | --- | --- |
| SZR-1005 | One trial is missing from the recording; from trial 121 on, trial numbers are one behind the trial-duration file | Trials 121 onward renumbered +1 |
| SZR-1016 | Same, from trial 59 | Trials 59 onward renumbered +1 |
| SZR-5011 | Same, from trial 184 | Trials 184 onward renumbered +1 |
| SZR-1008 | The originally shared file was a copy of SZR-5005's | Replaced with the correct recording; the run stops if the copy is still in place |

After correcting, the script checks that each participant's final hand positions track the target positions (correlation of at least 0.95, and no more than 10 reaches ending over 10 cm from the target). It also checks that each trial's recorded duration (End minus Start) matches its logged `trialDuration`; a file that is out of step gives a run of mismatching trials, and no corrected file has more than one in a row. A file that fails either check stops the run. Samples are written in trial and sample order.

## Methods in brief

- **Reach window:** from movement onset (the last rise through 10% of peak speed before the peak, so an early drift that slows again is not counted) to the last recorded sample (target contact). Trials more than 2.5 SD from the participant's mean duration, or with fewer than 16 samples after onset, are excluded from movement time and jerk. If a recording is shorter than its logged duration, the whole recording is used. Directness and smoothness use reaches with at least 8 samples and more than 3 cm of net displacement. Trial counts for each exclusion, by group, are written to `results/q1_final_trial_counts.csv`.
- **Jerk:** hand positions are low-pass filtered at 10 Hz (second-order, zero-lag Butterworth) over each trial's whole recording, then differentiated three times on each axis and combined into a magnitude; the reach window is then cut out. Each segment's jerk is summarized as mean squared jerk, the integrated squared jerk divided by the number of samples, then log-transformed. Each reach is split into four quarters with equal numbers of samples, with any remainder in the last quarter.
- **Directness:** path length divided by straight-line distance. **Smoothness:** R² of a cubic fit pinned at the start and end points.
- **Carryover:** slope of left-right endpoint error on the distance between the previous and current targets, with the current target's position as a second predictor so that a tendency to land nearer the middle is not counted as carryover; per participant and visibility transition, slopes more than 2.5 SD from the group mean excluded.
- **Movement time:** added to regressions of jerk in each quarter, smoothness and directness on group.
- **Checks in `stats_all.py`:** visibility effects on jerk adjusted for each trial's movement time within each person; main results repeated using only trials in which the hand was not already moving at the start of the recording window.
- **Statistics:** one value per participant. Shapiro-Wilk decides between t-tests and Mann-Whitney U (groups) or paired t-tests and Wilcoxon signed-rank tests (visibility). Mixed ANOVA with Greenhouse-Geisser correction; equivalence tests at d = ±0.5; JZS Bayes factors (r = 0.707).

## Tested with

Python 3.13 and the package versions in `requirements.txt`.
