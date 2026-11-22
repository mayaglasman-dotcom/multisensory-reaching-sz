"""Runs the whole analysis in order. Usage: python run_all.py [--allow-missing]

Tables and statistics go to results/ (one log per step in results/logs/), figures to figures/.
"""
import os, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from config import RAW_CONTROL_DIR, RAW_PATIENT_DIR, DUR_CONTROL, DUR_PATIENT, CLINICAL_XLSX, RESULTS_DIR, TASK_IMAGE

STEPS = [  # (script, what it produces)
    ("prepare_data.py", "corrected trajectory files and alignment check"),
    ("q1_final.py", "jerk in each quarter of the reach (Table 1, Figure 3)"),
    ("jerk_filter_checks.py", "checks on the 10 Hz filter: cutoff and end-of-recording handling"),
    ("reach_definitions.py", "directness and smoothness per trial"),
    ("duration_window_tests.py", "per-person directness and smoothness (Table 2)"),
    ("final_measures.py", "whole-reach jerk, session halves"),
    ("carryover57.py", "carryover slopes (Table 3, Figure 4)"),
    ("equivalence.py", "equivalence tests and Bayes factors (Table S2)"),
    ("recompute.py", "independent recomputation of the main measures"),
    ("analyze.py", "statistics on the independent recomputation"),
    ("stats_all.py", "every statistic reported in the paper"),
    ("path_speed_check.py", "is the directness-movement time link only the extra path length?"),
    ("extra_checks.py", "age, diagnosis and pre-movement checks"),
    ("thresh.py", "results at 5%, 10% and 20% onset thresholds (Table S3)"),
    ("make_figs_onset.py", "Figures 2-4 and S1-S3"),
    ("make_fig1_onset.py", "Figure 1"),
    ("table_s1.py", "participant characteristics (Table S1)"),
    ("smoothness_vs_jerk.py", "how path smoothness relates to jerk, across and within participants"),
    ("dimensionless_jerk.py", "speed-independent (dimensionless) jerk as a check on the main jerk measure"),
    ("sparc_check.py", "spectral arc length (SPARC) as a second speed-independent smoothness check"),
    ("figure2_same_speed.py", "Figure 2 with movement time accounted for"),
    ("path_measures_vs_movement_time.py", "path smoothness and directness vs movement time, across and within participants"),
    ("example_reaches.py", "example reaches with similar path smoothness but different jerk"),
    ("curved_reaches.py", "occasional strongly curved reaches (exploratory)"),
]
NEEDS_CLINICAL = {"extra_checks.py", "table_s1.py"}


def fname(s):
    s = str(s)
    return "SZR-" + s if len(s) <= 3 else s


def check_data():
    problems = []
    for p in (DUR_CONTROL, DUR_PATIENT, RAW_CONTROL_DIR, RAW_PATIENT_DIR, TASK_IMAGE):
        if not Path(p).exists():
            problems.append(f"missing: {p}")
    if problems:
        return problems
    for folder, dpath in ((RAW_CONTROL_DIR, DUR_CONTROL), (RAW_PATIENT_DIR, DUR_PATIENT)):
        for sub in pd.read_csv(dpath)["Subject"].astype(str).unique():
            f = Path(folder) / f"raw_{fname(sub)}_Trajectories.csv"
            if not f.exists():
                problems.append(f"no trajectory file for {sub}: {f}")
    return problems


def main():
    problems = check_data()
    if problems:
        print("Data check:\n  " + "\n  ".join(problems))
        if "--allow-missing" not in sys.argv:
            sys.exit("Stopping. Participants without a trajectory file would be silently dropped. "
                     "Fix the data folder, or rerun with --allow-missing to continue anyway.")
        print("Continuing without them (--allow-missing).\n")
    logs = RESULTS_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MPLBACKEND="Agg")
    for script, what in STEPS:
        if script in NEEDS_CLINICAL and not CLINICAL_XLSX.exists():
            print(f"skip  {script:28s} (needs {CLINICAL_XLSX.name})")
            continue
        print(f"run   {script:28s} {what}", flush=True)
        with open(logs / script.replace(".py", ".txt"), "w") as log:
            r = subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=RESULTS_DIR, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(open(logs / script.replace(".py", ".txt")).read()[-1500:])
            sys.exit(f"{script} failed; see {logs / script.replace('.py', '.txt')}")
    print(f"\nDone. Statistics: {logs / 'stats_all.txt'}   Figures: {ROOT / 'figures'}")


if __name__ == "__main__":
    main()
