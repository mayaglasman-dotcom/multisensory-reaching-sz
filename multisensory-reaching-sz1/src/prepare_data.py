"""Step 0: copy the raw trajectory files into data/corrected_trajectories/, applying the known corrections,
then check that every participant's reaches line up with their targets.

Corrections:
  - SZR-1005, SZR-1016, SZR-5011: one trial is missing from the recording partway through, so from that point
    the trajectory file's trial numbers are one behind the trial-duration file. Trials from the listed trial
    onward are renumbered +1.
  - SZR-1008: the originally shared file was a byte-for-byte copy of SZR-5005's. The data folder must hold the
    corrected file; this step stops if the two are still identical.
"""
import hashlib, sys
import numpy as np, pandas as pd
from config import RAW_CONTROL_DIR, RAW_PATIENT_DIR, CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT

SHIFT_FROM = {"SZR-1005": 121, "SZR-1016": 59, "SZR-5011": 184}   # first trajectory trial to renumber +1
# Trials line up when final hand positions track the targets. After correction every participant has r >= 0.97
# and at most 5 reaches ending more than 10 cm from the target; out-of-step files fail one or both limits.
MIN_R, MAX_MISSES, MISS_M = 0.95, 10, 0.10
# Second check: each trial's recorded duration (End - Start) should equal its logged trialDuration. A file that is
# out of step gives a run of mismatching trials; after correction no participant has more than one in a row.
DUR_TOL_S, MAX_RUN = 0.02, 2


def fname(s):
    s = str(s)
    return "SZR-" + s if len(s) <= 3 else s


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def longest_run(flags):
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def main():
    problems = []
    a, b = RAW_CONTROL_DIR / "raw_SZR-1008_Trajectories.csv", RAW_PATIENT_DIR / "raw_SZR-5005_Trajectories.csv"
    if a.exists() and b.exists() and md5(a) == md5(b):
        problems.append("raw_SZR-1008_Trajectories.csv is still a copy of SZR-5005's file; replace it with the corrected file")

    for src, dst, dpath in ((RAW_CONTROL_DIR, CONTROL_DIR, DUR_CONTROL), (RAW_PATIENT_DIR, PATIENT_DIR, DUR_PATIENT)):
        dst.mkdir(parents=True, exist_ok=True)
        for old in dst.glob("raw_*_Trajectories.csv"):   # a copy whose raw file is gone must not be analysed
            if not (src / old.name).exists():
                old.unlink(); print(f"removed stale corrected copy {old.name}")
        dur = pd.read_csv(dpath)
        for sub in dur["Subject"].astype(str).unique():
            name = f"raw_{fname(sub)}_Trajectories.csv"
            if not (src / name).exists():
                continue
            t = pd.read_csv(src / name)
            if fname(sub) in SHIFT_FROM:
                k = SHIFT_FROM[fname(sub)]
                t.loc[t.Trial >= k, "Trial"] += 1
                print(f"{fname(sub)}: trials {k}+ renumbered +1")
            # every later step reads these copies in file order, so store samples in trial and sample order
            t = t.sort_values(["Trial", "sample"], kind="stable")
            t.to_csv(dst / name, index=False)

            # alignment check: each trial's final hand position should track its target's left-right position
            end_x = t.groupby("Trial")["Hand Pos X"].last()
            j = dur[dur["Subject"].astype(str) == sub].set_index("Trial").join(end_x.rename("end_x"), how="inner")
            r = np.corrcoef(j["Ball Pos X"], j["end_x"])[0, 1]
            misses = int(((j["end_x"] - j["Ball Pos X"]).abs() > MISS_M).sum())
            if not (r >= MIN_R and misses <= MAX_MISSES):
                problems.append(f"{fname(sub)}: final hand positions don't track targets (r = {r:.2f}, {misses} reaches "
                                f"ending >10 cm away); trials may be out of step")
            if {"Start", "End"} <= set(t.columns):
                rec = t.groupby("Trial").agg(s=("Start", "first"), e=("End", "first"))
                jd = dur[dur["Subject"].astype(str) == sub].set_index("Trial")[["trialDuration"]].join(rec, how="inner").sort_index()
                run = longest_run(((jd.e - jd.s) - jd.trialDuration).abs() > DUR_TOL_S)
                if run > MAX_RUN:
                    problems.append(f"{fname(sub)}: {run} trials in a row whose recorded duration doesn't match the "
                                    f"trial log; trials may be out of step")

    if problems:
        sys.exit("Data problems:\n  " + "\n  ".join(problems))
    print("All participants' reaches line up with their targets.")


if __name__ == "__main__":
    main()
