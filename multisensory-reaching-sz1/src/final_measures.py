import os
import numpy as np
import pandas as pd
from scipy import stats
from q1_final import (onset_trim, fname, jerk, jerk_power_per_sample, add_lowpass, LP, CONTROL_DIR, PATIENT_DIR,
                      DUR_CONTROL, DUR_PATIENT, EXCLUDE, SR)

def process(folder, dpath, group):
    dur = pd.read_csv(dpath)
    rows = []
    for sub in dur["Subject"].astype(str).unique():
        if sub in EXCLUDE:
            continue
        path = os.path.join(folder, f"raw_{fname(sub)}_Trajectories.csv")
        if not os.path.exists(path):
            continue
        traj = pd.read_csv(path)
        sd = dur[dur["Subject"].astype(str) == sub]
        m, s = sd.trialDuration.mean(), sd.trialDuration.std(ddof=1)
        for _, r in sd[sd.trialDuration.between(m - 2.5 * s, m + 2.5 * s)].iterrows():
            tdf = traj[traj.Trial == r.Trial]
            if len(tdf) < 2:
                continue
            tdf = add_lowpass(tdf)
            n = int(np.floor(r.trialDuration * SR))
            t = tdf.iloc[-(n + 1):] if n < len(tdf) else tdf
            t = onset_trim(t)
            if len(t) < 16:
                continue
            j = t["LP jerk"].to_numpy()
            rows.append({"Group": group, "Subject": sub, "Trial": r.Trial,
                         "Visibility": t["Visibility"].iloc[0],
                         "Half": "first" if r.Trial <= sd.Trial.max() / 2 else "second",   # split at each person's own midpoint
                         "hand_error": r["Hand error"], "trial_duration": r.trialDuration,   # whole trial, not movement time
                         "whole": np.log(jerk_power_per_sample(j))})
    return pd.DataFrame(rows)


df = pd.concat([process(CONTROL_DIR, DUR_CONTROL, "control"),
                process(PATIENT_DIR, DUR_PATIENT, "patient")], ignore_index=True)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["whole"])
df.to_csv("final_measures_trials.csv", index=False)

subj = df.groupby(["Group", "Subject"])[["whole", "trial_duration"]].mean().reset_index()
subj.to_csv("final_measures_subject.csv", index=False)
for col in ["whole"]:
    c, p = subj.loc[subj.Group == "control", col], subj.loc[subj.Group == "patient", col]
    print(col, "control", round(c.mean(), 3), "patient", round(p.mean(), 3),
          "t p", round(stats.ttest_ind(c, p).pvalue, 3), "MW p", round(stats.mannwhitneyu(c, p).pvalue, 3),
          "shapiro", round(stats.shapiro(c).pvalue, 3), round(stats.shapiro(p).pvalue, 3))

vis = df.groupby(["Subject", "Visibility"])["whole"].mean().unstack().dropna()
print("visible vs invisible whole jerk paired p", round(stats.ttest_rel(vis.iloc[:, 0], vis.iloc[:, 1]).pvalue, 3))

df["log_error"] = np.log(df["hand_error"].where(df["hand_error"] > 0))
half = df.groupby(["Group", "Subject", "Half"])[["whole", "log_error"]].mean().reset_index()
half.to_csv("final_fatigue_subject.csv", index=False)
for col in ["whole", "log_error"]:
    pv = half.pivot_table(index=["Group", "Subject"], columns="Half", values=col).dropna().reset_index()
    pv["change"] = pv["second"] - pv["first"]
    for g in ["control", "patient"]:
        d = pv[pv.Group == g]
        print(col, g, "first", round(d["first"].mean(), 3), "second", round(d["second"].mean(), 3),
              "paired p", round(stats.ttest_rel(d["first"], d["second"]).pvalue, 3))
    print(col, "group diff in change p",
          round(stats.ttest_ind(pv[pv.Group == "control"].change, pv[pv.Group == "patient"].change).pvalue, 3))
