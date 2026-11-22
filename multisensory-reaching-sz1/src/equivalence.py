import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

BOUND_D = 0.5  # smallest effect of interest (Cohen's d)


def tost(c, p, bound_d=BOUND_D):
    nc, np_ = len(c), len(p)
    sp = np.sqrt(((nc - 1) * c.var() + (np_ - 1) * p.var()) / (nc + np_ - 2))
    diff = p.mean() - c.mean()
    se = sp * np.sqrt(1 / nc + 1 / np_)
    df = nc + np_ - 2
    bound = bound_d * sp
    p_lower = stats.t.sf((diff + bound) / se, df)
    p_upper = stats.t.cdf((diff - bound) / se, df)
    d = diff / sp
    se_d = np.sqrt((nc + np_) / (nc * np_) + d**2 / (2 * df))
    z = stats.t.ppf(0.95, df)
    return d, d - z * se_d, d + z * se_d, max(p_lower, p_upper)


def row(name, c, p):
    c, p = pd.Series(c).dropna(), pd.Series(p).dropna()
    t, p_t = stats.ttest_ind(c, p)
    d, lo, hi, p_tost = tost(c, p)
    bf10 = float(pg.bayesfactor_ttest(t, len(c), len(p), r=0.707))
    return {"measure": name, "n_control": len(c), "n_patient": len(p),
            "p_difference": p_t, "cohens_d": d, "d_90ci_low": lo, "d_90ci_high": hi,
            "p_tost_d0.5": p_tost, "BF01": 1 / bf10}


rows = []
q = pd.read_csv("q1_final_subject.csv")
for col, name in [("duration", "Movement time"), ("Q1", "Jerk, first quarter"), ("Q2", "Jerk, second quarter")]:
    rows.append(row(name, q.loc[q.Group == "control", col], q.loc[q.Group == "patient", col]))
fm = pd.read_csv("final_measures_subject.csv")
for col, name in [("whole", "Whole-reach jerk")]:
    rows.append(row(name, fm.loc[fm.Group == "control", col], fm.loc[fm.Group == "patient", col]))

ds = pd.read_csv("subject_directness_smoothness_duration.csv")
for col, name in [("dir_duration", "Directness (median)"), ("smooth_duration", "Smoothness (median)")]:
    rows.append(row(name, ds.loc[ds.Group == "control", col], ds.loc[ds.Group == "patient", col]))
tr = pd.read_csv("reach_definitions_trials.csv"); tr = tr[tr.kept].dropna(subset=["dir_duration", "smooth_duration"])
dm = tr.groupby(["Group", "Subject"])[["dir_duration", "smooth_duration"]].mean().reset_index()
for col, name in [("dir_duration", "Directness (mean)"), ("smooth_duration", "Smoothness (mean)")]:
    rows.append(row(name, dm.loc[dm.Group == "control", col], dm.loc[dm.Group == "patient", col]))


co = pd.read_csv("carryover_slopes_by_transition_no_outliers.csv")
labels = {"visible_to_visible": "Carryover V→V", "invisible_to_invisible": "Carryover I→I",
          "visible_to_invisible": "Carryover V→I", "invisible_to_visible": "Carryover I→V"}
for tr, name in labels.items():
    s = co[co.transition == tr]
    rows.append(row(name, s.loc[s.Group == "control", "slope"], s.loc[s.Group == "patient", "slope"]))

fa = pd.read_csv("final_fatigue_subject.csv")
for col, name in [("whole", "Fatigue: change in jerk"), ("log_error", "Fatigue: change in hand error")]:
    pv = fa.pivot_table(index=["Group", "Subject"], columns="Half", values=col).dropna().reset_index()
    pv["change"] = pv["second"] - pv["first"]
    rows.append(row(name, pv.loc[pv.Group == "control", "change"], pv.loc[pv.Group == "patient", "change"]))

out = pd.DataFrame(rows)
out["equivalent_at_d0.5"] = out["p_tost_d0.5"] < 0.05
out["BF_reading"] = pd.cut(out["BF01"], [0, 1 / 3, 1, 3, 10, np.inf],
                           labels=["favours difference", "inconclusive (leans difference)",
                                   "inconclusive (leans no difference)", "moderate evidence of no difference",
                                   "strong evidence of no difference"])
out = out.round(3)
out.to_csv("equivalence_bayes.csv", index=False)
print(out.to_string(index=False))

# power of TOST at this sample size if the true effect were zero (normal approximation throughout)
nc, np_ = int((q.Group == "control").sum()), int((q.Group == "patient").sum())
se = np.sqrt(1 / nc + 1 / np_)
z = stats.norm.ppf(0.95)
print(f"TOST power (true d=0, bound 0.5, n={nc}/{np_}):", round(2 * stats.norm.cdf(0.5 / se - z) - 1, 2))
for target in [0.8]:
    for nn in range(20, 200):
        if 2 * stats.norm.cdf(0.5 / np.sqrt(2 / nn) - stats.norm.ppf(0.95)) - 1 >= target:
            print("n per group for 80% TOST power:", nn); break
