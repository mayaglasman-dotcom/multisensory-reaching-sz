"""Occasional strongly curved reaches (exploratory): do they explain why mean and median directness disagree?
Reads reach_definitions_trials.csv; prints group and visibility comparisons."""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats

r = pd.read_csv("reach_definitions_trials.csv"); r = r[r.kept].dropna(subset=["dir_duration"])


def summ(d):
    s = d.dir_duration
    return pd.Series({"mean": s.mean(), "median": s.median(), "gap": s.mean() - s.median(),
                      "over_1p5": (s > 1.5).mean() * 100, "p90": s.quantile(0.9)})


def mw(c, p):
    u = stats.mannwhitneyu(c, p); return f"U = {min(u.statistic, len(c) * len(p) - u.statistic):.0f}, p = {u.pvalue:.3f}"


P = r.groupby(["Group", "Subject"]).apply(summ, include_groups=False).reset_index()
print("Patients vs controls (per-person values; Mann-Whitney)")
for c in ["mean", "median", "gap", "over_1p5", "p90"]:
    a, b = P[P.Group == "control"][c], P[P.Group == "patient"][c]
    print(f"  {c}: controls {a.mean():.3f}, patients {b.mean():.3f}; {mw(a, b)}")
r["curved"] = (r.dir_duration > 1.5).astype(int)
g = smf.gee("curved ~ Group", groups="Subject", data=r, family=sm.families.Binomial()).fit()
print(f"  trial level, reaches above 1.5 (GEE, clustered by participant): odds ratio {np.exp(g.params['Group[T.patient]']):.2f}, "
      f"p = {g.pvalues['Group[T.patient]']:.3f}")

V = r.groupby(["Subject", "Visibility"]).apply(summ, include_groups=False).unstack()
print("Seeing the hand (visible vs hidden, all participants; Wilcoxon)")
for c in ["mean", "median", "over_1p5", "p90"]:
    a, b = V[(c, "Visible Hand")], V[(c, "Invisible Hand")]; w = stats.wilcoxon(a, b)
    print(f"  {c}: visible {a.mean():.3f}, hidden {b.mean():.3f}; W = {w.statistic:g}, p = {w.pvalue:.2g}")
