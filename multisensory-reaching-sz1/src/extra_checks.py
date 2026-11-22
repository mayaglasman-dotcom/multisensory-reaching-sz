# Extra checks reported in the paper: age, schizophrenia-only and leave-one-out checks, pre-movement analysis of the original windows,
# and carryover with all participants included. Run after the pipeline and recompute.py (needs trials.csv).
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy import stats
warnings.filterwarnings("ignore")

q = pd.read_csv("q1_final_subject.csv")

# demographics (Supplementary Table S1 source file)
from config import CLINICAL_XLSX as X, CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT
d = pd.read_excel(X, "Demographics"); d = d[d["Record ID"].notna()].copy(); d["Record ID"] = d["Record ID"].astype(str).str.replace(r"\s*\(.*\)\s*$", "", regex=True).str.strip()   # drop notes such as "PC-C19 (also SZR-1019)"
ID = {'C02':'PC-C16','C03':'PC-C18','C04':'PC-C19','C05':'PC-C21','C06':'PC-C22','C07':'PC-C20','C08':'PC-C24','C09':'PC-C25',
      'S02':'PC-S02','S05':'PC-S06','S06':'PC-S16','S07':'PC-S22','S09':'PC-S12','S10':'PC-S23'}
v = pd.read_csv("reach_definitions_trials.csv"); v = v[v.kept]
P = v.groupby("Subject").agg(smooth_mean=("smooth_duration", "mean"), smooth_median=("smooth_duration", "median"),
                             dir_mean=("dir_duration", "mean"), dir_median=("dir_duration", "median")).reset_index()
S = q[["Group", "Subject", "duration", "Q1", "Q2", "Q3", "Q4"]].merge(P, on="Subject")
S["RID"] = S.Subject.map(lambda s: ID.get(s, s))
S = S.merge(d[["Record ID", "Age", "Primary Diagnosis"]], left_on="RID", right_on="Record ID", how="left")
S["Age"] = pd.to_numeric(S.Age, errors="coerce").where(lambda v: v >= 0)   # negative codes (e.g. -888) are missing
S["dx"] = S["Primary Diagnosis"].astype(str).str.strip()
MEAS = ["duration", "smooth_mean", "smooth_median", "dir_mean", "dir_median", "Q1", "Q2", "Q3", "Q4"]

# age: is it related to the measures, across participants and within each group?
A = S.dropna(subset=["Age"])
print("age n:", A.groupby("Group").size().to_dict())
for y in MEAS:
    line = f"age vs {y}:"
    for g, Z in [("all", A), *A.groupby("Group")]:
        r = stats.spearmanr(Z.Age, Z[y]); line += f" {g} rho={r[0]:+.2f} p={r[1]:.3f};"
    print(line)
print("age smallest p:", round(min(stats.spearmanr(Z.Age, Z[y])[1] for y in MEAS for Z in [A, *[z for _, z in A.groupby("Group")]]), 3))

# age and jerk: group difference with age in the model, and the patients' larger late-reach visibility effect
# (same movement time: per-person regression of trial jerk on visibility and log movement time) with age in the model
tq = pd.read_csv("q1_final_trials.csv"); tq["lmt"] = np.log(tq.duration); tq["vis"] = (tq.Visibility == "Visible Hand").astype(float)
E = pd.DataFrame([dict(Subject=s, **{k: smf.ols(f"{k} ~ vis + lmt", z).fit().params["vis"] for k in ["Q1", "Q2", "Q3", "Q4"]})
                  for s, z in tq.groupby("Subject")])
AE = A[["Group", "Subject", "Age", "duration", "Q1", "Q2", "Q3", "Q4"]].merge(E, on="Subject", suffixes=("", "_vis"))
for k in ["Q1", "Q2", "Q3", "Q4"]:
    m = smf.ols(f"{k} ~ Group + Age", AE).fit(); g = smf.ols(f"{k}_vis ~ Group + Age", AE).fit()
    print(f"age model {k}: jerk group p = {m.pvalues['Group[T.patient]']:.3f}; same-speed visibility effect group p = {g.pvalues['Group[T.patient]']:.3f} (age p = {g.pvalues['Age']:.3f})")

# schizophrenia diagnoses only, and leave-one-out
for y in ["duration", "smooth_mean", "smooth_median", "dir_mean"]:
    Z = S[(S.Group == "control") | S.dx.isin(["SZ", "SZ/SZA"])]; c, p = Z[Z.Group == "control"][y], Z[Z.Group == "patient"][y]
    u = stats.mannwhitneyu(c, p); print("SZ only", y, "n", len(c), len(p), "U", min(u.statistic, len(c)*len(p)-u.statistic), "p", round(u.pvalue, 4))
for y, test in [("duration", "MW"), ("smooth_mean", "MW"), ("smooth_median", "MW")]:
    ps = []
    for s in S.Subject:
        Z = S[S.Subject != s]; c, p = Z[Z.Group == "control"][y], Z[Z.Group == "patient"][y]
        ps.append(stats.ttest_ind(c, p).pvalue if test == "t" else stats.mannwhitneyu(c, p).pvalue)
    print("LOO max p", y, round(max(ps), 4))

# carryover visible->visible with everyone included
from carryover_by_visibility_transition import get_all_transition_pairs, carryover_slope
tr = pd.concat([get_all_transition_pairs(CONTROL_DIR, DUR_CONTROL, "control"),
                get_all_transition_pairs(PATIENT_DIR, DUR_PATIENT, "patient")])
sl = [dict(Group=g, slope=carryover_slope(x))
      for (g, s, k), x in tr.groupby(["Group", "Subject", "transition"]) if k == "visible_to_visible" and len(x) >= 10]
sl = pd.DataFrame(sl); c, p = sl[sl.Group == "control"].slope, sl[sl.Group == "patient"].slope
t = stats.ttest_ind(p, c); u = stats.mannwhitneyu(c, p)
print("V->V all included: t", round(t.statistic, 2), "p", round(t.pvalue, 3), "U", min(u.statistic, len(c)*len(p)-u.statistic), "p", round(u.pvalue, 3))

# pre-movement period in the ORIGINAL (duration-based) windows; uses trials.csv from recompute.py
T = pd.read_csv("trials.csv")
X = T[T.kept & (T.nwin >= 16)].copy()   # the trials used in the original analysis
X["pre"] = X.on / 72; X["frac"] = X.on / X.nwin; X["q1move"] = X.on < (X.nwin // 4)
print("mean pre-movement share", round(X.frac.mean(), 3), "| speed at contact / peak", round(X.endfrac.mean(), 3))
P = X.groupby(["Group", "Subject"])[["pre", "q1move", "frac"]].mean().reset_index(); c = P.Group == "control"
for k in ["pre", "q1move"]:
    u = stats.mannwhitneyu(P[c][k], P[~c][k]); print(k, round(P[c][k].mean(), 3), round(P[~c][k].mean(), 3), "U", min(u.statistic, c.sum()*(~c).sum()-u.statistic), "p", round(u.pvalue, 3))
O = X.dropna(subset=["Q1orig"]).groupby(["Group", "Subject"])[["Q1orig", "frac"]].mean().reset_index()
print("Q1 (original window) vs pre-movement share r", round(stats.pearsonr(O.Q1orig, O.frac)[0], 2),
      "| group p controlling for it", round(smf.ols("Q1orig ~ Group + frac", O).fit().pvalues["Group[T.patient]"], 2))
