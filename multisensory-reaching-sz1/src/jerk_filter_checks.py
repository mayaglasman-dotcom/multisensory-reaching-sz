"""Robustness checks for the 10 Hz low-pass filter used before computing jerk.

1. How much of each reach's hand-speed signal lies below 6 and 10 Hz.
2. The jerk results at 6 and 10 Hz cutoffs, with two ways of filtering the end of the recording
   ("odd": the default padding used in the main analysis; "gust": Gustafsson's method).
"""
import os
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.fft import rfft, rfftfreq
from q1_final import process, onset_index, fname, SR, DT, XYZ, CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT

Q = ["Q1", "Q2", "Q3", "Q4"]
GROUPS = [(CONTROL_DIR, DUR_CONTROL, "control"), (PATIENT_DIR, DUR_PATIENT, "patient")]


def two(c, p):
    if min(stats.shapiro(c).pvalue, stats.shapiro(p).pvalue) >= 0.05:
        r = stats.ttest_ind(p, c); return f"t({len(c) + len(p) - 2}) = {r.statistic:.2f}, p = {r.pvalue:.4f}"
    r = stats.mannwhitneyu(c, p); return f"U = {min(r.statistic, len(c) * len(p) - r.statistic):.0f}, p = {r.pvalue:.4f}"


def paired(d):
    if stats.shapiro(d).pvalue >= 0.05:
        r = stats.ttest_1samp(d, 0); return f"t({len(d) - 1}) = {r.statistic:.2f}, p = {r.pvalue:.2g}"
    r = stats.wilcoxon(d); return f"W = {r.statistic:.0f}, p = {r.pvalue:.2g}"


def speed_spectrum():
    out = []
    for folder, dpath, g in GROUPS:
        dur = pd.read_csv(dpath)
        for sub in dur.Subject.astype(str).unique():
            path = os.path.join(folder, f"raw_{fname(sub)}_Trajectories.csv")
            if not os.path.exists(path):
                continue
            traj = pd.read_csv(path)
            for _, r in dur[dur.Subject.astype(str) == sub].iterrows():
                tdf = traj[traj.Trial == r.Trial]; n = int(np.floor(r.trialDuration * SR))
                P = tdf[XYZ].to_numpy()[-(n + 1):]
                if len(P) < 3:
                    continue
                X = P[onset_index(*P.T):]
                if len(X) < 16:
                    continue
                V = np.gradient(X, DT, axis=0); f = rfftfreq(len(X), DT); w = np.hanning(len(X))
                pw = sum(np.abs(rfft((V[:, k] - V[:, k].mean()) * w))**2 for k in range(3))
                out.append(dict(below6=pw[f <= 6].sum() / pw.sum(), below10=pw[f <= 10].sum() / pw.sum()))
    d = pd.DataFrame(out)
    print(f"Share of hand-speed power per reach (median of {len(d)} reaches): below 6 Hz {d.below6.median():.3f}, below 10 Hz {d.below10.median():.3f}")


def summarize(T, label):
    S = T.groupby(["Group", "Subject"])[Q].mean().reset_index()
    V = T.groupby(["Group", "Subject", "Visibility"])[Q].mean().unstack("Visibility")
    D = (V.xs("Visible Hand", axis=1, level=1) - V.xs("Invisible Hand", axis=1, level=1)).reset_index()
    c, p = S.Group == "control", S.Group == "patient"; dc, dp = D.Group == "control", D.Group == "patient"
    print(f"\n{label}")
    for q in Q:
        print(f"  {q}: group {two(S.loc[c, q], S.loc[p, q])} | visible - hidden {D[q].mean():+.2f}, {paired(D[q])}"
              f" | group x visibility {two(D.loc[dc, q], D.loc[dp, q])}")
    L = S.melt(id_vars=["Group", "Subject"], value_vars=Q, var_name="Quarter", value_name="jerk")
    i = pg.mixed_anova(data=L, dv="jerk", within="Quarter", between="Group", subject="Subject", correction=True)
    i = i[i.Source == "Interaction"].iloc[0]
    print(f"  group x quarter F({i.DF1:.0f}, {i.DF2:.0f}) = {i.F:.2f}, Greenhouse-Geisser p = {i.p_GG_corr:.3f}")


if __name__ == "__main__":
    speed_spectrum()
    for cutoff in (10, 6):
        for edge in ("odd", "gust"):
            T = pd.concat([process(f, d, g, cutoff, edge) for f, d, g in GROUPS], ignore_index=True)
            summarize(T.replace([np.inf, -np.inf], np.nan).dropna(subset=Q), f"{cutoff} Hz, end of recording: {edge}")
