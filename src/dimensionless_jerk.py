"""Speed-independent (dimensionless) jerk, as a check on the main jerk measure.

Mean squared jerk grows steeply with movement speed, so faster reaches score higher even when they are just as even.
Dimensionless jerk (Hogan and Sternad 2009) scales the integrated squared jerk by movement time and peak speed:
    DJ = T^3 / v_peak^2 * integral(jerk^2 dt)
T and v_peak are those of the whole reach, so the four quarters are on one scale and add up to the whole reach.
A version scaled by reach amplitude instead of peak speed (DJ = T^5 / A^2 * integral(jerk^2 dt)) is also computed.
Values are log-transformed; higher means less smooth. Same trials, filter and onset as q1_final.py."""
import os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from config import CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT
from q1_final import SR, DT, LP, fname, add_lowpass, onset_trim


def trials_for(folder, dpath, group):
    dur = pd.read_csv(dpath); rows = []
    for sub in dur["Subject"].astype(str).unique():
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
            P = t[LP].to_numpy()
            v = np.linalg.norm(np.gradient(P, DT, axis=0), axis=1)
            T, vpk, A = len(t) / SR, v.max(), np.linalg.norm(P[-1] - P[0])
            j2 = t["LP jerk"].to_numpy() ** 2
            q = len(j2) // 4
            row = dict(Group=group, Subject=sub, Trial=r.Trial, Visibility=t["Visibility"].iloc[0], duration=T)
            for name, seg in (("Q1", j2[:q]), ("Q2", j2[q:2 * q]), ("Q3", j2[2 * q:3 * q]), ("Q4", j2[3 * q:]), ("whole", j2)):
                integral = seg.sum() * DT
                row[name] = np.log(T ** 3 / vpk ** 2 * integral)
                row[name + "_amp"] = np.log(T ** 5 / A ** 2 * integral) if A > 0.03 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def two_group(c, p):
    """Same rule as the paper: t-test if both groups look normal, otherwise Mann-Whitney."""
    if stats.shapiro(c).pvalue > 0.05 and stats.shapiro(p).pvalue > 0.05:
        r = stats.ttest_ind(p, c); return f"t({len(c) + len(p) - 2}) = {r.statistic:.2f}, p = {r.pvalue:.3g}"
    r = stats.mannwhitneyu(c, p); return f"U = {r.statistic:.0f}, p = {r.pvalue:.3g}"


def paired(a, b):
    d = (a - b).dropna()
    if stats.shapiro(d).pvalue > 0.05:
        r = stats.ttest_1samp(d, 0); return f"{d.mean():+.2f}, t({len(d) - 1}) = {r.statistic:.2f}, p = {r.pvalue:.3g}"
    r = stats.wilcoxon(d); return f"{d.mean():+.2f}, W = {r.statistic:.0f}, p = {r.pvalue:.3g}"


if __name__ == "__main__":
    t = pd.concat([trials_for(CONTROL_DIR, DUR_CONTROL, "control"), trials_for(PATIENT_DIR, DUR_PATIENT, "patient")], ignore_index=True)
    t = t.replace([np.inf, -np.inf], np.nan)
    t.to_csv("dimensionless_jerk_trials.csv", index=False)
    print(f"trials: {len(t)}\n")
    cols = ["Q1", "Q2", "Q3", "Q4", "whole"]

    for suffix, label in (("", "DIMENSIONLESS JERK, PEAK-SPEED VERSION (main check)"), ("_amp", "DIMENSIONLESS JERK, AMPLITUDE VERSION")):
        cc = [c + suffix for c in cols]
        S = t.groupby(["Group", "Subject"])[cc + ["duration"]].mean().reset_index()
        V = t.groupby(["Group", "Subject", "Visibility"])[cc].mean().unstack()
        print(label)
        print("  does it still depend on speed? participant-level Spearman with movement time:")
        for c in cc:
            r = stats.spearmanr(S[c], S.duration); print(f"    {c}: rho = {r[0]:.2f}, p = {r[1]:.3g}")
        print("  group (patients vs controls), means:")
        for c in cc:
            a, b = S[S.Group == "control"][c], S[S.Group == "patient"][c]
            print(f"    {c}: controls {a.mean():.2f}, patients {b.mean():.2f}; {two_group(a, b)}")
        print("  seeing the hand (visible minus hidden, all participants):")
        for c in cc:
            print(f"    {c}: {paired(V[(c, 'Visible Hand')], V[(c, 'Invisible Hand')])}")
        print("  group x visibility (visibility effect, patients vs controls):")
        D = (V.xs("Visible Hand", axis=1, level=1) - V.xs("Invisible Hand", axis=1, level=1)).reset_index()
        for c in cc:
            a, b = D[D.Group == "control"][c].dropna(), D[D.Group == "patient"][c].dropna()
            print(f"    {c}: controls {a.mean():+.2f}, patients {b.mean():+.2f}; {two_group(a, b)}")
        print("  same movement time: seeing the hand (per-person regression on visibility and log movement time), and group with movement time:")
        t["lmt"] = np.log(t.duration); t["vis"] = (t.Visibility == "Visible Hand").astype(float)
        for c in cc:
            E = []
            for (g, s), d in t.dropna(subset=[c]).groupby(["Group", "Subject"]):
                E.append((g, smf.ols(f"{c} ~ vis + lmt", d).fit().params["vis"]))
            E = pd.DataFrame(E, columns=["Group", "b"])
            gm = smf.ols(f"{c} ~ Group + np.log(duration)", S).fit()
            print(f"    {c}: seeing the hand {paired(E.b, 0 * E.b)}; patients vs controls in that effect: "
                  f"{two_group(E[E.Group == 'control'].b, E[E.Group == 'patient'].b)}; group with movement time p = {gm.pvalues['Group[T.patient]']:.3g}")
        print()
