"""Spectral arc length (SPARC; Balasubramanian et al. 2015) as a speed-independent smoothness check.

SPARC is the length of the speed profile's normalized frequency spectrum up to 10 Hz (amplitude threshold 0.05).
Values are negative; closer to zero is smoother. On simulated 72 Hz minimum-jerk reaches it is the same for every
movement time (-1.40) and drops when a corrective sub-movement is added (-1.58), so it is not tied to speed.
The reach window ends at target contact, often before the hand has stopped; that truncation lowers SPARC, so each
trial's end speed (as a fraction of its peak) is also recorded and checked. Same trials, filter and onset as q1_final.py."""
import os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from config import CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT
from q1_final import SR, DT, LP, fname, add_lowpass, onset_trim


def sparc(v, fs=SR, padlevel=4, fc=10.0, amp_th=0.05):
    nfft = int(2 ** (np.ceil(np.log2(len(v))) + padlevel))
    f = np.arange(nfft) * fs / nfft
    M = np.abs(np.fft.fft(v, nfft)); M = M / M.max()
    keep = f <= fc; f, M = f[keep], M[keep]
    idx = np.flatnonzero(M >= amp_th); f, M = f[idx[0]:idx[-1] + 1], M[idx[0]:idx[-1] + 1]
    return -np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(M) ** 2))


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
            t = onset_trim(tdf.iloc[-(n + 1):] if n < len(tdf) else tdf)
            if len(t) < 16:
                continue
            v = np.linalg.norm(np.gradient(t[LP].to_numpy(), DT, axis=0), axis=1)
            h = len(v) // 2
            rows.append(dict(Group=group, Subject=sub, Trial=r.Trial, Visibility=t["Visibility"].iloc[0],
                             duration=len(t) / SR, end_speed=v[-1] / v.max(),
                             sparc=sparc(v), sparc_first=sparc(v[:h]), sparc_second=sparc(v[h:])))
    return pd.DataFrame(rows)


def two_group(c, p):
    if stats.shapiro(c).pvalue > 0.05 and stats.shapiro(p).pvalue > 0.05:
        r = stats.ttest_ind(p, c); return f"t({len(c) + len(p) - 2}) = {r.statistic:.2f}, p = {r.pvalue:.3g}"
    r = stats.mannwhitneyu(c, p); return f"U = {r.statistic:.0f}, p = {r.pvalue:.3g}"


def one_sample(d):
    d = pd.Series(d).dropna()
    if stats.shapiro(d).pvalue > 0.05:
        r = stats.ttest_1samp(d, 0); return f"{d.mean():+.3f}, t({len(d) - 1}) = {r.statistic:.2f}, p = {r.pvalue:.3g}"
    r = stats.wilcoxon(d); return f"{d.mean():+.3f}, W = {r.statistic:.0f}, p = {r.pvalue:.3g}"


if __name__ == "__main__":
    t = pd.concat([trials_for(CONTROL_DIR, DUR_CONTROL, "control"), trials_for(PATIENT_DIR, DUR_PATIENT, "patient")], ignore_index=True)
    t["log_mt"] = np.log(t.duration); t["vis"] = (t.Visibility == "Visible Hand").astype(int)
    t.to_csv("sparc_trials.csv", index=False)
    print(f"trials: {len(t)}; SPARC is negative, closer to zero = smoother\n")
    cols = ["sparc", "sparc_first", "sparc_second"]
    S = t.groupby(["Group", "Subject"])[cols + ["duration", "end_speed"]].mean().reset_index(); S["log_mt"] = np.log(S.duration)

    print("Tied to movement time? participant-level Spearman")
    for c in cols:
        r = stats.spearmanr(S[c], S.duration); print(f"  {c}: rho = {r[0]:.2f}, p = {r[1]:.3g}")
    w = t.groupby("Subject").apply(lambda d: stats.spearmanr(d.sparc, d.duration)[0], include_groups=False)
    print(f"  within participants, trial by trial (whole reach): median rho = {w.median():.2f}")
    print(f"  end speed, fraction of peak: controls {S[S.Group=='control'].end_speed.mean():.2f}, patients {S[S.Group=='patient'].end_speed.mean():.2f}; {two_group(S[S.Group=='control'].end_speed, S[S.Group=='patient'].end_speed)}")

    print("\nGroup (patients vs controls)")
    for c in cols:
        a, b = S[S.Group == "control"][c], S[S.Group == "patient"][c]
        f = smf.ols(f"{c} ~ Group + log_mt", S).fit(); g = smf.ols(f"{c} ~ Group + log_mt + end_speed", S).fit()
        print(f"  {c}: controls {a.mean():.3f}, patients {b.mean():.3f}; {two_group(a, b)}; "
              f"with movement time p = {f.pvalues['Group[T.patient]']:.3g}; also with end speed p = {g.pvalues['Group[T.patient]']:.3g}")

    print("\nSeeing the hand (visible minus hidden), raw and with movement time held constant (per-person regression)")
    rows = []
    for (g, s), d in t.groupby(["Group", "Subject"]):
        row = dict(Group=g, Subject=s)
        for c in cols:
            row[c] = d[d.vis == 1][c].mean() - d[d.vis == 0][c].mean()
            row[c + "_adj"] = smf.ols(f"{c} ~ vis + log_mt", d).fit().params["vis"]
            row[c + "_adj2"] = smf.ols(f"{c} ~ vis + log_mt + end_speed", d).fit().params["vis"]
        rows.append(row)
    V = pd.DataFrame(rows)
    for c in cols:
        print(f"  {c}: raw {one_sample(V[c])}; same movement time {one_sample(V[c + '_adj'])}; also same end speed {one_sample(V[c + '_adj2'])}")

    print("\nGroup x visibility (visibility effect, patients minus controls)")
    for c in cols:
        for k, lab in ((c, "raw"), (c + "_adj", "same movement time")):
            a, b = V[V.Group == "control"][k], V[V.Group == "patient"][k]
            print(f"  {c}, {lab}: controls {a.mean():+.3f}, patients {b.mean():+.3f}; {two_group(a, b)}")

    # agreement with path smoothness (cubic R²) trial by trial, movement time held constant
    sm = pd.read_csv("reach_definitions_trials.csv"); sm = sm[sm.kept].dropna(subset=["smooth_duration"]).assign(Trial=lambda d: d.Trial.astype(int))
    u = t.merge(sm[["Subject", "Trial", "smooth_duration"]], on=["Subject", "Trial"])

    def held(d):
        rk = d[["sparc", "smooth_duration", "log_mt"]].rank()
        res = lambda y: y - np.polyval(np.polyfit(rk.log_mt, y, 1), rk.log_mt)
        return stats.pearsonr(res(rk.sparc), res(rk.smooth_duration))[0]
    w = u.groupby("Subject").apply(held, include_groups=False)
    print(f"\nSPARC vs path smoothness within participants, movement time held constant: median rho = {w.median():.2f}, "
          f"{int((w > 0).sum())}/{len(w)} positive, Wilcoxon p = {stats.wilcoxon(w).pvalue:.3g}")
