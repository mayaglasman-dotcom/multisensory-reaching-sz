import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
import statsmodels.formula.api as smf

from config import CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT
EXCLUDE = set()

SR = 72
DT = 1 / SR
LOWPASS_HZ = 10        # positions are low-pass filtered at this cutoff before differentiating
XYZ = ["Hand Pos X", "Hand Pos Y", "Hand Pos Z"]
LP = ["LP X", "LP Y", "LP Z"]


def fname(s):
    s = str(s)
    return "SZR-" + s if len(s) <= 3 else s


def onset_index(x, y, z, frac=0.1):
    if len(x) < 3:
        return 0
    v = np.sqrt(np.gradient(x, DT)**2 + np.gradient(y, DT)**2 + np.gradient(z, DT)**2)
    # Onset is where speed last rises through `frac` of its peak before the peak. Taking the first crossing instead
    # would pick up a small early drift that slows down again before the real reach begins.
    pk = int(np.argmax(v))
    below = np.flatnonzero(v[:pk + 1] <= frac * v[pk])
    return int(below[-1] + 1) if below.size else 0   # 0: already above threshold from the start of the window


def onset_trim(t):
    i = onset_index(*(t[c].to_numpy() for c in ("Hand Pos X", "Hand Pos Y", "Hand Pos Z")))
    return t.iloc[i:]


def add_lowpass(tdf, cutoff=LOWPASS_HZ, edge="odd"):
    """Add low-pass filtered positions (2nd-order zero-lag Butterworth) for one trial's whole recording.
    Filtering the whole recording, before the reach window is cut out, leaves the window start away from the filter's edge."""
    P = tdf[XYZ].to_numpy()
    b, a = butter(2, cutoff / (SR / 2))
    if edge == "gust":
        F = filtfilt(b, a, P, axis=0, method="gust")
    else:
        F = filtfilt(b, a, P, axis=0, padlen=min(3 * max(len(a), len(b)), len(P) - 1))
    out = tdf.assign(**{c: F[:, k] for k, c in enumerate(LP)})
    out["LP jerk"] = jerk(*F.T)   # differentiated over the whole recording, so the reach window has no edge at onset
    return out


def jerk(x, y, z):
    d = lambda a: np.gradient(np.gradient(np.gradient(a, DT), DT), DT)
    return np.sqrt(d(x)**2 + d(y)**2 + d(z)**2)


def jerk_power_per_sample(sig):
    """Mean squared jerk: the integrated squared jerk divided by the number of samples,
    so a segment does not score higher just because it lasts longer."""
    return float(np.mean(np.asarray(sig) ** 2))


COUNTS = {}   # trial bookkeeping per group, filled by process()


def process(folder, dpath, group, cutoff=LOWPASS_HZ, edge="odd"):
    dur = pd.read_csv(dpath)
    rows = []
    cnt = COUNTS.setdefault(group, dict(logged=0, duration_outlier=0, no_recording=0, short_recording=0, under_16_samples=0, kept=0, moving_at_window_start=0))
    for sub in dur["Subject"].astype(str).unique():
        if sub in EXCLUDE:
            continue
        path = os.path.join(folder, f"raw_{fname(sub)}_Trajectories.csv")
        if not os.path.exists(path):
            continue
        traj = pd.read_csv(path)
        sd = dur[dur["Subject"].astype(str) == sub]
        m, s = sd.trialDuration.mean(), sd.trialDuration.std(ddof=1)
        inside = sd[sd.trialDuration.between(m - 2.5 * s, m + 2.5 * s)]
        cnt["logged"] += len(sd); cnt["duration_outlier"] += len(sd) - len(inside)
        for _, r in inside.iterrows():
            tdf = traj[traj.Trial == r.Trial]
            if len(tdf) < 2:
                cnt["no_recording"] += 1
                continue
            tdf = add_lowpass(tdf, cutoff, edge)
            n = int(np.floor(r.trialDuration * SR))
            if n >= len(tdf):
                cnt["short_recording"] += 1      # recording shorter than the logged trial: the whole recording is used
            t = tdf.iloc[-(n + 1):] if n < len(tdf) else tdf
            n_window = len(t)
            t = onset_trim(t)                      # onset from the unfiltered positions, as for every other measure
            if len(t) < 16:
                cnt["under_16_samples"] += 1
                continue
            cnt["kept"] += 1
            cnt["moving_at_window_start"] += int(len(t) == n_window)   # first sample already above 10% of peak speed
            j = t["LP jerk"].to_numpy()
            q = len(j) // 4
            row = {"Group": group, "Subject": sub, "Trial": r.Trial, "Visibility": t["Visibility"].iloc[0],
                   "moving_at_start": len(t) == n_window,
                   "duration": len(t) / SR, "trial_duration": r.trialDuration}
            for i, seg in enumerate([j[:q], j[q:2 * q], j[2 * q:3 * q], j[3 * q:]], start=1):
                row[f"Q{i}"] = np.log(jerk_power_per_sample(seg))
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    trials = pd.concat([process(CONTROL_DIR, DUR_CONTROL, "control"),
                        process(PATIENT_DIR, DUR_PATIENT, "patient")], ignore_index=True)
    trials = trials.replace([np.inf, -np.inf], np.nan).dropna()
    trials.to_csv("q1_final_trials.csv", index=False)
    counts = pd.DataFrame(COUNTS).T
    counts.to_csv("q1_final_trial_counts.csv")
    print(counts.to_string()); print()

    subj = trials.groupby(["Group", "Subject"])[["Q1", "Q2", "Q3", "Q4", "duration"]].mean().reset_index()
    subj.to_csv("q1_final_subject.csv", index=False)

    group_rows = []
    for col in ["Q1", "Q2", "Q3", "Q4", "duration"]:
        c, p = subj.loc[subj.Group == "control", col], subj.loc[subj.Group == "patient", col]
        t, p_t = stats.ttest_ind(p, c)   # patient minus control, same sign as d
        _, p_mw = stats.mannwhitneyu(c, p)
        d = (p.mean() - c.mean()) / np.sqrt(((len(c) - 1) * c.var() + (len(p) - 1) * p.var()) / (len(c) + len(p) - 2))
        group_rows.append({"measure": col, "control_mean": c.mean(), "patient_mean": p.mean(),
                           "t": t, "p_ttest": p_t, "p_mannwhitney": p_mw, "cohens_d": d})
    group_tbl = pd.DataFrame(group_rows).round(3)
    group_tbl.to_csv("q1_final_group_tests.csv", index=False)
    print(group_tbl.to_string(index=False))

    rel_rows = []
    for g in ["control", "patient"]:
        s = subj[subj.Group == g]
        lr = stats.linregress(s["Q1"], s["duration"])
        rel_rows.append({"group": g, "n": len(s), "slope": lr.slope, "r": lr.rvalue, "p": lr.pvalue})
    rel = pd.DataFrame(rel_rows).round(3)
    inter = smf.ols("duration ~ Q1 * Group", subj).fit()
    rel.to_csv("q1_final_jerk_vs_duration.csv", index=False)
    print()
    print(rel.to_string(index=False))
    print("slope difference between groups, p =", round(inter.pvalues["Q1:Group[T.patient]"], 3))

    colors = {"control": "#4C72B0", "patient": "#C44E52"}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(0)
    q1p = group_tbl.set_index("measure").loc["Q1", "p_mannwhitney"]
    for i, g in enumerate(["control", "patient"]):
        v = subj.loc[subj.Group == g, "Q1"]
        ax[0].scatter(i + rng.normal(0, 0.06, len(v)), v, color=colors[g], s=40, edgecolor="white")
        ax[0].errorbar(i + 0.3, v.mean(), yerr=v.sem(), fmt="o", color="black", capsize=5)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["control", "patient"]); ax[0].set_xlim(-0.5, 1.6)
    ax[0].set_ylabel("Quarter-1 jerk (log mean squared jerk)")
    ax[0].set_title(f"Quarter-1 jerk by group\nMann-Whitney p = {q1p:.3f} (black: mean ± SEM)")
    txt = []
    for g in ["control", "patient"]:
        s = subj[subj.Group == g]
        lr = stats.linregress(s["Q1"], s["duration"])
        xx = np.linspace(s["Q1"].min(), s["Q1"].max(), 50)
        ax[1].scatter(s["Q1"], s["duration"], color=colors[g], s=40, edgecolor="white", label=g)
        ax[1].plot(xx, lr.intercept + lr.slope * xx, color=colors[g], lw=2)
        txt.append(f"{g}: r = {lr.rvalue:.2f}, p = {lr.pvalue:.2f}")
    ax[1].set_xlabel("Quarter-1 jerk (log mean squared jerk)")
    ax[1].set_ylabel("Mean reach duration (s)")
    ax[1].set_title("Quarter-1 jerk vs. reach duration\n" + " | ".join(txt), fontsize=10.5)
    ax[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig("q1_final.png", dpi=150)
