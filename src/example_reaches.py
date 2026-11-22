"""Example reaches with similar path smoothness but different jerk.

Each row compares reaches by two different people, with the same hand visibility, distance, direction and path smoothness.
Row A: same movement time; one speeds up and slows down in one smooth burst, the other unevenly (more jerk).
Row B: both one smooth burst; one at least 1.6 times as fast as the other (the fast one has more jerk only because it is faster).
Reads results from earlier steps and the corrected trajectories; writes figures/example_reaches.png."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS_DIR, FIG_DIR, CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT
from q1_final import SR, DT, LP, fname, add_lowpass, onset_trim

R = str(RESULTS_DIR) + "/"; FIG_DIR.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e6e5e0"
COL = ("#2a78d6", "#eb6834")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

q = pd.read_csv(R + "q1_final_trials.csv")
j = pd.read_csv(R + "final_measures_trials.csv")[["Subject", "Trial", "whole"]]
s = pd.read_csv(R + "reach_definitions_trials.csv")
s = s[s.kept].dropna(subset=["smooth_duration"]).assign(Trial=lambda d: d.Trial.astype(int))[["Subject", "Trial", "smooth_duration"]]
t = q[~q.moving_at_start].merge(j, on=["Subject", "Trial"]).merge(s, on=["Subject", "Trial"])
t = t[t.smooth_duration >= 0.98].reset_index(drop=True)   # typical, smooth-looking paths


def trace(row):
    folder, dpath = (CONTROL_DIR, DUR_CONTROL) if row.Group == "control" else (PATIENT_DIR, DUR_PATIENT)
    key = (folder, str(row.Subject))
    if key not in _cache:
        dur = pd.read_csv(dpath)
        _cache[key] = (pd.read_csv(os.path.join(folder, f"raw_{fname(row.Subject)}_Trajectories.csv")),
                       dur[dur.Subject.astype(str) == str(row.Subject)].set_index("Trial").trialDuration)
    traj, dur = _cache[key]
    tdf = add_lowpass(traj[traj.Trial == row.Trial])
    n = int(np.floor(dur[row.Trial] * SR))
    w = onset_trim(tdf.iloc[-(n + 1):] if n < len(tdf) else tdf)
    P = w[LP].to_numpy()
    return P, np.linalg.norm(np.gradient(P, DT, axis=0), axis=1), w["LP jerk"].to_numpy()


_cache = {}


def n_peaks(v):
    """Speed peaks above 20% of the trial's peak speed: 1 for a single smooth burst, more for extra sub-movements."""
    i = np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > 0.2 * v.max())) + 1
    return len(i) + int(v[-1] > v[-2] and v[-1] > 0.2 * v.max() and (len(i) == 0 or v[-1] < v.max()))


feat = []
for _, row in t.iterrows():
    P, v, _ = trace(row)
    dxz = P[-1, [0, 2]] - P[0, [0, 2]]
    feat.append((np.linalg.norm(P[-1] - P[0]), np.degrees(np.arctan2(dxz[0], dxz[1])), n_peaks(v),
                 v[0] / v.max(), int(np.argmax(v)) < len(v) - 3))
t[["dist", "angle", "peaks", "start_frac", "peak_inside"]] = feat
lo_j, hi_j = t.whole.quantile([0.05, 0.95])
# clean, typical reaches only: starts slowly (no tracking jump at onset), speed peaks before the end, jerk in the middle 90%
t = t[(t.start_frac < 0.3) & t.peak_inside.astype(bool) & t.whole.between(lo_j, hi_j)].reset_index(drop=True)
print(f"candidate reaches: {len(t)}")


def best_pair(kind):
    """Search pairs of reaches by two different people with the same hand visibility."""
    best = None
    for vis, d in t.groupby("Visibility"):
        d = d.reset_index(drop=True)
        sub, sm, dist, ang = np.asarray(d.Subject.astype(str), dtype=object), d.smooth_duration.values, d.dist.values, d.angle.values
        dur, jk, pk = d.duration.values, d.whole.values, d.peaks.values
        for i0 in range(0, len(d), 400):
            a = np.arange(i0, min(i0 + 400, len(d)))[:, None]; b = np.arange(len(d))[None, :]
            ok = ((sub[a] != sub[b]) & (np.abs(sm[a] - sm[b]) < 0.002)
                  & (np.abs(dist[a] - dist[b]) < 0.1 * np.minimum(dist[a], dist[b])) & (np.abs(ang[a] - ang[b]) < 10))
            dj = jk[a] - jk[b]
            if kind == "same_speed":   # same movement time; a has uneven speed and more jerk, b one smooth burst
                ok &= (np.abs(dur[a] - dur[b]) <= 2 * DT + 1e-9) & (pk[a] >= 3) & (pk[b] == 1) & (dj > 0)
            else:                      # both one smooth burst; a at least 1.6 times as fast as b and with more jerk
                ok &= (pk[a] == 1) & (pk[b] == 1) & (dur[b] >= 1.6 * dur[a]) & (dj > 0)
            score = np.where(ok, dj, -np.inf)
            k = np.unravel_index(np.argmax(score), score.shape)
            if np.isfinite(score[k]) and (best is None or score[k] > best[0]):
                best = (score[k], d.iloc[int(a[k[0], 0])], d.iloc[int(b[0, k[1]])])
    return best[1], best[2]


pairs = [("A", "Same movement time and path smoothness, uneven vs. even speed", best_pair("same_speed")),
         ("B", "Same path smoothness, both even speed, one fast and one slow", best_pair("speed"))]
fig, axs = plt.subplots(2, 3, figsize=(13, 9))
fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.13, hspace=0.95, wspace=0.32)
for r, (lab, title, (hi, lo)) in enumerate(pairs):
    print(f"{lab}: {hi.Visibility}; {hi.Group} {hi.Subject} trial {hi.Trial} vs {lo.Group} {lo.Subject} trial {lo.Trial}")
    for k, row in enumerate((hi, lo)):
        P, v, jk = trace(row)
        tt = np.arange(len(v)) * DT * 1000
        name = f"{'Person 1' if k == 0 else 'Person 2'} ({'control' if row.Group == 'control' else 'patient'}), {'more' if k == 0 else 'less'} jerk: jerk {row.whole:.1f}, path R² {row.smooth_duration:.3f}, {row.duration:.2f} s, {row.dist*100:.0f} cm, {'one smooth speed burst' if row.peaks == 1 else 'uneven speed'}"
        print("   ", name)
        axs[r, 0].plot((P[:, 0] - P[0, 0]) * 100, (P[:, 2] - P[0, 2]) * 100, color=COL[k], marker="o", ms=3, label=name)
        axs[r, 1].plot(tt, v * 100, color=COL[k], marker="o", ms=3)
        axs[r, 2].plot(tt, jk, color=COL[k], marker="o", ms=3)
    axs[r, 0].set(xlabel="Left–right (cm)", ylabel="Forward (cm)"); axs[r, 0].set_aspect("equal", adjustable="datalim")
    axs[r, 0].legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.22), frameon=False)
    axs[r, 1].set(xlabel="Time from movement onset (ms)", ylabel="Hand speed (cm/s)")
    axs[r, 2].set(xlabel="Time from movement onset (ms)", ylabel="Jerk (m/s³)")
    axs[r, 0].set_title(f"{lab}  {title}\nPath seen from above", loc="left", fontsize=11)
    axs[r, 1].set_title("Speed", loc="left", fontsize=11); axs[r, 2].set_title("Jerk", loc="left", fontsize=11)
fig.savefig(str(FIG_DIR / "example_reaches.png"), dpi=300)
print("saved", FIG_DIR / "example_reaches.png")
