# Is the link between directness and movement time just the extra distance of a curved path?
# Per trial: path length, straight-line distance, movement time, and speed along the path (path / time).
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from q1_final import onset_index, fname, CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT, EXCLUDE, SR


def process(folder, dpath, group):
    dur = pd.read_csv(dpath); rows = []
    for sub in dur["Subject"].astype(str).unique():
        if sub in EXCLUDE:
            continue
        p = os.path.join(folder, f"raw_{fname(sub)}_Trajectories.csv")
        if not os.path.exists(p):
            continue
        traj = pd.read_csv(p).sort_values(["Trial", "sample"])
        sd = dur[dur["Subject"].astype(str) == sub]
        m, s = sd.trialDuration.mean(), sd.trialDuration.std(ddof=1)
        for _, r in sd[sd.trialDuration.between(m - 2.5 * s, m + 2.5 * s)].iterrows():
            g = traj[traj.Trial == r.Trial]
            x, y, z = (g[c].to_numpy() for c in ("Hand Pos X", "Hand Pos Y", "Hand Pos Z"))
            k = min(len(x), int(np.floor(r.trialDuration * SR)) + 1)
            x, y, z = x[-k:], y[-k:], z[-k:]
            i0 = onset_index(x, y, z); x, y, z = x[i0:], y[i0:], z[i0:]
            if len(x) < 8:
                continue
            straight = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2 + (z[-1]-z[0])**2)
            if straight <= 0.03:
                continue
            path = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
            mt = len(x) / SR
            rows.append(dict(Group=group, Subject=sub, directness=path / straight, path=path, straight=straight,
                             mt=mt, path_speed=path / mt))
    return pd.DataFrame(rows)


T = pd.concat([process(CONTROL_DIR, DUR_CONTROL, "control"), process(PATIENT_DIR, DUR_PATIENT, "patient")], ignore_index=True)
S = T.groupby(["Group", "Subject"])[["directness", "path", "straight", "mt", "path_speed"]].median().reset_index()
r = stats.spearmanr(S.directness, S.mt); print(f"directness vs movement time: rho = {r.statistic:.2f}, p = {r.pvalue:.4f}")
r = stats.spearmanr(S.directness, S.path_speed); print(f"directness vs speed along the path: rho = {r.statistic:.2f}, p = {r.pvalue:.4f}")
# share of the movement-time difference explained by extra distance: log MT = log path - log speed
S["lmt"], S["lpath"], S["lspeed"] = np.log(S.mt), np.log(S.path), np.log(S.path_speed)
m = smf.ols("lmt ~ np.log(directness)", S).fit(); ms = smf.ols("lspeed ~ np.log(directness)", S).fit()
print(f"log MT on log directness: slope = {m.params.iloc[1]:.2f} (1.0 = time grows only by the extra distance), p = {m.pvalues.iloc[1]:.4f}")
print(f"log path speed on log directness: slope = {ms.params.iloc[1]:.2f}, p = {ms.pvalues.iloc[1]:.4f}")
print("median directness", S.directness.median().round(3), "IQR", S.directness.quantile([.25, .75]).round(3).tolist())
