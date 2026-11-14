import os
import numpy as np
import pandas as pd
from directness_smoothness import cubic_r2
from q1_final import onset_index, fname, CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT, EXCLUDE, SR


def measures(x, y, z):
    if len(x) < 8:
        return np.nan, np.nan
    straight = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2 + (z[-1]-z[0])**2)
    if straight <= 0.03:
        return np.nan, np.nan
    path = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    t = np.linspace(0, 1, len(x))
    res = tot = 0.0
    for a in (x, y, z):
        r_, t_ = cubic_r2(t, a)
        res += r_; tot += t_
    return path / straight, 1 - res / tot


def process(folder, dpath, group):
    dur = pd.read_csv(dpath)
    rows = []
    for sub in dur["Subject"].astype(str).unique():
        if sub in EXCLUDE:
            continue
        p = os.path.join(folder, f"raw_{fname(sub)}_Trajectories.csv")
        if not os.path.exists(p):
            continue
        traj = pd.read_csv(p).sort_values(["Trial", "sample"])
        sd = dur[dur["Subject"].astype(str) == sub]
        m, s = sd.trialDuration.mean(), sd.trialDuration.std(ddof=1)
        keep = set(sd[sd.trialDuration.between(m - 2.5 * s, m + 2.5 * s)].Trial)
        durs = sd.set_index("Trial").trialDuration
        for trial, g in traj.groupby("Trial"):
            x, y, z = (g[c].to_numpy() for c in ("Hand Pos X", "Hand Pos Y", "Hand Pos Z"))
            r = {"Group": group, "Subject": sub, "Trial": trial, "Visibility": g.Visibility.iloc[0],
                 "kept": trial in keep}
            if trial in durs.index:
                k = min(len(x), int(np.floor(durs[trial] * SR)) + 1)
                xs, ys, zs = x[-k:], y[-k:], z[-k:]
                i0 = onset_index(xs, ys, zs)
                r["dir_duration"], r["smooth_duration"] = measures(xs[i0:], ys[i0:], zs[i0:])
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.concat([process(CONTROL_DIR, DUR_CONTROL, "control"),
                    process(PATIENT_DIR, DUR_PATIENT, "patient")], ignore_index=True)
    df.to_csv("reach_definitions_trials.csv", index=False)
    print(df.groupby("Group").size())
