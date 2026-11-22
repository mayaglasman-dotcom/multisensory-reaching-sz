# Builds trial pairs 1, 2 or 3 trials apart (carryover by lag).
import os
import numpy as np
import pandas as pd


def resolve_filename_subject(subject_id: str) -> str:
    subject_id = str(subject_id)
    if len(subject_id) <= 3:
        return "SZR-" + subject_id
    return subject_id


def get_endpoints(folder, durations_df):
    """Returns {(subject_id, trial): endpoint_x} for every trial."""
    endpoints = {}
    for subject_id in durations_df["Subject"].unique():
        subject_id = str(subject_id)
        fname_subject = resolve_filename_subject(subject_id)
        traj_path = os.path.join(folder, f"raw_{fname_subject}_Trajectories.csv")
        if not os.path.exists(traj_path):
            continue
        traj_df = pd.read_csv(traj_path)
        for trial, tdf in traj_df.groupby("Trial"):
            endpoints[(subject_id, trial)] = tdf["Hand Pos X"].iloc[-1]
    return endpoints


def build_lag_pairs(durations_df, group_name, lag, endpoints):
    """Returns a DataFrame with one row per (trial, trial-lag) pair:
    Group, Subject, lag, trial_curr, trial_prev, error,
    prev_minus_curr_target."""
    records = []
    for subject_id in durations_df["Subject"].unique():
        subject_id = str(subject_id)
        sub_durations = durations_df[durations_df["Subject"].astype(str) == subject_id].set_index("Trial")
        logged = set(sub_durations.index)

        for t_curr in sorted(logged):
            t_prev = t_curr - lag   # exactly `lag` trials earlier, by trial number
            if t_prev not in logged:
                continue

            endpoint_x = endpoints.get((subject_id, t_curr))
            if endpoint_x is None:
                continue

            curr_target = sub_durations.loc[t_curr, "Ball Pos X"]
            prev_target = sub_durations.loc[t_prev, "Ball Pos X"]

            records.append({
                "Group": group_name, "Subject": subject_id, "lag": lag,
                "trial_curr": t_curr, "trial_prev": t_prev,
                "error": endpoint_x - curr_target,
                "prev_minus_curr_target": prev_target - curr_target,
                "curr_target": curr_target,
            })

    return pd.DataFrame(records)
