# Builds consecutive-trial pairs for all four visibility transitions (visible/hidden -> visible/hidden).
import os
import numpy as np
import pandas as pd


def resolve_filename_subject(subject_id: str) -> str:
    subject_id = str(subject_id)
    if len(subject_id) <= 3:
        return "SZR-" + subject_id
    return subject_id


def get_all_transition_pairs(folder, durations_path, group_name):
    """Returns a DataFrame with one row per consecutive trial pair
    (any visibility combination): Group, Subject, trial_curr, trial_next,
    transition (e.g. 'visible_to_invisible'), error, prev_minus_curr_target."""
    durations_df = pd.read_csv(durations_path)
    records = []

    for subject_id in durations_df["Subject"].unique():
        subject_id = str(subject_id)
        fname_subject = resolve_filename_subject(subject_id)
        traj_path = os.path.join(folder, f"raw_{fname_subject}_Trajectories.csv")
        if not os.path.exists(traj_path):
            continue

        traj_df = pd.read_csv(traj_path)
        sub_durations = durations_df[durations_df["Subject"].astype(str) == subject_id].set_index("Trial")
        vis_per_trial = traj_df.groupby("Trial")["Visibility"].first()
        trials_sorted = sorted(vis_per_trial.index)

        for i in range(len(trials_sorted) - 1):
            t_curr, t_next = trials_sorted[i], trials_sorted[i + 1]
            if t_next != t_curr + 1:
                continue
            if t_curr not in sub_durations.index or t_next not in sub_durations.index:
                continue

            vis_curr = "visible" if vis_per_trial[t_curr] == "Visible Hand" else "invisible"
            vis_next = "visible" if vis_per_trial[t_next] == "Visible Hand" else "invisible"
            transition = f"{vis_curr}_to_{vis_next}"

            next_traj = traj_df[traj_df["Trial"] == t_next]
            if len(next_traj) == 0:
                continue

            endpoint_x = next_traj["Hand Pos X"].iloc[-1]
            target_x = sub_durations.loc[t_next, "Ball Pos X"]
            prev_target_x = sub_durations.loc[t_curr, "Ball Pos X"]

            records.append({
                "Group": group_name, "Subject": subject_id,
                "trial_curr": t_curr, "trial_next": t_next, "transition": transition,
                "error": endpoint_x - target_x,
                "prev_minus_curr_target": prev_target_x - target_x,
                "curr_target": target_x,
            })

    return pd.DataFrame(records)


def carryover_slope(d):
    """Slope of the reach's left-right error on (previous - current target), holding the current target fixed.
    Both the error and (previous - current) contain the current target, so a simple tendency to land nearer the
    middle of the arc would give a slope on its own; including the current target as a second predictor removes it."""
    X = np.column_stack([np.ones(len(d)), d["prev_minus_curr_target"], d["curr_target"]])
    coef, *_ = np.linalg.lstsq(X, d["error"].to_numpy(), rcond=None)
    return float(coef[1])
