# Data (not included)

Expected layout:

```
data/
  trial_durations_controls.csv      one row per trial: Trial, trialDuration, Hand error, Subject, Ball Pos X/Y/Z
  trial_durations_patients.csv
  Interesting_Data - SZR.xlsx       demographics and clinical scores (sheet "Demographics"); optional
  raw_trajectories/
    controls/raw_<subject>_Trajectories.csv
    patients/raw_<subject>_Trajectories.csv
```

Each trajectory file has one row per sample: `Hand Pos X`, `Hand Pos Y`, `Hand Pos Z` (meters), `sample`, `End`, `Start`, `Visibility` ("Visible Hand" / "Invisible Hand"), `Trial`.

The participants analyzed are the ones listed in the trial-duration files; trajectory files for anyone not listed there are ignored. `prepare_data.py` writes corrected copies of the trajectory files to `corrected_trajectories/` (see Data corrections in the main README); the analyses read those.
