"""Paths used by every script. Change DATA_DIR here, or set the REACH_DATA_DIR environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("REACH_DATA_DIR", ROOT / "data"))

RAW_CONTROL_DIR = DATA_DIR / "raw_trajectories" / "controls"   # raw_<subject>_Trajectories.csv files, as exported
RAW_PATIENT_DIR = DATA_DIR / "raw_trajectories" / "patients"
CONTROL_DIR = DATA_DIR / "corrected_trajectories" / "controls"  # written by prepare_data.py; every analysis reads these
PATIENT_DIR = DATA_DIR / "corrected_trajectories" / "patients"
DUR_CONTROL = DATA_DIR / "trial_durations_controls.csv"
DUR_PATIENT = DATA_DIR / "trial_durations_patients.csv"
CLINICAL_XLSX = DATA_DIR / "Interesting_Data - SZR.xlsx"     # demographics and clinical scores

RESULTS_DIR = ROOT / "results"   # intermediate tables and statistics
FIG_DIR = ROOT / "figures"
TASK_IMAGE = ROOT / "assets" / "task_schematic.png"
