import sys
import numpy as np
import pandas as pd
from scipy import stats
from carryover_by_visibility_transition import get_all_transition_pairs, carryover_slope
from carryover_lag_analysis import get_endpoints, build_lag_pairs

from config import CONTROL_DIR as CTRL, PATIENT_DIR as PT, DUR_CONTROL as DC, DUR_PATIENT as DP
drop = set(sys.argv[1:])  # optional subjects to drop


def slopes(df, key):
    rows = []
    for (g, s, k), d in df.groupby(["Group", "Subject", key]):
        if len(d) >= 10:   # enough pairs for a stable slope with two predictors
            rows.append({"Group": g, "Subject": s, key: k, "slope": carryover_slope(d), "n_pairs": len(d)})
    out = pd.DataFrame(rows)
    keep = []
    for (g, k), d in out.groupby(["Group", key]):
        m, sd = d.slope.mean(), d.slope.std(ddof=1)
        keep.append(d[(d.slope - m).abs() <= 2.5 * sd])
    return pd.concat(keep)


def test(sl, key):
    rows = []
    for k, d in sl.groupby(key):
        c, p = d.loc[d.Group == "control", "slope"], d.loc[d.Group == "patient", "slope"]
        rows.append({key: k, "n_c": len(c), "mean_c": c.mean(), "n_p": len(p), "mean_p": p.mean(),
                     "p": stats.ttest_ind(c, p).pvalue})
    return pd.DataFrame(rows)


tr = pd.concat([get_all_transition_pairs(CTRL, DC, "control"), get_all_transition_pairs(PT, DP, "patient")])
tr = tr[~tr.Subject.isin(drop)]
sl = slopes(tr, "transition")
sl.to_csv("carryover_slopes_by_transition_no_outliers.csv", index=False)
tt = test(sl, "transition")
tt.to_csv("transition_table.csv", index=False)
print(tt.round(4).to_string(index=False))

dc, dp = pd.read_csv(DC), pd.read_csv(DP)
ec, ep = get_endpoints(CTRL, dc), get_endpoints(PT, dp)
lag = pd.concat([pd.concat([build_lag_pairs(dc, "control", L, ec), build_lag_pairs(dp, "patient", L, ep)]) for L in (1, 2, 3)])
lag = lag[~lag.Subject.isin(drop)]
lt = test(slopes(lag, "lag"), "lag")
lt.to_csv("carryover_lag_subject_tests.csv", index=False)
print(lt.round(4).to_string(index=False))
