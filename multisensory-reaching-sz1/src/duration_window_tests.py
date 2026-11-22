import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import pingouin as pg

df = pd.read_csv("reach_definitions_trials.csv")
df = df[df.kept].dropna(subset=["dir_duration", "smooth_duration"])
print("trials", len(df))
subj = df.groupby(["Group", "Subject"])[["dir_duration", "smooth_duration"]].median().reset_index()
vis = df.groupby(["Group", "Subject", "Visibility"])[["dir_duration", "smooth_duration"]].median().reset_index()
rows = []
for m in ["dir_duration", "smooth_duration"]:
    for cond, d in [("all", subj), ("Visible Hand", vis[vis.Visibility == "Visible Hand"]), ("Invisible Hand", vis[vis.Visibility == "Invisible Hand"])]:
        c, p = d.loc[d.Group == "control", m], d.loc[d.Group == "patient", m]
        t, pt = stats.ttest_ind(c, p)
        sp = np.sqrt(((len(c)-1)*c.var() + (len(p)-1)*p.var())/(len(c)+len(p)-2))
        rows.append(dict(measure=m, trials=cond, ctrl_median=c.median(), pt_median=p.median(),
                         p_mw=stats.mannwhitneyu(c, p).pvalue, p_t=pt, d=(p.mean()-c.mean())/sp,
                         BF01=1/float(pg.bayesfactor_ttest(t, len(c), len(p)))))
    w = vis.pivot_table(index=["Group", "Subject"], columns="Visibility", values=m).dropna().reset_index()
    for g in ["all", "control", "patient"]:
        s = w if g == "all" else w[w.Group == g]
        rows.append(dict(measure=m, trials=f"visible vs invisible ({g})",
                         visible_median=s["Visible Hand"].median(), hidden_median=s["Invisible Hand"].median(),
                         p_t=stats.ttest_rel(s["Visible Hand"], s["Invisible Hand"]).pvalue,
                         p_mw=stats.wilcoxon(s["Visible Hand"], s["Invisible Hand"]).pvalue))
    mm = smf.mixedlm(f"{m} ~ Group * Visibility", df, groups=df.Subject).fit()
    print(m, "mixed: group p=%.3f vis p=%.2g inter p=%.3f" % (mm.pvalues["Group[T.patient]"], mm.pvalues["Visibility[T.Visible Hand]"], mm.pvalues["Group[T.patient]:Visibility[T.Visible Hand]"]))
out = pd.DataFrame(rows).round(4)
out.to_csv("duration_window_tests.csv", index=False)
print(out.to_string(index=False))
print("dir-smooth r (subject):", round(stats.pearsonr(subj.dir_duration, subj.smooth_duration)[0], 3))
subj.to_csv("subject_directness_smoothness_duration.csv", index=False)
