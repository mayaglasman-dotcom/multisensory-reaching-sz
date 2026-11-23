"""Figure 2: the effect of seeing the hand. Panel A shows jerk for reaches of the same movement time; panels B and C show
path smoothness and directness as measured (per-person means), because a path's shape partly sets how long a reach takes.

For each participant, each trial's measure is modelled from visibility and log movement time:
    measure ~ visible + log(movement time)
(least squares, since all three use per-person means).
Each participant's visible and hidden values are then read off the fit at that participant's own average log movement time,
so the gap between them is the effect of seeing the hand at equal speed, and each person keeps their usual speed level.
This is the same per-participant regression as the speed-adjusted jerk results in stats_all.py.
The same-movement-time statistics are printed for all three measures.
Writes figures/Figure2_visibility.png."""
import warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from config import RESULTS_DIR, FIG_DIR
warnings.filterwarnings("ignore")

R = str(RESULTS_DIR) + "/"; FIG_DIR.mkdir(parents=True, exist_ok=True)
C = {"control": "#2a78d6", "patient": "#eb6834"}; LAB = {"control": "Controls", "patient": "Patients"}; G = ["control", "patient"]
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def fmtp(p):
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def one_sample(d):   # the paper's rule: t-test if the differences look normal, otherwise Wilcoxon
    d = pd.Series(d).dropna()
    if stats.shapiro(d).pvalue >= 0.05:
        r = stats.ttest_1samp(d, 0); return f"t({len(d) - 1}) = {r.statistic:.2f}", r.pvalue
    r = stats.wilcoxon(d); return f"W = {r.statistic:.0f}", r.pvalue


def two_group(c, p):
    if min(stats.shapiro(c).pvalue, stats.shapiro(p).pvalue) >= 0.05:
        r = stats.ttest_ind(p, c); return f"t({len(c) + len(p) - 2}) = {r.statistic:.2f}", r.pvalue
    r = stats.mannwhitneyu(c, p); return f"U = {min(r.statistic, len(c) * len(p) - r.statistic):.0f}", r.pvalue


def adjusted(df, col, median=False):
    """Per participant: visible and hidden values at that participant's mean log movement time."""
    rows = []
    for (g, s), d in df.groupby(["Group", "Subject"]):
        d = d.dropna(subset=[col])
        if d.vis.nunique() < 2:
            continue
        f = (smf.quantreg(f"{col} ~ vis + lmt", d).fit(q=0.5) if median else smf.ols(f"{col} ~ vis + lmt", d).fit()).params
        base = f["Intercept"] + f["lmt"] * d.lmt.mean()
        rows.append(dict(Group=g, Subject=s, visible=base + f["vis"], hidden=base, effect=f["vis"]))
    return pd.DataFrame(rows)


q = pd.read_csv(R + "q1_final_trials.csv")
q["lmt"] = np.log(q.duration); q["vis"] = (q.Visibility == "Visible Hand").astype(float)
v = pd.read_csv(R + "reach_definitions_trials.csv")
v = v[v.kept].dropna(subset=["dir_duration", "smooth_duration"]).assign(Trial=lambda d: d.Trial.astype(int))
pv = v.merge(q[["Subject", "Trial", "lmt", "vis"]], on=["Subject", "Trial"])   # path measures for trials with a movement time
print(f"path-measure trials with a movement time: {len(pv)} of {len(v)}\n")

Qs = ["Q1", "Q2", "Q3", "Q4"]
A = {k: adjusted(q, k) for k in Qs}
Badj = adjusted(pv, "smooth_duration")
Cadj = adjusted(pv, "dir_duration")

print("Seeing the hand at equal movement time (visible minus hidden), and patients vs controls in that effect")
res = {}
for name, d in [*[(f"jerk {k}", A[k]) for k in Qs], ("path smoothness (mean)", Badj), ("directness (mean)", Cadj)]:
    st, p = one_sample(d.effect)
    gst, gp = two_group(d[d.Group == "control"].effect, d[d.Group == "patient"].effect)
    res[name] = (p, gp)
    print(f"  {name}: {d.effect.mean():+.4f}, {st}, {fmtp(p)} | controls {d[d.Group=='control'].effect.mean():+.4f}, "
          f"patients {d[d.Group=='patient'].effect.mean():+.4f}, {gst}, {fmtp(gp)}")

# panels B and C: per-person means as measured (all kept trials)
vv = pd.read_csv(R + "reach_definitions_trials.csv"); vv = vv[vv.kept]
raw = {}
for col in ("smooth_duration", "dir_duration"):
    m = vv.dropna(subset=[col]).groupby(["Group", "Subject", "Visibility"])[col].mean().unstack().dropna()
    st, p = one_sample(m["Visible Hand"] - m["Invisible Hand"])
    raw[col] = (m, p)
    print(f"as measured, {col} (per-person means): visible {m['Visible Hand'].mean():.4f}, hidden {m['Invisible Hand'].mean():.4f}, {st}, {fmtp(p)}")

fig, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [1.6, 1, 1]})
ax = axs[0]
for g in G:
    for cond, ls, fc in (("visible", "-", C[g]), ("hidden", "--", "white")):
        m = np.array([A[k][A[k].Group == g][cond].mean() for k in Qs]); e = np.array([A[k][A[k].Group == g][cond].sem() for k in Qs])
        off = 0   # groups and conditions plotted at the same x positions
        ax.errorbar(np.arange(4) + off, m, yerr=e, color=C[g], ls=ls, marker="o", mfc=fc, mec=C[g], ms=7, mew=1.6, capsize=3,
                    label=f"{LAB[g]}, hand {'visible' if cond == 'visible' else 'invisible'}")
ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\n(start)", "Q2", "Q3", "Q4\n(end)"]); ax.set_ylabel("Jerk (log mean squared jerk)")
ax.set_title("Jerk across the reach, movement time controlled", loc="left", fontsize=11)
ax.legend(fontsize=8, loc="lower center", ncol=2)
pmax = max(res[f"jerk {k}"][0] for k in Qs)
ax.text(0.5, 0.97, f"visible > invisible in every quarter (all p ≤ {pmax:.3f})" if all(A[k].effect.mean() > 0 for k in Qs) and pmax < 0.05
        else "visible − invisible: " + ", ".join(f"{k} {fmtp(res[f'jerk {k}'][0])}" for k in Qs),
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED)
lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.35 * (hi - lo), hi + 0.12 * (hi - lo))
for ax, col, ttl, lab in ((axs[1], "smooth_duration", "Path smoothness", "Path smoothness (R², per-person mean)"),
                          (axs[2], "dir_duration", "Directness", "Directness (per-person mean)")):
    m, p = raw[col]
    for g in G:
        dd = m.loc[g]; off = 0
        ax.errorbar(np.array([0, 1]) + off, [dd["Visible Hand"].mean(), dd["Invisible Hand"].mean()],
                    yerr=[dd["Visible Hand"].sem(), dd["Invisible Hand"].sem()], color=C[g], marker="o", ms=7, capsize=3, label=LAB[g])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Hand\nvisible", "Hand\ninvisible"]); ax.set_xlim(-0.4, 1.4); ax.set_ylabel(lab)
    ax.set_title(f"{ttl}\nvisible vs. invisible: {fmtp(p)}", loc="left", fontsize=11)
axs[2].legend(fontsize=8)
for ax, l in zip(axs, "ABC"):
    ax.text(-0.14, 1.06, l, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
plt.tight_layout(); plt.savefig(str(FIG_DIR / "Figure2_visibility.png")); plt.close()
print("saved", FIG_DIR / "Figure2_visibility.png")
