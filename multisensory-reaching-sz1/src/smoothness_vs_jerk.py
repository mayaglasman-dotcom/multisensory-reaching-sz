"""How path smoothness (cubic R²) relates to jerk, across participants and across trials within a participant.
Reads results written by earlier steps; writes figures/smoothness_vs_jerk.png and prints the statistics."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from config import RESULTS_DIR, FIG_DIR

R = str(RESULTS_DIR) + "/"; FIG_DIR.mkdir(parents=True, exist_ok=True)
C = {"control": "#2a78d6", "patient": "#eb6834"}; LAB = {"control": "Controls", "patient": "Patients"}
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

# trial-level jerk (whole reach), movement time, and smoothness, matched by participant and trial
jerk = pd.read_csv(R + "final_measures_trials.csv")[["Group", "Subject", "Trial", "Visibility", "whole"]]
mt = pd.read_csv(R + "q1_final_trials.csv")[["Subject", "Trial", "duration"]]
sm = pd.read_csv(R + "reach_definitions_trials.csv")
sm = sm[sm.kept].dropna(subset=["smooth_duration"]).assign(Trial=lambda d: d.Trial.astype(int))[["Subject", "Trial", "smooth_duration"]]
t = jerk.merge(mt, on=["Subject", "Trial"]).merge(sm, on=["Subject", "Trial"])
t["log_mt"] = np.log(t.duration)

# participant level: same summaries as the paper (mean jerk, mean movement time, mean smoothness)
S = (jerk.groupby(["Group", "Subject"]).whole.mean().to_frame()
     .join(mt.merge(jerk[["Subject", "Trial", "Group"]]).groupby(["Group", "Subject"]).duration.mean())
     .join(sm.merge(jerk[["Subject", "Trial", "Group"]]).groupby(["Group", "Subject"]).smooth_duration.mean())
     .reset_index())
S["log_mt"] = np.log(S.duration)


def resid(y, x):
    """What is left of y after a straight-line fit on x."""
    X = np.column_stack([np.ones(len(x)), x])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def partial_spearman(a, b, z):
    """Spearman correlation of a and b with z held constant (correlation of rank residuals)."""
    ra, rb, rz = (pd.Series(v).rank().to_numpy() for v in (a, b, z))
    return stats.pearsonr(resid(ra, rz), resid(rb, rz))


def fmt(r, p):
    return f"ρ = {r:.2f}, p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}"


out = []
r, p = stats.spearmanr(S.whole, S.smooth_duration); out.append(("participants: jerk vs smoothness", r, p))
for g, d in S.groupby("Group"):
    out.append((f"  {g}", *stats.spearmanr(d.whole, d.smooth_duration)))
out.append(("participants: jerk vs movement time", *stats.spearmanr(S.whole, S.duration)))
out.append(("participants: smoothness vs movement time", *stats.spearmanr(S.smooth_duration, S.duration)))
pr, pp = partial_spearman(S.whole, S.smooth_duration, S.log_mt); out.append(("participants: jerk vs smoothness, movement time held constant", pr, pp))
for g, d in S.groupby("Group"):
    out.append((f"  {g}", *partial_spearman(d.whole, d.smooth_duration, d.log_mt)))

# within each participant, trial by trial
W = []
for (g, s), d in t.groupby(["Group", "Subject"]):
    W.append(dict(Group=g, Subject=s, raw=stats.spearmanr(d.whole, d.smooth_duration)[0],
                  held=partial_spearman(d.whole, d.smooth_duration, d.log_mt)[0]))
W = pd.DataFrame(W)
for c, lab in (("raw", "within participants: jerk vs smoothness"), ("held", "within participants, movement time held constant")):
    w = stats.wilcoxon(W[c])
    out.append((f"{lab} (median of participants; Wilcoxon vs 0, W = {w.statistic:.0f}; {int((W[c] < 0).sum())}/{len(W)} negative)", W[c].median(), w.pvalue))

for lab, r, p in out:
    print(f"{lab}: rho = {r:.3f}, p = {p:.3g}")
W.to_csv(R + "smoothness_vs_jerk_within.csv", index=False)

# figure
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
ax = axs[0]
for g in C:
    d = S[S.Group == g]; ax.scatter(d.whole, d.smooth_duration, s=30, color=C[g], edgecolor="white", linewidth=0.6, label=LAB[g], zorder=3)
ax.set(xlabel="Whole-reach jerk (log)", ylabel="Path smoothness (mean R²)")
ax.set_title(f"A  Participants\n{fmt(*stats.spearmanr(S.whole, S.smooth_duration))}", loc="left", fontsize=11)
ax.legend(frameon=False, loc="lower right")

ax = axs[1]
S["jerk_adj"] = resid(S.whole.to_numpy(), S.log_mt.to_numpy()); S["smooth_adj"] = resid(S.smooth_duration.to_numpy(), S.log_mt.to_numpy())
for g in C:
    d = S[S.Group == g]; ax.scatter(d.jerk_adj, d.smooth_adj, s=30, color=C[g], edgecolor="white", linewidth=0.6, zorder=3)
lr = stats.linregress(S.jerk_adj, S.smooth_adj); xs = np.linspace(S.jerk_adj.min(), S.jerk_adj.max(), 20)
ax.plot(xs, lr.intercept + lr.slope * xs, color=INK, lw=1.2)
ax.set(xlabel="Jerk, movement time held constant", ylabel="Smoothness, movement time held constant")
ax.set_title(f"B  Participants, same movement time\n{fmt(pr, pp)}", loc="left", fontsize=11)

ax = axs[2]
rng = np.random.default_rng(0)
for _, w in W.iterrows():
    ax.plot([0, 1], [w.raw, w.held], color=C[w.Group], alpha=0.25, lw=1)
for i, c in enumerate(("raw", "held")):
    for g in C:
        v = W[W.Group == g][c]; ax.scatter(i + rng.uniform(-0.06, 0.06, len(v)), v, s=18, color=C[g], edgecolor="white", linewidth=0.5, zorder=3)
    ax.plot([i - 0.2, i + 0.2], [W[c].median()] * 2, color=INK, lw=2.5, zorder=4)
ax.axhline(0, color=MUTED, lw=0.8, ls=":")
ax.set_xticks([0, 1], ["All trials", "Same movement\ntime"]); ax.set_xlim(-0.5, 1.5); ax.grid(axis="x", visible=False)
ax.set_ylabel("Trial-by-trial correlation, jerk vs. smoothness (ρ)")
ax.set_title("C  Within each participant", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "smoothness_vs_jerk.png"), dpi=300)
print("saved", FIG_DIR / "smoothness_vs_jerk.png")
