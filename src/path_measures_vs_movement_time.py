"""How path smoothness and directness relate to movement time, across participants and trial by trial within each.
Reads results from earlier steps; writes figures/path_measures_vs_movement_time.png and prints the statistics."""
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


def fmt(r, p):
    return f"ρ = {r:.2f}, p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}"


q = pd.read_csv(R + "q1_final_trials.csv")[["Group", "Subject", "Trial", "duration"]]
v = pd.read_csv(R + "reach_definitions_trials.csv")
v = v[v.kept].dropna(subset=["dir_duration", "smooth_duration"]).assign(Trial=lambda d: d.Trial.astype(int))

# participant level, as in the paper: movement time = mean over jerk trials; smoothness and directness = means (directness median also shown)
S = (q.groupby(["Group", "Subject"]).duration.mean().to_frame()
     .join(v.groupby(["Group", "Subject"]).agg(smooth=("smooth_duration", "mean"), dir_mean=("dir_duration", "mean"),
                                                dir_median=("dir_duration", "median")))
     .reset_index())
MEAS = [("smooth", "Path smoothness (mean R²)"), ("dir_mean", "Directness (mean)"), ("dir_median", "Directness (median)")]
print("Across participants (Spearman with mean movement time)")
for c, lab in MEAS:
    r = stats.spearmanr(S[c], S.duration)
    line = f"  {lab}: all {fmt(*r)}"
    for g in C:
        d = S[S.Group == g]; line += f"; {g} {fmt(*stats.spearmanr(d[c], d.duration))}"
    print(line)

# within participants: trial-by-trial Spearman with movement time
t = v.merge(q[["Subject", "Trial", "duration"]], on=["Subject", "Trial"])
W = t.groupby(["Group", "Subject"]).apply(lambda d: pd.Series({
    "smooth": stats.spearmanr(d.smooth_duration, d.duration)[0],
    "dir": stats.spearmanr(d.dir_duration, d.duration)[0]}), include_groups=False).reset_index()
print("\nWithin participants (each person's trial-by-trial Spearman with movement time)")
for c, lab in (("smooth", "Path smoothness"), ("dir", "Directness")):
    w = stats.wilcoxon(W[c]); line = f"  {lab}: median rho {W[c].median():.2f}, {int((W[c] < 0).sum())}/{len(W)} negative, Wilcoxon p = {w.pvalue:.3g}"
    a, b = W[W.Group == "control"][c], W[W.Group == "patient"][c]
    line += f"; controls {a.median():.2f}, patients {b.median():.2f}, group difference U = {stats.mannwhitneyu(a, b).statistic:.0f}, p = {stats.mannwhitneyu(a, b).pvalue:.3f}"
    print(line)
W.to_csv(R + "path_measures_vs_mt_within.csv", index=False)

fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
for k, (c, lab) in enumerate(MEAS[:2]):
    ax = axs[0, k]
    for g in C:
        d = S[S.Group == g]
        ax.scatter(d.duration, d[c], s=30, color=C[g], edgecolor="white", linewidth=0.6, label=LAB[g], zorder=3)
        lr = stats.linregress(d.duration, d[c]); xs = np.linspace(d.duration.min(), d.duration.max(), 20)
        ax.plot(xs, lr.intercept + lr.slope * xs, color=C[g], lw=1.5)
    ax.set(xlabel="Mean movement time (s)", ylabel=lab)
    ax.set_title(f"{'AB'[k]}  Participants\n{fmt(*stats.spearmanr(S[c], S.duration))}", loc="left", fontsize=11)
    if k == 0:
        ax.legend(frameon=False)
rng = np.random.default_rng(0)
for k, (c, lab) in enumerate((("smooth", "path smoothness"), ("dir", "directness"))):
    ax = axs[1, k]
    for i, g in enumerate(C):
        d = W[W.Group == g][c]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=22, color=C[g], edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [d.median()] * 2, color=INK, lw=2.5, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks([0, 1], [LAB[g] for g in C]); ax.set_xlim(-0.6, 1.6); ax.grid(axis="x", visible=False)
    ax.set_ylabel(f"Trial-by-trial ρ, {lab} vs. movement time")
    ax.set_title(f"{'CD'[k]}  Within each participant\nmedian ρ = {W[c].median():.2f}", loc="left", fontsize=11)
fig.tight_layout(h_pad=2.5)
fig.savefig(str(FIG_DIR / "path_measures_vs_movement_time.png"), dpi=300)
print("saved", FIG_DIR / "path_measures_vs_movement_time.png")
