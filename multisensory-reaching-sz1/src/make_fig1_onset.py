# Figure 1: task schematic (A), an example reach seen from above (B), and its speed profile split into quarters (C).
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from config import CONTROL_DIR, DUR_CONTROL, TASK_IMAGE, FIG_DIR
panelA = Image.open(TASK_IMAGE).convert('RGB')

dur = pd.read_csv(DUR_CONTROL); sub = 'SZR-1002'   # example: this control's median-duration trial
t = pd.read_csv(CONTROL_DIR / f'raw_{sub}_Trajectories.csv')
dd = dur[dur.Subject == sub].set_index('Trial')
med = dd.trialDuration.median(); tr = (dd.trialDuration - med).abs().idxmin()
seg = t[t.Trial == tr]; n = int(np.floor(dd.loc[tr, 'trialDuration'] * 72)) + 1; seg = seg.tail(n)
x, y, z = (seg[c].values for c in ['Hand Pos X', 'Hand Pos Y', 'Hand Pos Z'])
dt = 1 / 72; tt = np.arange(len(seg)) * dt
vel = np.sqrt(np.gradient(x, dt)**2 + np.gradient(y, dt)**2 + np.gradient(z, dt)**2) * 100
pk = int(np.argmax(vel)); on = pk
while on > 0 and vel[on - 1] > 0.1 * vel[pk]: on -= 1   # last rise through 10% of peak before the peak
L = len(seg) - on; k = L // 4; ar = np.arange(L)
q = [on + ar[:k], on + ar[k:2*k], on + ar[2*k:3*k], on + ar[3*k:]]   # same split as the analysis (remainder in Q4)
print('trial', tr, 'onset', on, 'samples per quarter', [len(i) for i in q])

QC = ['#9cc6f5', '#5d9de9', '#2b74d6', '#173f80']; PRE = '#a8a7a1'
INK = "#2b2b2a"; MUTED = "#6b6a64"; BAND = 0.30
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 300, "savefig.bbox": "tight"})
fig = plt.figure(figsize=(11, 8.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.22, wspace=0.28)
a = fig.add_subplot(gs[0, :]); a.imshow(panelA); a.axis('off')
a.text(-0.02, 1.0, 'A', transform=a.transAxes, fontsize=14, fontweight='bold', va='top')
b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])

# C: bands span each quarter's samples (edges halfway between samples), dots = individual samples
edge = lambda i: tt[i] - dt / 2
c.axvspan(0, edge(on), color='#e4e3dd', lw=0)
c.text(edge(on) / 2, vel.max() * 0.55, 'before\nonset\n(excluded)', ha='center', fontsize=8, color=MUTED)
c.plot(tt, vel, color=INK, lw=1.4, zorder=3)
c.scatter(tt[:on], vel[:on], s=14, color=PRE, zorder=4, lw=0)
for i, idx in enumerate(q):
    lo = edge(idx[0]); hi = tt[idx[-1]] if i == 3 else edge(idx[-1] + 1)
    c.axvspan(lo, hi, color=QC[i], alpha=BAND, lw=0)
    if i > 0: c.axvline(lo, color='white', lw=1.5, zorder=2)
    c.scatter(tt[idx], vel[idx], s=18, color=QC[i], edgecolor=INK, lw=0.4, zorder=4)
    c.text((lo + hi) / 2, vel.max() * 1.06, f'Q{i+1}\nn = {len(idx)}', ha='center', va='bottom', color=INK, fontsize=8.5)
c.axhline(0.1 * vel.max(), color=MUTED, ls=':', lw=1.2)
c.text(0.005, 0.1 * vel.max() + 5, '10% of peak speed', color=MUTED, fontsize=8, va='bottom')
c.annotate('movement onset', xy=(tt[on], vel[on]), xytext=(tt[on] - 0.12, vel.max() * 0.30),
           fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='->', color=INK, lw=0.8))
c.set_xlim(0, tt[-1] + dt / 2); c.set_ylim(0, vel.max() * 1.22)
c.set_xlabel('Time since target appeared (s)'); c.set_ylabel('Hand speed (cm/s)')
c.set_title('Same reach: quarters have equal numbers of samples', loc='left', fontsize=11)
c.text(-0.18, 1.06, 'C', transform=c.transAxes, fontsize=14, fontweight='bold')

# B: path seen from above, dots = individual samples
b.plot(x[:on+1]*100, z[:on+1]*100, color=PRE, lw=3, label='Before onset (excluded)')
b.scatter(x[:on]*100, z[:on]*100, s=14, color=PRE, zorder=3, lw=0)
for i, idx in enumerate(q):
    j = np.r_[idx, idx[-1] + 1] if i < 3 else idx
    dist = np.sum(np.hypot(np.diff(x[j]), np.diff(z[j]))) * 100
    b.plot(x[j]*100, z[j]*100, color=QC[i], lw=3, solid_capstyle='round', label=f'Q{i+1}: {len(idx)} samples, {dist:.0f} cm')
    b.scatter(x[idx]*100, z[idx]*100, s=18, color=QC[i], edgecolor=INK, lw=0.4, zorder=4)
b.plot(x[0]*100, z[0]*100, 'o', color=INK, ms=6, zorder=5); b.plot(x[-1]*100, z[-1]*100, 's', color=INK, ms=6, zorder=5)
b.plot(dd.loc[tr, 'Ball Pos X']*100, dd.loc[tr, 'Ball Pos Z']*100, '*', color='#e87ba4', ms=14, label='Target')
h, l = b.get_legend_handles_labels(); order = [1, 2, 3, 4, 0, 5]
b.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=8, loc='upper right', bbox_to_anchor=(1.04, 1.03), title='Same time per quarter,\ndifferent distance', title_fontsize=8.5, alignment='left')
b.set_xlabel('Left–right position (cm)'); b.set_ylabel('Forward position (cm)'); b.set_aspect('equal', adjustable='datalim')
b.set_title('Example reach, seen from above', loc='left', fontsize=11)

b.text(-0.18, 1.06, 'B', transform=b.transAxes, fontsize=14, fontweight='bold')

FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / 'Figure1_task.png')
