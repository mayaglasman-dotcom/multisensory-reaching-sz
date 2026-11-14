import numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
from config import RESULTS_DIR, FIG_DIR
R=str(RESULTS_DIR)+"/"; OUT=str(FIG_DIR)+"/"; FIG_DIR.mkdir(parents=True, exist_ok=True)
C={"control":"#2a78d6","patient":"#eb6834"}; LAB={"control":"Controls","patient":"Patients"}; G=["control","patient"]
INK="#2b2b2a"; MUTED="#6b6a64"; GRID="#e6e5e0"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
  "axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"text.color":INK,
  "axes.grid":True,"axes.grid.axis":"y","grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True,
  "lines.linewidth":2,"legend.frameon":False,"savefig.dpi":300,"savefig.bbox":"tight"})
rng=np.random.default_rng(1)
def fmtp(p): return "p < 0.0001" if p<1e-4 else f"p = {p:.4f}" if p<0.001 else f"p = {p:.3f}" if p<0.01 else f"p = {p:.2f}" if p>=0.1 else f"p = {p:.3f}"
def panel(ax,l): ax.text(-0.14,1.06,l,transform=ax.transAxes,fontsize=13,fontweight="bold",va="bottom")
def strip(ax,vals,stat="mean",ylabel=""):
    for i,g in enumerate(G):
        v=vals[g].dropna(); x=i+rng.uniform(-0.12,0.12,len(v))
        ax.scatter(x,v,s=22,color=C[g],alpha=0.75,edgecolor="white",linewidth=0.6,zorder=3)
        m=v.median() if stat=="median" else v.mean()
        ax.plot([i-0.25,i+0.25],[m,m],color=INK,lw=2.5,zorder=4)
    ax.set_xticks([0,1]); ax.set_xticklabels([LAB[g] for g in G]); ax.set_xlim(-0.6,1.6); ax.set_ylabel(ylabel)

q=pd.read_csv(R+"q1_final_subject.csv")
t=pd.read_csv(R+"q1_final_trials.csv"); v=pd.read_csv(R+"reach_definitions_trials.csv")
mm=t   # q1_final_trials.csv carries each trial's Visibility
import pingouin as pg
def vis_p(d):   # visibility test by the paper's rule
    return stats.ttest_1samp(d,0).pvalue if stats.shapiro(d).pvalue>=0.05 else stats.wilcoxon(d).pvalue
def grp_p(c,p):
    return stats.ttest_ind(p,c).pvalue if min(stats.shapiro(c).pvalue,stats.shapiro(p).pvalue)>=0.05 else stats.mannwhitneyu(c,p).pvalue
def grp_txt(c,p):
    if min(stats.shapiro(c).pvalue,stats.shapiro(p).pvalue)>=0.05:
        r=stats.ttest_ind(p,c); return f"t({len(c)+len(p)-2}) = {r.statistic:.2f}, {fmtp(r.pvalue)}"
    r=stats.mannwhitneyu(c,p); return f"U = {min(r.statistic,len(c)*len(p)-r.statistic):.0f}, {fmtp(r.pvalue)}"
ds=v[v.kept].dropna(subset=["dir_duration","smooth_duration"])
Qs=["Q1","Q2","Q3","Q4"]; QL=["Q1\n(start)","Q2","Q3","Q4\n(end)"]

# ---------- Figure 2: visibility
fig,axs=plt.subplots(1,3,figsize=(12,4),gridspec_kw={"width_ratios":[1.6,1,1]})
ax=axs[0]; panel(ax,"A")
sv=mm.groupby(["Group","Subject","Visibility"])[Qs].mean().reset_index()
for g in G:
    for vv,ls,mk,fc in [("Visible Hand","-","o",C[g]),("Invisible Hand","--","o","white")]:
        d=sv[(sv.Group==g)&(sv.Visibility==vv)][Qs]
        off=(-0.06 if g=="control" else 0.06)+(-0.02 if vv=="Visible Hand" else 0.02)
        ax.errorbar(np.arange(4)+off,d.mean(),yerr=d.sem(),color=C[g],ls=ls,marker=mk,mfc=fc,mec=C[g],ms=7,mew=1.6,capsize=3,
                    label=f"{LAB[g]}, hand {'visible' if vv=='Visible Hand' else 'hidden'}")
ax.set_xticks(range(4)); ax.set_xticklabels(QL); ax.set_ylabel("Jerk (log mean squared jerk)")
ax.set_title("Jerk across the reach",loc="left",fontsize=11)
ax.legend(fontsize=8,loc="lower center",ncol=2,frameon=False)
w=sv.pivot_table(index="Subject",columns="Visibility",values=Qs); dv={k:(w[(k,"Visible Hand")]-w[(k,"Invisible Hand")]).dropna() for k in Qs}
pv={k:vis_p(dv[k]) for k in Qs}; pmax=max(pv.values())
if all(dv[k].mean()>0 for k in Qs) and pmax<0.05:   # only claim it when every quarter shows it
    vtxt="visible > hidden in every quarter (all p < 0.001)" if pmax<0.001 else f"visible > hidden in every quarter (all p ≤ {pmax:.3f})"
else:
    vtxt="visible − hidden: "+", ".join(f"{k} {fmtp(pv[k])}" for k in Qs)
ax.text(0.5,0.97,vtxt,transform=ax.transAxes,ha="center",va="top",fontsize=8.5,color=MUTED)
lo,hi=ax.get_ylim(); ax.set_ylim(lo-0.35*(hi-lo),hi+0.12*(hi-lo))
for ax,col,ttl,lab in [(axs[1],"smooth_duration","Smoothness","Smoothness (R², per-person median)"),
                       (axs[2],"dir_duration","Directness","Directness (per-person median)")]:
    panel(ax,"B" if col=="smooth_duration" else "C")
    s=ds.groupby(["Group","Subject","Visibility"])[col].median().unstack()
    for g in G:
        d=s.loc[g]; x=[0,1] ; off=-0.05 if g=="control" else 0.05
        m=[d["Visible Hand"].mean(),d["Invisible Hand"].mean()]; e=[d["Visible Hand"].sem(),d["Invisible Hand"].sem()]
        ax.errorbar(np.array(x)+off,m,yerr=e,color=C[g],marker="o",ms=7,capsize=3,label=LAB[g])
    ax.set_xticks([0,1]); ax.set_xticklabels(["Hand\nvisible","Hand\nhidden"]); ax.set_xlim(-0.4,1.4); ax.set_ylabel(lab)
    p=vis_p((s["Visible Hand"]-s["Invisible Hand"]).dropna())   # computed, same rule as the paper
    ax.set_title(f"{ttl}\nvisible vs. hidden: {fmtp(p)}",loc="left",fontsize=11)
axs[1].legend(fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT+"Figure2_visibility.png"); plt.close()

# ---------- Figure 3: group by quarter
fig,ax=plt.subplots(figsize=(6.5,4.5))
for g in G:
    d=q[q.Group==g][Qs]; off=-0.05 if g=="control" else 0.05
    ax.errorbar(np.arange(4)+off,d.mean(),yerr=d.sem(),color=C[g],marker="o",ms=7,capsize=3,label=f"{LAB[g]} (n = {len(d)})")
ax.set_xticks(range(4)); ax.set_xticklabels(QL); ax.set_ylabel("Jerk (log mean squared jerk)")
gp={k:grp_p(q[q.Group=="control"][k],q[q.Group=="patient"][k]) for k in Qs}; sig=[k for k in Qs if gp[k]<0.05]
pmin=min(gp[k] for k in Qs if k not in sig) if len(sig)<len(Qs) else None
gtxt=("No quarter differs between groups (all p ≥ %.2f)"%pmin if not sig else "Patients lower in "+", ".join(f"{k} ({fmtp(gp[k])})" for k in sig)+("; other quarters p ≥ %.2f"%pmin if pmin is not None else ""))
Lq=q.melt(id_vars=["Group","Subject"],value_vars=Qs,var_name="Quarter",value_name="jerk")
ai=pg.mixed_anova(data=Lq,dv="jerk",within="Quarter",between="Group",subject="Subject", correction=True); ai=ai[ai.Source=="Interaction"].iloc[0]
ax.text(0.03,0.04,gtxt,transform=ax.transAxes,fontsize=9,color=MUTED,ha="left",va="bottom")
lo,hi=ax.get_ylim(); ax.set_ylim(lo-0.2*(hi-lo),hi)
ax.set_title(f"Jerk across the reach by group\nGroup × quarter interaction: F({ai.DF1:.0f}, {ai.DF2:.0f}) = {ai.F:.2f}, Greenhouse-Geisser {fmtp(ai.p_GG_corr)}",loc="left",fontsize=11)
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig(OUT+"Figure3_group_by_quarter.png"); plt.close()

# ---------- Figure 4: carryover
s=pd.read_csv(R+"carryover_slopes_by_transition_no_outliers.csv"); tt=pd.read_csv(R+"transition_table.csv").set_index("transition")
order=[("visible_to_visible","Visible → visible"),("invisible_to_invisible","Hidden → hidden"),
       ("visible_to_invisible","Visible → hidden"),("invisible_to_visible","Hidden → visible")]
fig,axs=plt.subplots(1,4,figsize=(13,4),sharey=True)
for i,(ax,(k,lab)) in enumerate(zip(axs,order)):
    d=s[s.transition==k]; strip(ax,{g:d[d.Group==g].slope.reset_index(drop=True) for g in G},"mean","Carryover slope" if i==0 else "")
    ax.axhline(0,color=MUTED,lw=0.8,ls=":")
    ax.set_title(f"{lab}\n{fmtp(tt.loc[k,'p'])}",loc="left",fontsize=11); panel(ax,"ABCD"[i])
axs[0].text(-0.55,axs[0].get_ylim()[1]*0.92,"pulled toward\nprevious target",fontsize=7.5,color=MUTED,va="top")
axs[0].text(-0.55,axs[0].get_ylim()[0]*0.92,"pushed away from\nprevious target",fontsize=7.5,color=MUTED,va="bottom")
plt.tight_layout(); plt.savefig(OUT+"Figure4_carryover.png"); plt.close()

# ---------- Figure S1: directness & smoothness
fig,axs=plt.subplots(2,2,figsize=(9,7.5))
def mw(x,y):   # smaller Mann-Whitney U, as reported in the paper
    r=stats.mannwhitneyu(x,y); return f"U = {min(r.statistic,len(x)*len(y)-r.statistic):.0f}, {fmtp(r.pvalue)}"
for r,(col,name) in enumerate([("dir_duration","Directness"),("smooth_duration","Smoothness (R²)")]):
    for c,agg in enumerate(["median","mean"]):
        ax=axs[r,c]; sv2=ds.groupby(["Group","Subject"])[col].agg(agg)
        strip(ax,{g:sv2[g].reset_index(drop=True) for g in G},agg,f"{name}, per-person {agg}")
        ax.set_title(f"{name}: {agg}s\n{mw(sv2['control'],sv2['patient'])}",loc="left",fontsize=11); panel(ax,"ABCD"[r*2+c])
plt.tight_layout(); plt.savefig(OUT+"FigureS1_directness_smoothness.png"); plt.close()

# ---------- Figure S2: whole-reach jerk, and jerk vs. movement time
f=pd.read_csv(R+"final_measures_subject.csv")
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
strip(axs[0],{g:f[f.Group==g].whole.reset_index(drop=True) for g in G},"mean","Whole-reach jerk (log mean squared jerk)")
axs[0].set_title(f"Whole reach\n{grp_txt(f[f.Group=='control'].whole,f[f.Group=='patient'].whole)}",loc="left",fontsize=11)
ax=axs[1]
for g in G:
    d=q[q.Group==g]; ax.scatter(d.duration,d.Q2,color=C[g],s=26,edgecolor="white",linewidth=0.6,label=LAB[g])
    lr=stats.linregress(d.duration,d.Q2); xs=np.linspace(d.duration.min(),d.duration.max(),20); ax.plot(xs,lr.intercept+lr.slope*xs,color=C[g],lw=1.5)
ax.set_xlabel("Mean movement time (s)"); ax.set_ylabel("Jerk, second quarter (log mean squared jerk)")
import statsmodels.formula.api as smf
mq=smf.ols("Q2 ~ Group + duration",q).fit()
ax.set_title(f"Second-quarter jerk vs. movement time\nmovement time: {fmtp(mq.pvalues['duration'])}; group with movement time: {fmtp(mq.pvalues['Group[T.patient]'])}",loc="left",fontsize=11); ax.legend(); ax.grid(axis="x",color=GRID)
for i,a_ in enumerate(axs): panel(a_,"AB"[i])
plt.tight_layout(); plt.savefig(OUT+"FigureS2_jerk_checks.png"); plt.close()

# ---------- Figure S3: session halves
h=pd.read_csv(R+"final_fatigue_subject.csv")
fig,axs=plt.subplots(1,2,figsize=(10,4.3))
pvh=h.pivot_table(index=["Group","Subject"],columns="Half",values="whole").dropna().reset_index(); ch=pvh["second"]-pvh["first"]
pve=h.pivot_table(index=["Group","Subject"],columns="Half",values="log_error").dropna().reset_index(); che=pve["second"]-pve["first"]
for i,(ax,col,lab,title) in enumerate([(axs[0],"whole","Whole-reach jerk (log mean squared jerk)",f"Jerk\ngroup difference in change: {grp_txt(ch[pvh.Group=='control'],ch[pvh.Group=='patient'])}"),
                                       (axs[1],"log_error","Hand error (log degrees)","Accuracy\ngroup difference in change: "+grp_txt(che[pve.Group=='control'],che[pve.Group=='patient']))]):
    pv=h.pivot_table(index=["Group","Subject"],columns="Half",values=col).dropna().reset_index()
    for g in G:
        d=pv[pv.Group==g]
        for _,r in d.iterrows(): ax.plot([0,1],[r["first"],r["second"]],color=C[g],alpha=0.18,lw=1)
        ax.plot([0,1],[d["first"].mean(),d["second"].mean()],color=C[g],lw=3,marker="o",ms=7,label=LAB[g])
    ax.set_xticks([0,1]); ax.set_xticklabels(["First half","Second half"]); ax.set_xlim(-0.3,1.3); ax.set_ylabel(lab)
    ax.set_title(title,loc="left",fontsize=11); panel(ax,"AB"[i])
axs[0].legend(); plt.tight_layout(); plt.savefig(OUT+"FigureS3_session_halves.png"); plt.close()
print("done")
