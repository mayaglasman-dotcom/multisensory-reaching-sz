import sys, numpy as np, pandas as pd, pingouin as pg, statsmodels.formula.api as smf, warnings
from scipy import stats
warnings.filterwarnings("ignore")
from config import RESULTS_DIR
R=(sys.argv[1] if len(sys.argv)>1 else str(RESULTS_DIR)).rstrip('/')+'/'; OUT={}
def put(k,v): OUT[k]=v
def sw(*xs): return min(stats.shapiro(x).pvalue for x in xs)
def grp(lab,c,p):
    c,p=pd.Series(c).dropna(),pd.Series(p).dropna()
    t=stats.ttest_ind(p,c); u=stats.mannwhitneyu(c,p); U=min(u.statistic,len(c)*len(p)-u.statistic)
    sp=np.sqrt(((len(c)-1)*c.var()+(len(p)-1)*p.var())/(len(c)+len(p)-2))
    put(lab,dict(nc=len(c),np=len(p),mc=round(c.mean(),3),mp=round(p.mean(),3),medc=round(c.median(),3),medp=round(p.median(),3),
        t=round(t.statistic,2),df=len(c)+len(p)-2,pt=round(t.pvalue,4),U=int(U),pU=round(u.pvalue,4),d=round((p.mean()-c.mean())/sp,2),
        rule='MW' if sw(c,p)<0.05 else 't'))
def paired(lab,a,b):
    ok=a.notna()&b.notna(); a,b=a[ok],b[ok]   # people missing a condition are left out of that test
    t=stats.ttest_rel(a,b); w=stats.wilcoxon(a,b); put(lab,dict(n=len(a),ma=round(a.mean(),3),mb=round(b.mean(),3),diff=round((a-b).mean(),3),t=round(t.statistic,2),df=len(a)-1,p=float('%.3g'%t.pvalue),W=int(w.statistic),pW=float('%.3g'%w.pvalue),rule='W' if stats.shapiro(a-b).pvalue<0.05 else 't'))
def bf01(c,p):
    t=stats.ttest_ind(c,p).statistic; return round(1/float(pg.bayesfactor_ttest(t,len(c),len(p),r=0.707)),2)
q=pd.read_csv(R+"q1_final_subject.csv"); C=q.Group=="control"
put('n',dict(c=int(C.sum()),p=int((~C).sum())))
grp('duration',q[C].duration,q[~C].duration)
for k in ["Q1","Q2","Q3","Q4"]: grp(k,q[C][k],q[~C][k])
put('BF01_Q1',bf01(q[C].Q1,q[~C].Q1)); put('BF01_Q2',bf01(q[C].Q2,q[~C].Q2))
L=q.melt(id_vars=["Group","Subject"],value_vars=["Q1","Q2","Q3","Q4"],var_name="Quarter",value_name="jerk")
def inter(d):
    a=pg.mixed_anova(data=d,dv="jerk",within="Quarter",between="Group",subject="Subject", correction=True); r=a[a.Source=="Interaction"].iloc[0]; return round(r.F,2),int(r.DF1),int(r.DF2),round(r.p_GG_corr,4),round(r.eps,2)
put('interaction',inter(L)); put('interaction_LOO_maxp',round(max(inter(L[L.Subject!=s])[3] for s in q.Subject),4))
q["rise"]=q[["Q2","Q3","Q4"]].mean(axis=1)-q.Q1; grp('rise',q[C].rise,q[~C].rise)
lo=[]
for s in q.Subject:
    d=q[q.Subject!=s]; lo.append(stats.mannwhitneyu(d[d.Group=="control"].Q1,d[d.Group=="patient"].Q1).pvalue)
put('Q1_LOO_maxMWp',round(max(lo),4))
m=smf.ols("Q1 ~ Group + duration",q).fit(); put('cov',dict(bG=round(m.params['Group[T.patient]'],2),tG=round(m.tvalues['Group[T.patient]'],2),pG=round(m.pvalues['Group[T.patient]'],4),bD=round(m.params['duration'],3),tD=round(m.tvalues['duration'],2),pD=round(m.pvalues['duration'],3),df=int(m.df_resid)))
for k in ["Q2","Q3","Q4"]:
    m=smf.ols(f"{k} ~ Group + duration",q).fit(); put(f'cov_{k}',dict(bG=round(m.params['Group[T.patient]'],2),tG=round(m.tvalues['Group[T.patient]'],2),pG=round(m.pvalues['Group[T.patient]'],4),bD=round(m.params['duration'],3),tD=round(m.tvalues['duration'],2),pD=float('%.3g'%m.pvalues['duration']),df=int(m.df_resid)))
f=pd.read_csv(R+"final_measures_subject.csv"); Cf=f.Group=="control"
grp('whole',f[Cf].whole,f[~Cf].whole)
t=pd.read_csv(R+"q1_final_trials.csv"); v=pd.read_csv(R+"reach_definitions_trials.csv")
mm=t   # q1_final_trials.csv carries each trial's Visibility
for k in ["Q1","Q2","Q3","Q4"]:
    s=mm.groupby(["Group","Subject","Visibility"])[k].mean().unstack(); g=s.index.get_level_values(0)
    paired('vis_'+k,s["Visible Hand"],s["Invisible Hand"]); d=s["Visible Hand"]-s["Invisible Hand"]; grp('gxv_'+k,d[g=="control"],d[g=="patient"])
    if k=="Q1": grp('Q1_visible',s["Visible Hand"][g=="control"],s["Visible Hand"][g=="patient"]); grp('Q1_hidden',s["Invisible Hand"][g=="control"],s["Invisible Hand"][g=="patient"])
ds=v[v.kept].dropna(subset=["dir_duration","smooth_duration"])
for agg in ["median","mean"]:
    s=ds.groupby(["Group","Subject","Visibility"])[["dir_duration","smooth_duration"]].agg(agg).unstack(); g=s.index.get_level_values(0)
    for col in ["dir_duration","smooth_duration"]:
        a,b=s[(col,"Visible Hand")],s[(col,"Invisible Hand")]; paired(f'vis_{col}_{agg}',a,b); d=a-b; grp(f'gxv_{col}_{agg}',d[g=="control"],d[g=="patient"])
    for col in ["dir_duration","smooth_duration"]:
        for cond in ["all","Visible Hand","Invisible Hand"]:
            dd=ds if cond=="all" else ds[ds.Visibility==cond]; ss=dd.groupby(["Group","Subject"])[col].agg(agg); grp(f'{col}_{agg}_{cond}',ss["control"],ss["patient"])
    ss=ds.groupby(["Group","Subject"])[["dir_duration","smooth_duration"]].agg(agg)
    put(f'BF01_dir_{agg}',bf01(ss.loc['control'].dir_duration,ss.loc['patient'].dir_duration)); put(f'BF01_smooth_{agg}',bf01(ss.loc['control'].smooth_duration,ss.loc['patient'].smooth_duration))
for name,fn in [("p90",lambda x:x.quantile(.9)),("frac15",lambda x:(x>1.5).mean())]:
    ss=ds.groupby(["Group","Subject"]).dir_duration.agg(fn); grp(name,ss["control"],ss["patient"])
co=pd.read_csv(R+"carryover_slopes_by_transition_no_outliers.csv")
for tr,d in co.groupby("transition"):
    grp('co_'+tr,d[d.Group=="control"].slope,d[d.Group=="patient"].slope); put('BF01_co_'+tr,bf01(d[d.Group=="control"].slope,d[d.Group=="patient"].slope))
lg=pd.read_csv(R+"carryover_lag_subject_tests.csv"); put('lag_p',[round(x,3) for x in lg.p])
h=pd.read_csv(R+"final_fatigue_subject.csv")
for col in ["whole","log_error"]:
    pv=h.pivot_table(index=["Group","Subject"],columns="Half",values=col).dropna().reset_index()
    for g in ["control","patient"]:
        d=pv[pv.Group==g]; paired(f'fat_{col}_{g}',d["second"],d["first"])
    pv["ch"]=pv["second"]-pv["first"]; grp(f'fat_{col}_change',pv[pv.Group=="control"].ch,pv[pv.Group=="patient"].ch)
# smoothness and directness vs movement time (per-person medians; movement time from onset)
sd_=ds.groupby("Subject")[["smooth_duration","dir_duration"]].median().reset_index().merge(q[["Group","Subject","duration"]],on="Subject")
for col,lab in [("smooth_duration","smooth"),("dir_duration","dir")]:
    r=stats.spearmanr(sd_[col],sd_.duration); m=smf.ols(f"{col} ~ Group + duration",sd_).fit()
    put(f'mt_{lab}',dict(rho=round(r.statistic,2),p_rho=float('%.3g'%r.pvalue),n=len(sd_),bG=round(m.params['Group[T.patient]'],4),tG=round(m.tvalues['Group[T.patient]'],2),pG=round(m.pvalues['Group[T.patient]'],3),bD=round(m.params['duration'],4),tD=round(m.tvalues['duration'],2),pD=float('%.3g'%m.pvalues['duration']),df=int(m.df_resid)))
    for g in ["control","patient"]:
        d=sd_[sd_.Group==g]; r=stats.spearmanr(d[col],d.duration); put(f'mt_{lab}_{g}',dict(rho=round(r.statistic,2),p=float('%.3g'%r.pvalue),n=len(d)))
# same with per-person means
sm_=ds.groupby("Subject")[["smooth_duration","dir_duration"]].mean().reset_index().merge(q[["Group","Subject","duration"]],on="Subject")
for col,lab in [("smooth_duration","smooth"),("dir_duration","dir")]:
    r=stats.spearmanr(sm_[col],sm_.duration); m=smf.ols(f"{col} ~ Group + duration",sm_).fit()
    put(f'mt_{lab}_mean',dict(rho=round(r.statistic,2),p_rho=float('%.3g'%r.pvalue),n=len(sm_),bG=round(m.params['Group[T.patient]'],4),tG=round(m.tvalues['Group[T.patient]'],2),pG=round(m.pvalues['Group[T.patient]'],3),bD=round(m.params['duration'],4),tD=round(m.tvalues['duration'],2),pD=float('%.3g'%m.pvalues['duration']),df=int(m.df_resid)))
    for g in ["control","patient"]:
        d=sm_[sm_.Group==g]; r=stats.spearmanr(d[col],d.duration); put(f'mt_{lab}_mean_{g}',dict(rho=round(r.statistic,2),p=float('%.3g'%r.pvalue),n=len(d)))
eq=pd.read_csv(R+"equivalence_bayes.csv"); put('equiv',eq[['measure','cohens_d','d_90ci_low','d_90ci_high','p_tost_d0.5','BF01']].round(3).values.tolist())
put('trials_kept',t.groupby('Group').size().to_dict()); put('median_samples',float(np.median(np.round(t.duration*72))))   # duration = samples/72, from onset
# --- seeing the hand also changes movement time: speed-adjusted visibility effects on jerk
s=t.groupby(["Group","Subject","Visibility"]).duration.mean().unstack(); g=s.index.get_level_values(0)
paired('vis_mt',s["Visible Hand"],s["Invisible Hand"]); d=s["Visible Hand"]-s["Invisible Hand"]; grp('gxv_mt',d[g=="control"],d[g=="patient"])
adj=[]
for (gg,sub),dd in t.groupby(["Group","Subject"]):
    if dd.Visibility.nunique()<2: continue
    row=dict(Group=gg,Subject=sub)
    for k in ["Q1","Q2","Q3","Q4"]:   # within each person: jerk ~ visible + log movement time
        m=smf.ols(f"{k} ~ vis + np.log(duration)",dd.assign(vis=(dd.Visibility=="Visible Hand").astype(float))).fit()
        row[k]=m.params["vis"]
    adj.append(row)
adj=pd.DataFrame(adj); ga=adj.Group
for k in ["Q1","Q2","Q3","Q4"]:
    x=adj[k]; put('visadj_'+k,dict(n=len(x),mean=round(x.mean(),3),rule='W' if stats.shapiro(x).pvalue<0.05 else 't',
        p_t=float('%.3g'%stats.ttest_1samp(x,0).pvalue),p_W=float('%.3g'%stats.wilcoxon(x).pvalue)))
    grp('gxvadj_'+k,x[ga=="control"],x[ga=="patient"])
# --- main results using only trials with a detected movement onset (hand not already moving when the recording window starts)
to=t[~t.moving_at_start.astype(bool)]
so=to.groupby(["Group","Subject"])[["Q1","Q2","Q3","Q4","duration"]].mean()
put('onset_only_trials',to.groupby('Group').size().to_dict())
for k in ["duration","Q1","Q2","Q3","Q4"]: grp('onset_only_'+k,so.loc["control"][k],so.loc["patient"][k])
for k in ["Q1","Q2","Q3","Q4"]:
    ss=to.groupby(["Group","Subject","Visibility"])[k].mean().unstack().dropna(); gg=ss.index.get_level_values(0)
    paired('onset_only_vis_'+k,ss["Visible Hand"],ss["Invisible Hand"]); d=ss["Visible Hand"]-ss["Invisible Hand"]; grp('onset_only_gxv_'+k,d[gg=="control"],d[gg=="patient"])
vo=v[v.kept].merge(to[["Group","Subject","Trial"]],on=["Group","Subject","Trial"]).dropna(subset=["smooth_duration","dir_duration"])
for col in ["smooth_duration","dir_duration"]:
    for agg in ("mean","median"):
        sm=vo.groupby(["Group","Subject"])[col].agg(agg); grp(f'onset_only_{col}_{agg}',sm["control"],sm["patient"])
for k,val in OUT.items(): print(k,'=',val)
