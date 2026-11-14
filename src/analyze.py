# Statistics from the independent recomputation (reads trials.csv from recompute.py).
import pandas as pd, numpy as np, pingouin as pg, statsmodels.formula.api as smf, warnings
from scipy import stats
warnings.filterwarnings("ignore")
T=pd.read_csv('trials.csv'); K=T[T.kept]
J=K.dropna(subset=['Q1']); print('jerk trials kept',J.groupby('Group').size().to_dict(),'median samples',np.median(J.mt*72),'s',np.median(J.mt))
def gt(name,c,p):
    c,p=c.dropna(),p.dropna(); sw=min(stats.shapiro(c).pvalue,stats.shapiro(p).pvalue)
    t=stats.ttest_ind(p,c); u=stats.mannwhitneyu(c,p); U=min(u.statistic,len(c)*len(p)-u.statistic)
    sp=np.sqrt(((len(c)-1)*c.var()+(len(p)-1)*p.var())/(len(c)+len(p)-2))
    print(f"{name:28s} C {c.mean():.4f} P {p.mean():.4f} | t({len(c)+len(p)-2})={t.statistic:.2f} p={t.pvalue:.4f} | U={U:.0f} p={u.pvalue:.4f} | d={(p.mean()-c.mean())/sp:.2f} | rule={'MW' if sw<.05 else 't'}")
def pt(name,a,b):
    ok=a.notna()&b.notna(); a,b=a[ok],b[ok]
    d=a-b; t=stats.ttest_rel(a,b); w=stats.wilcoxon(a,b)
    print(f"{name:28s} {a.mean():.4f} vs {b.mean():.4f} diff {d.mean():.4f} | t({len(d)-1})={t.statistic:.2f} p={t.pvalue:.2g} | W={w.statistic:.0f} p={w.pvalue:.2g} | rule={'W' if stats.shapiro(d).pvalue<.05 else 't'}")
S=J.groupby(['Group','Subject'])[['Q1','Q2','Q3','Q4','whole','mt']].mean().reset_index(); C=S.Group=='control'
for k in ['mt','Q1','Q2','Q3','Q4','whole']: gt(k,S[C][k],S[~C][k])
L=S.melt(id_vars=['Group','Subject'],value_vars=['Q1','Q2','Q3','Q4'],var_name='Q',value_name='j')
a=pg.mixed_anova(data=L,dv='j',within='Q',between='Group',subject='Subject', correction=True); r=a[a.Source=='Interaction'].iloc[0]; print('interaction F',round(r.F,2),r.DF1,r.DF2,'pGG',round(r['p_GG_corr'],4),'eps',round(r.eps,2))
m=smf.ols('Q1 ~ Group + mt',S).fit(); print('Q1~Group+mt', m.params.round(3).to_dict(), m.tvalues.round(2).to_dict(), m.pvalues.round(4).to_dict())
for k in ['Q1','Q2','Q3','Q4']:
    v=J.groupby(['Group','Subject','vis'])[k].mean().unstack(); g=v.index.get_level_values(0)
    pt('vis '+k,v['Visible Hand'],v['Invisible Hand']); d=v['Visible Hand']-v['Invisible Hand']; gt('gxv '+k,d[g=='control'],d[g=='patient'])
D=K.dropna(subset=['dir','smooth'])
for agg in ['median','mean']:
    s=D.groupby(['Group','Subject'])[['dir','smooth']].agg(agg).reset_index(); c=s.Group=='control'
    gt('dir '+agg,s[c].dir,s[~c].dir); gt('smooth '+agg,s[c].smooth,s[~c].smooth)
    for vv in ['Visible Hand','Invisible Hand']:
        s2=D[D.vis==vv].groupby(['Group','Subject'])[['dir','smooth']].agg(agg).reset_index(); c2=s2.Group=='control'
        gt(f'dir {agg} {vv}',s2[c2].dir,s2[~c2].dir); gt(f'smooth {agg} {vv}',s2[c2].smooth,s2[~c2].smooth)
    v=D.groupby(['Group','Subject','vis'])[['dir','smooth']].agg(agg).unstack(); g=v.index.get_level_values(0)
    for col in ['dir','smooth']:
        pt(f'vis {col} {agg}',v[(col,'Visible Hand')],v[(col,'Invisible Hand')]); d=v[(col,'Visible Hand')]-v[(col,'Invisible Hand')]; gt(f'gxv {col} {agg}',d[g=='control'],d[g=='patient'])
s=D.groupby(['Group','Subject']).dir.agg([lambda x:(x>1.5).mean(),lambda x:x.quantile(.9)]).reset_index(); s.columns=['Group','Subject','f15','p90']; c=s.Group=='control'
gt('frac>1.5',s[c].f15,s[~c].f15); gt('p90',s[c].p90,s[~c].p90)
# LOO
for name,col,test in [('mt',None,'t'),('smooth median',None,'MW')]:
    s=(S[['Group','Subject','mt']] if name=='mt' else D.groupby(['Group','Subject']).smooth.median().reset_index().rename(columns={'smooth':'mt'}))
    ps=[]
    for sub in s.Subject:
        x=s[s.Subject!=sub]; c,p=x[x.Group=='control'].mt,x[x.Group=='patient'].mt
        ps.append(stats.ttest_ind(c,p).pvalue if test=='t' else stats.mannwhitneyu(c,p).pvalue)
    print('LOO max p',name,round(max(ps),4))
# (pre-movement checks on the original windows: see extra_checks.py)
# fatigue
F=J.copy(); F['le']=np.log(F.err.where(F.err>0))
for col in ['whole','le']:
    h=F.groupby(['Group','Subject','half'])[col].mean().unstack().dropna(); g=h.index.get_level_values(0)
    for gg in ['control','patient']: pt(f'fat {col} {gg} (second-first)',h['second'][g==gg],h['first'][g==gg])
    ch=h['second']-h['first']; gt(f'fat {col} change',ch[g=='control'],ch[g=='patient'])
S.to_csv('subject.csv',index=False); D.groupby(['Group','Subject']).smooth.median().reset_index().to_csv('smooth_med.csv',index=False)
