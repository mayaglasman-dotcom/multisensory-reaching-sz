# Supplementary Table S1: demographics and clinical characteristics by group.
import pandas as pd, numpy as np, warnings
from scipy import stats
warnings.filterwarnings("ignore")
from config import CLINICAL_XLSX as x
d=pd.read_excel(x,'Demographics'); d=d[d['Record ID'].notna()].copy(); d['Record ID']=d['Record ID'].astype(str).str.replace(r'\s*\(.*\)\s*$','',regex=True).str.strip()   # drop notes such as 'PC-C19 (also SZR-1019)'
m={'C02':'PC-C16','C03':'PC-C18','C04':'PC-C19','C05':'PC-C21','C06':'PC-C22','C07':'PC-C20','C08':'PC-C24','C09':'PC-C25',
   'S02':'PC-S02','S05':'PC-S06','S06':'PC-S16','S07':'PC-S22','S09':'PC-S12','S10':'PC-S23'}
q=pd.read_csv('q1_final_subject.csv')[['Group','Subject']]   # the 57 analyzed participants
q['RID']=q.Subject.map(lambda s:m.get(s,s))
t=q.merge(d,left_on='RID',right_on='Record ID',how='left')
print('no demographics row:',t[t['Record ID'].isna()].Subject.tolist())
def num(s): return pd.to_numeric(s.astype(str).str.replace(' mg','').str.strip(),errors='coerce').where(lambda v:v>=0)
for c in ['Age','Years of Education (mom)','Years of Education (dad)','WRAT-5  Standard Score','Chlorpromazine equivalent','Olanzapine Equivalent',
          'Avg Five Factor - Positive','Avg Five Factor - Negative','Avg Five Factor - Disorganized','Avg Five Factor - Excitement','Avg Five Factor - Depression','Sum MADRS Depression','YMRS Sum','Fagerstrom Sum ( w/o PRISM)']:
    t[c]=num(t[c])
    g=[t[t.Group==k][c].dropna() for k in ['control','patient']]
    s=f"{c}: C n={len(g[0])} {g[0].mean():.1f} ({g[0].std():.1f}) | P n={len(g[1])} {g[1].mean():.2f} ({g[1].std():.2f}) median {g[1].median():.1f} [{g[1].quantile(.25):.1f}-{g[1].quantile(.75):.1f}]"
    if len(g[0])>2 and len(g[1])>2:
        sw=min(stats.shapiro(g[0]).pvalue,stats.shapiro(g[1]).pvalue)
        s+= f" | t({len(g[0])+len(g[1])-2})={stats.ttest_ind(g[1],g[0]).statistic:.2f} p={stats.ttest_ind(g[1],g[0]).pvalue:.3f}; U={min(stats.mannwhitneyu(g[0],g[1]).statistic,len(g[0])*len(g[1])-stats.mannwhitneyu(g[0],g[1]).statistic):.0f} p={stats.mannwhitneyu(g[0],g[1]).pvalue:.3f} SW={sw:.3f}"
    print(s)
for c in ['Sex Assigned at birth (M/F)','Race','Handedness','Primary Diagnosis']:
    t[c]=t[c].astype(str).str.strip().str.capitalize()
    ct=pd.crosstab(t[c],t.Group); print(ct)
sx=pd.crosstab(t['Sex Assigned at birth (M/F)'],t.Group); sx=sx.loc[['F','M']]; print('sex fisher',stats.fisher_exact(sx.values), 'chi2',stats.chi2_contingency(sx.values)[:2])
print(t[t.Group=='patient'][['Subject','Primary Diagnosis','Chlorpromazine equivalent']].to_string())
