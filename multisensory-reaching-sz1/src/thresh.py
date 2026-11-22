from config import CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, os
from scipy import stats
from q1_final import jerk, jerk_power_per_sample, fname, add_lowpass, LP, onset_index
import reach_definitions as rd
# Main group results at movement-onset thresholds of 5%, 10% and 20% of peak speed (Supplementary Table S3).
rows=[]
for grp,folder,dpath in [('control',CONTROL_DIR,DUR_CONTROL),('patient',PATIENT_DIR,DUR_PATIENT)]:
    dur=pd.read_csv(dpath)
    for sub,dd in dur.groupby('Subject'):
        p=os.path.join(folder,f'raw_{fname(sub)}_Trajectories.csv')
        if not os.path.exists(p): continue
        t=pd.read_csv(p); m,s=dd.trialDuration.mean(),dd.trialDuration.std()
        kt=set(dd[dd.trialDuration.between(m-2.5*s,m+2.5*s)].Trial); dmap=dict(zip(dd.Trial,dd.trialDuration))
        for tr,g in t.groupby('Trial'):
            if tr not in kt: continue
            if len(g)<16: continue
            n=int(np.floor(dmap[tr]*72))+1; g=add_lowpass(g).tail(n)
            if len(g)<16: continue
            x,y,z=(g[c].to_numpy() for c in ['Hand Pos X','Hand Pos Y','Hand Pos Z']); jl=g['LP jerk'].to_numpy()
            v=np.sqrt(np.gradient(x,1/72)**2+np.gradient(y,1/72)**2+np.gradient(z,1/72)**2)
            for fr in [0.05,0.1,0.2]:
                i=onset_index(x,y,z,fr); xs,ys,zs=x[i:],y[i:],z[i:]
                if len(xs)<16: continue
                j=jl[i:]; q=len(j)//4
                dirn,sm=rd.measures(xs,ys,zs)
                rows.append(dict(Group=grp,Subject=sub,thr=fr,mt=len(xs)/72,Q1=np.log(jerk_power_per_sample(j[:q])),dir=dirn,smooth=sm))
r=pd.DataFrame(rows)   # missing directness/smoothness drops only those values, not the trial's movement time or jerk
for fr,d in r.groupby('thr'):
    s=d.groupby(['Group','Subject']).agg(mt=('mt','mean'),Q1=('Q1','mean'),dir_mean=('dir','mean'),dir_med=('dir','median'),sm_mean=('smooth','mean'),sm_med=('smooth','median'))
    out=[f'thr {fr}']
    for c in s.columns:
        a,b=s.loc['control'][c],s.loc['patient'][c]; out.append(f"{c}: C {a.mean():.4f} P {b.mean():.4f} MW p={stats.mannwhitneyu(a,b).pvalue:.3f}")
    print(' | '.join(out))
