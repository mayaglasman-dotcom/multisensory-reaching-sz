# Independent recomputation of the movement-onset results from raw trajectories.
import os, numpy as np, pandas as pd, warnings
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt
SOS=butter(2,10,fs=72,output='sos')   # 10 Hz low-pass, applied to each trial's whole recording before jerk
warnings.filterwarnings("ignore")
SR=72; DT=1/SR
from config import CONTROL_DIR, PATIENT_DIR, DUR_CONTROL, DUR_PATIENT
G={'control':(CONTROL_DIR,DUR_CONTROL),
   'patient':(PATIENT_DIR,DUR_PATIENT)}
def fn(s): s=str(s); return "SZR-"+s if len(s)<=3 else s
def deriv(a,k): 
    for _ in range(k): a=np.gradient(a,DT)
    return a
def jpow(j):   # mean squared jerk
    return float(np.sum(np.square(j))/len(j))
def cubic(x,y,z):
    t=np.linspace(0,1,len(x)); B=np.c_[t*(1-t),t**2*(1-t)]; res=tot=0
    for a in (x,y,z):
        r=a-(a[0]+(a[-1]-a[0])*t); c,*_=np.linalg.lstsq(B,r,rcond=None); res+=np.sum((r-B@c)**2); tot+=np.sum((a-a.mean())**2)
    return 1-res/tot
rows=[]
for g,(folder,dp) in G.items():
    dur=pd.read_csv(dp)
    for sub,sd in dur.groupby(dur.Subject.astype(str)):
        p=os.path.join(folder,f"raw_{fn(sub)}_Trajectories.csv")
        if not os.path.exists(p): continue
        tr=pd.read_csv(p).sort_values(['Trial','sample'])
        m,s=sd.trialDuration.mean(),sd.trialDuration.std()
        for _,r in sd.iterrows():
            seg=tr[tr.Trial==r.Trial]
            if len(seg)==0: continue
            kept=abs(r.trialDuration-m)<=2.5*s
            full=seg[['Hand Pos X','Hand Pos Y','Hand Pos Z']].to_numpy().T
            XF=sosfiltfilt(SOS,full,axis=1,padlen=min(9,full.shape[1]-1)) if full.shape[1]>1 else full
            n=int(np.floor(r.trialDuration*SR))+1; seg=seg.iloc[-n:] if n<len(seg) else seg
            X=seg[['Hand Pos X','Hand Pos Y','Hand Pos Z']].to_numpy().T
            JF=np.sqrt(sum(deriv(a,3)**2 for a in XF))[-X.shape[1]:] if XF.shape[1]>1 else np.zeros(X.shape[1])   # jerk of the filtered whole recording, cut to the window
            if X.shape[1]<3: continue
            sp=np.sqrt(sum(np.gradient(a,DT)**2 for a in X)); pk=int(np.argmax(sp)); on=pk
            while on>0 and sp[on-1]>0.1*sp[pk]: on-=1   # walk back from the peak to the last sample above 10%
            row=dict(Group=g,Subject=sub,Trial=r.Trial,vis=seg.Visibility.iloc[0],kept=kept,nwin=X.shape[1],on=on,
                     endfrac=sp[-1]/sp.max(),half='first' if r.Trial<=100 else 'second',err=r['Hand error'])
            # original-window Q1 (for confound check)
            if X.shape[1]>=16:
                J=JF; q=len(J)//4; row['Q1orig']=np.log(jpow(J[:q]))
            Y=X[:,on:]; L=Y.shape[1]; row['mt']=L/SR
            if L>=16:
                J=JF[on:]; q=L//4
                for i,(a,b) in enumerate([(0,q),(q,2*q),(2*q,3*q),(3*q,L)]): row[f'Q{i+1}']=np.log(jpow(J[a:b]))
                row['whole']=np.log(jpow(J))
            if L>=8:
                st=np.linalg.norm(Y[:,-1]-Y[:,0])
                if st>0.03: row['dir']=np.sum(np.linalg.norm(np.diff(Y,axis=1),axis=0))/st; row['smooth']=cubic(*Y)
            rows.append(row)
T=pd.DataFrame(rows); T.to_csv('trials.csv',index=False); print(T.groupby('Group').size())
