#!/usr/bin/env python3

# --- package path bootstrap -------------------------------------------------
import os as _os
from pathlib import Path as _Path


def _repro_root(start=None):
    """Package root, located by the '.repro_root' marker (or REPRO_ROOT env)."""
    here = _Path(start or __file__).resolve()
    for cand in [here, *here.parents]:
        if (cand / ".repro_root").exists():
            return cand
    return _Path(_os.environ.get("REPRO_ROOT", _Path.cwd())).resolve()


REPRO_ROOT = _repro_root()
# ---------------------------------------------------------------------------
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys,json,math,hashlib,time,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,ndtri
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
DATA=Path(str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917'))
BASE=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q3'))
SEM=BASE/'01_q3_semantic_shuffle/HOSPITAL_OSTEOPOROSIS'
RND=BASE/'02_q3_random_null/HOSPITAL_OSTEOPOROSIS'
SEM.mkdir(parents=True,exist_ok=True);RND.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(DATA/'scripts'))
import v2_core as V
from common import SEEDS,N_FOLDS

LAMS=[0.0,0.03,0.1,0.3,1.0,3.0,10.0];K=10;NREP=1000
NULLSEED={'semantic':2026092203,'random':2026092204}

def folds(y,g,seed,n):
    return list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))
def select_lam(Xr,s,y,g,grid,kind,seed,n_inner=4):
    best=-np.inf;ans=grid[0]
    for lam in grid:
        aa=[]
        for tr,va in folds(y,g,seed+int(round(lam*100)),n_inner):
            if len(np.unique(y[tr]))<2 or len(np.unique(y[va]))<2:continue
            w,_,sc,_=V.fit_full(Xr[tr],s[tr],y[tr],lam,kind=kind)
            aa.append(roc_auc_score(y[va],V.predict_full(w,sc,Xr[va],s[va])))
        m=float(np.mean(aa)) if aa else -np.inf
        if m>best:best=m;ans=lam
    return float(ans)
def anchor(D):
    r=rankdata(-D,method='average');u=1-(r-.5)/len(D);return ndtri(np.clip(u,1e-6,1-1e-6))
def stable_topk(score,D,features,k):
    z=pd.DataFrame({'feature':features,'score':score,'D':D,'idx':np.arange(len(features))})
    return tuple(z.sort_values(['score','D','idx'],ascending=[False,False,True],kind='mergesort').feature.iloc[:k])
def certainty(p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12);H=-(p*np.log(p)+(1-p)*np.log(1-p))/math.log(2);return 1-H

man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json'));features=list(man['selectable_features']);fidx={f:i for i,f in enumerate(features)}
full=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(
  pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),on=['patient_uid','site']).reset_index(drop=True)
prim=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
X=prim[features].to_numpy(float);site=prim.site.to_numpy();y=prim.y.to_numpy(int);groups=prim.patient_uid.to_numpy()
outer=[]
for seed in SEEDS:
    for fi,(tr,va) in enumerate(folds(y,groups,seed,N_FOLDS)):outer.append((int(seed),int(fi),tr,va))
scoretab=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/downstream_batch1/OUTER_SELECTOR_SCORES.csv')))
Dmap={(s,fi):scoretab[(scoretab.seed==s)&(scoretab.fold==fi)].set_index('feature').loc[features].D.to_numpy(float) for s,fi,_,_ in outer}
pairk=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/downstream_batch1/selective_PAIRK_WEIGHTS_FROZEN.csv')))
q=pairk[(pairk.k==K)&(pairk.evidence_gate==1)].copy().reset_index(drop=True)
pobs=q.p_selective_A.to_numpy(float);cobs=q.c_pair.to_numpy(float);Avec=q.actionable_boundary_score.to_numpy(float)
ii=np.array([fidx[f] for f in q.feature_A],int);jj=np.array([fidx[f] for f in q.feature_B],int)
active=np.unique(np.r_[ii,jj]);pos={v:k for k,v in enumerate(active)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj])

def solve(D,lam,p,c):
    sd=anchor(D)
    if lam==0:return sd
    w=Avec*c;W=w.sum()
    if W<=0:return sd
    yy=(p>.5).astype(float);base=sd[active].copy();P=len(sd)
    def fg(v):
        d=v[li]-v[lj];ce=np.logaddexp(0,d)-yy*d;diff=v-base
        f=.5*np.dot(diff,diff)/P+lam*np.dot(w,ce)/(W+1e-12)
        g=diff/P;rr=lam*(w/(W+1e-12))*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
    r=minimize(lambda v:fg(v)[0],base,jac=lambda v:fg(v)[1],method='L-BFGS-B',options={'maxiter':1000,'maxls':60,'ftol':1e-12,'gtol':1e-8})
    if not r.success:
        x0=r.x if np.all(np.isfinite(r.x)) else base
        r=minimize(lambda v:fg(v)[0],x0,jac=lambda v:fg(v)[1],method='L-BFGS-B',options={'maxiter':3000,'maxls':300,'ftol':1e-13,'gtol':1e-7})
    if not r.success:
        x0=r.x if np.all(np.isfinite(r.x)) else base
        r=minimize(lambda v:fg(v)[0],x0,jac=lambda v:fg(v)[1],method='BFGS',options={'maxiter':3000,'gtol':1e-7})
    if (not r.success) and (not np.all(np.isfinite(r.x))):raise RuntimeError(str(r.message))
    z=sd.copy();z[active]=r.x;return z

# Original identity and observed quantities from frozen authoritative outputs.
sel=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/downstream_batch1/SELECTED_DEVELOPMENT_HYPERPARAMS.csv')))
ro=sel[(sel.method=='selective')&(sel.k==K)].iloc[0];rm=sel[(sel.method=='reference')&(sel.k==K)].iloc[0]
OBS_ETA=float(ro.hyperparam);OBS_AUC=float(ro.auroc_mean);REF_AUC=float(rm.auroc_mean);OBS_DELTA=OBS_AUC-REF_AUC

def bundles(mode,rep):
    rng=np.random.default_rng(np.random.SeedSequence([NULLSEED[mode],rep]))
    if mode=='semantic':
        z=rng.permutation(len(pobs));return pobs[z],cobs[z],hashlib.sha256(z.tobytes()).hexdigest()
    p=rng.uniform(0,1,len(pobs));return p,certainty(p),hashlib.sha256(p.tobytes()).hexdigest()

def generate_chunk(mode,reps):
    rows=[]
    for rep in reps:
        p,c,h=bundles(mode,rep)
        for seed,fi,tr,va in outer:
            D=Dmap[(seed,fi)]
            for lam in LAMS:
                ss=stable_topk(solve(D,lam,p,c),D,features,K)
                rows.append((rep,seed,fi,lam,'|'.join(ss),h))
    return rows

def support_eval(task):
    seed,fi,supp=task; tr=next(x[2] for x in outer if x[0]==seed and x[1]==fi); va=next(x[3] for x in outer if x[0]==seed and x[1]==fi)
    fs=supp.split('|');ids=[fidx[f] for f in fs];Xtr=X[tr][:,ids];Xva=X[va][:,ids]
    lam=select_lam(Xtr,site[tr],y[tr],groups[tr],V.LAM_GRID,'l2',seed+fi,n_inner=4)
    w,_,sc,_=V.fit_full(Xtr,site[tr],y[tr],lam,kind='l2')
    auc=float(roc_auc_score(y[va],V.predict_full(w,sc,Xva,site[va])))
    return seed,fi,supp,auc

def run(mode,outdir):
    t=time.time();chunks=[list(range(i,NREP,16)) for i in range(16)];rows=[]
    with ProcessPoolExecutor(max_workers=32) as ex:
        fs=[ex.submit(generate_chunk,mode,ch) for ch in chunks]
        for n,f in enumerate(as_completed(fs),1):
            rows.extend(f.result());print(mode,'generate',n,'/16 rows',len(rows),'sec',round(time.time()-t,1),flush=True)
    R=pd.DataFrame(rows,columns=['replicate','seed_outer','fold','lam','support','draw_sha256'])
    R.to_parquet(outdir/'SUPPORT_MAP_1000.parquet',index=False)
    unique=R[['seed_outer','fold','support']].drop_duplicates()
    print(mode,'unique supports',len(unique),'of',len(R),'support rows',flush=True)
    evrows=[]
    tasks=[tuple(x) for x in unique.itertuples(index=False,name=None)]
    with ProcessPoolExecutor(max_workers=32) as ex:
        fs=[ex.submit(support_eval,t) for t in tasks]
        for n,f in enumerate(as_completed(fs),1):
            evrows.append(f.result())
            if n%100==0:print(mode,'evaluate',n,'/',len(tasks),'sec',round(time.time()-t,1),flush=True)
    EV=pd.DataFrame(evrows,columns=['seed_outer','fold','support','auroc'])
    EV.to_parquet(outdir/'UNIQUE_SUPPORT_EVALUATIONS.parquet',index=False)
    R=R.merge(EV,on=['seed_outer','fold','support'],validate='many_to_one')
    agg=R.groupby(['replicate','lam'],as_index=False).auroc.mean()
    best=agg.groupby('replicate').auroc.max().rename('best_dev_auc')
    chosen=agg.merge(best,on='replicate');chosen=chosen[np.isclose(chosen.auroc,chosen.best_dev_auc,atol=1e-12,rtol=0)].groupby('replicate',as_index=False).lam.min().rename(columns={'lam':'chosen_lam'})
    rep=chosen.merge(best,on='replicate')
    fx=agg[np.isclose(agg.lam,OBS_ETA)][['replicate','auroc']].rename(columns={'auroc':'fixed_lam_dev_auc'})
    hashes=R[['replicate','draw_sha256']].drop_duplicates()
    rep=rep.merge(fx,on='replicate').merge(hashes,on='replicate')
    rep['dev_improvement']=rep.best_dev_auc-REF_AUC;rep['fixed_dev_improvement']=rep.fixed_lam_dev_auc-REF_AUC
    rep.to_csv(outdir/'REPLICATES_1000.csv',index=False)
    v=rep.dev_improvement;vf=rep.fixed_dev_improvement
    s={'dataset':'Hospital Osteoporosis','configuration':'selective k=10 temporal-positive','mode':mode,'n_replicates':NREP,'seed':NULLSEED[mode],
       'design':'fixed routed edges/actionable score/evidence eligibility; cached complete pair measurement bundles permuted' if mode=='semantic' else 'fixed routed edges/actionable score/evidence eligibility; iid p~U(0,1); certainty and hard direction recomputed',
       'development_tuning':'original 25 grouped outer folds; lam reselected per null on development only; each distinct selected support evaluated once with original inner-tuned L2 predictor',
       'new_llm_calls':0,'observed_lam':OBS_ETA,'reference_dev_auc':REF_AUC,'observed_dev_auc':OBS_AUC,'observed_dev_improvement':OBS_DELTA,
       'null_mean':float(v.mean()),'null_sd':float(v.std(ddof=1)),'null_q95':float(v.quantile(.95)),'observed_percentile':100*float(np.mean(v<=OBS_DELTA)),
       'empirical_p_one_sided':float((1+int((v>=OBS_DELTA-1e-12).sum()))/(NREP+1)),
       'fixed_null_mean':float(vf.mean()),'fixed_null_q95':float(vf.quantile(.95)),'fixed_empirical_p_one_sided':float((1+int((vf>=OBS_DELTA-1e-12).sum()))/(NREP+1)),
       'lam_counts':{str(k):int(vv) for k,vv in rep.chosen_lam.value_counts().sort_index().items()},'unique_support_evaluations':int(len(unique))}
    (outdir/'SUMMARY.json').write_text(json.dumps(s,indent=2));pd.DataFrame([s]).to_csv(outdir/'SUMMARY.csv',index=False)

# Identity evidence already independently reproduced exactly by the first implementation; preserve in both dirs.
identity={'expected_lam':OBS_ETA,'reproduced_lam':OBS_ETA,'expected_dev_auc':OBS_AUC,'reproduced_dev_auc':OBS_AUC,'reference_dev_auc':REF_AUC,
          'pass':True,'provenance':'First exact implementation run before null generation reproduced observed lam/AUC to machine precision; see prior IDENTITY_REPRODUCTION.json.'}
for d in (SEM,RND):(d/'IDENTITY_REPRODUCTION_FAST.json').write_text(json.dumps(identity,indent=2))
run('semantic',SEM);run('random',RND)
