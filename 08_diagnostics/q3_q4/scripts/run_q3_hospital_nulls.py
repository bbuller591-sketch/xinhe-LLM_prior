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
import sys,json,math,hashlib,time,warnings
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,ndtri
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings('ignore')

ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
DATA=Path(str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917'))
OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4'))
SEM=OUT/'01_q3_semantic_shuffle/HOSPITAL_OSTEOPOROSIS';RND=OUT/'02_q3_random_null/HOSPITAL_OSTEOPOROSIS'
SEM.mkdir(parents=True,exist_ok=True);RND.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(DATA/'scripts'))
import v2_core as V
from common import SEEDS,N_FOLDS
LAMS=[0.0,0.03,0.1,0.3,1.0,3.0,10.0];K=10;NREP=1000
SEEDS_NULL={'semantic':2026092203,'random':2026092204}

def folds(y,g,seed,n_splits):
    return list(StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))
def metrics(y,p):
    return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,(p>=.5).astype(int))))
def select_lam(Xr,s,y,g,grid,kind,seed,n_inner=4):
    best,best_lam=-np.inf,grid[0]
    for lam in grid:
        aucs=[]
        for tri,vai in folds(y,g,seed+int(round(lam*100)),n_inner):
            if len(np.unique(y[tri]))<2 or len(np.unique(y[vai]))<2:continue
            w,_,sc,_=V.fit_full(Xr[tri],s[tri],y[tri],lam,kind=kind)
            aucs.append(roc_auc_score(y[vai],V.predict_full(w,sc,Xr[vai],s[vai])))
        m=float(np.mean(aucs)) if aucs else -np.inf
        if m>best:best,best_lam=m,lam
    return float(best_lam)
def selective_anchor(D):
    r=rankdata(-np.asarray(D,float),method='average');u=1-(r-.5)/len(D)
    return ndtri(np.clip(u,1e-6,1-1e-6))
def stable_topk(score,D,features,k):
    z=pd.DataFrame({'feature':features,'score':score,'D':D,'idx':np.arange(len(features))})
    return tuple(z.sort_values(['score','D','idx'],ascending=[False,False,True],kind='mergesort').feature.iloc[:k])
def certainty(p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12);H=-(p*np.log(p)+(1-p)*np.log(1-p))/math.log(2)
    return 1-H

man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json'));features=list(man['selectable_features'])
full=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(
    pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),on=['patient_uid','site']).reset_index(drop=True)
prim=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
X=prim[features].to_numpy(float);site=prim.site.to_numpy();y=prim.y.to_numpy(int);groups=prim.patient_uid.to_numpy()
assert len(prim)==892 and len(features)==37
# Recreate frozen outer list and load exact frozen selector D by seed/fold.
outer=[]
for seed in SEEDS:
    for fi,(tri,vai) in enumerate(folds(y,groups,seed,N_FOLDS)):outer.append((int(seed),int(fi),tri,vai))
scoretab=pd.read_csv(ROOT/'09_BATCH1_DOWNSTREAM_V1_9/OUTER_SELECTOR_SCORES.csv')
Dmap={}
for seed,fi,tri,vai in outer:
    q=scoretab[(scoretab.seed==seed)&(scoretab.fold==fi)].set_index('feature')
    Dmap[(seed,fi)]=q.loc[features].D.to_numpy(float)

# k=10 fixed route and true cached bundle.
pairk=pd.read_csv(ROOT/'09_BATCH1_DOWNSTREAM_V1_9/selective_PAIRK_WEIGHTS_FROZEN.csv')
q10=pairk[(pairk.k==K)&(pairk.evidence_gate==1)].copy().reset_index(drop=True)
assert len(q10)>0 and q10.pair_id.is_unique
pairs=q10.pair_id.astype(str).tolist()
p_obs=q10.p_selective_A.to_numpy(float);c_obs=q10.c_pair.to_numpy(float)
A=q10.actionable_boundary_score.to_numpy(float)
idx={f:i for i,f in enumerate(features)}
ii=np.array([idx[f] for f in q10.feature_A],int);jj=np.array([idx[f] for f in q10.feature_B],int)
active=np.unique(np.r_[ii,jj]);apos={v:k for k,v in enumerate(active)};li=np.array([apos[v] for v in ii]);lj=np.array([apos[v] for v in jj])

def solve(D,lam,p,c):
    sd=selective_anchor(D)
    if lam==0:return sd
    w=A*c;W=float(w.sum())
    if W<=0:return sd
    base=sd[active].copy();yy=(p>.5).astype(float);P=len(sd)
    def fg(v):
        d=v[li]-v[lj];ce=np.logaddexp(0,d)-yy*d;diff=v-base
        loss=.5*np.dot(diff,diff)/P+lam*np.dot(w,ce)/(W+1e-12)
        grad=diff/P;rr=lam*(w/(W+1e-12))*(expit(d)-yy);np.add.at(grad,li,rr);np.add.at(grad,lj,-rr)
        return float(loss),grad
    r=minimize(lambda v:fg(v)[0],base,jac=lambda v:fg(v)[1],method='BFGS',options={'maxiter':1000,'gtol':1e-8})
    if not r.success:
        r=minimize(lambda v:fg(v)[0],r.x,jac=lambda v:fg(v)[1],method='L-BFGS-B',options={'maxiter':2000,'gtol':1e-8,'ftol':1e-12})
    if not r.success:raise RuntimeError('hospital optimize failed '+str(r.message))
    out=sd.copy();out[active]=r.x;return out

eval_cache={}
def eval_support(seed,fi,tri,vai,ss):
    key=(seed,fi,tuple(ss))
    if key in eval_cache:return eval_cache[key]
    inds=[features.index(f) for f in ss];Xtr=X[tri][:,inds];Xva=X[vai][:,inds]
    lam=select_lam(Xtr,site[tri],y[tri],groups[tri],V.LAM_GRID,'l2',seed+fi,n_inner=4)
    w,info,sc,_=V.fit_full(Xtr,site[tri],y[tri],lam,kind='l2')
    pv=V.predict_full(w,sc,Xva,site[vai]);val=float(roc_auc_score(y[vai],pv))
    eval_cache[key]=val;return val

def curve(p,c):
    vals=[]
    for lam in LAMS:
        aa=[]
        for seed,fi,tri,vai in outer:
            D=Dmap[(seed,fi)];z=solve(D,lam,p,c);ss=stable_topk(z,D,features,K)
            aa.append(eval_support(seed,fi,tri,vai,ss))
        vals.append((lam,float(np.mean(aa))))
    best=max(v for _,v in vals);chosen=min(e for e,v in vals if np.isclose(v,best,atol=1e-12,rtol=0))
    return chosen,best,float(dict(vals)[OBS_ETA]),vals

sel=pd.read_csv(ROOT/'09_BATCH1_DOWNSTREAM_V1_9/SELECTED_DEVELOPMENT_HYPERPARAMS.csv')
r=sel[(sel.method=='selective')&(sel.k==K)].iloc[0];OBS_ETA=float(r.hyperparam);OBS_AUC=float(r.auroc_mean)
reference=sel[(sel.method=='reference')&(sel.k==K)].iloc[0];REF_AUC=float(reference.auroc_mean)
ig,ia,ifix,icurve=curve(p_obs,c_obs)
identity={'expected_lam':OBS_ETA,'reproduced_lam':ig,'expected_dev_auc':OBS_AUC,'reproduced_dev_auc':ia,'reference_dev_auc':REF_AUC,
          'pass':bool(np.isclose(ig,OBS_ETA)&np.isclose(ia,OBS_AUC,atol=1e-12))}
for d in [SEM,RND]:(d/'IDENTITY_REPRODUCTION.json').write_text(json.dumps(identity,indent=2))
if not identity['pass']:raise RuntimeError('hospital identity reproduction failed')
OBS_DELTA=OBS_AUC-REF_AUC

def run(mode,outdir):
    rows=[];seed0=SEEDS_NULL[mode];t0=time.time()
    for rep in range(NREP):
        rng=np.random.default_rng(np.random.SeedSequence([seed0,rep]))
        if mode=='semantic':
            perm=rng.permutation(len(p_obs));p=p_obs[perm];c=c_obs[perm];hh=hashlib.sha256(perm.tobytes()).hexdigest()
        else:
            p=rng.uniform(0,1,len(p_obs));c=certainty(p);hh=hashlib.sha256(p.tobytes()).hexdigest()
        chosen,best,fixed,_=curve(p,c)
        rows.append({'replicate':rep,'mode':mode,'draw_sha256':hh,'chosen_lam':chosen,'best_dev_auc':best,'dev_improvement':best-REF_AUC,
                     'fixed_observed_lam':OBS_ETA,'fixed_lam_dev_auc':fixed,'fixed_dev_improvement':fixed-REF_AUC})
        if (rep+1)%25==0:
            pd.DataFrame(rows).to_csv(outdir/'PARTIAL.csv',index=False);print(mode,rep+1,'evalcache',len(eval_cache),'sec',round(time.time()-t0,1),flush=True)
    R=pd.DataFrame(rows);R.to_csv(outdir/'REPLICATES_1000.csv',index=False);v=R.dev_improvement;fx=R.fixed_dev_improvement
    S={'dataset':'Hospital Osteoporosis','configuration':'selective k=10 temporal-positive','mode':mode,'n_replicates':NREP,'seed':seed0,
       'design':'fixed k=10 routed edges/actionable scores/evidence eligibility; cached complete (p,c) bundle permutation' if mode=='semantic' else 'fixed k=10 routed edges/actionable scores/evidence eligibility; iid p~U(0,1), certainty recomputed; hard target derived as 1[p>0.5]',
       'development_tuning':'original 25 grouped outer folds; lam reselected per replicate on development only','new_llm_calls':0,
       'observed_lam':OBS_ETA,'reference_dev_auc':REF_AUC,'observed_dev_auc':OBS_AUC,'observed_dev_improvement':OBS_DELTA,
       'null_mean':float(v.mean()),'null_sd':float(v.std(ddof=1)),'null_q95':float(v.quantile(.95)),
       'observed_percentile':100*float(np.mean(v<=OBS_DELTA)),'empirical_p_one_sided':float((1+int((v>=OBS_DELTA-1e-12).sum()))/(NREP+1)),
       'fixed_null_mean':float(fx.mean()),'fixed_null_q95':float(fx.quantile(.95)),
       'fixed_empirical_p_one_sided':float((1+int((fx>=OBS_DELTA-1e-12).sum()))/(NREP+1)),
       'lam_counts':{str(k):int(vv) for k,vv in R.chosen_lam.value_counts().sort_index().items()}}
    (outdir/'SUMMARY.json').write_text(json.dumps(S,indent=2));pd.DataFrame([S]).to_csv(outdir/'SUMMARY.csv',index=False)

run('semantic',SEM);run('random',RND)
