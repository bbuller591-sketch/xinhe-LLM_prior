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
import json, math, hashlib, time
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SRC=Path(str(REPRO_ROOT / '04_credit_g'))
OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4'))
SEM=OUT/'01_q3_semantic_shuffle/CREDIT_G'; RND=OUT/'02_q3_random_null/CREDIT_G'
SEM.mkdir(parents=True,exist_ok=True); RND.mkdir(parents=True,exist_ok=True)
FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[x for x in FEATURES if x not in CAT]
LAMBDAS=[0,0.025,0.05,0.075,0.10,0.15,0.20,0.30]
TARGETS=['GBM_PERM','ELASTIC_NET']; K=10; NREP=1000
DEV_SEED=20260918
SEEDS={'semantic':2026092201,'random':2026092202}

def prep(cols):
    cols=list(cols); num=[c for c in NUM if c in cols]; cat=[c for c in CAT if c in cols]; tr=[]
    if num: tr.append(('num',StandardScaler(),num))
    if cat: tr.append(('cat',Pipeline([('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),('scale',StandardScaler())]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)

def fit_eval(Xtr,ytr,Xev,yev,sel):
    sel=list(sel)
    pipe=Pipeline([('prep',prep(sel)),('clf',LogisticRegression(penalty='l2',solver='lbfgs',C=0.01,max_iter=5000,tol=1e-5))])
    pipe.fit(Xtr[sel],ytr)
    return float(roc_auc_score(yev,pipe.predict_proba(Xev[sel])[:,1]))

def entropy_c(p):
    p=float(np.clip(p,1e-12,1-1e-12)); H=-(p*math.log(p)+(1-p)*math.log(1-p))
    return 1-H/math.log(2)

X=pd.read_csv(SRC/'01_DATA_AND_SPLITS/X.csv'); y=pd.read_csv(SRC/'01_DATA_AND_SPLITS/y.csv')['label'].to_numpy()
dev=np.load(SRC/'01_DATA_AND_SPLITS/modern_dev_indices_seed20260918.npy'); hold=np.load(SRC/'01_DATA_AND_SPLITS/modern_holdout_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]; Xh=X.iloc[hold].reset_index(drop=True); yh=y[hold]
ranks=pd.read_csv(SRC/'06_RESULTS/reference_RESAMPLE_RANKS.csv')
m0res=pd.read_csv(SRC/'06_RESULTS/reference_RESAMPLE_RESULTS.csv')
graph=pd.read_csv(SRC/'02_PROTOCOLS/selective_SELECTIVE_GRAPH_FREEZE.csv')
pair=pd.read_csv(SRC/'06_RESULTS/selective_PAIR_MEASUREMENT.csv')
fullrank=pd.read_csv(SRC/'06_RESULTS/FINAL_DATA_ONLY_RANKINGS_FREEZE.csv')
auth_lam=json.loads((SRC/'06_RESULTS/GAMMA_FREEZE.json').read_text())['chosen_lam']
auth_dev=pd.read_csv(SRC/'06_RESULTS/GAMMA_SELECTION_SUMMARY.csv')
auth_hold=pd.read_csv(SRC/'06_RESULTS/FINAL_HOLDOUT_RESULTS.csv')

# fixed 300 development splits exactly as original
outer=list(StratifiedShuffleSplit(n_splits=300,train_size=.8,test_size=.2,random_state=DEV_SEED).split(Xd,yd))
# data utilities by selector/resample
udata={}
for rr in range(300):
    for s in TARGETS:
        q=ranks[(ranks['resample']==rr)&(ranks.selector==s)].set_index('feature')
        rmap=q['rank'].to_dict()
        udata[(rr,s)]={f:(len(FEATURES)-rmap[f])/(len(FEATURES)-1) for f in FEATURES}

# pair measurement orientation table and graph relation
P=pair.copy()
assert len(P)==3
pair_ids=P.pair_id.astype(str).tolist()
p_obs=P.p_canonical_a_order_neutral.to_numpy(float); c_obs=P.c_edge_entropy.to_numpy(float)
pm_index={pid:i for i,pid in enumerate(pair_ids)}

def hmap(selector,pvals,cvals):
    key=selector.lower(); num={f:0.0 for f in FEATURES}; den={f:0.0 for f in FEATURES}
    for r in graph.itertuples():
        if not bool(r.formal_selective_measurement_eligible) or not bool(getattr(r,f'requested_{key}')): continue
        k=pm_index[str(r.pair_id)]
        p=float(pvals[k]); c=float(cvals[k]); u=float(getattr(r,f'u_data_{key}')); w=u*c; sig=2*p-1
        num[r.feature_a]+=w*sig; den[r.feature_a]+=w
        num[r.feature_b]-=w*sig; den[r.feature_b]+=w
    return {f:(num[f]/den[f] if den[f]>0 else 0.0) for f in FEATURES}

# cache support evaluation across all null replicates
dev_eval_cache={}
def dev_auc(rr,sel):
    key=(rr,tuple(sel))
    if key not in dev_eval_cache:
        tr,va=outer[rr]
        dev_eval_cache[key]=fit_eval(Xd.iloc[tr].reset_index(drop=True),yd[tr],Xd.iloc[va].reset_index(drop=True),yd[va],sel)
    return dev_eval_cache[key]

hold_cache={}
def hold_auc(sel):
    key=tuple(sel)
    if key not in hold_cache: hold_cache[key]=fit_eval(Xd,yd,Xh,yh,sel)
    return hold_cache[key]

full_u={}
for s in TARGETS:
    q=fullrank[fullrank.selector==s].set_index('feature')
    rmap=q.data_rank.to_dict()
    full_u[s]={f:(len(FEATURES)-rmap[f])/(len(FEATURES)-1) for f in FEATURES}

def support_from_u(u,h,lam):
    util={f:u[f]+lam*h.get(f,0.0) for f in FEATURES}
    return tuple(sorted(sorted(FEATURES,key=lambda f:(-util[f],FEATURES.index(f)))[:K],key=FEATURES.index))

def evaluate_bundle(pvals,cvals):
    out={}
    for s in TARGETS:
        h=hmap(s,pvals,cvals)
        curves=[]
        for g in LAMBDAS:
            av=[]
            for rr in range(300):
                ss=support_from_u(udata[(rr,s)],h,g)
                av.append(dev_auc(rr,ss))
            curves.append((g,float(np.mean(av))))
        best=max(v for _,v in curves); chosen=min(g for g,v in curves if v>=best-1e-4)
        fixed=float(dict(curves)[float(auth_lam[s]['selective'])])
        fs=support_from_u(full_u[s],h,chosen)
        out[s]={'chosen_lam':chosen,'best_dev_auc':best,'fixed_obs_lam_dev_auc':fixed,'full_support':'|'.join(fs),'holdout_auc':hold_auc(fs)}
    return out

# Identity gate: exact cached measurement must reproduce observed selected lam and metrics.
identity=evaluate_bundle(p_obs,c_obs)
checks=[]
for s in TARGETS:
    obs_g=float(auth_lam[s]['selective'])
    obs_dev=float(auth_dev[(auth_dev.selector==s)&(auth_dev.method=='selective')& (auth_dev.selected_lam==True)].iloc[0].mean_auc)
    obs_hold=float(auth_hold[(auth_hold.selector==s)&(auth_hold.method=='selective')].iloc[0].holdout_auroc)
    checks.append({'selector':s,'lam_expected':obs_g,'lam_reproduced':identity[s]['chosen_lam'],
                   'dev_expected':obs_dev,'dev_reproduced':identity[s]['best_dev_auc'],
                   'holdout_expected':obs_hold,'holdout_reproduced':identity[s]['holdout_auc']})
idc=pd.DataFrame(checks)
idc['pass']=(np.isclose(idc.lam_expected,idc.lam_reproduced)&np.isclose(idc.dev_expected,idc.dev_reproduced,atol=1e-12)&np.isclose(idc.holdout_expected,idc.holdout_reproduced,atol=1e-12))
idc.to_csv(SEM/'IDENTITY_REPRODUCTION.csv',index=False)
(RND/'IDENTITY_REPRODUCTION.csv').write_bytes((SEM/'IDENTITY_REPRODUCTION.csv').read_bytes())
if not idc['pass'].all(): raise RuntimeError('CREDIT identity reproduction failed')

obs={}
for s in TARGETS:
    base=float(auth_dev[(auth_dev.selector==s)&(auth_dev.method=='selective')&np.isclose(auth_dev.lam,0)].iloc[0].mean_auc)
    obs[s]={'reference_dev_auc':base,'observed_dev_auc':identity[s]['best_dev_auc'],'observed_dev_improvement':identity[s]['best_dev_auc']-base,
            'observed_lam':identity[s]['chosen_lam'],'observed_holdout_auc':identity[s]['holdout_auc']}

def run(mode,outdir):
    rows=[]; seed=SEEDS[mode]; t=time.time()
    for rep in range(NREP):
        rng=np.random.default_rng(np.random.SeedSequence([seed,rep]))
        if mode=='semantic':
            perm=rng.permutation(len(p_obs)); pv=p_obs[perm]; cv=c_obs[perm]; token=hashlib.sha256(perm.tobytes()).hexdigest()
        else:
            pv=rng.uniform(0,1,len(p_obs)); cv=np.array([entropy_c(x) for x in pv]); token=hashlib.sha256(pv.tobytes()).hexdigest()
        z=evaluate_bundle(pv,cv)
        for s in TARGETS:
            base=obs[s]['reference_dev_auc']
            rows.append({'replicate':rep,'selector':s,'mode':mode,'draw_sha256':token,
                         'chosen_lam':z[s]['chosen_lam'],'best_dev_auc':z[s]['best_dev_auc'],'dev_improvement':z[s]['best_dev_auc']-base,
                         'fixed_observed_lam':obs[s]['observed_lam'],'fixed_observed_lam_dev_auc':z[s]['fixed_obs_lam_dev_auc'],
                         'fixed_dev_improvement':z[s]['fixed_obs_lam_dev_auc']-base,'holdout_auc_frozen_support':z[s]['holdout_auc'],
                         'full_support':z[s]['full_support']})
        if (rep+1)%100==0:
            pd.DataFrame(rows).to_csv(outdir/'PARTIAL.csv',index=False)
            print(mode,rep+1,'cache',len(dev_eval_cache),'elapsed',round(time.time()-t,1),flush=True)
    R=pd.DataFrame(rows); R.to_csv(outdir/'REPLICATES_1000.csv',index=False)
    summ=[]
    for s in TARGETS:
        q=R[R.selector==s]; O=obs[s]; v=q.dev_improvement; fx=q.fixed_dev_improvement
        summ.append({'dataset':'CREDIT-G','selector':s,'mode':mode,'n_replicates':NREP,'seed':seed,
                     'observed_lam':O['observed_lam'],'observed_dev_auc':O['observed_dev_auc'],'reference_dev_auc':O['reference_dev_auc'],
                     'observed_dev_improvement':O['observed_dev_improvement'],'null_mean':v.mean(),'null_sd':v.std(ddof=1),'null_q95':v.quantile(.95),
                     'observed_percentile':100*float(np.mean(v<=O['observed_dev_improvement'])),
                     'empirical_p_one_sided':(1+int((v>=O['observed_dev_improvement']-1e-12).sum()))/(NREP+1),
                     'fixed_null_mean':fx.mean(),'fixed_null_q95':fx.quantile(.95),
                     'fixed_empirical_p_one_sided':(1+int((fx>=O['observed_dev_improvement']-1e-12).sum()))/(NREP+1),
                     'observed_holdout_auc':O['observed_holdout_auc'],'null_holdout_mean':q.holdout_auc_frozen_support.mean(),
                     'trust_counts':json.dumps({str(k):int(vv) for k,vv in q.chosen_lam.value_counts().sort_index().items()})})
    S=pd.DataFrame(summ); S.to_csv(outdir/'SUMMARY.csv',index=False)
    (outdir/'SUMMARY.json').write_text(json.dumps({'design':'fixed routed edges/U/evidence eligibility; cached LLM bundle permuted' if mode=='semantic' else 'fixed routed edges/U/evidence eligibility; iid p~U(0,1), certainty recomputed from binary entropy',
        'development_tuning':'original 300 development resamples; lam reselected per replicate; within 1e-4 of best mean choose smallest lam',
        'final_evaluation':'holdout evaluated only after replicate development lam/support freeze; never used for tuning','new_llm_calls':0,
        'summary':summ},indent=2))
    return S

run('semantic',SEM)
run('random',RND)
