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
import sys,io,json,hashlib,math,warnings
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from concurrent.futures import ProcessPoolExecutor,as_completed

warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
BASE=Path(str(REPRO_ROOT / 'dataset_screening_20260921'))
DOWN=ROOT/'formal_outputs/02_downstream_dev_v3_2'
MEAS=ROOT/'formal_outputs/measurements/v3_2'
FREEZE=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2/FINAL'
EXTFREEZE=ROOT/'formal_outputs/04_pre_external_final_freeze_v3_2'
OUT=ROOT/'formal_outputs/07_random_llm_probability_null_v3_2';OUT.mkdir(parents=True,exist_ok=True)
SEED=20261044;MODEL_SEED=20260921;GRID=[0.,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.];K=50;P=2000;NREP=1000

# Formal observed results are frozen.
obs=json.loads((DOWN/'PRE_EXTERNAL_FREEZE_SUMMARY.json').read_text())
obs_ext=json.loads((ROOT/'formal_outputs/05_external_evaluation_v3_2/EXTERNAL_EVALUATION_SUMMARY.json').read_text())
assert obs['external_used'] is False
assert obs_ext['no_post_external_tuning'] is True

sys.path.insert(0,str(BASE/'src'))
from geo_utils import read_geo_metadata,read_geo_expression

# Frozen GPL570 mapping.
txt=(BASE/'data/GPL570_full.txt').read_text(errors='replace').replace('\r','')
block=txt.split('!platform_table_begin\n',1)[1].split('!platform_table_end',1)[0]
ann=pd.read_csv(io.StringIO(block),sep='\t',dtype=str)[['ID','Gene Symbol']].dropna()
ann=ann[~ann['Gene Symbol'].isin(['---',''])]
ann['Gene Symbol']=ann['Gene Symbol'].str.split(' /// ').str[0].str.strip()
probe2gene=dict(zip(ann.ID.astype(str),ann['Gene Symbol'].astype(str)))

def load(acc):
    m=read_geo_metadata(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz'))
    e=read_geo_expression(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz')).set_index('feature_id')
    if acc=='GSE36059':
        lab=m['diagnosis_tcmr_abmr_mixed_non_rejecting'].astype(str)
        m=m[lab.isin(['TCMR','ABMR','MIXED','non-rejecting'])].copy()
        m['y']=(m['diagnosis_tcmr_abmr_mixed_non_rejecting']=='TCMR').astype(int)
    else:
        lab=m['diagnosis_tcmr_non_tcmr_nephrectomies'].astype(str)
        m=m[lab.isin(['TCMR','non-TCMR'])].copy()
        m['y']=(m['diagnosis_tcmr_non_tcmr_nephrectomies']=='TCMR').astype(int)
    sam=[s for s in m.geo_accession if s in e.columns]
    m=m.set_index('geo_accession').loc[sam]
    sub=e[sam].copy();sub['gene']=[probe2gene.get(str(i),'') for i in sub.index];sub=sub[sub.gene!='']
    ge=sub.groupby('gene',sort=False).median(numeric_only=True)
    return m,ge,sam

md,gd,sd=load('GSE36059')
me,ge,se=load('GSE48581')
genes=pd.read_csv(ROOT/'CANDIDATE_UNIVERSE_FROZEN.csv').gene.astype(str).to_numpy()
Xd=gd.loc[genes,sd].T.to_numpy(float);yd=md.y.to_numpy(int)
Xe=ge.loc[genes,se].T.to_numpy(float);ye=me.y.to_numpy(int)
gidx={g:i for i,g in enumerate(genes)}
med=np.nanmedian(Xd,axis=0)
for A in [Xd,Xe]:
    rr,cc=np.where(~np.isfinite(A));A[rr,cc]=med[cc]
assert Xd.shape==(403,2000) and Xe.shape==(300,2000)

# Frozen fold-local scores/splits from formal observed tuning.
folds=[]
for f in range(1,6):
    q=np.load(DOWN/f'CV_FOLD{f}_DATA_SCORE.npz',allow_pickle=True)
    tr=q['train_idx'].astype(int);va=q['val_idx'].astype(int);raw=q['raw'].astype(float);sd0=q['sD'].astype(float)
    fmed=q['median_impute'].astype(float)
    Atr=Xd[tr].copy();Ava=Xd[va].copy()
    rr,cc=np.where(~np.isfinite(Atr));Atr[rr,cc]=fmed[cc]
    rr,cc=np.where(~np.isfinite(Ava));Ava[rr,cc]=fmed[cc]
    folds.append((f,tr,va,Atr,Ava,raw,sd0))

# Full development Reference score (already frozen).
refscore=pd.read_csv(DOWN/'REFERENCE_SCORES_FROZEN.csv')
assert refscore.gene.astype(str).tolist()==list(genes)
Dfull=refscore.reference_raw_score.to_numpy(float)
sDfull=refscore.reference_score_normalized.to_numpy(float)

# Routing tables.
selroute=pd.read_csv(FREEZE/'SELECTIVE_PAIRSET_FROZEN_V3_2.csv')
glroute=pd.read_csv(FREEZE/'GLOBAL_PAIRSET_FROZEN_V3_2.csv')
def edge_arrays(route):
    ii=np.asarray([gidx[g] for g in route.feature_i],int)
    jj=np.asarray([gidx[g] for g in route.feature_j],int)
    nodes=np.unique(np.r_[ii,jj]);pos={v:k for k,v in enumerate(nodes)}
    li=np.asarray([pos[x] for x in ii]);lj=np.asarray([pos[x] for x in jj])
    return ii,jj,nodes,li,lj
SII,SJJ,SN,SLI,SLJ=edge_arrays(selroute)
GII,GJJ,GN,GLI,GLJ=edge_arrays(glroute)
U=selroute.U.to_numpy(float)

def certainty(p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    H=-(p*np.log(p)+(1-p)*np.log(1-p))
    return 1-H/math.log(2)

def correct(sd,p,c,lam,method):
    if method=='Global':
        nodes,li,lj,w=GN,GLI,GLJ,c
    elif method=='selective':
        nodes,li,lj,w=SN,SLI,SLJ,U*c
    elif method=='selective_no_certainty-NoC':
        nodes,li,lj,w=SN,SLI,SLJ,U
    else:raise ValueError(method)
    if lam==0 or float(np.sum(w))<=0:return sd.copy()
    ws=float(np.sum(w));base=sd[nodes].copy()
    def fg(v):
        z=v[li]-v[lj]
        ce=np.logaddexp(0.0,z)-p*z
        diff=v-base
        f=.5*np.sum(diff*diff)/P+lam*float(np.dot(w,ce))/(ws+1e-12)
        sig=1/(1+np.exp(-np.clip(z,-50,50)))
        gr=diff/P;vv=lam*(w/(ws+1e-12))*(sig-p)
        np.add.at(gr,li,vv);np.add.at(gr,lj,-vv)
        return float(f),gr
    r=minimize(lambda v:fg(v)[0],base,jac=lambda v:fg(v)[1],method='L-BFGS-B',
               options={'maxiter':300,'ftol':1e-11,'gtol':1e-8})
    s=sd.copy();s[nodes]=r.x
    return s

def topk(s,raw):return np.lexsort((np.arange(P),-raw,-s))[:K]

def dev_auc(fold,ids,cache):
    f,tr,va,Atr,Ava,raw,sd0=fold
    key=(f,tuple(sorted(ids.tolist())))
    if key in cache:return cache[key]
    sc=StandardScaler().fit(Atr[:,ids])
    mdl=LogisticRegression(C=1.,penalty='l2',solver='liblinear',class_weight='balanced',
        max_iter=2000,random_state=SEED).fit(sc.transform(Atr[:,ids]),yd[tr])
    pr=mdl.predict_proba(sc.transform(Ava[:,ids]))[:,1]
    val=float(roc_auc_score(yd[va],pr));cache[key]=val;return val

def external_metrics(ids):
    sc=StandardScaler().fit(Xd[:,ids])
    mdl=LogisticRegression(C=1.,penalty='l2',solver='liblinear',class_weight='balanced',
        max_iter=2000,random_state=SEED).fit(sc.transform(Xd[:,ids]),yd)
    pr=mdl.predict_proba(sc.transform(Xe[:,ids]))[:,1]
    yh=(pr>=.5).astype(int)
    return float(roc_auc_score(ye,pr)),float(average_precision_score(ye,pr)),float(balanced_accuracy_score(ye,yh))

def generate_probs(rng,mode,n):
    if mode=='DIRECT_UNIFORM_PAIR_P':
        p=rng.random(n)
        return p,certainty(p)
    elif mode=='UNIFORM_ABBA_CALLS':
        pab=rng.random(n)
        pba_semantic_i=rng.random(n)  # directly p_BA(B), i.e. probability of semantic gene_i in BA presentation
        p=.5*(pab+pba_semantic_i)
        return p,certainty(p)
    else:raise ValueError(mode)

def run_rep(arg):
    mode,rep=arg
    rng=np.random.default_rng(SEED+700000+(0 if mode=='DIRECT_UNIFORM_PAIR_P' else 1000000)+rep)
    ps,cs=generate_probs(rng,mode,len(selroute))
    pg,cg=generate_probs(rng,mode,len(glroute))
    out={'mode':mode,'replicate':rep,
         'selective_random_p_sha256':hashlib.sha256(ps.tobytes()).hexdigest(),
         'global_random_p_sha256':hashlib.sha256(pg.tobytes()).hexdigest()}
    cache={}
    selected={}
    for method in ['Global','selective','selective_no_certainty-NoC']:
        p,c=(pg,cg) if method=='Global' else (ps,cs)
        vals=[]
        for lam in GRID:
            av=[]
            for fold in folds:
                _,_,_,_,_,raw,sd0=fold
                s=correct(sd0,p,c,lam,method);ids=topk(s,raw)
                av.append(dev_auc(fold,ids,cache))
            vals.append((lam,float(np.mean(av))))
        best=max(v for _,v in vals);chosen=min(l for l,v in vals if np.isclose(v,best,atol=1e-12,rtol=0))
        # Only after development-only lambda choice, form full-dev support and evaluate external.
        sfull=correct(sDfull,p,c,chosen,method);ids=topk(sfull,Dfull)
        eauroc,eauprc,ebal=external_metrics(ids)
        pref={'Global':'global','selective':'selective','selective_no_certainty-NoC':'m4'}[method]
        out.update({f'{pref}_chosen_lambda':chosen,f'{pref}_dev_best_auroc':best,
                    f'{pref}_external_auroc':eauroc,f'{pref}_external_auprc':eauprc,
                    f'{pref}_external_balacc':ebal,
                    f'{pref}_support_sha256':hashlib.sha256('\n'.join(genes[ids]).encode()).hexdigest()})
    return out

modes=['DIRECT_UNIFORM_PAIR_P','UNIFORM_ABBA_CALLS']
jobs=[(mode,r) for mode in modes for r in range(NREP)]
rows=[]
with ProcessPoolExecutor(max_workers=8) as ex:
    fs={ex.submit(run_rep,j):j for j in jobs}
    for n,f in enumerate(as_completed(fs),1):
        rows.append(f.result())
        if n%50==0:
            print(n,'/',len(jobs),flush=True)
            pd.DataFrame(rows).to_csv(OUT/'RANDOM_LLM_NULL_PARTIAL.csv',index=False)
R=pd.DataFrame(rows).sort_values(['mode','replicate'])
R.to_csv(OUT/'RANDOM_LLM_PROBABILITY_NULL_2000_ROWS.csv',index=False)

# Observed points from formal run.
obs_dev={}
agg=pd.read_csv(DOWN/'FOLDLOCAL_LAMBDA_CV_AGGREGATE.csv')
for m in ['Global','selective','selective_no_certainty-NoC']:
    lam=float(obs['selected_lambda'][m])
    obs_dev[m]=float(agg[(agg.method==m)&np.isclose(agg['lambda'],lam)].auroc.iloc[0])
obs_ext_m={m:float(obs_ext['metrics'][m]['auroc']) for m in ['Global','selective','selective_no_certainty-NoC']}
ref_dev=float(agg[(agg.method=='Reference')&(agg['lambda']==0)].auroc.iloc[0])
ref_ext=float(obs_ext['metrics']['Reference']['auroc'])

S={'version':'RANDOM_LLM_PROBABILITY_NULL_V3_2','n_per_mode':NREP,'modes':{},
   'design':'Fixed pair identities and fixed U_e. Randomly generate LLM probabilities instead of permuting observed probabilities. Re-select lambda on development CV only for each replicate; then evaluate that frozen random-guidance support on GSE48581. No new LLM calls.',
   'observed_reference':{'development_cv_auroc':ref_dev,'external_auroc':ref_ext},
   'observed_guided':{m:{'selected_lambda':float(obs['selected_lambda'][m]),'development_cv_auroc':obs_dev[m],
                         'external_auroc':obs_ext_m[m]} for m in ['Global','selective','selective_no_certainty-NoC']},
   'posthoc_diagnostic_note':'This null was requested after the formal external run and is a post-hoc diagnostic. External outcome is not used for lambda selection within any replicate.'}
for mode in modes:
    z=R[R['mode']==mode]
    S['modes'][mode]={}
    for m,pref in [('Global','global'),('selective','selective'),('selective_no_certainty-NoC','m4')]:
        dv=z[f'{pref}_dev_best_auroc'];ev=z[f'{pref}_external_auroc']
        od=obs_dev[m];oe=obs_ext_m[m]
        S['modes'][mode][m]={
          'random_p_generation':'p_e~Uniform(0,1)' if mode=='DIRECT_UNIFORM_PAIR_P' else 'p_AB(A),p_BA(B) iid Uniform(0,1), then p_e=0.5*(p_AB(A)+p_BA(B))',
          'null_dev_mean_auroc':float(dv.mean()),'null_dev_sd':float(dv.std(ddof=1)),
          'null_dev_q05':float(dv.quantile(.05)),'null_dev_q50':float(dv.quantile(.5)),'null_dev_q95':float(dv.quantile(.95)),
          'observed_dev_auroc':od,'empirical_p_dev_ge_observed':float((1+int((dv>=od-1e-12).sum()))/(NREP+1)),
          'null_external_mean_auroc':float(ev.mean()),'null_external_sd':float(ev.std(ddof=1)),
          'null_external_q05':float(ev.quantile(.05)),'null_external_q50':float(ev.quantile(.5)),'null_external_q95':float(ev.quantile(.95)),
          'observed_external_auroc':oe,'empirical_p_external_ge_observed':float((1+int((ev>=oe-1e-12).sum()))/(NREP+1)),
          'null_external_mean_delta_vs_ref':float((ev-ref_ext).mean()),
          'observed_external_delta_vs_ref':float(oe-ref_ext),
          'lambda_counts':{str(k):int(v) for k,v in z[f'{pref}_chosen_lambda'].value_counts().sort_index().items()}
        }
(OUT/'RANDOM_LLM_PROBABILITY_NULL_SUMMARY.json').write_text(json.dumps(S,indent=2)+'\n')
with (OUT/'SHA256SUMS.txt').open('w') as h:
    for p in [OUT/'RANDOM_LLM_PROBABILITY_NULL_2000_ROWS.csv',OUT/'RANDOM_LLM_PROBABILITY_NULL_SUMMARY.json']:
        h.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print(json.dumps(S,indent=2))
