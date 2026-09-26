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
import sys,io,json,hashlib,math,os,warnings
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor,as_completed

warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
BASE=Path(str(REPRO_ROOT / 'dataset_screening_20260921'))
DOWN=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/02_RENAL/downstream_dev'))
MEAS=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/02_RENAL/measurements_compat'))
FREEZE=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2/FINAL'
OUT=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/02_RENAL/semantic_shuffle'));OUT.mkdir(parents=True,exist_ok=True)
SEED=20261044;MODEL_SEED=20260921;GRID=[0.,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.];K=50;P=2000;NREP=1000

# Frozen observed result.
summary=json.loads((DOWN/'PRE_EXTERNAL_FREEZE_SUMMARY.json').read_text())
assert summary['external_used'] is False and summary['semantic_shuffle_required'] is True
obs_lambda={'selective':float(summary['selected_lambda']['selective']),'selective_no_certainty-NoC':float(summary['selected_lambda']['selective_no_certainty-NoC'])}
agg=pd.read_csv(DOWN/'FOLDLOCAL_LAMBDA_CV_AGGREGATE.csv')
obs_ref=float(agg[(agg.method=='Reference')&(agg['lambda']==0)].auroc.iloc[0])
obs_best={}
for m in ['selective','selective_no_certainty-NoC']:
    obs_best[m]=float(agg[(agg.method==m)&np.isclose(agg['lambda'],obs_lambda[m])].auroc.iloc[0])

# Reconstruct development matrix only. External GSE48581 is not opened.
sys.path.insert(0,str(BASE/'src'))
from geo_utils import read_geo_metadata,read_geo_expression
txt=(BASE/'data/GPL570_full.txt').read_text(errors='replace').replace('\r','')
block=txt.split('!platform_table_begin\n',1)[1].split('!platform_table_end',1)[0]
ann=pd.read_csv(io.StringIO(block),sep='\t',dtype=str)[['ID','Gene Symbol']].dropna()
ann=ann[~ann['Gene Symbol'].isin(['---',''])]
ann['Gene Symbol']=ann['Gene Symbol'].str.split(' /// ').str[0].str.strip()
probe2gene=dict(zip(ann.ID.astype(str),ann['Gene Symbol'].astype(str)))
m=read_geo_metadata(str(BASE/'data/GSE36059/GSE36059_series_matrix.txt.gz'))
e=read_geo_expression(str(BASE/'data/GSE36059/GSE36059_series_matrix.txt.gz')).set_index('feature_id')
lab=m['diagnosis_tcmr_abmr_mixed_non_rejecting'].astype(str)
m=m[lab.isin(['TCMR','ABMR','MIXED','non-rejecting'])].copy()
m['y']=(m['diagnosis_tcmr_abmr_mixed_non_rejecting']=='TCMR').astype(int)
sam=[s for s in m.geo_accession if s in e.columns]
m=m.set_index('geo_accession').loc[sam]
sub=e[sam].copy();sub['gene']=[probe2gene.get(str(i),'') for i in sub.index];sub=sub[sub.gene!='']
ge=sub.groupby('gene',sort=False).median(numeric_only=True)
genes=pd.read_csv(ROOT/'formal_outputs/00_r200_refresh/candidate_features.csv').feature.astype(str).to_numpy()
X=ge.loc[genes,sam].T.to_numpy(float);y=m.y.to_numpy(int);gidx={g:i for i,g in enumerate(genes)}
assert X.shape==(403,2000) and y.sum()==35

# Frozen fold-local raw/sD and exact splits from the observed tuning.
folds=[]
for f in range(1,6):
    q=np.load(DOWN/f'CV_FOLD{f}_DATA_SCORE.npz',allow_pickle=True)
    tr=q['train_idx'].astype(int);va=q['val_idx'].astype(int);raw=q['raw'].astype(float);sd=q['sD'].astype(float)
    med=q['median_impute'].astype(float)
    Atr=X[tr].copy();Ava=X[va].copy()
    rr,cc=np.where(~np.isfinite(Atr));Atr[rr,cc]=med[cc]
    rr,cc=np.where(~np.isfinite(Ava));Ava[rr,cc]=med[cc]
    folds.append((f,tr,va,Atr,Ava,raw,sd))

A=pd.read_csv(MEAS/'SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv')
route=pd.read_csv(FREEZE/'SELECTIVE_PAIRSET_FROZEN_V3_2.csv')
assert len(A)==len(route)==100
for a,b in zip(A.itertuples(),route.itertuples()): assert a.gene_i==b.feature_i and a.gene_j==b.feature_j
ii=np.asarray([gidx[g] for g in A.gene_i],int);jj=np.asarray([gidx[g] for g in A.gene_j],int)
U=route.U.to_numpy(float);obs_p=A.p_e_gene_i_gt_gene_j.to_numpy(float);obs_C=A.C_e.to_numpy(float)
nodes=np.unique(np.r_[ii,jj]);nodepos={v:k for k,v in enumerate(nodes)}
li=np.asarray([nodepos[x] for x in ii]);lj=np.asarray([nodepos[x] for x in jj])

def correct(sd,p,c,lam,method):
    if lam==0:return sd.copy()
    w=U*c if method=='selective' else U
    ws=float(w.sum())
    base=sd[nodes].copy()
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

def topk(s,raw): return np.lexsort((np.arange(len(s)),-raw,-s))[:K]

def auc_support(fold,ids,cache):
    f,tr,va,Atr,Ava,raw,sd=fold
    key=(f,tuple(sorted(ids.tolist())))
    if key in cache:return cache[key]
    sc=StandardScaler().fit(Atr[:,ids])
    mdl=LogisticRegression(C=1.,penalty='l2',solver='liblinear',class_weight='balanced',
        max_iter=2000,random_state=SEED).fit(sc.transform(Atr[:,ids]),y[tr])
    pr=mdl.predict_proba(sc.transform(Ava[:,ids]))[:,1]
    val=float(roc_auc_score(y[va],pr));cache[key]=val;return val

def run_rep(rep):
    rng=np.random.default_rng(SEED+300000+rep)
    perm=rng.permutation(len(obs_p));p=obs_p[perm];c=obs_C[perm]
    out={'replicate':rep,'perm_sha256':hashlib.sha256(perm.tobytes()).hexdigest()}
    cache={}
    for method in ['selective','selective_no_certainty-NoC']:
        vals=[]
        for lam in GRID:
            av=[]
            for fold in folds:
                _,_,_,_,_,raw,sd=fold
                s=correct(sd,p,c,lam,method);ids=topk(s,raw)
                av.append(auc_support(fold,ids,cache))
            vals.append((lam,float(np.mean(av))))
        best=max(v for _,v in vals)
        chosen=min(l for l,v in vals if np.isclose(v,best,atol=1e-12,rtol=0))
        fixed=dict(vals)[obs_lambda[method]]
        pref='selective' if method=='selective' else 'm4'
        out.update({f'{pref}_chosen_lambda':chosen,f'{pref}_best_cv_auroc':best,
                    f'{pref}_delta_vs_reference':best-obs_ref,
                    f'{pref}_fixed_observed_lambda':obs_lambda[method],
                    f'{pref}_fixed_lambda_cv_auroc':fixed,
                    f'{pref}_fixed_lambda_delta_vs_reference':fixed-obs_ref})
    return out

rows=[]
with ProcessPoolExecutor(max_workers=4) as ex:
    fs={ex.submit(run_rep,r):r for r in range(NREP)}
    for n,f in enumerate(as_completed(fs),1):
        rows.append(f.result())
        if n%50==0:
            print(n,'/',NREP,flush=True)
            pd.DataFrame(rows).to_csv(OUT/'SEMANTIC_SHUFFLE_PARTIAL.csv',index=False)
R=pd.DataFrame(rows).sort_values('replicate')
R.to_csv(OUT/'SEMANTIC_SHUFFLE_1000.csv',index=False)

S={'version':'KIDNEY_TCMR_SEMANTIC_SHUFFLE_V3_2','n_replicates':NREP,
   'shuffle':'permute complete cached (p_e,C_e) bundles across fixed 100 Selective semantic edges; keep edge identities and U_e fixed; selective_no_certainty ignores C_e; no new LLM calls',
   'lambda_reselected_each_shuffle':True,'grid':GRID,'cv_seed':SEED,'external_used':False,
   'observed_reference_cv_auroc':obs_ref,'new_llm_calls':0}
for method,pref in [('selective','selective'),('selective_no_certainty-NoC','m4')]:
    obs_delta=obs_best[method]-obs_ref
    ret=R[f'{pref}_delta_vs_reference'];fix=R[f'{pref}_fixed_lambda_delta_vs_reference']
    S[method]={
      'observed_selected_lambda':obs_lambda[method],
      'observed_cv_auroc':obs_best[method],
      'observed_delta_vs_reference':obs_delta,
      'null_mean_best_delta':float(ret.mean()),
      'null_sd_best_delta':float(ret.std(ddof=1)),
      'null_q95_best_delta':float(ret.quantile(.95)),
      'empirical_p_retuned':float((1+int((ret>=obs_delta-1e-12).sum()))/(NREP+1)),
      'null_mean_fixed_lambda_delta':float(fix.mean()),
      'null_q95_fixed_lambda_delta':float(fix.quantile(.95)),
      'empirical_p_fixed_lambda':float((1+int((fix>=obs_delta-1e-12).sum()))/(NREP+1)),
      'lambda_counts':{str(k):int(v) for k,v in R[f'{pref}_chosen_lambda'].value_counts().sort_index().items()}
    }
(OUT/'SEMANTIC_SHUFFLE_SUMMARY.json').write_text(json.dumps(S,indent=2)+'\n')
with (OUT/'SHA256SUMS.txt').open('w') as h:
    for p in [OUT/'SEMANTIC_SHUFFLE_1000.csv',OUT/'SEMANTIC_SHUFFLE_SUMMARY.json']:
        h.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print(json.dumps(S,indent=2))
