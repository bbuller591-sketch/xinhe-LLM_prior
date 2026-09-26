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
import sys,io,json,hashlib,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
BASE=Path(str(REPRO_ROOT / 'dataset_screening_20260921'))
DOWN=ROOT/'formal_outputs/02_downstream_dev_v3_2'; MEAS=ROOT/'formal_outputs/measurements/v3_2'
FREEZE=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2/FINAL'
SRC=ROOT/'formal_outputs/03_semantic_shuffle_v3_2/SEMANTIC_SHUFFLE_1000.csv'
OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion/artifacts/RENAL_Q3_SEMANTIC_EXTERNAL'))
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261044;K=50;P=2000

sys.path.insert(0,str(BASE/'src'))
from geo_utils import read_geo_metadata,read_geo_expression
txt=(BASE/'data/GPL570_full.txt').read_text(errors='replace').replace('\r','')
block=txt.split('!platform_table_begin\n',1)[1].split('!platform_table_end',1)[0]
ann=pd.read_csv(io.StringIO(block),sep='\t',dtype=str)[['ID','Gene Symbol']].dropna()
ann=ann[~ann['Gene Symbol'].isin(['---',''])];ann['Gene Symbol']=ann['Gene Symbol'].str.split(' /// ').str[0].str.strip()
probe2gene=dict(zip(ann.ID.astype(str),ann['Gene Symbol'].astype(str)))
def load(acc):
    m=read_geo_metadata(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz'))
    e=read_geo_expression(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz')).set_index('feature_id')
    if acc=='GSE36059':
        lab=m['diagnosis_tcmr_abmr_mixed_non_rejecting'].astype(str);m=m[lab.isin(['TCMR','ABMR','MIXED','non-rejecting'])].copy()
        m['y']=(m['diagnosis_tcmr_abmr_mixed_non_rejecting']=='TCMR').astype(int)
    else:
        lab=m['diagnosis_tcmr_non_tcmr_nephrectomies'].astype(str);m=m[lab.isin(['TCMR','non-TCMR'])].copy()
        m['y']=(m['diagnosis_tcmr_non_tcmr_nephrectomies']=='TCMR').astype(int)
    sam=[s for s in m.geo_accession if s in e.columns];m=m.set_index('geo_accession').loc[sam]
    sub=e[sam].copy();sub['gene']=[probe2gene.get(str(i),'') for i in sub.index];sub=sub[sub.gene!='']
    ge=sub.groupby('gene',sort=False).median(numeric_only=True)
    return m,ge,sam
md,gd,sd=load('GSE36059');me,ge,se=load('GSE48581')
genes=pd.read_csv(ROOT/'CANDIDATE_UNIVERSE_FROZEN.csv').gene.astype(str).to_numpy()
Xd=gd.loc[genes,sd].T.to_numpy(float); yd=md.y.to_numpy(int); Xe=ge.loc[genes,se].T.to_numpy(float); ye=me.y.to_numpy(int)
med=np.nanmedian(Xd,axis=0)
for A in (Xd,Xe):
    rr,cc=np.where(~np.isfinite(A));A[rr,cc]=med[cc]
gidx={g:i for i,g in enumerate(genes)}
refscore=pd.read_csv(DOWN/'REFERENCE_SCORES_FROZEN.csv');Dfull=refscore.reference_raw_score.to_numpy(float);sDfull=refscore.reference_score_normalized.to_numpy(float)
route=pd.read_csv(FREEZE/'SELECTIVE_PAIRSET_FROZEN_V3_2.csv');M=pd.read_csv(MEAS/'SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv')
assert len(route)==len(M)==100
for a,b in zip(M.itertuples(),route.itertuples()): assert a.gene_i==b.feature_i and a.gene_j==b.feature_j
ii=np.asarray([gidx[g] for g in M.gene_i],int);jj=np.asarray([gidx[g] for g in M.gene_j],int);nodes=np.unique(np.r_[ii,jj]);pos={v:k for k,v in enumerate(nodes)}
li=np.asarray([pos[x] for x in ii]);lj=np.asarray([pos[x] for x in jj]);U=route.U.to_numpy(float)
pobs=M.p_e_gene_i_gt_gene_j.to_numpy(float); cobs=M.C_e.to_numpy(float)
def correct(sd,p,c,lam):
    if lam==0:return sd.copy()
    w=U*c;W=float(w.sum());base=sd[nodes].copy()
    def fg(v):
        z=v[li]-v[lj]; ce=np.logaddexp(0,z)-p*z; diff=v-base
        f=.5*np.sum(diff*diff)/P+lam*np.dot(w,ce)/(W+1e-12)
        sig=1/(1+np.exp(-np.clip(z,-50,50)));g=diff/P;rr=lam*(w/(W+1e-12))*(sig-p)
        np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
    rr=minimize(lambda v:fg(v)[0],base,jac=lambda v:fg(v)[1],method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12,'gtol':1e-8})
    if not rr.success: raise RuntimeError(rr.message)
    out=sd.copy();out[nodes]=rr.x;return out
def topk(z,D): return np.lexsort((np.arange(P),-D,-z))[:K]
def ext_metrics(ids):
    sc=StandardScaler().fit(Xd[:,ids]);mod=LogisticRegression(C=1.,penalty='l2',solver='liblinear',class_weight='balanced',max_iter=2000,random_state=SEED).fit(sc.transform(Xd[:,ids]),yd)
    pr=mod.predict_proba(sc.transform(Xe[:,ids]))[:,1]
    return float(roc_auc_score(ye,pr)),float(average_precision_score(ye,pr)),float(balanced_accuracy_score(ye,pr>=.5))
R=pd.read_csv(SRC).sort_values('replicate')
rows=[]
for n,r in enumerate(R.itertuples(),1):
    rep=int(r.replicate);rng=np.random.default_rng(SEED+300000+rep);perm=rng.permutation(len(pobs))
    h=hashlib.sha256(perm.tobytes()).hexdigest()
    if h!=r.perm_sha256: raise RuntimeError(f'perm identity fail rep {rep}')
    p=pobs[perm];c=cobs[perm];lam=float(r.selective_chosen_lambda)
    ids=topk(correct(sDfull,p,c,lam),Dfull); au,ap,ba=ext_metrics(ids)
    rows.append({'replicate':rep,'chosen_lambda_from_original_dev_null':lam,'external_auroc':au,'external_auprc':ap,'external_balacc':ba,
                 'support_sha256':hashlib.sha256('\n'.join(genes[ids]).encode()).hexdigest()})
    if n%100==0: print(n,flush=True)
Z=pd.DataFrame(rows);Z.to_csv(OUT/'SEMANTIC_SHUFFLE_EXTERNAL_1000.csv',index=False)
obs=json.load(open(ROOT/'formal_outputs/05_external_evaluation_v3_2/EXTERNAL_EVALUATION_SUMMARY.json'))
oe=float(obs['metrics']['selective']['auroc']);oap=float(obs['metrics']['selective']['auprc']);ref=float(obs['metrics']['Reference']['auroc'])
summary={'dataset':'Renal GSE36059->GSE48581 TCMR','mode':'semantic_shuffle','n_replicates':1000,'new_llm_calls':0,
 'chronology':'post-hoc external evaluation of the already-frozen 2026-09-21 semantic-shuffle null; replicate lambda choices are exactly the original development-only choices; external outcomes never enter tuning',
 'observed_external_auroc':oe,'observed_external_auprc':oap,'reference_external_auroc':ref,
 'null_external_mean_auroc':float(Z.external_auroc.mean()),'null_external_sd_auroc':float(Z.external_auroc.std(ddof=1)),
 'null_external_q025_auroc':float(Z.external_auroc.quantile(.025)),'null_external_q975_auroc':float(Z.external_auroc.quantile(.975)),
 'empirical_p_external_ge_observed':float((1+(Z.external_auroc>=oe-1e-12).sum())/(len(Z)+1)),
 'n_lower':int((Z.external_auroc<oe-1e-12).sum()),'n_equal':int(np.isclose(Z.external_auroc,oe,atol=1e-12,rtol=0).sum()),'n_greater':int((Z.external_auroc>oe+1e-12).sum()),
 'null_external_mean_delta_vs_ref':float((Z.external_auroc-ref).mean()),
 'null_external_mean_auprc':float(Z.external_auprc.mean()),'empirical_p_external_auprc_ge_observed':float((1+(Z.external_auprc>=oap-1e-12).sum())/(len(Z)+1))}
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
pd.DataFrame([summary]).to_csv(OUT/'SUMMARY.csv',index=False)
print(json.dumps(summary,indent=2))
