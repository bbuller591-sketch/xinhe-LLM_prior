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
import sys,io,json,hashlib,math
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
BASE=Path(str(REPRO_ROOT / 'dataset_screening_20260921'))
OUT=ROOT/'formal_outputs/02_downstream_dev_v3_2'
MEAS=ROOT/'formal_outputs/measurements/v3_2'
FREEZE=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2/FINAL'
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261044
MODEL_SEED=20260921
K=50
P=2000
GRID=[0.,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.]
EN_C=.3
EN_RATIO=.8

# Hard preconditions: formal study calls complete; external is still not loaded by this script.
gate=json.loads((FREEZE/'STOP_GATE.json').read_text())
assert gate['status']=='PASS'
ms=json.loads((MEAS/'FORMAL_DEEPSEEK_V3_2_SUMMARY.json').read_text())
assert ms['status']=='COMPLETE' and ms['n_calls']==400
qa=json.loads((MEAS/'FORMAL_PAIR_MEASUREMENT_QA_V3_2.json').read_text())
assert qa['status']=='PASS' and qa['n_selective']==100 and qa['n_global']==100

sys.path.insert(0,str(BASE/'src'))
from geo_utils import read_geo_metadata,read_geo_expression

# Frozen GPL570 mapping.
txt=(BASE/'data/GPL570_full.txt').read_text(errors='replace').replace('\r','')
block=txt.split('!platform_table_begin\n',1)[1].split('!platform_table_end',1)[0]
ann=pd.read_csv(io.StringIO(block),sep='\t',dtype=str)
ann=ann[['ID','Gene Symbol']].dropna()
ann=ann[~ann['Gene Symbol'].isin(['---',''])]
ann['Gene Symbol']=ann['Gene Symbol'].str.split(' /// ').str[0].str.strip()
probe2gene=dict(zip(ann.ID.astype(str),ann['Gene Symbol'].astype(str)))

m=read_geo_metadata(str(BASE/'data/GSE36059/GSE36059_series_matrix.txt.gz'))
e=read_geo_expression(str(BASE/'data/GSE36059/GSE36059_series_matrix.txt.gz')).set_index('feature_id')
lab=m['diagnosis_tcmr_abmr_mixed_non_rejecting'].astype(str)
keep=lab.isin(['TCMR','ABMR','MIXED','non-rejecting'])
m=m[keep].copy()
m['y']=(m['diagnosis_tcmr_abmr_mixed_non_rejecting']=='TCMR').astype(int)
sam=[s for s in m.geo_accession if s in e.columns]
m=m.set_index('geo_accession').loc[sam]
sub=e[sam].copy()
sub['gene']=[probe2gene.get(str(i),'') for i in sub.index]
sub=sub[sub.gene!='']
ge=sub.groupby('gene',sort=False).median(numeric_only=True)

cand=pd.read_csv(ROOT/'formal_outputs/00_r200_refresh/candidate_features.csv').feature.astype(str).tolist()
assert len(cand)==P and len(set(cand))==P
missing=[g for g in cand if g not in ge.index]
assert not missing,missing[:20]
X=ge.loc[cand,sam].T.to_numpy(float)
y=m.y.to_numpy(int)
genes=np.asarray(cand,str)
idx={g:i for i,g in enumerate(genes)}
assert X.shape==(403,2000) and int(y.sum())==35

# Median imputation from the current training data is applied fold-locally below. Full-dev has no expected NAs,
# but keep deterministic imputation for audit consistency.
def impute_train_val(Atr,Ava=None):
    med=np.nanmedian(Atr,axis=0)
    tr=Atr.copy()
    rr,cc=np.where(~np.isfinite(tr));tr[rr,cc]=med[cc]
    if Ava is None:return tr,med
    va=Ava.copy();rr,cc=np.where(~np.isfinite(va));va[rr,cc]=med[cc]
    return tr,va,med

def sis_score(A,b):
    b=np.asarray(b,float);yc=b-b.mean();xc=A-np.nanmean(A,axis=0)
    den=np.sqrt(np.nansum(xc*xc,axis=0)*np.sum(yc*yc))
    return np.nan_to_num(np.abs(np.nansum(xc*yc[:,None],axis=0)/np.where(den==0,np.nan,den)),nan=0.)

def data_score(A,b,seed):
    # Same frozen primary Elastic-Net specification as R200 Reference.
    sc=StandardScaler()
    Z=sc.fit_transform(A)
    lr=LogisticRegression(C=EN_C,penalty='elasticnet',l1_ratio=EN_RATIO,solver='saga',
        class_weight='balanced',max_iter=3000,tol=1e-3,random_state=seed,n_jobs=1)
    lr.fit(Z,b)
    raw=np.abs(lr.coef_[0])+1e-8*sis_score(A,b)
    return raw

def anchor(raw):
    mx=float(np.nanmax(raw))
    if not np.isfinite(mx) or mx<=0:raise RuntimeError('invalid reference max')
    return raw/mx

# Freeze full-development Reference anchor before lambda tuning outputs.
Xfull,_=impute_train_val(X)
Dfull=data_score(Xfull,y,MODEL_SEED)
sDfull=anchor(Dfull)
ref_order=np.lexsort((np.arange(P),-Dfull,-sDfull))
ref_rank=np.empty(P,int);ref_rank[ref_order]=np.arange(1,P+1)
ref=pd.DataFrame({'candidate_order':np.arange(1,P+1),'gene':genes,'reference_raw_score':Dfull,
                  'reference_score_normalized':sDfull,'reference_rank':ref_rank})
ref.to_csv(OUT/'REFERENCE_SCORES_FROZEN.csv',index=False)

# Frozen measurement + routing tables.
selm=pd.read_csv(MEAS/'SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv')
glm=pd.read_csv(MEAS/'GLOBAL_MEASUREMENTS_V3_2.csv')
selp=pd.read_csv(FREEZE/'SELECTIVE_PAIRSET_FROZEN_V3_2.csv')
glp=pd.read_csv(FREEZE/'GLOBAL_PAIRSET_FROZEN_V3_2.csv')
# Exact order identity checks.
assert len(selm)==len(selp)==100 and len(glm)==len(glp)==100
for a,b in zip(selm.itertuples(),selp.itertuples()):
    assert a.gene_i==b.feature_i and a.gene_j==b.feature_j
for a,b in zip(glm.itertuples(),glp.itertuples()):
    assert a.gene_i==b.feature_i and a.gene_j==b.feature_j
selm=selm.copy();selm['U']=selp.U.to_numpy(float)

def edges(method):
    if method=='Global':
        d=glm.copy();d['p']=d.p_e_gene_i_gt_gene_j.astype(float);d['w0']=d.C_e.astype(float)
    elif method=='selective':
        d=selm.copy();d['p']=d.p_e_gene_i_gt_gene_j.astype(float);d['w0']=d.U.astype(float)*d.C_e.astype(float)
    elif method=='selective_no_certainty-NoC':
        d=selm.copy();d['p']=d.p_e_gene_i_gt_gene_j.astype(float);d['w0']=d.U.astype(float)
    else:
        return pd.DataFrame()
    assert np.isfinite(d.p).all() and np.isfinite(d.w0).all() and (d.w0>=0).all()
    return d

def correct(sD,lam,d):
    # Established normalized anchored objective:
    # 1/(2P)||s-sD||^2 + lam * sum w_e CE(p_e,sigmoid(s_i-s_j))/(sum w_e + 1e-12).
    if lam==0 or d.empty or float(d.w0.sum())<=0:
        return sD.copy(),True,0,0.0
    ii=np.asarray([idx[g] for g in d.gene_i],int)
    jj=np.asarray([idx[g] for g in d.gene_j],int)
    pp=d.p.to_numpy(float);w=d.w0.to_numpy(float);ws=float(w.sum())
    active=np.unique(np.r_[ii,jj])
    amap={v:k for k,v in enumerate(active)}
    ai=np.asarray([amap[v] for v in ii],int);aj=np.asarray([amap[v] for v in jj],int)
    x0=sD[active].copy()
    def fg(x):
        dif=x-x0
        f=.5*np.sum(dif*dif)/P
        gr=dif/P
        zz=x[ai]-x[aj]
        # stable CE(p,sigmoid(z)) = logaddexp(0,z)-p*z
        ce=np.logaddexp(0.0,zz)-pp*zz
        f+=lam*float(np.dot(w,ce))/(ws+1e-12)
        sig=1/(1+np.exp(-np.clip(zz,-50,50)))
        v=lam*(w/(ws+1e-12))*(sig-pp)
        np.add.at(gr,ai,v);np.add.at(gr,aj,-v)
        return float(f),gr
    r=minimize(lambda x:fg(x)[0],x0,jac=lambda x:fg(x)[1],method='L-BFGS-B',
        options={'maxiter':1000,'ftol':1e-12,'gtol':1e-9})
    s=sD.copy();s[active]=r.x
    max_unqueried=float(np.max(np.abs(s[np.setdiff1d(np.arange(P),active)]-sD[np.setdiff1d(np.arange(P),active)]))) if len(active)<P else 0.
    assert max_unqueried==0.
    return s,bool(r.success),int(r.nit),float(r.fun)

def topk(s,raw):
    # primary corrected score, then frozen data-only raw score, then candidate order.
    return np.lexsort((np.arange(P),-raw,-s))[:K]

def evaluate(Atr,btr,Ava,bva,ids):
    sc=StandardScaler().fit(Atr[:,ids])
    mdl=LogisticRegression(C=1.,penalty='l2',solver='liblinear',class_weight='balanced',
        max_iter=3000,random_state=SEED)
    mdl.fit(sc.transform(Atr[:,ids]),btr)
    prob=mdl.predict_proba(sc.transform(Ava[:,ids]))[:,1]
    return {'auroc':float(roc_auc_score(bva,prob)),
            'auprc':float(average_precision_score(bva,prob)),
            'balanced_accuracy':float(balanced_accuracy_score(bva,prob>=.5))}

cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
splits=list(cv.split(X,y));foldobj=[]
for f,(tr,va) in enumerate(splits,1):
    Atr,Ava,med=impute_train_val(X[tr],X[va])
    raw=data_score(Atr,y[tr],MODEL_SEED+f)
    sd=anchor(raw)
    foldobj.append((f,tr,va,Atr,Ava,raw,sd))
    np.savez_compressed(OUT/f'CV_FOLD{f}_DATA_SCORE.npz',train_idx=tr,val_idx=va,raw=raw,sD=sd,
                        selected_reference_idx=topk(sd,raw),median_impute=med)

rows=[]
for method in ['Reference','Global','selective','selective_no_certainty-NoC']:
    ed=edges(method)
    lams=[0.] if method=='Reference' else GRID
    for lam in lams:
        mets=[];changes=[];oks=[];nits=[]
        for f,tr,va,Atr,Ava,raw,sd in foldobj:
            zz,ok,nit,fun=correct(sd,lam,ed)
            ids=topk(zz,raw);rid=topk(sd,raw)
            mm=evaluate(Atr,y[tr],Ava,y[va],ids)
            mets.append(mm);changes.append(K-len(set(ids)&set(rid)));oks.append(ok);nits.append(nit)
            rows.append({'record_type':'fold','method':method,'lambda':lam,'fold':f,**mm,
                         'n_changed_vs_fold_reference':changes[-1],'opt_success':ok,'opt_nit':nit,'opt_fun':fun,
                         'selected_genes':'|'.join(genes[ids])})
        rows.append({'record_type':'aggregate','method':method,'lambda':lam,'fold':0,
                     'auroc':float(np.mean([x['auroc'] for x in mets])),
                     'auprc':float(np.mean([x['auprc'] for x in mets])),
                     'balanced_accuracy':float(np.mean([x['balanced_accuracy'] for x in mets])),
                     'n_changed_vs_fold_reference':float(np.mean(changes)),
                     'opt_success':all(oks),'opt_nit':float(np.mean(nits)),'opt_fun':np.nan,'selected_genes':''})
out=pd.DataFrame(rows)
out.to_csv(OUT/'FOLDLOCAL_LAMBDA_CV_ALL.csv',index=False)
agg=out[out.record_type=='aggregate'].copy()
agg.to_csv(OUT/'FOLDLOCAL_LAMBDA_CV_AGGREGATE.csv',index=False)

choices={'Reference':0.}
for method in ['Global','selective','selective_no_certainty-NoC']:
    z=agg[agg.method==method].copy()
    mx=float(z.auroc.max())
    mask=np.isclose(z.auroc.to_numpy(float),mx,atol=1e-12,rtol=0)
    choices[method]=float(z.loc[mask].sort_values('lambda').iloc[0]['lambda'])

# Full-development supports only after lambda choices.
srows=[];score={'Reference':sDfull.copy()}
refids=topk(sDfull,Dfull)
for r,i in enumerate(refids,1):
    srows.append({'method':'Reference','selected_lambda':0.,'selected_rank':r,'gene':genes[i],
                  'corrected_score':sDfull[i],'reference_rank':int(ref_rank[i])})
opt_full={}
for method in ['Global','selective','selective_no_certainty-NoC']:
    zz,ok,nit,fun=correct(sDfull,choices[method],edges(method))
    ids=topk(zz,Dfull);score[method]=zz;opt_full[method]={'success':ok,'nit':nit,'fun':fun}
    for r,i in enumerate(ids,1):
        srows.append({'method':method,'selected_lambda':choices[method],'selected_rank':r,'gene':genes[i],
                      'corrected_score':zz[i],'reference_rank':int(ref_rank[i])})
sup=pd.DataFrame(srows)
sup.to_csv(OUT/'SELECTED_SUPPORTS_FROZEN.csv',index=False)
pd.DataFrame({'candidate_order':np.arange(1,P+1),'gene':genes,**score}).to_csv(OUT/'CORRECTED_SCORES_FROZEN.csv',index=False)

sets={m:set(sup[sup.method==m].gene) for m in sup.method.unique()}
aud=[]
for method in ['Reference','Global','selective','selective_no_certainty-NoC']:
    ss=sets[method];rr=sets['Reference']
    aud.append({'method':method,'selected_lambda':choices[method],
                'intersection_reference':len(ss&rr),'jaccard_reference':len(ss&rr)/len(ss|rr),
                'added_vs_reference':'|'.join(sorted(ss-rr)),'removed_vs_reference':'|'.join(sorted(rr-ss))})
pd.DataFrame(aud).to_csv(OUT/'SUPPORT_OVERLAP.csv',index=False)

# Explicit selective/selective_no_certainty measurement equality audit.
shared_hash=hashlib.sha256((MEAS/'SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv').read_bytes()).hexdigest()
summary={
 'version':'KIDNEY_TCMR_DOWNSTREAM_V3_2',
 'selected_lambda':choices,'lambda_grid':GRID,'selection':'5-fold GSE36059 development CV; each fold recomputes Elastic-Net raw score + SIS and max-normalized anchor on fold-training only; frozen pair identities/U/LLM p,C remain fixed; exact AUROC ties -> smaller lambda',
 'objective':'1/(2p)||s-sD||^2 + lambda * weighted_mean_edge_CE',
 'reference_anchor':'sD=(abs(ElasticNet beta)+1e-8*SIS)/max; C=0.3,l1_ratio=0.8',
 'edge_weights':{'Global':'C_e','selective':'U_e*C_e','selective_no_certainty-NoC':'U_e'},
 'cv_seed':SEED,'model_seed_base':MODEL_SEED,'k':K,'p':P,
 'external_used':False,
 'selective_m4_shared_measurement_sha256':shared_hash,
 'semantic_shuffle_required':bool(choices['selective']>0 or choices['selective_no_certainty-NoC']>0),
 'semantic_shuffle_n':1000 if (choices['selective']>0 or choices['selective_no_certainty-NoC']>0) else 0,
 'recipient_cv_note':'CV is biopsy-row-level because public recipient mapping is unavailable; do not call patient-grouped.',
 'full_optimization':opt_full,
 'study_specific_llm_calls_reused':400,'new_llm_calls_in_downstream':0
}
(OUT/'PRE_EXTERNAL_FREEZE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')

# Freeze all development-only downstream artifacts + hashes before any external access.
files=[OUT/'REFERENCE_SCORES_FROZEN.csv',OUT/'FOLDLOCAL_LAMBDA_CV_ALL.csv',OUT/'FOLDLOCAL_LAMBDA_CV_AGGREGATE.csv',
       OUT/'SELECTED_SUPPORTS_FROZEN.csv',OUT/'CORRECTED_SCORES_FROZEN.csv',OUT/'SUPPORT_OVERLAP.csv',
       OUT/'PRE_EXTERNAL_FREEZE_SUMMARY.json']+[OUT/f'CV_FOLD{i}_DATA_SCORE.npz' for i in range(1,6)]
with (OUT/'PRE_EXTERNAL_SHA256.txt').open('w') as h:
    for p in files:h.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print(json.dumps(summary,indent=2))
print('\nAGGREGATE\n'+agg.to_string(index=False))
print('\nSUPPORT OVERLAP\n'+pd.DataFrame(aud).to_string(index=False))
