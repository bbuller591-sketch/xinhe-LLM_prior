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
import json, math, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
DATA=Path(str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917'))
CANON=Path(str(REPRO_ROOT / 'hospital_osteoporosis_canonical_20260917'))
sys.path.insert(0,str(DATA/'scripts'))
import v2_core as V

OUT=ROOT/'11_BATCH2_FINAL_EVALUATION_V2_0_1'
OUT.mkdir(parents=True,exist_ok=True)

freeze=json.load(open(ROOT/'10_PRE_BATCH2_FREEZE/PRE_BATCH2_FREEZE_MANIFEST_V2.json'))
if freeze.get('status')!='COMPLETE_METHOD_AND_PREDICTOR_FREEZE_BEFORE_FIRST_BATCH2_OPEN':
    raise RuntimeError('PRE_BATCH2_FREEZE_NOT_COMPLETE')
predman=json.load(open(ROOT/'10_PRE_BATCH2_FREEZE/FINAL_PREDICTORS/FINAL_PREDICTOR_FREEZE_MANIFEST.json'))
if predman.get('status')!='FINAL_PREDICTORS_FROZEN_BEFORE_BATCH2_OPEN':
    raise RuntimeError('PREDICTOR_FREEZE_NOT_COMPLETE')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

# Verify sealed input identity immediately after opening.
b2raw=CANON/'source/data_clean/batch2_full.csv'
b2supp=CANON/'canonical/batch2_supplementary_features.csv'
assert sha(b2raw)=='864832034dc8f15f6ffa92021e2662f324ee753fe166e455c71e7e761499964a'
assert sha(b2supp)=='92457c448386eb8c4e7f76805143e30a4328e8fd89e4e1f5df0b0acbc5cdc77e'

Xall=pd.read_csv(CANON/'canonical/site_level_X.csv')
Y=pd.read_csv(CANON/'canonical/site_level_y.csv')
assert len(Xall)==len(Y)
Xall=Xall.copy(); Xall['y']=pd.to_numeric(Y.iloc[:,0]).astype(int).to_numpy()
b2=Xall[Xall.cohort.astype(str).eq('batch2')].copy()
prim=b2[b2.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
assert len(prim)==727 and prim.patient_uid.nunique()==538
y=prim.y.to_numpy(int); site=prim.site.to_numpy(); uid=prim.patient_uid.astype(str).to_numpy()

index=pd.read_csv(ROOT/'10_PRE_BATCH2_FREEZE/FINAL_PREDICTORS/FINAL_PREDICTOR_INDEX.csv')
preds={}
model_metrics={}
missing_rows=[]
for mid,g in index.groupby('model_id'):
    r=g.iloc[0]
    feats=str(r.feature_order).split('|')
    z=prim[feats].copy()
    for c in feats:
        if c=='DXA_性别':
            z[c]=z[c].astype(str).str.strip().map({'女':1.0,'男':0.0})
        else:
            z[c]=pd.to_numeric(z[c],errors='coerce')
    if 'DXA_性别' in feats and z['DXA_性别'].isna().any():
        raise RuntimeError('SEX_MAPPING_FAILURE')
    X=z.to_numpy(float)
    for j,f in enumerate(feats):
        missing_rows.append({'model_id':mid,'feature':f,'n_missing':int(np.isnan(X[:,j]).sum()),'missing_rate':float(np.isnan(X[:,j]).mean())})
    z=np.load(ROOT/f'10_PRE_BATCH2_FREEZE/FINAL_PREDICTORS/{mid}_MODEL.npz')
    w=z['w']; sc=(z['med'],z['mu'],z['sd'])
    p=V.predict_full(w,sc,X,site)
    preds[mid]=p
    model_metrics[mid]={
      'auroc':float(roc_auc_score(y,p)),
      'auprc':float(average_precision_score(y,p)),
      'balanced_accuracy':float(balanced_accuracy_score(y,(p>=0.5).astype(int)))
    }
pd.DataFrame(missing_rows).to_csv(OUT/'BATCH2_SELECTED_FEATURE_MISSINGNESS.csv',index=False)

rows=[]
for r in index.itertuples():
    m=model_metrics[r.model_id]
    rows.append({'method':r.method,'k':int(r.k),'method_hyperparam':float(r.method_hyperparam),
                 'model_id':r.model_id,'predictor_l2_lambda':float(r.predictor_l2_lambda),
                 'selected_set':r.feature_order,'auroc':m['auroc'],'auprc':m['auprc'],'balanced_accuracy':m['balanced_accuracy']})
res=pd.DataFrame(rows).sort_values(['k','method'])
# Delta versus same-k reference
for metric in ['auroc','auprc','balanced_accuracy']:
    base={int(k):float(g[g.method=='reference'].iloc[0][metric]) for k,g in res.groupby('k')}
    res['delta_'+metric+'_vs_reference']=res.apply(lambda r:float(r[metric])-base[int(r.k)],axis=1)
res.to_csv(OUT/'BATCH2_PRIMARY_RESULTS.csv',index=False)

# Site-specific diagnostics (no retuning).
site_rows=[]
for s in V.SITE_PRIMARY:
    mask=(prim.site.to_numpy()==s)
    ys=y[mask]
    for r in index.itertuples():
        pp=preds[r.model_id][mask]
        site_rows.append({'site':s,'method':r.method,'k':int(r.k),'model_id':r.model_id,'n_rows':int(mask.sum()),
                          'n_positive':int(ys.sum()),'auroc':float(roc_auc_score(ys,pp)),
                          'auprc':float(average_precision_score(ys,pp)),
                          'balanced_accuracy':float(balanced_accuracy_score(ys,(pp>=0.5).astype(int)))})
pd.DataFrame(site_rows).to_csv(OUT/'BATCH2_SITE_STRATIFIED_DIAGNOSTICS.csv',index=False)

# Patient-cluster paired bootstrap for metric uncertainty and delta-vs-reference. Reporting only.
rng=np.random.default_rng(20260918)
patients=np.unique(uid); B=2000
boot=[]
for b in range(B):
    samp=rng.choice(patients,size=len(patients),replace=True)
    # repeat all site rows for each sampled patient occurrence
    inds=np.concatenate([np.where(uid==u)[0] for u in samp])
    yy=y[inds]
    if len(np.unique(yy))<2: continue
    for r in index.itertuples():
        pp=preds[r.model_id][inds]
        boot.append({'b':b,'method':r.method,'k':int(r.k),
                     'auroc':float(roc_auc_score(yy,pp)),
                     'auprc':float(average_precision_score(yy,pp)),
                     'balanced_accuracy':float(balanced_accuracy_score(yy,(pp>=0.5).astype(int)))})
bd=pd.DataFrame(boot)
ci=[]
for (method,k),g in bd.groupby(['method','k']):
    rec={'method':method,'k':int(k)}
    for metric in ['auroc','auprc','balanced_accuracy']:
        v=g[metric].to_numpy()
        rec[metric+'_lo']=float(np.quantile(v,.025)); rec[metric+'_hi']=float(np.quantile(v,.975))
        base=bd[(bd.method=='reference')&(bd.k==k)][['b',metric]].rename(columns={metric:'base'})
        z=g[['b',metric]].merge(base,on='b')
        dv=(z[metric]-z['base']).to_numpy()
        rec['delta_'+metric+'_vs_reference_lo']=float(np.quantile(dv,.025)); rec['delta_'+metric+'_vs_reference_hi']=float(np.quantile(dv,.975))
        rec['prob_delta_'+metric+'_gt0_bootstrap']=float((dv>0).mean())
    ci.append(rec)
ci=pd.DataFrame(ci)
res_ci=res.merge(ci,on=['method','k'],how='left')
res_ci.to_csv(OUT/'BATCH2_PRIMARY_RESULTS_WITH_CLUSTER_BOOTSTRAP.csv',index=False)

audit={
 'status':'PASS_FINAL_TEMPORAL_EXTERNAL_EVALUATION_COMPLETE',
 'pre_batch2_freeze_manifest_sha256_v2':freeze['freeze_manifest_sha256_v2'],
 'predictor_freeze_manifest_sha256':predman['manifest_sha256'],
 'batch2_input_hashes_verified':True,
 'batch2_all_site_rows':int(len(b2)),'batch2_all_site_patients':int(b2.patient_uid.nunique()),
 'primary_rows':int(len(prim)),'primary_patients':int(prim.patient_uid.nunique()),'primary_positive':int(y.sum()),
 'primary_prevalence':float(y.mean()),'site_counts':prim.site.value_counts().to_dict(),
 'n_method_k_rows':int(len(res)),'n_unique_predictor_models':int(index.model_id.nunique()),
 'bootstrap':'2000 patient-cluster paired bootstrap replicates, seed 20260918; reporting only',
 'no_batch2_reselection_or_retuning':True,
 'no_batch2_llm_calls':True,'preprocessing_correction':'BATCH2_EVALUATION_PREPROCESSING_CORRECTION_V2_0_1.md','sex_mapping':'女=1.0, 男=0.0','revokes_version':'V2_0'
}
(OUT/'BATCH2_EVALUATION_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

# Compact report
lines=['# Batch2 final temporal external evaluation','',
'Batch2 was opened only after the complete method + predictor freeze. No Batch2-based feature reselection, lam/lam retuning, predictor-lambda retuning, evidence change, prompt change, or LLM call was performed.','',
f"Primary scope: {len(prim)} lumbar/hip site rows from {prim.patient_uid.nunique()} patients; prevalence={y.mean():.4f}.",'',
'| method | k | hp | AUROC | AUPRC | Balanced Acc | ΔAUROC vs reference | ΔAUPRC vs reference |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for r in res.itertuples():
    lines.append(f"| {r.method} | {r.k} | {r.method_hyperparam:g} | {r.auroc:.4f} | {r.auprc:.4f} | {r.balanced_accuracy:.4f} | {r.delta_auroc_vs_reference:+.4f} | {r.delta_auprc_vs_reference:+.4f} |")
lines += ['','Cluster-bootstrap uncertainty is stored in BATCH2_PRIMARY_RESULTS_WITH_CLUSTER_BOOTSTRAP.csv. Site-stratified diagnostics are descriptive only.']
(OUT/'BATCH2_FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
print('\nPRIMARY RESULTS\n'+res.to_string(index=False))
print('\nWITH CIs\n'+res_ci[['method','k','auroc','auroc_lo','auroc_hi','delta_auroc_vs_reference','delta_auroc_vs_reference_lo','delta_auroc_vs_reference_hi','auprc','auprc_lo','auprc_hi','delta_auprc_vs_reference','delta_auprc_vs_reference_lo','delta_auprc_vs_reference_hi']].to_string(index=False))
