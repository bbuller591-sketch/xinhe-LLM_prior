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
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,confusion_matrix

warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
BASE=Path(str(REPRO_ROOT / 'dataset_screening_20260921'))
FREEZE=ROOT/'formal_outputs/04_pre_external_final_freeze_v3_2'
OUT=ROOT/'formal_outputs/05_external_evaluation_v3_2';OUT.mkdir(parents=True,exist_ok=True)
SEED=20261044;BOOT=2000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
fz=json.loads((FREEZE/'PRE_EXTERNAL_FINAL_FREEZE.json').read_text())
assert fz['status']=='PASS_EXTERNAL_UNSEAL_ALLOWED'
assert fz['external_outcome_accessed_in_method_selection'] is False
# Integrity of the freeze itself.
expect=(FREEZE/'PRE_EXTERNAL_FINAL_FREEZE.sha256').read_text().split()[0]
assert sha(FREEZE/'PRE_EXTERNAL_FINAL_FREEZE.json')==expect

sys.path.insert(0,str(BASE/'src'))
from geo_utils import read_geo_metadata,read_geo_expression

# Frozen GPL570 probe->symbol mapping.
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
    elif acc=='GSE48581':
        lab=m['diagnosis_tcmr_non_tcmr_nephrectomies'].astype(str)
        m=m[lab.isin(['TCMR','non-TCMR'])].copy()
        m['y']=(m['diagnosis_tcmr_non_tcmr_nephrectomies']=='TCMR').astype(int)
    sam=[s for s in m.geo_accession if s in e.columns]
    m=m.set_index('geo_accession').loc[sam]
    sub=e[sam].copy();sub['gene']=[probe2gene.get(str(i),'') for i in sub.index];sub=sub[sub.gene!='']
    ge=sub.groupby('gene',sort=False).median(numeric_only=True)
    return m,ge,sam

# This is the first script in the formal branch that reads GSE48581 outcomes.
md,gd,sd=load('GSE36059')
me,ge,se=load('GSE48581')
assert len(md)==403 and md.y.sum()==35
assert len(me)==300 and me.y.sum()==32

cand=pd.read_csv(ROOT/'CANDIDATE_UNIVERSE_FROZEN.csv')
genes=cand.gene.astype(str).tolist()
assert len(genes)==2000
assert all(g in gd.index for g in genes) and all(g in ge.index for g in genes)
Xd=gd.loc[genes,sd].T.to_numpy(float);yd=md.y.to_numpy(int)
Xe=ge.loc[genes,se].T.to_numpy(float);ye=me.y.to_numpy(int)
gidx={g:i for i,g in enumerate(genes)}

# Development-only median imputation.
med=np.nanmedian(Xd,axis=0)
for A in [Xd,Xe]:
    rr,cc=np.where(~np.isfinite(A));A[rr,cc]=med[cc]

methods=['Reference','Global','selective','selective_no_certainty-NoC']
preds={'geo_accession':np.asarray(se,str),'y_true':ye}
metrics=[]
support_sets={}
for method in methods:
    sp=FREEZE/f'{method.replace("-","_")}_TOP50_FROZEN.txt'
    assert sha(sp)==fz['support_sha256'][method]
    gs=[x.strip() for x in sp.read_text().splitlines() if x.strip()]
    assert len(gs)==50 and len(set(gs))==50
    ids=np.asarray([gidx[g] for g in gs],int);support_sets[method]=set(gs)
    scaler=StandardScaler().fit(Xd[:,ids])
    mdl=LogisticRegression(C=1.,penalty='l2',solver='liblinear',class_weight='balanced',
        max_iter=3000,random_state=SEED).fit(scaler.transform(Xd[:,ids]),yd)
    p=mdl.predict_proba(scaler.transform(Xe[:,ids]))[:,1]
    preds['p_'+method.replace('-','_')]=p
    yh=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(ye,yh,labels=[0,1]).ravel()
    metrics.append({'method':method,'selected_lambda':float(fz['selected_lambda'][method]),'k':50,
      'external_n':len(ye),'external_positive':int(ye.sum()),
      'auroc':float(roc_auc_score(ye,p)),'auprc':float(average_precision_score(ye,p)),
      'balanced_accuracy':float(balanced_accuracy_score(ye,yh)),
      'sensitivity':float(tp/(tp+fn)) if tp+fn else np.nan,
      'specificity':float(tn/(tn+fp)) if tn+fp else np.nan,
      'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn)})
pred=pd.DataFrame(preds);pred.to_csv(OUT/'EXTERNAL_PREDICTIONS_GSE48581.csv',index=False)
met=pd.DataFrame(metrics)

# Paired stratified biopsy-row bootstrap. Same resample indices across methods.
rng=np.random.default_rng(SEED+900000)
neg=np.where(ye==0)[0];pos=np.where(ye==1)[0]
probs={m:pred['p_'+m.replace('-','_')].to_numpy(float) for m in methods}
br=[]
for b in range(BOOT):
    ix=np.r_[rng.choice(neg,len(neg),replace=True),rng.choice(pos,len(pos),replace=True)]
    yy=ye[ix]
    row={'replicate':b}
    for m in methods:
        pp=probs[m][ix];yh=(pp>=.5).astype(int)
        row[f'{m}_auroc']=roc_auc_score(yy,pp)
        row[f'{m}_auprc']=average_precision_score(yy,pp)
        row[f'{m}_balanced_accuracy']=balanced_accuracy_score(yy,yh)
    for m in methods[1:]:
        for metric in ['auroc','auprc','balanced_accuracy']:
            row[f'{m}_minus_Reference_{metric}']=row[f'{m}_{metric}']-row[f'Reference_{metric}']
    br.append(row)
boot=pd.DataFrame(br);boot.to_csv(OUT/'EXTERNAL_BOOTSTRAP_BIOPSY_LEVEL_2000.csv',index=False)

# CI table.
rows=[]
for m in methods:
    point=met[met.method==m].iloc[0]
    rr={'method':m,'selected_lambda':float(point.selected_lambda),'k':50}
    for metric in ['auroc','auprc','balanced_accuracy']:
        vals=boot[f'{m}_{metric}']
        rr[metric]=float(point[metric]);rr[metric+'_ci025']=float(vals.quantile(.025));rr[metric+'_ci975']=float(vals.quantile(.975))
    if m!='Reference':
        for metric in ['auroc','auprc','balanced_accuracy']:
            vals=boot[f'{m}_minus_Reference_{metric}']
            rr['delta_'+metric]=float(point[metric]-met[met.method=='Reference'].iloc[0][metric])
            rr['delta_'+metric+'_ci025']=float(vals.quantile(.025));rr['delta_'+metric+'_ci975']=float(vals.quantile(.975))
    rows.append(rr)
ci=pd.DataFrame(rows);ci.to_csv(OUT/'EXTERNAL_METRICS_WITH_BIOPSY_BOOTSTRAP_CI.csv',index=False)
met.to_csv(OUT/'EXTERNAL_METRICS_POINT.csv',index=False)

over=[]
ref=support_sets['Reference']
for m in methods:
    s=support_sets[m]
    over.append({'method':m,'intersection_reference':len(s&ref),'jaccard_reference':len(s&ref)/len(s|ref),
                 'added_vs_reference':'|'.join(sorted(s-ref)),'removed_vs_reference':'|'.join(sorted(ref-s))})
pd.DataFrame(over).to_csv(OUT/'EXTERNAL_SELECTED_SET_OVERLAP.csv',index=False)

result={
 'version':'KIDNEY_TCMR_EXTERNAL_EVAL_V3_2',
 'status':'COMPLETE_NO_POST_EXTERNAL_TUNING',
 'development':{'dataset':'GSE36059','n':403,'positive':35},
 'external':{'dataset':'GSE48581','n':300,'positive':32},
 'selected_lambda':fz['selected_lambda'],
 'k':50,
 'metrics':{r['method']:{k:v for k,v in r.items() if k not in ('method','k','external_n','external_positive')} for r in metrics},
 'bootstrap':{'n':BOOT,'type':'paired stratified biopsy-row bootstrap','seed':SEED+900000,
              'limitation':'recipient mapping unavailable; these are not patient-clustered confidence intervals'},
 'no_post_external_tuning':True,
 'external_outcome_first_read_stage':'05_external_evaluation_v3_2',
 'pre_external_freeze_sha256':sha(FREEZE/'PRE_EXTERNAL_FINAL_FREEZE.json')
}
(OUT/'EXTERNAL_EVALUATION_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
files=[OUT/'EXTERNAL_PREDICTIONS_GSE48581.csv',OUT/'EXTERNAL_METRICS_POINT.csv',
       OUT/'EXTERNAL_METRICS_WITH_BIOPSY_BOOTSTRAP_CI.csv',OUT/'EXTERNAL_BOOTSTRAP_BIOPSY_LEVEL_2000.csv',
       OUT/'EXTERNAL_SELECTED_SET_OVERLAP.csv',OUT/'EXTERNAL_EVALUATION_SUMMARY.json']
with (OUT/'SHA256SUMS.txt').open('w') as h:
    for p in files:h.write(sha(p)+'  '+p.name+'\n')
print(met.to_string(index=False))
print('\nCI/DELTAS\n'+ci.to_string(index=False))
print('\nSUMMARY\n'+json.dumps(result,indent=2))
