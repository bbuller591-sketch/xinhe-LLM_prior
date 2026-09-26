#!/usr/bin/env python3
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

PKG=Path(__file__).resolve().parents[1]
TASK='BREAST_GSE25055_GSE25065'
SEED=2026091901
DATA=PKG/'DATA/FROZEN'/TASK
REF=PKG/'RESULTS/SEALED_VALIDATION_PRIMARY_RESULTS.csv'
OUT=PKG/'REPRODUCE/OUTPUT'
OUT.mkdir(parents=True,exist_ok=True)

def sis_score(X,y):
    yc=y-y.mean(); xc=X-X.mean(axis=0)
    den=np.sqrt((xc*xc).sum(axis=0)*(yc*yc).sum())
    return np.nan_to_num(np.abs((xc*yc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0)

def sparse_score(X,y,C,ratio):
    sc=StandardScaler().fit(X); Z=sc.transform(X)
    penalty='l1' if float(ratio)==1.0 else 'elasticnet'
    m=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=penalty,
        l1_ratio=None if penalty=='l1' else float(ratio),max_iter=5000,tol=1e-4,
        random_state=SEED,fit_intercept=True,n_jobs=1)
    m.fit(Z,y)
    if int(m.n_iter_[0])>=m.max_iter:
        raise RuntimeError(f'full-development {penalty} did not converge')
    return np.abs(m.coef_[0])

def topk(D,k):
    return np.lexsort((np.arange(len(D)),-D))[:k]

def eval_support(Xtr,ytr,Xte,yte,cols):
    sc=StandardScaler().fit(Xtr[:,cols])
    A=sc.transform(Xtr[:,cols]); B=sc.transform(Xte[:,cols])
    m=LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',class_weight='balanced',
                         max_iter=5000,random_state=SEED)
    m.fit(A,ytr)
    p=m.predict_proba(B)[:,1]
    app=average_precision_score(yte,p)
    apn=average_precision_score(1-yte,1-p)
    return {
      'auroc':float(roc_auc_score(yte,p)),
      'ap_positive':float(app),
      'ap_negative':float(apn),
      'macro_ap':float((app+apn)/2),
      'balanced_accuracy':float(balanced_accuracy_score(yte,p>=0.5))
    }

Xtr=np.load(DATA/'X_development.npy').astype(float)
ytr=np.load(DATA/'y_development.npy').astype(int)
Xte=np.load(DATA/'X_sealed_validation.npy').astype(float)
yte=np.load(DATA/'y_sealed_validation.npy').astype(int)
feat=pd.read_csv(DATA/'features_p2000.csv')
pars=pd.read_csv(PKG/'METHOD_ARTIFACTS/selective_FINAL/FINAL_SELECTOR_PARAMS.csv')
selective=pd.read_csv(PKG/'METHOD_ARTIFACTS/selective_FINAL/FINAL_selective_SELECTED_FEATURES.csv')
m12=pd.read_csv(PKG/'METHOD_ARTIFACTS/global_INTERNAL_D20/global_FINAL_SELECTED_FEATURES.csv')
ref=pd.read_csv(REF)

supports={}
for sel in ['LASSO','ELASTICNET','SIS']:
    rr=pars[pars.selector==sel].iloc[0]
    if sel=='SIS':
        D=sis_score(Xtr,ytr)
    else:
        D=sparse_score(Xtr,ytr,float(rr.C),float(rr.l1_ratio))
    for k in [10,20,30]:
        supports[('reference',sel,k)]=topk(D,k)

for (sel,k),g in selective.groupby(['selector','k']):
    g=g.sort_values('rank')
    supports[('selective',str(sel),int(k))]=g.feature_index.to_numpy(int)

for r in m12.itertuples():
    supports[(str(r.method),str(r.selector),int(r.k))]=np.array([int(x) for x in str(r.selected_indices).split('|')],int)

rows=[]
for method in ['reference','global','global_certainty','selective']:
    for sel in ['LASSO','ELASTICNET','SIS']:
        for k in [10,20,30]:
            cols=supports[(method,sel,k)]
            if len(cols)!=k: raise RuntimeError(f'support size mismatch {method} {sel} {k}')
            met=eval_support(Xtr,ytr,Xte,yte,cols)
            rows.append({'method':method,'selector':sel,'k':k,**met,
                         'selected_indices':'|'.join(map(str,cols.tolist())),
                         'selected_genes':'|'.join(feat.iloc[cols].gene_symbol.astype(str).tolist())})
out=pd.DataFrame(rows)
out.to_csv(OUT/'REPRODUCED_SEALED_PRIMARY_RESULTS.csv',index=False)

cmp=out.merge(ref[['method','selector','k','auroc','ap_positive','ap_negative','macro_ap','balanced_accuracy','selected_indices']],
              on=['method','selector','k'],suffixes=('_reproduced','_reference'),validate='one_to_one')
metric_cols=['auroc','ap_positive','ap_negative','macro_ap','balanced_accuracy']
for c in metric_cols:
    cmp[f'abs_diff_{c}']=(cmp[f'{c}_reproduced']-cmp[f'{c}_reference']).abs()
cmp['support_exact']=cmp.selected_indices_reproduced==cmp.selected_indices_reference
maxdiff=max(float(cmp[f'abs_diff_{c}'].max()) for c in metric_cols)
ok=bool(maxdiff<=1e-12 and cmp.support_exact.all() and len(cmp)==36)
cmp.to_csv(OUT/'REPRODUCTION_COMPARISON.csv',index=False)
status={
 'status':'PASS' if ok else 'FAIL',
 'n_primary_cells':len(cmp),
 'all_supports_exact':bool(cmp.support_exact.all()),
 'max_abs_metric_difference':maxdiff,
 'tolerance':1e-12,
 'sklearn_reproduction_note':'Expected exact/near-machine-precision match under the frozen environment.'
}
(OUT/'REPRODUCTION_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
if not ok: raise SystemExit(2)
