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
import hashlib,json
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ORIG=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
W=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap'))
TASK='BREAST_GSE25055_GSE25065'
TUNE=W/'04_BREAST/final_development_tuning'/TASK
OUT=W/'04_BREAST/sealed_validation';OUT.mkdir(parents=True,exist_ok=True)
SRC=ORIG/'03_FROZEN_DATA'/TASK
SEED=2026091901

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sis_score(X,y):
 yc=y-y.mean();xc=X-X.mean(0);den=np.sqrt((xc*xc).sum(0)*(yc*yc).sum())
 return np.nan_to_num(np.abs((xc*yc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0.)
def sparse_score(X,y,C,ratio):
 sc=StandardScaler().fit(X);Z=sc.transform(X);pen='l1' if float(ratio)==1 else 'elasticnet'
 m=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=pen,
   l1_ratio=None if pen=='l1' else float(ratio),max_iter=5000,tol=1e-4,random_state=SEED,n_jobs=1).fit(Z,y)
 if int(m.n_iter_[0])>=m.max_iter: raise RuntimeError('FINAL_reference_SPARSE_NONCONVERGENCE')
 return np.abs(m.coef_[0])
def top(D,k):return np.lexsort((np.arange(len(D)),-D))[:k]
def downstream(Xtr,ytr,Xte,yte,cols):
 sc=StandardScaler().fit(Xtr[:,cols]);A=sc.transform(Xtr[:,cols]);B=sc.transform(Xte[:,cols])
 m=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=SEED).fit(A,ytr)
 p=m.predict_proba(B)[:,1];ap=average_precision_score(yte,p);apn=average_precision_score(1-yte,1-p)
 return {'auroc':float(roc_auc_score(yte,p)),'ap_positive':float(ap),'ap_negative':float(apn),
         'macro_ap':float((ap+apn)/2),'balanced_accuracy':float(balanced_accuracy_score(yte,p>=.5))}

# Development-only support formation and freeze BEFORE opening sealed outcomes.
Xtr=np.load(SRC/'X_development.npy').astype(float);ytr=np.load(SRC/'y_development.npy').astype(int)
feat=pd.read_csv(SRC/'features_p2000.csv')
pars=pd.read_csv(TUNE/'FINAL_SELECTOR_PARAMS.csv')
selective=pd.read_csv(TUNE/'FINAL_selective_SELECTED_FEATURES.csv')
supports=[];support_rows=[]
for sel in ['LASSO','ELASTICNET','SIS']:
 rr=pars[pars.selector==sel].iloc[0]
 D=sis_score(Xtr,ytr) if sel=='SIS' else sparse_score(Xtr,ytr,float(rr.C),float(rr.l1_ratio))
 for k in [10,20,30]:
  ids=top(D,k);supports.append(('reference',sel,k,0.,ids))
  q=selective[(selective.selector==sel)&(selective.k==k)].sort_values('rank')
  mid=q.feature_index.to_numpy(int);lam=float(q.chosen_lam.iloc[0])
  if len(mid)!=k:raise RuntimeError('selective_SUPPORT_SIZE')
  supports.append(('selective',sel,k,lam,mid))
for method,sel,k,hp,ids in supports:
 for rank,j in enumerate(ids,1):
  support_rows.append({'method':method,'selector':sel,'k':k,'rank':rank,'feature_index':int(j),
                       'gene_symbol':str(feat.iloc[j].gene_symbol),'tuning_strength':hp})
sf=pd.DataFrame(support_rows);sf.to_csv(OUT/'FROZEN_reference_selective_SUPPORTS_PRE_SEALED.csv',index=False)
freeze={'status':'QWEN3_32B_reference_selective_SUPPORTS_FROZEN_BEFORE_SEALED_EVALUATION','task':TASK,
        'uses_sealed_for_tuning':False,'support_sha256':sha(OUT/'FROZEN_reference_selective_SUPPORTS_PRE_SEALED.csv'),
        'final_selected_lam_sha256':sha(TUNE/'FINAL_SELECTED_ETA.csv'),
        'final_selective_support_sha256':sha(TUNE/'FINAL_selective_SELECTED_FEATURES.csv')}
(OUT/'PRE_SEALED_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')

# One-time evaluation with no retuning.
Xte=np.load(SRC/'X_sealed_validation.npy').astype(float);yte=np.load(SRC/'y_sealed_validation.npy').astype(int)
rows=[]
for method,sel,k,hp,ids in supports:
 met=downstream(Xtr,ytr,Xte,yte,ids)
 rows.append({'task':TASK,'method':method,'selector':sel,'k':k,'tuning_strength':hp,
              'n_development':len(ytr),'n_test':len(yte),**met,
              'selected_indices':'|'.join(map(str,ids.tolist())),
              'selected_genes':'|'.join(feat.iloc[ids].gene_symbol.astype(str))})
res=pd.DataFrame(rows)
for metric in ['auroc','macro_ap','balanced_accuracy']:
 base=res[res.method=='reference'].set_index(['selector','k'])[metric].to_dict()
 res['delta_'+metric+'_vs_reference']=[float(r[metric])-float(base[(r.selector,r.k)]) for _,r in res.iterrows()]
res.to_csv(OUT/'SEALED_reference_selective_RESULTS.csv',index=False)
status={'status':'COMPLETE_NO_TEST_RETUNING','task':TASK,'n_rows':len(res),'test_n':len(yte),
        'test_positive':int(yte.sum()),'no_test_based_tuning':True,'results_sha256':sha(OUT/'SEALED_reference_selective_RESULTS.csv')}
(OUT/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print(res.to_string(index=False))
