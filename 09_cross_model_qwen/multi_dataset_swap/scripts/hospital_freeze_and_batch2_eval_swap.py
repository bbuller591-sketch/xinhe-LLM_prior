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
import json,hashlib,sys
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT))
W=ROOT/'selective_qwen3_32b_local_modelswap_20260924'
DATA=ROOT/'hospital_osteoporosis_dataonly_pilot_20260917'
CAN=ROOT/'hospital_osteoporosis_canonical_20260917'
B1=W/'06_HOSPITAL/downstream_batch1'
OUT=W/'06_HOSPITAL/final_temporal_eval';PRED=OUT/'FROZEN_PREDICTORS';PRED.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(DATA/'scripts'));import v2_core as V

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

sets=pd.read_csv(B1/'FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv')
man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json'))
all_features=list(man['selectable_features'])
full=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(
 pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),
 on=['patient_uid','site']).reset_index(drop=True)
prim=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
assert len(prim)==892 and prim.patient_uid.nunique()==664
Xall=prim[all_features].to_numpy(float);site=prim.site.to_numpy();y=prim.y.to_numpy(int);groups=prim.patient_uid.to_numpy()

def choose_lam(X,s,y,g,seed=91019):
 rows=[];best=(-1,None)
 cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
 for lam in V.LAM_GRID:
  vals=[]
  for tr,va in cv.split(np.zeros(len(y)),y,g):
   ww,_,sc,_=V.fit_full(X[tr],s[tr],y[tr],lam,kind='l2')
   vals.append(float(roc_auc_score(y[va],V.predict_full(ww,sc,X[va],s[va]))))
  m=float(np.mean(vals));rows.append({'lambda':float(lam),'mean_auroc':m,'fold_aurocs':'|'.join(map(str,vals))})
  if m>best[0]:best=(m,float(lam))
 return best[1],best[0],pd.DataFrame(rows)

# Freeze model artifacts using Batch1 only.
unique={};index=[]
for r in sets.itertuples():
 chosen=set(str(r.selected_set).split('|'));ordered=tuple(f for f in all_features if f in chosen)
 assert len(ordered)==int(r.k)
 unique.setdefault(ordered,[]).append((str(r.method),int(r.k),float(r.selected_hyperparam)))
for n,(fts,uses) in enumerate(unique.items(),1):
 ids=[all_features.index(f) for f in fts];X=Xall[:,ids];lam,cv,curve=choose_lam(X,site,y,groups);mid=f'PREDSET{n:02d}'
 curve.to_csv(PRED/f'{mid}_L2_LAMBDA_CV.csv',index=False)
 ww,info,sc,nctx=V.fit_full(X,site,y,lam,kind='l2');med,mu,sd=sc
 np.savez_compressed(PRED/f'{mid}_MODEL.npz',w=ww,med=med,mu=mu,sd=sd,n_ctx=np.array([nctx],int))
 meta={'model_id':mid,'feature_order':list(fts),'uses':[{'method':a,'k':b,'method_hyperparam':c} for a,b,c in uses],
       'predictor_l2_lambda':lam,'batch1_inner_cv_mean_auroc':cv,'training_scope':'Batch1 only'}
 (PRED/f'{mid}_META.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 for method,k,hp in uses:index.append({'method':method,'k':k,'method_hyperparam':hp,'model_id':mid,'feature_order':'|'.join(fts),'predictor_l2_lambda':lam,'batch1_inner_cv_mean_auroc':cv})
idx=pd.DataFrame(index).sort_values(['method','k']);idx.to_csv(PRED/'FINAL_PREDICTOR_INDEX.csv',index=False)
freeze={'status':'FROZEN_BEFORE_BATCH2_EVALUATION','batch2_used_for_selection':False,
        'selected_sets_sha256':sha(B1/'FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv'),
        'predictor_index_sha256':sha(PRED/'FINAL_PREDICTOR_INDEX.csv'),'n_rows':len(idx),'n_unique_models':idx.model_id.nunique()}
(OUT/'PRE_BATCH2_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')

# First outcome-dependent operation: evaluate frozen predictors on Batch2.
Xsite=pd.read_csv(CAN/'canonical/site_level_X.csv');Y=pd.read_csv(CAN/'canonical/site_level_y.csv')
Xsite=Xsite.copy();Xsite['y']=pd.to_numeric(Y.iloc[:,0]).astype(int).to_numpy()
b2=Xsite[Xsite.cohort.astype(str).eq('batch2')].copy()
ext=b2[b2.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
assert len(ext)==727 and ext.patient_uid.nunique()==538
ye=ext.y.to_numpy(int);se=ext.site.to_numpy();uid=ext.patient_uid.astype(str).to_numpy()
preds={};modelmetrics={}
for mid,g in idx.groupby('model_id'):
 r=g.iloc[0];fts=str(r.feature_order).split('|');Z=ext[fts].copy()
 for c in fts:
  Z[c]=Z[c].astype(str).str.strip().map({'女':1.0,'男':0.0}) if c=='DXA_性别' else pd.to_numeric(Z[c],errors='coerce')
 a=np.load(PRED/f'{mid}_MODEL.npz');p=V.predict_full(a['w'],(a['med'],a['mu'],a['sd']),Z.to_numpy(float),se);preds[mid]=p
 modelmetrics[mid]={'auroc':float(roc_auc_score(ye,p)),'auprc':float(average_precision_score(ye,p)),
                    'balanced_accuracy':float(balanced_accuracy_score(ye,p>=.5))}
rows=[]
for r in idx.itertuples():
 m=modelmetrics[r.model_id];rows.append({'method':r.method,'k':int(r.k),'method_hyperparam':float(r.method_hyperparam),
 'model_id':r.model_id,'predictor_l2_lambda':float(r.predictor_l2_lambda),'selected_set':r.feature_order,**m})
res=pd.DataFrame(rows).sort_values(['k','method'])
for metric in ['auroc','auprc','balanced_accuracy']:
 base={int(k):float(g[g.method=='reference'].iloc[0][metric]) for k,g in res.groupby('k')}
 res['delta_'+metric+'_vs_reference']=[float(r[metric])-base[int(r.k)] for _,r in res.iterrows()]
res.to_csv(OUT/'BATCH2_PRIMARY_RESULTS.csv',index=False)

# Same original 2000 patient-cluster paired bootstrap, reporting only.
rng=np.random.default_rng(20260918);patients=np.unique(uid);boot=[]
for bb in range(2000):
 samp=rng.choice(patients,size=len(patients),replace=True);inds=np.concatenate([np.where(uid==u)[0] for u in samp]);yy=ye[inds]
 if len(np.unique(yy))<2:continue
 for r in idx.itertuples():
  pp=preds[r.model_id][inds]
  boot.append({'b':bb,'method':r.method,'k':int(r.k),'auroc':float(roc_auc_score(yy,pp)),
               'auprc':float(average_precision_score(yy,pp)),'balanced_accuracy':float(balanced_accuracy_score(yy,pp>=.5))})
bd=pd.DataFrame(boot);ci=[]
for (method,k),g in bd.groupby(['method','k']):
 rec={'method':method,'k':int(k)}
 for metric in ['auroc','auprc','balanced_accuracy']:
  v=g[metric].to_numpy();rec[metric+'_lo']=float(np.quantile(v,.025));rec[metric+'_hi']=float(np.quantile(v,.975))
  base=bd[(bd.method=='reference')&(bd.k==k)][['b',metric]].rename(columns={metric:'base'});z=g[['b',metric]].merge(base,on='b');dv=(z[metric]-z.base).to_numpy()
  rec['delta_'+metric+'_vs_reference_lo']=float(np.quantile(dv,.025));rec['delta_'+metric+'_vs_reference_hi']=float(np.quantile(dv,.975))
 ci.append(rec)
res.merge(pd.DataFrame(ci),on=['method','k'],how='left').to_csv(OUT/'BATCH2_PRIMARY_RESULTS_WITH_CLUSTER_BOOTSTRAP.csv',index=False)
status={'status':'COMPLETE_NO_BATCH2_RETUNING','n_rows':len(res),'batch2_rows':len(ext),'batch2_patients':int(ext.patient_uid.nunique()),
        'no_batch2_reselection_or_retuning':True,'results_sha256':sha(OUT/'BATCH2_PRIMARY_RESULTS.csv')}
(OUT/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print(res.to_string(index=False))
