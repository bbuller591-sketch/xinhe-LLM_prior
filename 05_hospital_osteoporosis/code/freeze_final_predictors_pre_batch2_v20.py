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
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
DATA=Path(str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917'))
sys.path.insert(0,str(DATA/'scripts'))
import v2_core as V

OUT=ROOT/'10_PRE_BATCH2_FREEZE/FINAL_PREDICTORS'
OUT.mkdir(parents=True,exist_ok=True)
sets=pd.read_csv(ROOT/'09_BATCH1_DOWNSTREAM_V1_9/FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv')
man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json'))
all_features=list(man['selectable_features'])
full=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(
    pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),
    on=['patient_uid','site']).reset_index(drop=True)
prim=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
assert len(prim)==892 and prim.patient_uid.nunique()==664
Xall=prim[all_features].to_numpy(float); site=prim.site.to_numpy(); y=prim.y.to_numpy(int); groups=prim.patient_uid.to_numpy()

def choose_lam(X,s,y,g,seed=91019):
    rows=[]; best=(-1,None)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    splits=list(cv.split(np.zeros(len(y)),y,g))
    for lam in V.LAM_GRID:
        aucs=[]
        for tri,vai in splits:
            w,info,sc,nctx=V.fit_full(X[tri],s[tri],y[tri],lam,kind='l2')
            p=V.predict_full(w,sc,X[vai],s[vai])
            aucs.append(float(roc_auc_score(y[vai],p)))
        m=float(np.mean(aucs))
        rows.append({'lambda':float(lam),'mean_auroc':m,'sd_auroc':float(np.std(aucs,ddof=1)),'fold_aurocs':'|'.join(map(str,aucs))})
        if m>best[0]: best=(m,float(lam))
    return best[1],best[0],pd.DataFrame(rows)

# Canonicalize to data feature order so identical sets share identical predictor artifact.
unique={}
for r in sets.itertuples():
    chosen=set(str(r.selected_set).split('|'))
    ordered=tuple(f for f in all_features if f in chosen)
    assert len(ordered)==int(r.k)
    unique.setdefault(ordered,[]).append((str(r.method),int(r.k),float(r.selected_hyperparam)))

model_index=[]
for mi,(feat_tuple,uses) in enumerate(unique.items(),1):
    inds=[all_features.index(f) for f in feat_tuple]
    X=Xall[:,inds]
    lam,inner,curve=choose_lam(X,site,y,groups)
    model_id='PREDSET%02d'%mi
    curve.to_csv(OUT/f'{model_id}_L2_LAMBDA_CV.csv',index=False)
    w,info,sc,nctx=V.fit_full(X,site,y,lam,kind='l2')
    med,mu,sd=sc
    np.savez_compressed(OUT/f'{model_id}_MODEL.npz',w=w,med=med,mu=mu,sd=sd,n_ctx=np.array([nctx],dtype=int))
    meta={'model_id':model_id,'feature_order':list(feat_tuple),'uses':[{'method':a,'k':b,'method_hyperparam':c} for a,b,c in uses],
          'predictor_kind':'masked_l2_logistic','predictor_l2_lambda':lam,'batch1_inner_cv_mean_auroc':inner,
          'lambda_grid':[float(x) for x in V.LAM_GRID],'inner_cv':'5-fold StratifiedGroupKFold by patient_uid, seed=91019',
          'training_scope':'full Batch1 primary lumbar+hip, 892 rows / 664 patients','n_ctx':int(nctx),
          'fit_info':{k:(float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else v) for k,v in info.items()}}
    (OUT/f'{model_id}_META.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    for method,k,hp in uses:
        model_index.append({'method':method,'k':k,'method_hyperparam':hp,'model_id':model_id,'feature_order':'|'.join(feat_tuple),
                            'predictor_l2_lambda':lam,'batch1_inner_cv_mean_auroc':inner})
idx=pd.DataFrame(model_index).sort_values(['method','k'])
idx.to_csv(OUT/'FINAL_PREDICTOR_INDEX.csv',index=False)
assert len(idx)==8

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
recs=[]
for p in sorted(OUT.iterdir()):
    if p.is_file():
        recs.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'status':'FINAL_PREDICTORS_FROZEN_BEFORE_BATCH2_OPEN','batch2_accessed':False,'n_unique_predictors':len(unique),'n_method_k_rows':8,
          'predictor_index':idx.to_dict('records'),'artifacts':recs}
raw=json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':'))
manifest['manifest_sha256']=hashlib.sha256(raw.encode()).hexdigest()
(OUT/'FINAL_PREDICTOR_FREEZE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':manifest['status'],'n_unique_predictors':len(unique),'manifest_sha256':manifest['manifest_sha256']},ensure_ascii=False,indent=2))
print(idx.to_string(index=False))
