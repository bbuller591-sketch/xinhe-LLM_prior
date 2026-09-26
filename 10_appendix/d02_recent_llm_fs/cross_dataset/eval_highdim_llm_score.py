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
import argparse,json,warnings
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
warnings.filterwarnings('ignore')
ROOT=Path(str(REPRO_ROOT));BASE=ROOT/'CROSS_DATASET_RECENT_LLM_FS_20260926'
ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['sepsis','breast'],required=True);a=ap.parse_args();ds=a.dataset
if ds=='sepsis':
    OUT=BASE/'GSE272769_SEPSIS';B=ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/01_FROZEN_DATA';X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);names=pd.read_csv(B/'features_p1500.csv').gene_symbol.astype(str).tolist();sp=pd.read_csv(B/'OUTER_SPLITS.csv');outer=[]
    for f in sorted(sp.fold.unique()):outer.append((sp[(sp.fold==f)&(sp.role=='outer_train')].sample_index.astype(int).to_numpy(),sp[(sp.fold==f)&(sp.role=='outer_validation')].sample_index.astype(int).to_numpy()))
    k=50;solver='lbfgs';seed=20260919;external=None
else:
    OUT=BASE/'BREAST_PCR';B=ROOT/'GSE25055_GSE25065_FORMAL_REPRODUCIBILITY_PACKAGE/DATA/FROZEN/BREAST_GSE25055_GSE25065';X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);Xe=np.load(B/'X_sealed_validation.npy').astype(float);ye=np.load(B/'y_sealed_validation.npy').astype(int);names=pd.read_csv(B/'features_p2000.csv').gene_symbol.astype(str).tolist();outer=[(np.arange(len(y)),None)];k=20;solver='lbfgs';seed=2026091901;external=(Xe,ye)
s=pd.read_csv(OUT/'LLM_SCORE/LLM_SCORE.csv').sort_values(['score','index'],ascending=[False,True]);cols=s['index'].astype(int).iloc[:k].to_numpy();sel=[names[j] for j in cols]
rows=[]
for fi,(tr,va) in enumerate(outer,1):
    sc=StandardScaler().fit(X[tr][:,cols]);m=LogisticRegression(C=1.,penalty='l2',solver=solver,class_weight='balanced',max_iter=5000,random_state=seed+fi).fit(sc.transform(X[tr][:,cols]),y[tr])
    if va is None:Bb,yb=external
    else:Bb,yb=X[va],y[va]
    p=m.predict_proba(sc.transform(Bb[:,cols]))[:,1];rows.append({'dataset':ds,'method':'LLM-Select Score','k':k,'fold':fi,'auroc':float(roc_auc_score(yb,p)),'ap':float(average_precision_score(yb,p)),'balanced_accuracy':float(balanced_accuracy_score(yb,p>=.5)),'selected_features':'|'.join(sel)})
df=pd.DataFrame(rows);df.to_csv(OUT/'LLM_SCORE_FOLD_RESULTS.csv',index=False);agg=df.groupby(['dataset','method','k'],as_index=False).agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),mean_ap=('ap','mean'));agg.to_csv(OUT/'LLM_SCORE_RESULT.csv',index=False);print(agg.to_string(index=False));print('TOP',sel)
