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
import sys,json,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
sys.path.insert(0,str(ROOT/'scripts'))
import run_strict_nested_selective_routing as route

SEED=2026091901
LAMS=[0,0.03,0.1,0.3,1,3,10]
KGRID=[10,20,30]
SELECTORS=['LASSO','ELASTICNET','SIS']

def reference_sparse_score(X,y,C,l1_ratio,fold):
    sc=StandardScaler().fit(X); Z=sc.transform(X)
    penalty='l1' if float(l1_ratio)==1.0 else 'elasticnet'
    mod=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=penalty,
        l1_ratio=None if penalty=='l1' else float(l1_ratio),max_iter=5000,tol=1e-4,
        random_state=SEED+fold,fit_intercept=True,n_jobs=1)
    mod.fit(Z,y)
    return np.abs(mod.coef_[0]),int(mod.n_iter_[0]),bool(int(mod.n_iter_[0])<mod.max_iter)

def sis_score(X,y):
    yc=y-y.mean(); xc=X-X.mean(axis=0)
    den=np.sqrt((xc*xc).sum(axis=0)*(yc*yc).sum())
    return np.nan_to_num(np.abs((xc*yc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0)

def order(D): return np.lexsort((np.arange(len(D)),-D))
def data_anchor(D):
    r=rankdata(-D,method='average'); u=1-(r-0.5)/len(D)
    return norm.ppf(np.clip(u,1e-6,1-1e-6))
def stable_topk(z,D,k): return np.lexsort((np.arange(len(z)),-D,-z))[:k]

for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
    print('\nTASK',task,flush=True)
    X=np.load(ROOT/'03_FROZEN_DATA'/task/'X_development.npy').astype(float)
    y=np.load(ROOT/'03_FROZEN_DATA'/task/'y_development.npy').astype(int)
    sp=pd.read_csv(ROOT/'04_reference'/task/'OUTER_SPLITS.csv')
    sa=pd.read_csv(ROOT/'04_reference'/task/'reference_SELECTOR_AUDIT.csv')
    fr=pd.read_csv(ROOT/'04_reference'/task/'reference_FOLD_RESULTS.csv')
    rows=[]
    for fold in range(1,6):
      tr=sp[(sp.fold==fold)&(sp.role=='outer_train')].sample_index.to_numpy(int)
      for sel in SELECTORS:
        rr=sa[(sa.fold==fold)&(sa.selector==sel)].iloc[0]
        if sel=='SIS':
            D=sis_score(X[tr],y[tr]); nit=0; conv=True
        else:
            D,nit,conv=reference_sparse_score(X[tr],y[tr],rr.chosen_C,rr.chosen_l1_ratio,fold)
        sD=data_anchor(D)
        for k in KGRID:
            got=stable_topk(sD,D,k)
            ref=fr[(fr.fold==fold)&(fr.selector==sel)&(fr.k==k)].iloc[0]
            exp=np.array([int(x) for x in str(ref.selected_indices).split('|')],int)
            exact=bool(np.array_equal(got,exp))
            rows.append({'task':task,'fold':fold,'selector':sel,'k':k,
                         'eta0_exact_reference':exact,'reference_solver_iter':nit,'reference_solver_converged':conv})
            if not exact:
                raise RuntimeError(f'ETA0_reference_REPRO_FAIL {task} fold={fold} {sel} k={k}')
    out=ROOT/'12_selective_INTERNAL'/task
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'ETA0_reference_REPRODUCTION_GATE.csv',index=False)
    print(pd.DataFrame(rows).groupby('selector').agg(n=('eta0_exact_reference','size'),all_exact=('eta0_exact_reference','all'),
          outer_saga_all_converged=('reference_solver_converged','all'),max_iter=('reference_solver_iter','max')).to_string(),flush=True)
print('\nALL_ETA0_REPRODUCTION_GATES_PASS',flush=True)
