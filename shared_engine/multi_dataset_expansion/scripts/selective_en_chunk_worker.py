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
import argparse,time,json,sys
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from skglm.penalties import L1_plus_L2
from skglm.solvers import ProxNewton
from skglm import GeneralizedLinearEstimator
sys.path.insert(0,str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion/scripts'))
import run_strict_nested_selective_routing as R

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
ap=argparse.ArgumentParser()
ap.add_argument('--fold',type=int,required=True)
ap.add_argument('--start',type=int,required=True)
ap.add_argument('--end',type=int,required=True)
ap.add_argument('--p0',type=int,default=2000)
args=ap.parse_args()
assert 0<=args.start<args.end<=R.B
fold=args.fold; p0=args.p0
task='SEPSIS_GSE65682'; selector='ELASTICNET'
X=np.load(ROOT/'03_FROZEN_DATA'/task/'X_development.npy').astype(float)
y=np.load(ROOT/'03_FROZEN_DATA'/task/'y_development.npy').astype(int)
sp=pd.read_csv(ROOT/'04_reference'/task/'OUTER_SPLITS.csv')
sa=pd.read_csv(ROOT/'04_reference'/task/'reference_SELECTOR_AUDIT.csv')
tr=sp[(sp.fold==fold)&(sp.role=='outer_train')].sample_index.to_numpy(int)
Xtr=X[tr]; ytr=y[tr]
row=sa[(sa.fold==fold)&(sa.selector==selector)].iloc[0]
C=float(row.chosen_C); ratio=float(row.chosen_l1_ratio)
nchunk=args.end-args.start
scores=np.zeros((nchunk,X.shape[1]),dtype=np.float32)
ranks=np.zeros_like(scores); nz=np.zeros(nchunk,dtype=np.int32); conv=np.ones(nchunk,dtype=np.int8)
stops=np.zeros(nchunk,dtype=np.float64)
t0=time.time()
for ii,b in enumerate(range(args.start,args.end)):
    rng=np.random.default_rng(R.BASE_SEED+1000000+fold*10000+b)
    take=[]
    for c in np.unique(ytr):
        idx=np.where(ytr==c)[0]; n=max(2,int(round(R.FRAC*len(idx))))
        take.extend(rng.choice(idx,size=min(n,len(idx)),replace=False).tolist())
    take=np.array(sorted(take),dtype=int)
    XX=Xtr[take]; yy=ytr[take]
    Z=StandardScaler().fit_transform(XX)
    n=len(yy); n1=int((yy==1).sum()); n0=n-n1
    sw=np.where(yy==1,n/(2*n1),n/(2*n0)).astype(np.float64)
    alpha=1/(C*sw.sum())
    est=GeneralizedLinearEstimator(datafit=R.WeightedLogistic(sw),penalty=L1_plus_L2(alpha,ratio),
        solver=ProxNewton(p0=p0,max_iter=50,max_pn_iter=5000,tol=1e-8,fit_intercept=True,warm_start=False,verbose=0))
    est.fit(Z,yy)
    D=np.abs(np.asarray(est.coef_).reshape(-1))
    scores[ii]=D.astype(np.float32); ranks[ii]=rankdata(-D,method='average').astype(np.float32)
    nz[ii]=int((D>1e-12).sum()); stops[ii]=float(est.stop_crit_); conv[ii]=int(stops[ii]<=1e-7)
od=ROOT/'06_selective_ROUTING'/task/f'fold{fold}_{selector}_CHUNKS_P0{p0}'
od.mkdir(parents=True,exist_ok=True)
fp=od/f'chunk_{args.start:03d}_{args.end:03d}.npz'
np.savez_compressed(fp,scores=scores,ranks=ranks,nz=nz,converged=conv,stop_crit=stops,
                    start=args.start,end=args.end,features=np.array(pd.read_csv(ROOT/'03_FROZEN_DATA'/task/'features_p2000.csv').gene_symbol.astype(str).tolist()))
status={'fold':fold,'selector':selector,'start':args.start,'end':args.end,'p0':p0,
        'all_converged':bool(np.all(conv==1)),'max_stop_crit':float(stops.max()),
        'runtime_sec':round(time.time()-t0,2),'C':C,'l1_ratio':ratio}
(od/f'chunk_{args.start:03d}_{args.end:03d}.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status))
