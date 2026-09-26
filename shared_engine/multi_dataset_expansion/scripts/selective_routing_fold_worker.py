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
import numpy as np, pandas as pd
sys.path.insert(0,str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion/scripts'))
import run_strict_nested_selective_routing as R

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
FROZEN=ROOT/'03_FROZEN_DATA'
reference=ROOT/'04_reference'
OUTROOT=ROOT/'06_selective_ROUTING'

ap=argparse.ArgumentParser()
ap.add_argument('--task',required=True)
ap.add_argument('--fold',type=int,required=True)
ap.add_argument('--selector',choices=['LASSO','ELASTICNET','SIS'],required=True)
args=ap.parse_args()

task=args.task; fold=args.fold; selector=args.selector
task_idx=0 if task.startswith('BREAST') else 1
src=FROZEN/task; reference=reference/task; out=OUTROOT/task
X=np.load(src/'X_development.npy').astype(float)
y=np.load(src/'y_development.npy').astype(int)
feat=pd.read_csv(src/'features_p2000.csv')
features=feat.gene_symbol.astype(str).tolist()
splits=pd.read_csv(reference/'OUTER_SPLITS.csv')
sa=pd.read_csv(reference/'reference_SELECTOR_AUDIT.csv')
tr=splits[(splits.fold==fold)&(splits.role=='outer_train')].sample_index.to_numpy(int)
Xtr=X[tr]; ytr=y[tr]
sdir=out/f'fold{fold}_{selector}'; sdir.mkdir(parents=True,exist_ok=True)
npz=sdir/'RESAMPLE_SCORES_RANKS.npz'
params={}
if selector!='SIS':
    row=sa[(sa.fold==fold)&(sa.selector==selector)].iloc[0]
    params={'C':float(row.chosen_C),'l1_ratio':float(row.chosen_l1_ratio)}
if npz.exists():
    print('CACHE_EXISTS',task,fold,selector,flush=True)
    raise SystemExit(0)

t=time.time()
scores,ranks,nz,conv=R.resample_scores(Xtr,ytr,selector,params,task_idx,fold)
tmp=sdir/'RESAMPLE_SCORES_RANKS.npz.tmp.npz'
np.savez_compressed(tmp,scores=scores,ranks=ranks,nz=nz,converged=conv,
                    features=np.array(features),train_indices=tr)
tmp.replace(npz)
status={
 'task':task,'fold':fold,'selector':selector,'B':R.B,'fraction':R.FRAC,
 'outer_train_n':int(len(tr)),'positive_n':int(ytr.sum()),'params':params,
 'all_converged':bool(np.all(conv==1)),
 'solver':('SIS_PEARSON' if selector=='SIS' else 'SKGLM_WEIGHTED_PROXNEWTON_EXACT'),
 'solver_tol':(None if selector=='SIS' else 1e-8),
 'solver_p0':(None if selector=='SIS' else (50 if params['l1_ratio']>=0.999999 else 400)),
 'mean_n_positive_scores':float(nz.mean()),'min_n_positive_scores':int(nz.min()),
 'runtime_sec':round(time.time()-t,2),'uses_outer_validation':False,
 'uses_sealed_validation':False,'uses_llm':False}
(sdir/'RESAMPLE_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
for k in R.KGRID:
    cf=R.confusion(scores,ranks,features,k)
    cf.insert(0,'selector',selector); cf.insert(0,'fold',fold); cf.insert(0,'task',task)
    fp=sdir/f'K{k}_PAIR_CONFUSION.csv'
    tmpcsv=sdir/f'K{k}_PAIR_CONFUSION.csv.tmp'
    cf.to_csv(tmpcsv,index=False); tmpcsv.replace(fp)
    nact=int((cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION').sum()) if len(cf) else 0
    print(task,fold,selector,'k',k,'relevant',len(cf),'actionable',nact,flush=True)
print('WORKER_DONE',json.dumps(status),flush=True)
