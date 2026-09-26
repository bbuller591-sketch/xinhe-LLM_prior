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
import argparse,json,time,sys
import numpy as np,pandas as pd
sys.path.insert(0,str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion/scripts'))
import run_strict_nested_selective_routing as R

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--p0',type=int,default=2000)
args=ap.parse_args(); fold=args.fold; p0=args.p0
task='SEPSIS_GSE65682'; selector='ELASTICNET'
chunkdir=ROOT/'06_selective_ROUTING'/task/f'fold{fold}_{selector}_CHUNKS_P0{p0}'
ranges=[(s,s+25) for s in range(0,200,25)]
missing=[str(chunkdir/f'chunk_{s:03d}_{e:03d}.npz') for s,e in ranges if not (chunkdir/f'chunk_{s:03d}_{e:03d}.npz').exists()]
if missing:
 print(json.dumps({'status':'INCOMPLETE','missing':missing},indent=2)); raise SystemExit(3)
parts=[]; statuses=[]; features=None
for s,e in ranges:
 z=np.load(chunkdir/f'chunk_{s:03d}_{e:03d}.npz',allow_pickle=True)
 assert int(z['start'])==s and int(z['end'])==e
 if features is None: features=[str(x) for x in z['features']]
 else: assert features==[str(x) for x in z['features']]
 parts.append((z['scores'],z['ranks'],z['nz'],z['converged'],z['stop_crit']))
 statuses.append(json.load(open(chunkdir/f'chunk_{s:03d}_{e:03d}.json')))
scores=np.concatenate([x[0] for x in parts],axis=0)
ranks=np.concatenate([x[1] for x in parts],axis=0)
nz=np.concatenate([x[2] for x in parts],axis=0)
conv=np.concatenate([x[3] for x in parts],axis=0)
stops=np.concatenate([x[4] for x in parts],axis=0)
assert scores.shape==(200,2000) and ranks.shape==(200,2000) and len(nz)==200
assert np.all(conv==1) and np.max(stops)<=1e-7

sp=pd.read_csv(ROOT/'04_reference'/task/'OUTER_SPLITS.csv')
sa=pd.read_csv(ROOT/'04_reference'/task/'reference_SELECTOR_AUDIT.csv')
tr=sp[(sp.fold==fold)&(sp.role=='outer_train')].sample_index.to_numpy(int)
row=sa[(sa.fold==fold)&(sa.selector==selector)].iloc[0]
sdir=ROOT/'06_selective_ROUTING'/task/f'fold{fold}_{selector}'
sdir.mkdir(parents=True,exist_ok=True)
np.savez_compressed(sdir/'RESAMPLE_SCORES_RANKS.npz',scores=scores,ranks=ranks,nz=nz,converged=conv,
                    features=np.array(features),train_indices=tr)
status={'task':task,'fold':fold,'selector':selector,'B':200,'fraction':0.8,
        'outer_train_n':int(len(tr)),'positive_n':int(pd.read_csv(ROOT/'03_FROZEN_DATA'/task/'development_samples.csv').iloc[tr].y.sum()),
        'params':{'C':float(row.chosen_C),'l1_ratio':float(row.chosen_l1_ratio)},
        'all_converged':True,'solver':'SKGLM_WEIGHTED_PROXNEWTON_EXACT',
        'solver_tol':1e-8,'solver_p0':p0,'parallel_chunking':'8x25 deterministic seed ranges',
        'max_stop_crit':float(stops.max()),'mean_n_positive_scores':float(nz.mean()),
        'min_n_positive_scores':int(nz.min()),
        'runtime_sec_sum_chunks':float(sum(x['runtime_sec'] for x in statuses)),
        'uses_outer_validation':False,'uses_sealed_validation':False,'uses_llm':False}
(sdir/'RESAMPLE_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
for k in R.KGRID:
 cf=R.confusion(scores,ranks,features,k)
 cf.insert(0,'selector',selector); cf.insert(0,'fold',fold); cf.insert(0,'task',task)
 cf.to_csv(sdir/f'K{k}_PAIR_CONFUSION.csv',index=False)
 print('fold',fold,'k',k,'relevant',len(cf),'actionable',int((cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION').sum()))
print(json.dumps(status,indent=2))
