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
import json, argparse
import pandas as pd

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
OUTROOT=ROOT/'06_selective_ROUTING'
KGRID=[10,20,30]
SELECTORS=['LASSO','ELASTICNET','SIS']

ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); args=ap.parse_args()
task=args.task; out=OUTROOT/task
summaries=[]; missing=[]
for fold in range(1,6):
  for selector in SELECTORS:
    sdir=out/f'fold{fold}_{selector}'
    sf=sdir/'RESAMPLE_STATUS.json'
    if not sf.exists():
      missing.append(str(sf)); continue
    st=json.load(open(sf))
    if not st.get('all_converged',False):
      raise RuntimeError(f'NONCONVERGED {sf}')
    for k in KGRID:
      fp=sdir/f'K{k}_PAIR_CONFUSION.csv'
      if not fp.exists():
        missing.append(str(fp)); continue
      cf=pd.read_csv(fp)
      nact=int((cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION').sum()) if len(cf) else 0
      summaries.append({'task':task,'fold':fold,'selector':selector,'k':k,
        'n_relevant_pairs':int(len(cf)),'n_actionable_pairs':nact,
        'max_actionable':float(cf.actionable_boundary_score.max()) if len(cf) else 0.0,
        'n_candidate_nodes':int(len(set(cf.feature_A)|set(cf.feature_B))) if len(cf) else 0,
        'min_n_positive_scores':int(st['min_n_positive_scores']),
        'all_converged':bool(st['all_converged'])})
if missing:
  print(json.dumps({'status':'INCOMPLETE','n_missing':len(missing),'missing':missing[:30]},indent=2))
  raise SystemExit(3)
sm=pd.DataFrame(summaries)
sm.to_csv(out/'ROUTING_SUMMARY.csv',index=False)
U=[]
for r in summaries:
  cf=pd.read_csv(out/f"fold{r['fold']}_{r['selector']}"/f"K{r['k']}_PAIR_CONFUSION.csv")
  if len(cf)==0 or 'pair_type' not in cf.columns:
    continue
  a=cf[cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION'].copy()
  if len(a): U.append(a)
U=pd.concat(U,ignore_index=True)
U['unordered_pair']=U.apply(lambda r:'||'.join(sorted([str(r.feature_A),str(r.feature_B)])),axis=1)
uu=(U.groupby('unordered_pair',as_index=False)
      .agg(feature_A=('feature_A','first'),feature_B=('feature_B','first'),
           n_fold_selector_k_routes=('unordered_pair','size'),
           n_outer_folds=('fold','nunique'),
           n_selectors=('selector','nunique'),
           n_k_values=('k','nunique'),
           max_actionable=('actionable_boundary_score','max'),
           mean_actionable=('actionable_boundary_score','mean')))
uu.to_csv(out/'ACTIONABLE_PAIR_UNION_FOR_CACHE.csv',index=False)
status={'task':task,'status':'ROUTING_COMPLETE','B':200,'fraction':0.8,'window':5,
 'actionable_threshold':0.25,'k_grid':KGRID,'selectors':SELECTORS,
 'n_unique_actionable_pairs_union':int(len(uu)),
 'n_actionable_route_instances':int(len(U)),
 'all_fold_selector_caches_present':True,'all_sparse_resamples_converged':True,
 'uses_outer_validation_for_routing':False,'uses_sealed_validation':False,'uses_llm':False}
(out/'ROUTING_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
print('\nBy selector/k:')
print(sm.groupby(['selector','k']).agg(actionable=('n_actionable_pairs','sum'),relevant=('n_relevant_pairs','sum')).to_string())
