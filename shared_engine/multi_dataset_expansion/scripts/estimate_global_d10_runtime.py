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
import json,numpy as np,pandas as pd
ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
B=ROOT/'14_global_BROAD_FREEZE'
bud=json.load(open(B/'BUDGET/BROAD_global_TOKEN_COST_ESTIMATE.json'))
cal=bud['calibration_by_arm']
ARMS=[
 ('BREAST_GSE25055_GSE25065','ARM_GSE41998'),
 ('BREAST_GSE25055_GSE25065','ARM_GSE32646'),
 ('SEPSIS_GSE65682','ARM_EMTAB4451'),
 ('SEPSIS_GSE65682','ARM_EMTAB7581')]

tot={'logical':0,'reuse':0,'new':0,'prompt':0.0,'prompt_hi':0.0,'completion':0.0}
rows=[]
for task,arm in ARMS:
    E10=set(pd.read_csv(B/task/arm/'EDGES_D10.csv',dtype=str).unordered_pair_id)
    q=pd.read_parquet(B/'QUERIES'/f'{task}_global_BROAD_D20_QUERIES.parquet',
                      columns=['query_id','arm','unordered_pair_id','prompt_chars','reusable_from_selective'])
    d=q[(q.arm==arm)&q.unordered_pair_id.isin(E10)].copy()
    assert len(d)==2*len(E10)
    c=cal[arm]
    est=np.maximum(1,c['intercept']+c['slope']*d.prompt_chars.astype(float))
    hi=np.maximum(1,c['intercept']+c['slope']*d.prompt_chars.astype(float)+2*c['rmse'])
    newmask=~d.reusable_from_selective.astype(bool)
    nlog=len(d); reuse=int((~newmask).sum()); nnew=int(newmask.sum())
    pt=float(est[newmask].sum()); ph=float(hi[newmask].sum()); comp=float(nnew*c['mean_completion_tokens'])
    rows.append({'task':task,'arm':arm,'logical_queries':nlog,'reusable':reuse,'new_queries':nnew,
                 'estimated_new_prompt_tokens':int(round(pt)),'estimated_new_prompt_tokens_2rmse':int(round(ph))})
    tot['logical']+=nlog; tot['reuse']+=reuse; tot['new']+=nnew; tot['prompt']+=pt; tot['prompt_hi']+=ph; tot['completion']+=comp
out={'status':'D10_ONLY_BUDGET_FALLBACK_NOT_PRIMARY','rows':rows,
     'total_logical_queries':tot['logical'],'total_reusable':tot['reuse'],'total_new_queries':tot['new'],
     'estimated_new_prompt_tokens':int(round(tot['prompt'])),
     'estimated_new_prompt_tokens_conservative_2rmse':int(round(tot['prompt_hi'])),
     'estimated_new_completion_tokens':int(round(tot['completion'])),
     'cost_estimates':{
       'off_peak_all_cache_miss_usd':tot['prompt']/1e6*.15+tot['completion']/1e6*.60,
       'peak_all_cache_miss_usd':tot['prompt']/1e6*.30+tot['completion']/1e6*1.20,
       'off_peak_conservative_2rmse_usd':tot['prompt_hi']/1e6*.15+tot['completion']/1e6*.60,
       'peak_conservative_2rmse_usd':tot['prompt_hi']/1e6*.30+tot['completion']/1e6*1.20}}
(B/'BUDGET/BROAD_D10_FALLBACK_BUDGET.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))

# Runtime calibration from retained selective calls.
c=pd.read_csv(ROOT/'10_LLM_MEASUREMENT/02_FULL_selective/summaries/CALLS_COMPLETE.csv')
lat=pd.to_numeric(c.latency_ms,errors='coerce').dropna().to_numpy(float)
rt={'selective_latency_n':len(lat),'mean_ms':float(lat.mean()),'median_ms':float(np.median(lat)),
    'p90_ms':float(np.quantile(lat,.9)),'p95_ms':float(np.quantile(lat,.95))}
for name,n in [('d20_new',bud['total_new_api_queries']),('d10_new',out['total_new_queries'])]:
    rt[name]={str(k):float(n*lat.mean()/1000/60/k) for k in [16,32,64]}
(B/'BUDGET/BROAD_RUNTIME_ESTIMATE.json').write_text(json.dumps(rt,indent=2),encoding='utf-8')
print('\nRUNTIME\n'+json.dumps(rt,indent=2))
