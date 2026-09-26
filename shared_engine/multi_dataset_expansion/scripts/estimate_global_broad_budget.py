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
OUT=B/'BUDGET'
OUT.mkdir(parents=True,exist_ok=True)

# Actual selective calibration from current experiment.
calls=pd.read_csv(ROOT/'10_LLM_MEASUREMENT/02_FULL_selective/summaries/CALLS_COMPLETE.csv')
parts=[]
for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
    q=pd.read_csv(ROOT/'08_PRE_LLM_QUERIES'/f'{task}_selective_QUERIES_PREAUTH.csv',usecols=['query_id','arm','prompt_text'])
    q['prompt_chars']=q.prompt_text.str.len()
    parts.append(q[['query_id','arm','prompt_chars']])
qall=pd.concat(parts,ignore_index=True)
z=qall.merge(calls[['query_id','prompt_tokens','completion_tokens']],on='query_id',validate='one_to_one')
z['prompt_tokens']=pd.to_numeric(z.prompt_tokens)
z['completion_tokens']=pd.to_numeric(z.completion_tokens)
cal={}
for arm,g in z.groupby('arm'):
    x=g.prompt_chars.to_numpy(float); y=g.prompt_tokens.to_numpy(float)
    b=np.cov(x,y,ddof=0)[0,1]/np.var(x)
    a=y.mean()-b*x.mean()
    pred=a+b*x
    cal[arm]={'n':len(g),'intercept':float(a),'slope':float(b),
              'rmse':float(np.sqrt(np.mean((pred-y)**2))),
              'mean_prompt_tokens':float(y.mean()),'mean_completion_tokens':float(g.completion_tokens.mean())}

summ=[]
for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
    p=B/'QUERIES'/f'{task}_global_BROAD_D20_QUERIES.parquet'
    d=pd.read_parquet(p,columns=['query_id','arm','prompt_chars','reusable_from_selective'])
    d['estimated_prompt_tokens']=0.0
    d['estimated_prompt_tokens_conservative']=0.0
    for arm,idx in d.groupby('arm').groups.items():
        c=cal[arm]
        x=d.loc[idx,'prompt_chars'].astype(float)
        d.loc[idx,'estimated_prompt_tokens']=np.maximum(1,c['intercept']+c['slope']*x)
        d.loc[idx,'estimated_prompt_tokens_conservative']=np.maximum(1,c['intercept']+c['slope']*x+2*c['rmse'])
    new=d[~d.reusable_from_selective.astype(bool)].copy()
    logical=d.copy()
    # For frozen budgeting, assume one output token per new query (observed current selective mean exactly below).
    prompt=float(new.estimated_prompt_tokens.sum())
    prompt_hi=float(new.estimated_prompt_tokens_conservative.sum())
    nnew=len(new)
    comp=nnew*max(cal[a]['mean_completion_tokens'] for a in new.arm.unique())
    # Same pricing frozen for the approved selective run; all-cache-miss is conservative.
    costs={
      'off_peak_all_cache_miss_usd':prompt/1e6*0.15+comp/1e6*0.60,
      'peak_all_cache_miss_usd':prompt/1e6*0.30+comp/1e6*1.20,
      'off_peak_conservative_2rmse_usd':prompt_hi/1e6*0.15+comp/1e6*0.60,
      'peak_conservative_2rmse_usd':prompt_hi/1e6*0.30+comp/1e6*1.20,
    }
    s={'task':task,'logical_queries_d20':len(d),'reusable_existing_selective_queries':int(d.reusable_from_selective.astype(bool).sum()),
       'new_api_queries':nnew,'estimated_new_prompt_tokens':int(round(prompt)),
       'estimated_new_prompt_tokens_conservative_2rmse':int(round(prompt_hi)),
       'estimated_new_completion_tokens':int(round(comp)),'cost_estimates':costs}
    summ.append(s)
    print(json.dumps(s,indent=2))

total={
 'status':'BROAD_global_PREAUTH_BUDGET',
 'calibration_by_arm':cal,
 'tasks':summ,
 'total_logical_queries':sum(x['logical_queries_d20'] for x in summ),
 'total_reusable_existing_selective_queries':sum(x['reusable_existing_selective_queries'] for x in summ),
 'total_new_api_queries':sum(x['new_api_queries'] for x in summ),
 'total_estimated_new_prompt_tokens':sum(x['estimated_new_prompt_tokens'] for x in summ),
 'total_estimated_new_prompt_tokens_conservative_2rmse':sum(x['estimated_new_prompt_tokens_conservative_2rmse'] for x in summ),
 'total_estimated_new_completion_tokens':sum(x['estimated_new_completion_tokens'] for x in summ)
}
pt=total['total_estimated_new_prompt_tokens']; ph=total['total_estimated_new_prompt_tokens_conservative_2rmse']; ct=total['total_estimated_new_completion_tokens']
total['combined_cost_estimates']={
 'off_peak_all_cache_miss_usd':pt/1e6*.15+ct/1e6*.60,
 'peak_all_cache_miss_usd':pt/1e6*.30+ct/1e6*1.20,
 'off_peak_conservative_2rmse_usd':ph/1e6*.15+ct/1e6*.60,
 'peak_conservative_2rmse_usd':ph/1e6*.30+ct/1e6*1.20}
(OUT/'BROAD_global_TOKEN_COST_ESTIMATE.json').write_text(json.dumps(total,indent=2),encoding='utf-8')
print('\nTOTAL\n'+json.dumps(total,indent=2))
