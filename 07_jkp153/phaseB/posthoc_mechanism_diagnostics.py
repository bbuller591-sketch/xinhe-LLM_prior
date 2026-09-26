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
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT=Path(str(REPRO_ROOT))
P=ROOT/'finance_llm_prior/experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
COR=ROOT/'JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321'
B=48.0; DEN=152.0

def r01(x):
    s=pd.Series(x); return ((s.rank(method='average')-1)/(len(s)-1)).to_numpy(float)
def sp(x,y): return float(spearmanr(np.asarray(x,float),np.asarray(y,float)).statistic)
def load(year):
    e=pd.read_csv(P/f'results/{year}/EDGE_PROBABILITIES_{year}.csv')
    if year==2024:
        s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv'); t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
    else:
        s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv'); t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
    t=t.set_index('factor_id').reindex(s.factor_id)
    return e,s,t
def hv(fids,e,arm,kind):
    pos={f:i for i,f in enumerate(fids)}; h=np.zeros(len(fids))
    z=e[e.selective_primary_H090.astype(bool)]
    for x in z.to_dict('records'):
        if not bool(x[f'{arm}_valid']): continue
        p=float(x[f'{arm}_p_i_gt_j'])
        if not np.isfinite(p) or p==.5: continue
        sign=1 if p>.5 else -1
        c=float(x[f'{arm}_certainty']); hd=float(x['HD_bits'])
        if kind=='LOCAL_HARD': w=1.0
        elif kind=='LOCAL_LLM_ENTROPY': w=c
        elif kind=='LOCAL_HD_ONLY': w=hd
        elif kind=='LOCAL_HD_X_LLM_ENTROPY': w=hd*c
        else: raise ValueError(kind)
        h[pos[x['factor_i']]] += sign*w/6.0
        h[pos[x['factor_j']]] -= sign*w/6.0
    return h
rows=[]
H={}
for year in [2024,2025]:
    e,s,t=load(year); fids=s.factor_id.astype(str).tolist(); raw=r01(s.raw_alpha.to_numpy(float)); y=t.future_CAPM_alpha.to_numpy(float)
    for arm in ['DQ','DQL']:
        for kind in ['LOCAL_HARD','LOCAL_LLM_ENTROPY','LOCAL_HD_ONLY','LOCAL_HD_X_LLM_ENTROPY']:
            h=hv(fids,e,arm,kind); H[(year,arm,kind)]=(fids,h)
            score=raw+(B/DEN)*h
            rows.append({'year':year,'arm':arm,'diagnostic':kind,'fixed_b':B,'rank_ic':sp(score,y),'delta_vs_raw':sp(score,y)-sp(raw,y),
                         'label':'2024_VALIDATION_POSTHOC_DIAGNOSTIC' if year==2024 else '2025_POST_FINAL_POSTHOC_DIAGNOSTIC'})
out=pd.DataFrame(rows); out.to_csv(P/'results/POSTHOC_LOCAL_WEIGHTING_DIAGNOSTICS.csv',index=False)

# graph overlap/stability
e24=pd.read_csv(P/'results/2024/EDGE_PROBABILITIES_2024.csv');e25=pd.read_csv(P/'results/2025/EDGE_PROBABILITIES_2025.csv')
def pkey(df):
    z=df[df.selective_primary_H090.astype(bool)].copy()
    z['pair_key']=z.apply(lambda r:'|'.join(sorted([str(r.factor_i),str(r.factor_j)])),axis=1)
    return z
z24=pkey(e24);z25=pkey(e25)
common=set(z24.pair_key)&set(z25.pair_key)
stab={'primary_edges_2024':len(z24),'primary_edges_2025':len(z25),'common_primary_pairs':len(common),
      'jaccard_primary_pairsets':len(common)/len(set(z24.pair_key)|set(z25.pair_key))}
for arm in ['DQ','DQL']:
    a=z24[z24.pair_key.isin(common)].set_index('pair_key'); b=z25[z25.pair_key.isin(common)].set_index('pair_key')
    idx=sorted(common); a=a.reindex(idx);b=b.reindex(idx)
    ok=a[f'{arm}_valid'].fillna(False).astype(bool)&b[f'{arm}_valid'].fillna(False).astype(bool)
    aa=a.loc[ok,f'{arm}_p_i_gt_j'].to_numpy(float); bb=b.loc[ok,f'{arm}_p_i_gt_j'].to_numpy(float)
    stab[f'{arm}_common_valid_pairs']=int(ok.sum())
    stab[f'{arm}_p_correlation_2024_2025']=float(np.corrcoef(aa,bb)[0,1]) if ok.sum()>2 else np.nan
    stab[f'{arm}_direction_agreement_2024_2025']=float(np.mean((aa>.5)==(bb>.5))) if ok.sum() else np.nan
# factor h stability
for arm in ['DQ','DQL']:
    for kind in ['LOCAL_HD_X_LLM_ENTROPY']:
        f24,h24=H[(2024,arm,kind)];f25,h25=H[(2025,arm,kind)]
        assert f24==f25
        stab[f'{arm}_factor_h_pearson_2024_2025']=float(np.corrcoef(h24,h25)[0,1])
        stab[f'{arm}_factor_h_spearman_2024_2025']=float(spearmanr(h24,h25).statistic)
(P/'results/GRAPH_AND_GUIDANCE_STABILITY.json').write_text(json.dumps(stab,indent=2)+'\n')
print(out.to_string(index=False))
print(json.dumps(stab,indent=2))
