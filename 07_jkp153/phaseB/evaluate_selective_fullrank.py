#!/usr/bin/env python3
from __future__ import annotations

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
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT=Path(str(REPRO_ROOT))
COR=ROOT/'JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321'
BGRID=[0,1,2,4,8,16,32]
THRESHOLDS=[0.85,0.90,0.95]
DEN=152.0

def rank01(x):
    s=pd.Series(x); return ((s.rank(method='average')-1)/(len(s)-1)).to_numpy(float)
def sp(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    return float(spearmanr(x[ok],y[ok]).statistic)
def build_h(fids,edges,arm,thr):
    pos={f:i for i,f in enumerate(fids)}; h=np.zeros(len(fids))
    for e in edges.to_dict('records'):
        if float(e['HD_bits'])<thr or not bool(e[f'{arm}_valid']): continue
        p=float(e[f'{arm}_p_i_gt_j']); c=float(e[f'{arm}_certainty']); u=float(e['HD_bits'])
        if not np.isfinite(p) or not np.isfinite(c): continue
        s=1 if p>0.5 else (-1 if p<0.5 else 0)
        w=u*c
        h[pos[e['factor_i']]] += w*s/6.0
        h[pos[e['factor_j']]] -= w*s/6.0
    return h
def data(year):
    if year==2024:
        s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv')
        t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
    else:
        s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv')
        t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
    t=t.set_index('factor_id').reindex(s.factor_id)
    return s,t
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--year',type=int,required=True); ap.add_argument('--edges',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--selected-b',type=float); a=ap.parse_args()
    s,t=data(a.year); fids=s.factor_id.astype(str).tolist(); rawrank=rank01(s.raw_alpha.to_numpy(float)); y=t.future_CAPM_alpha.to_numpy(float)
    e=pd.read_csv(a.edges); rows=[]; hcache={}
    for thr in THRESHOLDS:
      for arm in ['DQ','DQL']: hcache[(thr,arm)]=build_h(fids,e,arm,thr)
      for b in BGRID:
        for arm in ['DQ','DQL']:
          score=rawrank+(b/DEN)*hcache[(thr,arm)]
          rows.append({'year':a.year,'threshold':thr,'b_rank_positions':b,'arm':arm,'rank_ic':sp(score,y)})
    grid=pd.DataFrame(rows); a.out_dir.mkdir(parents=True,exist_ok=True); grid.to_csv(a.out_dir/f'SELECTIVE_GRID_{a.year}.csv',index=False)
    result={'year':a.year,'raw_ic':sp(rawrank,y)}
    if a.year==2024:
        p=grid[grid.threshold.eq(0.90)].groupby('b_rank_positions').rank_ic.mean().reset_index(name='mean_DQ_DQL_IC')
        p=p.sort_values(['mean_DQ_DQL_IC','b_rank_positions'],ascending=[False,True])
        bstar=float(p.iloc[0].b_rank_positions); result['selected_b_shared_2024']=bstar
        p.to_csv(a.out_dir/'B_SELECTION_2024.csv',index=False)
    else:
        if a.selected_b is None: raise SystemExit('--selected-b required for 2025')
        bstar=float(a.selected_b); result['frozen_b_from_2024']=bstar
    sel=grid[(grid.threshold.eq(0.90))&(grid.b_rank_positions.eq(bstar))].copy()
    result['selected_rows']=sel.to_dict('records')
    if len(sel)==2:
        m=dict(zip(sel.arm,sel.rank_ic)); result['DQL_minus_DQ']=m['DQL']-m['DQ']
    (a.out_dir/f'SELECTIVE_RESULT_{a.year}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
