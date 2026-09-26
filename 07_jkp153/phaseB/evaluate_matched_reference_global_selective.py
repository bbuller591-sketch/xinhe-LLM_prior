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
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr

ROOT=Path(str(REPRO_ROOT))
COR=ROOT/'JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321'
BGRID=[0,1,2,4,8,16,32,48,64,96,128,152];DEN=152.0

def r01(x):
    s=pd.Series(x);return ((s.rank(method='average')-1)/(len(s)-1)).to_numpy(float)
def sp(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);return float(spearmanr(x[ok],y[ok]).statistic)
def load_data(year):
    if year==2024:
        s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv');t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
    else:
        s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv');t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
    t=t.set_index('factor_id').reindex(s.factor_id)
    return s,t
def hvec(fids,e,arm,method):
    pos={f:i for i,f in enumerate(fids)};h=np.zeros(len(fids))
    if method.startswith('global') or method.startswith('global_certainty'):
        z=e[e.in_broad.astype(bool)]
    elif method=='selective_H085':
        z=e[e.in_selective_union_H085.astype(bool)]
    elif method=='selective_H090':
        z=e[e.selective_primary_H090.astype(bool)]
    elif method=='selective_H095':
        z=e[e.selective_strict_H095.astype(bool)]
    else: raise ValueError(method)
    for x in z.to_dict('records'):
        if not bool(x[f'{arm}_valid']):continue
        p=float(x[f'{arm}_p_i_gt_j'])
        if not np.isfinite(p) or p==.5:continue
        sign=1 if p>.5 else -1
        if method.startswith('global'):w=1.0
        elif method.startswith('global_certainty'):w=float(x[f'{arm}_certainty'])
        else:w=float(x['HD_bits'])*float(x[f'{arm}_certainty'])
        h[pos[x['factor_i']]] += sign*w/6.0
        h[pos[x['factor_j']]] -= sign*w/6.0
    return h
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--year',type=int,required=True);ap.add_argument('--edge-probs',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--b-freeze',type=Path);a=ap.parse_args()
    s,t=load_data(a.year);fids=s.factor_id.astype(str).tolist();raw=r01(s.raw_alpha.to_numpy(float));y=t.future_CAPM_alpha.to_numpy(float);e=pd.read_csv(a.edge_probs)
    methods=['global_BROAD_HARD','global_certainty_BROAD_ENTROPY','selective_H085','selective_H090','selective_H095']
    hs={(m,arm):hvec(fids,e,arm,m) for m in methods for arm in ['DQ','DQL']}
    rows=[]
    for m in methods:
      for b in BGRID:
        for arm in ['DQ','DQL']:
          score=raw+(b/DEN)*hs[(m,arm)]
          rows.append({'year':a.year,'method':m,'arm':arm,'b_rank_positions':b,'rank_ic':sp(score,y)})
    grid=pd.DataFrame(rows);a.out_dir.mkdir(parents=True,exist_ok=True);grid.to_csv(a.out_dir/f'MATCHED_GRID_{a.year}.csv',index=False)
    if a.year==2024:
        freeze={}
        for m in methods:
            q=grid[grid.method.eq(m)].groupby('b_rank_positions').rank_ic.mean().reset_index(name='mean_DQ_DQL_IC')
            q=q.sort_values(['mean_DQ_DQL_IC','b_rank_positions'],ascending=[False,True])
            freeze[m]=float(q.iloc[0].b_rank_positions)
            q.to_csv(a.out_dir/f'B_SELECTION_2024_{m}.csv',index=False)
        (a.out_dir/'B_FREEZE_2024.json').write_text(json.dumps(freeze,indent=2)+'\n')
    else:
        if a.b_freeze is None:raise SystemExit('--b-freeze required for 2025')
        freeze=json.loads(a.b_freeze.read_text())
    sel=[]
    for m in methods:
        b=freeze[m];x=grid[(grid.method==m)&(grid.b_rank_positions==b)].copy()
        dd=dict(zip(x.arm,x.rank_ic))
        sel.append({'year':a.year,'method':m,'b':b,'DQ_ic':dd['DQ'],'DQL_ic':dd['DQL'],'DQL_minus_DQ':dd['DQL']-dd['DQ'],'Raw_ic':sp(raw,y)})
    res=pd.DataFrame(sel);res.to_csv(a.out_dir/f'MATCHED_SELECTED_{a.year}.csv',index=False)
    summary={'year':a.year,'label':'2024_VALIDATION' if a.year==2024 else 'POST_FINAL_EXPLORATORY_REUSE','Raw_ic':sp(raw,y),'selected':sel}
    (a.out_dir/f'MATCHED_SUMMARY_{a.year}.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
