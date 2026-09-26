

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
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(str(REPRO_ROOT))
COR=ROOT/'JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321'
HERE=Path(str(REPRO_ROOT / '07_jkp153/qwen25/dl_extension'))
BGRID=[0,1,2,4,8,16,32,48,64,96,128,152]; DEN=152.
def r01(x):
 s=pd.Series(x);return ((s.rank(method='average')-1)/(len(s)-1)).to_numpy(float)
def sp(x,y): return float(spearmanr(np.asarray(x,float),np.asarray(y,float)).statistic)
def load(year):
 if year==2024:s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv');t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
 else:s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv');t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
 return s,t.set_index('factor_id').reindex(s.factor_id)
def hvec(fids,e,m):
 pos={f:i for i,f in enumerate(fids)};h=np.zeros(len(fids))
 if m=='global':z=e[e.in_broad.astype(bool)]
 elif m=='global_certainty':z=e[e.in_broad.astype(bool)]
 elif m=='selective_H085':z=e[e.in_selective_union_H085.astype(bool)]
 elif m=='selective_H090':z=e[e.selective_primary_H090.astype(bool)]
 elif m=='selective_H095':z=e[e.selective_strict_H095.astype(bool)]
 for x in z.to_dict('records'):
  if not bool(x['DL_valid']):continue
  p=float(x['DL_p_i_gt_j'])
  if not np.isfinite(p) or p==.5:continue
  sign=1 if p>.5 else -1
  w=1.0 if m=='global' else float(x['DL_certainty']) if m=='global_certainty' else float(x['HD_bits'])*float(x['DL_certainty'])
  h[pos[x['factor_i']]]+=sign*w/6;h[pos[x['factor_j']]]-=sign*w/6
 return h
freeze={}
rows=[]
for year in [2024,2025]:
 s,t=load(year);f=s.factor_id.astype(str).tolist();raw=r01(s.raw_alpha);y=t.future_CAPM_alpha.to_numpy(float);e=pd.read_csv(HERE/f'results/DL_EDGE_PROBABILITIES_{year}.csv')
 for m in ['global','global_certainty','selective_H085','selective_H090','selective_H095']:
  h=hvec(f,e,m)
  for b in BGRID: rows.append({'year':year,'method':m,'b':b,'ic':sp(raw+(b/DEN)*h,y)})
grid=pd.DataFrame(rows);grid.to_csv(HERE/'results/DL_BGRID_POSTHOC_DIAGNOSTIC.csv',index=False)
for m in grid.method.unique():
 q=grid[(grid.year==2024)&(grid.method==m)].sort_values(['ic','b'],ascending=[False,True]);freeze[m]=float(q.iloc[0].b)
print('2024 posthoc best b',freeze)
for year in [2024,2025]:
 print('\nYEAR',year)
 for m,b in freeze.items():
  x=grid[(grid.year==year)&(grid.method==m)&(grid.b==b)].iloc[0]
  print(m,'b',b,'IC',x.ic)
