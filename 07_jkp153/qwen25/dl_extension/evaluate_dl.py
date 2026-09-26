

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
B={'global_BROAD_HARD':16.0,'global_certainty_BROAD_ENTROPY':16.0,'selective_H085':32.0,'selective_H090':32.0,'selective_H095':48.0}; DEN=152.0
def r01(x):
 s=pd.Series(x); return ((s.rank(method='average')-1)/(len(s)-1)).to_numpy(float)
def sp(x,y): return float(spearmanr(np.asarray(x,float),np.asarray(y,float)).statistic)
def data(year):
 if year==2024: s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv'); t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
 else: s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv'); t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
 return s,t.set_index('factor_id').reindex(s.factor_id)
def hvec(fids,e,m):
 pos={f:i for i,f in enumerate(fids)}; h=np.zeros(len(fids))
 if m.startswith('global') or m.startswith('global_certainty'): z=e[e.in_broad.astype(bool)]
 elif m=='selective_H085': z=e[e.in_selective_union_H085.astype(bool)]
 elif m=='selective_H090': z=e[e.selective_primary_H090.astype(bool)]
 elif m=='selective_H095': z=e[e.selective_strict_H095.astype(bool)]
 for x in z.to_dict('records'):
  if not bool(x['DL_valid']): continue
  p=float(x['DL_p_i_gt_j'])
  if not np.isfinite(p) or p==.5: continue
  sign=1 if p>.5 else -1
  if m.startswith('global'): w=1.0
  elif m.startswith('global_certainty'): w=float(x['DL_certainty'])
  else: w=float(x['HD_bits'])*float(x['DL_certainty'])
  h[pos[x['factor_i']]]+=sign*w/6; h[pos[x['factor_j']]]-=sign*w/6
 return h
ap=argparse.ArgumentParser(); ap.add_argument('--year',type=int,required=True); ap.add_argument('--edge-probs',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
s,t=data(a.year); f=s.factor_id.astype(str).tolist(); raw=r01(s.raw_alpha); y=t.future_CAPM_alpha.to_numpy(float); e=pd.read_csv(a.edge_probs)
rawic=sp(raw,y); rows=[]
for m,b in B.items():
 h=hvec(f,e,m); score=raw+(b/DEN)*h
 rows.append({'year':a.year,'method':m,'arm':'DL','b_frozen_from_DQ_DQL':b,'rank_ic':sp(score,y),'Raw_ic':rawic,'delta_vs_Raw':sp(score,y)-rawic})
res=pd.DataFrame(rows); a.out.parent.mkdir(parents=True,exist_ok=True); res.to_csv(a.out,index=False)
print(res.to_string(index=False))
