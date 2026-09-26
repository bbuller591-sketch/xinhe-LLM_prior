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
import argparse,json,numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(str(REPRO_ROOT))
COR=ROOT/'JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321'
BGRID=[0,1,2,4,8,16,32,48,64,96,128,152]; DEN=152.0
METHODS=['BROAD_HARD','BROAD_ENT','SEL_HARD','SEL_ENT','SEL_HD_HARD','SEL_HD_ENT']
def r01(x):
 s=pd.Series(x);return ((s.rank(method='average')-1)/(len(s)-1)).to_numpy(float)
def sp(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);return float(spearmanr(x[ok],y[ok]).statistic)
def data(year):
 if year==2024:
  s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv');t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
 else:
  s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv');t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
 return s,t.set_index('factor_id').reindex(s.factor_id)
def hvec(fids,e,arm,method):
 pos={f:i for i,f in enumerate(fids)};h=np.zeros(len(fids))
 if method.startswith('BROAD'): z=e[e.in_broad.astype(bool)]
 else: z=e[e.selective_primary_H090.astype(bool)]
 for x in z.to_dict('records'):
  if not bool(x[f'{arm}_valid']):continue
  p=float(x[f'{arm}_p_i_gt_j']);c=float(x[f'{arm}_certainty'])
  if not np.isfinite(p) or p==.5:continue
  sign=1 if p>.5 else -1
  w=1.0
  if method in ['BROAD_ENT','SEL_ENT','SEL_HD_ENT']:w*=c
  if method in ['SEL_HD_HARD','SEL_HD_ENT']:w*=float(x['HD_bits'])
  h[pos[x['factor_i']]]+=sign*w/6;h[pos[x['factor_j']]]-=sign*w/6
 return h
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--year',type=int,required=True);ap.add_argument('--edge-probs',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--freeze',type=Path);a=ap.parse_args()
 s,t=data(a.year);fids=s.factor_id.astype(str).tolist();raw=r01(s.raw_alpha);y=t.future_CAPM_alpha.to_numpy(float);e=pd.read_csv(a.edge_probs)
 rows=[]
 for m in METHODS:
  for arm in ['DQ','DQL']:
   h=hvec(fids,e,arm,m)
   for b in BGRID: rows.append({'year':a.year,'method':m,'arm':arm,'b':b,'rank_ic':sp(raw+(b/DEN)*h,y)})
 grid=pd.DataFrame(rows);a.out_dir.mkdir(parents=True,exist_ok=True);grid.to_csv(a.out_dir/f'ROUTING_ABLATION_GRID_{a.year}.csv',index=False)
 if a.year==2024:
  fr={}
  for m in METHODS:
   q=grid[grid.method==m].groupby('b').rank_ic.mean().reset_index(name='mean').sort_values(['mean','b'],ascending=[False,True])
   fr[m]=float(q.iloc[0].b)
  (a.out_dir/'ROUTING_ABLATION_B_FREEZE_2024.json').write_text(json.dumps(fr,indent=2)+'\n')
 else:
  fr=json.loads(a.freeze.read_text())
 selected=[]
 for m in METHODS:
  z=grid[(grid.method==m)&(grid.b==fr[m])];d=dict(zip(z.arm,z.rank_ic))
  selected.append({'year':a.year,'method':m,'b':fr[m],'DQ_ic':d['DQ'],'DQL_ic':d['DQL'],'mean':(d['DQ']+d['DQL'])/2,'DQL_minus_DQ':d['DQL']-d['DQ'],'Raw_ic':sp(raw,y)})
 pd.DataFrame(selected).to_csv(a.out_dir/f'ROUTING_ABLATION_SELECTED_{a.year}.csv',index=False)
 print(pd.DataFrame(selected).to_string(index=False))
if __name__=='__main__':main()
