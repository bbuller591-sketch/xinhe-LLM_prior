

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
import numpy as np,pandas as pd,json
ROOT=Path(str(REPRO_ROOT));P=ROOT/'finance_llm_prior/experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
COR=ROOT/'JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321'
def load(year):
 if year==2024:
  s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv');t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
 else:
  s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv');t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
 return s.set_index('factor_id'),t.set_index('factor_id')
rows=[]
for year in [2024,2025]:
 s,t=load(year);e=pd.read_csv(P/f'results/{year}/EDGE_PROBABILITIES_{year}.csv')
 for graph,name in [('in_broad','BROAD'),('selective_primary_H090','SELECTIVE_H090')]:
  z=e[e[graph].astype(bool)]
  for arm in ['DQ','DQL']:
   n=correct=rawcorr=0; weighted_num=weighted_den=0.
   agree_with_raw=0
   for x in z.to_dict('records'):
    if not bool(x[f'{arm}_valid']):continue
    i,j=x['factor_i'],x['factor_j']
    target=float(t.loc[i,'future_CAPM_alpha']-t.loc[j,'future_CAPM_alpha'])
    if target==0:continue
    pred=float(x[f'{arm}_p_i_gt_j'])-.5
    if pred==0:continue
    raw=float(s.loc[i,'raw_alpha']-s.loc[j,'raw_alpha'])
    ps=1 if pred>0 else -1; ts=1 if target>0 else -1; rs=1 if raw>0 else -1
    n+=1; correct+=ps==ts; rawcorr+=rs==ts; agree_with_raw+=ps==rs
    w=float(x[f'{arm}_certainty'])
    weighted_num+=w*(ps==ts);weighted_den+=w
   rows.append({'year':year,'graph':name,'arm':arm,'n_pairs':n,'LLM_direction_future_accuracy':correct/n if n else np.nan,
                'Raw_direction_future_accuracy_same_pairs':rawcorr/n if n else np.nan,
                'LLM_minus_Raw_pair_accuracy':(correct-rawcorr)/n if n else np.nan,
                'LLM_raw_direction_agreement':agree_with_raw/n if n else np.nan,
                'certainty_weighted_LLM_accuracy':weighted_num/weighted_den if weighted_den else np.nan})
out=pd.DataFrame(rows);out.to_csv(P/'results/PAIR_RESOLUTION_DIAGNOSTIC.csv',index=False)
print(out.to_string(index=False))
