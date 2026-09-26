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
import json, math, numpy as np, pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
ROOT=Path(str(REPRO_ROOT)); P=Path('.')
COR=ROOT/'JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321'
B=48.0; DEN=152.0

def lg(p):
    p=float(np.clip(p,1e-6,1-1e-6));return math.log(p/(1-p))
def hb(p):
    p=float(np.clip(p,1e-12,1-1e-12));return -(p*math.log2(p)+(1-p)*math.log2(1-p))
def r01(x):
    s=pd.Series(x);return ((s.rank(method='average')-1)/(len(s)-1)).to_numpy(float)
def sp(x,y):return float(spearmanr(np.asarray(x,float),np.asarray(y,float)).statistic)
rows=[]
for year in [2024,2025]:
    calls=pd.DataFrame([json.loads(x) for x in (P/f'runs/{year}/PHASEB_UNION_CALLS.jsonl').read_text().splitlines() if x.strip()]).drop_duplicates('measurement_call_id',keep='last')
    e=pd.read_csv(P/f'results/{year}/EDGE_PROBABILITIES_{year}.csv')
    for arm in ['DQ','DQL']:
        vals={}
        for edge in e.measurement_edge_id:
            usearm=arm
            dqlreq=bool(e.loc[e.measurement_edge_id.eq(edge),'DQL_call_required'].iloc[0])
            if arm=='DQL' and not dqlreq: usearm='DQ'
            c=calls[(calls.edge_id==edge)&(calls.arm==usearm)&(calls.final_status=='SUCCESS')]
            lr=[]
            for rep in [1,2,3]:
                ab=c[(c.order=='AB')&(c.measurement_repeat==rep)]
                ba=c[(c.order=='BA')&(c.measurement_repeat==rep)]
                if len(ab)!=1 or len(ba)!=1:continue
                A=ab.iloc[0]; Z=ba.iloc[0]
                if A.hard_token not in ('A','B') or Z.hard_token not in ('A','B'):continue
                # require hard semantic agreement after mapping BA back to canonical i-vs-j
                iwin_ab=(A.hard_token=='A'); iwin_ba=(Z.hard_token=='B')
                if iwin_ab!=iwin_ba:continue
                if pd.isna(A.pA_cond_AB) or pd.isna(Z.pA_cond_AB):continue
                lr.append((lg(A.pA_cond_AB)-lg(Z.pA_cond_AB))/2)
            if len(lr)>=2:
                p=float(expit(np.mean(lr))); vals[edge]=(True,p,1-hb(p),len(lr))
            else: vals[edge]=(False,np.nan,np.nan,len(lr))
        e[f'{arm}_OC_valid']=[vals[x][0] for x in e.measurement_edge_id]
        e[f'{arm}_OC_p']=[vals[x][1] for x in e.measurement_edge_id]
        e[f'{arm}_OC_c']=[vals[x][2] for x in e.measurement_edge_id]
        e[f'{arm}_OC_reps']=[vals[x][3] for x in e.measurement_edge_id]
    if year==2024:
        s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv');t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
    else:
        s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv');t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
    t=t.set_index('factor_id').reindex(s.factor_id);fids=s.factor_id.astype(str).tolist();pos={f:i for i,f in enumerate(fids)}
    raw=r01(s.raw_alpha.to_numpy(float));y=t.future_CAPM_alpha.to_numpy(float)
    rec={'year':year,'Raw_ic':sp(raw,y),'fixed_b':B,'label':'POSTHOC_ORDER_CONSISTENT_SENSITIVITY'}
    for arm in ['DQ','DQL']:
        h=np.zeros(len(fids));z=e[e.selective_primary_H090.astype(bool)]
        for x in z.to_dict('records'):
            if not bool(x[f'{arm}_OC_valid']):continue
            p=float(x[f'{arm}_OC_p'])
            if p==.5:continue
            sign=1 if p>.5 else -1
            w=float(x['HD_bits'])*float(x[f'{arm}_OC_c'])
            h[pos[x['factor_i']]] += sign*w/6
            h[pos[x['factor_j']]] -= sign*w/6
        score=raw+(B/DEN)*h
        rec[f'{arm}_valid_edges']=int(z[f'{arm}_OC_valid'].sum())
        rec[f'{arm}_ic']=sp(score,y)
    rec['DQL_minus_DQ']=rec['DQL_ic']-rec['DQ_ic'];rows.append(rec)
    e.to_csv(P/f'results/{year}/EDGE_PROBABILITIES_ORDER_CONSISTENT_SENSITIVITY_{year}.csv',index=False)
out=pd.DataFrame(rows);out.to_csv(P/'results/POSTHOC_ORDER_CONSISTENT_SENSITIVITY.csv',index=False)
print(out.to_string(index=False))
