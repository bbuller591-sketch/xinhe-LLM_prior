

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
import math, json
import numpy as np
import pandas as pd

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
R=WS/'06_RESULTS'
primary=pd.read_csv(R/'global_DEEPSEEK_MEASUREMENT.csv')
ctrl=pd.read_csv(R/'POST_FINAL_BENCHMARK_CONTROL.csv')
sent=set(ctrl.pair_id)

def logit(p):
    p=min(max(float(p),1e-4),1-1e-4); return math.log(p/(1-p))
def sig(x): return 1/(1+math.exp(-x))

def summarize(d,arm):
    if arm=='CONSTRUCT_ONLY_NO_RETRIEVED_EVIDENCE':
        q=d[(d.pair_id.isin(sent))&(d.primary_measurement.astype(bool))&(d.repeat_index==0)].copy()
    else:
        q=d[(d.arm==arm)&(d.primary_measurement.astype(bool))].copy()
    rows=[]
    for pid,g in q.groupby('pair_id'):
        ab=g[g.presentation_order=='AB'].iloc[0]; ba=g[g.presentation_order=='BA'].iloc[0]
        abst=(ab.first_token=='U') or (ba.first_token=='U')
        p=np.nan; gap=np.nan
        if not abst:
            pab=float(ab.p_presented_A_cond_AB)
            pba=1-float(ba.p_presented_A_cond_AB)
            la,lb=logit(pab),logit(pba)
            p=sig((la+lb)/2); gap=abs(la-lb)
        rows.append({'arm':arm,'pair_id':pid,'feature_a':ab.canonical_feature_a,'feature_b':ab.canonical_feature_b,
                     'p_canonical_a':p,'order_gap_logit':gap,'abstain':abst})
    return pd.DataFrame(rows)

tabs=[
 summarize(primary,'CONSTRUCT_ONLY_NO_RETRIEVED_EVIDENCE'),
 summarize(ctrl,'RAW_FEATURE_NAMES_GENERIC_TASK'),
 summarize(ctrl,'EXPLICIT_BENCHMARK_CONTEXT')
]
x=pd.concat(tabs,ignore_index=True)
x.to_csv(R/'POST_FINAL_BENCHMARK_CONTROL_PAIR_SUMMARY.csv',index=False)
wide=x.pivot(index=['pair_id','feature_a','feature_b'],columns='arm',values='p_canonical_a').reset_index()
wide.to_csv(R/'POST_FINAL_BENCHMARK_CONTROL_COMPARISON.csv',index=False)

base='CONSTRUCT_ONLY_NO_RETRIEVED_EVIDENCE'
summ={}
for arm in ['RAW_FEATURE_NAMES_GENERIC_TASK','EXPLICIT_BENCHMARK_CONTEXT']:
    vals=[]
    flips=0; comparable=0
    for _,r in wide.iterrows():
        a=r.get(base); b=r.get(arm)
        if pd.notna(a) and pd.notna(b):
            vals.append(abs(float(a)-float(b)))
            flips += int((float(a)>=.5)!=(float(b)>=.5))
            comparable+=1
    summ[arm]={
      'n_comparable_pairs':comparable,
      'mean_abs_probability_shift':float(np.mean(vals)) if vals else None,
      'median_abs_probability_shift':float(np.median(vals)) if vals else None,
      'semantic_direction_flips':int(flips),
    }
summ['abstentions_by_arm']=x.groupby('arm').abstain.sum().astype(int).to_dict()
(R/'POST_FINAL_BENCHMARK_CONTROL_AUDIT.json').write_text(json.dumps(summ,indent=2)+'\n')
print(json.dumps(summ,indent=2))
print('\n',wide.to_string(index=False))
