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
import pandas as pd, numpy as np, math, json, re
from scipy.special import expit
W=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap'))

def ent(p):
    return 0.0 if p<=0 or p>=1 else -(p*math.log(p,2)+(1-p)*math.log(1-p,2))
def logit(p):
    p=min(max(float(p),1e-12),1-1e-12);return math.log(p/(1-p))

# Hospital Q2
O=W/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2'
d=pd.read_csv(O/'measurement/CALLS_COMPLETE.csv')
rows=[]
for pid,g in d.groupby('pair_id',sort=False):
    if len(g)!=2 or set(g.order.astype(str))!={'AB','BA'}: raise RuntimeError(('hospital orient',pid))
    ab=g[g.order=='AB'].iloc[0];ba=g[g.order=='BA'].iloc[0]
    if str(ab.parse_status)!='PASS' or str(ba.parse_status)!='PASS': raise RuntimeError(('hospital fail',pid))
    p1=float(ab.p_semantic_feature_A);p2=float(ba.p_semantic_feature_A)
    ell=.5*(logit(p1)+logit(p2));p=float(expit(ell));H=ent(p)
    rows.append({'pair_id':pid,'feature_A':ab.semantic_feature_A,'feature_B':ab.semantic_feature_B,
                 'p_pair_semantic_A':p,'H_pair':H,'c_pair':1-H,
                 'hard_order_disagreement':int((p1>.5)!=(p2>.5)),
                 'abs_order_logit_gap':abs(logit(p1)-logit(p2))})
h=pd.DataFrame(rows);h.to_csv(O/'Q2_GLOBAL_PAIR_MEASUREMENTS_GPT.csv',index=False)

# Darmanis Q2
O=W/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2'
d=pd.read_csv(O/'measurement/CALLS_COMPLETE.csv')
rows=[]
for (pair,arm),g in d.groupby(['unordered_pair_id','arm'],sort=False):
    if len(g)!=2 or set(g.order.astype(str))!={'AB','BA'}: raise RuntimeError(('darm orient',pair,arm))
    ab=g[g.order=='AB'].iloc[0];ba=g[g.order=='BA'].iloc[0]
    ell=.5*((float(ab.logp_A)-float(ab.logp_B))-(float(ba.logp_A)-float(ba.logp_B)))
    p=float(expit(ell));H=ent(p)
    m=re.fullmatch(r'N(\d+)_N(\d+)',str(pair))
    if not m: raise RuntimeError(pair)
    i,j=map(int,m.groups())
    rows.append({'unordered_pair_id':pair,'arm':arm,'gene_i':ab.gene_A,'gene_j':ab.gene_B,
                 'symmetrized_logit_i_vs_j':ell,'p_i_over_j':p,
                 'hard_y_i_over_j':1.0 if p>.5 else 0.0 if p<.5 else .5,
                 'H_AB_bits':H,'certainty_1_minus_H':1-H,
                 'either_U':bool(str(ab.first_token)=='U' or str(ba.first_token)=='U'),
                 'order_probability_gap':abs(float(ab.pA_vs_B)-(1-float(ba.pA_vs_B))),
                 'i':i,'j':j})
a=pd.DataFrame(rows);a.to_csv(O/'Q2_GLOBAL_PAIR_SOURCE_MEASUREMENTS_GPT.csv',index=False)
for arm,g in a.groupby('arm'):
    z=g[['i','j','unordered_pair_id','hard_y_i_over_j','certainty_1_minus_H']].copy()
    z.columns=['i','j','pair','hard_y','certainty']
    z.to_csv(O/f'{arm}_GPT_BROAD.csv',index=False)

print(json.dumps({'Hospital':len(h),'Darmanis':len(a),'Darmanis_by_arm':a.arm.value_counts().to_dict()},indent=2))
