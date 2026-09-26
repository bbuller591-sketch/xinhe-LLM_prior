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
import pandas as pd, numpy as np, math, json
from scipy.special import expit
W=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap'))

def ent(p):
 return 0. if p<=0 or p>=1 else -(p*math.log(p,2)+(1-p)*math.log(1-p,2))
def logit(p):
 p=min(max(float(p),1e-12),1-1e-12);return math.log(p/(1-p))
def good(r):return str(r.parse_status).startswith('PASS') and pd.notna(r.logp_A) and pd.notna(r.logp_B)
def agg_abba(calls):
 rows=[];drops=[]
 for (pair,arm),g in calls.groupby(['unordered_pair_id','arm'],sort=False):
  if len(g)!=2 or set(g.order.astype(str))!={'AB','BA'}:
   drops.append((pair,arm,'orientation'));continue
  ab=g[g.order=='AB'].iloc[0];ba=g[g.order=='BA'].iloc[0]
  if not good(ab) or not good(ba):
   drops.append((pair,arm,'top20'));continue
  ell=.5*((float(ab.logp_A)-float(ab.logp_B))-(float(ba.logp_A)-float(ba.logp_B)))
  p=float(expit(ell));H=ent(p);C=1-H
  rows.append({'unordered_pair_id':pair,'arm':arm,'gene_i':ab.gene_A,'gene_j':ab.gene_B,
    'symmetrized_logit_i_vs_j':ell,'p_i_over_j':p,'hard_y_i_over_j':1. if p>.5 else 0. if p<.5 else .5,
    'H_AB_bits':H,'certainty_1_minus_H':C,'either_U':bool(ab.first_token=='U' or ba.first_token=='U'),
    'order_probability_gap':abs(float(ab.pA_vs_B)-(1-float(ba.pA_vs_B)))})
 return pd.DataFrame(rows),drops

# Sepsis
base=W/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q2'
d=pd.read_csv(base/'measurement/CALLS_COMPLETE.csv');a,dr=agg_abba(d)
a.to_csv(base/'Q2_GLOBAL_MATCHED_PAIR_SOURCE_MEASUREMENTS_GPT.csv',index=False)

# Breast
base=W/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2'
d=pd.read_csv(base/'measurement/CALLS_COMPLETE.csv');b,db=agg_abba(d)
b.to_csv(base/'Q2_GLOBAL_PAIR_SOURCE_MEASUREMENTS_GPT.csv',index=False)

# Darmanis
base=W/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2'
d=pd.read_csv(base/'measurement/CALLS_COMPLETE.csv');c,dc=agg_abba(d)
c.to_csv(base/'Q2_GLOBAL_PAIR_SOURCE_MEASUREMENTS_GPT.csv',index=False)

# Hospital
base=W/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2'
d=pd.read_csv(base/'measurement/CALLS_COMPLETE.csv')
rows=[];dh=[]
for pid,g in d.groupby('pair_id',sort=False):
 if len(g)!=2 or set(g.order.astype(str))!={'AB','BA'}:dh.append((pid,'orient'));continue
 ab=g[g.order=='AB'].iloc[0];ba=g[g.order=='BA'].iloc[0]
 if str(ab.parse_status)!='PASS' or str(ba.parse_status)!='PASS':dh.append((pid,'top20'));continue
 p1=float(ab.p_semantic_feature_A);p2=float(ba.p_semantic_feature_A)
 ell=.5*(logit(p1)+logit(p2));p=float(expit(ell));H=ent(p)
 rows.append({'pair_id':pid,'feature_A':ab.semantic_feature_A,'feature_B':ab.semantic_feature_B,
              'p_pair_semantic_A':p,'H_pair':H,'c_pair':1-H,
              'hard_order_disagreement':int((p1>.5)!=(p2>.5)),'abs_order_logit_gap':abs(logit(p1)-logit(p2))})
h=pd.DataFrame(rows);h.to_csv(base/'Q2_GLOBAL_PAIR_MEASUREMENTS_GPT.csv',index=False)
summary={'Sepsis':{'usable':len(a),'dropped':len(dr)},'Breast':{'usable':len(b),'dropped':len(db)},
         'Darmanis':{'usable':len(c),'dropped':len(dc)},'Hospital':{'usable':len(h),'dropped':len(dh)}}
(W/'08_Q2_Q5_DIAGNOSTICS/Q2_MEASUREMENT_AGGREGATION_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
