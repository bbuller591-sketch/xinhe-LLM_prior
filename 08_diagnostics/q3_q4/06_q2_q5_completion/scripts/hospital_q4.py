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
import json
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,ndtri
from scipy.stats import rankdata
R=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
O=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/03_q4_protection/HOSPITAL_OSTEOPOROSIS'));O.mkdir(parents=True,exist_ok=True)
S=pd.read_csv(R/"09_BATCH1_DOWNSTREAM_V1_9/FINAL_FULL_BATCH1_DATA_SELECTOR_SCORES.csv"); fs=S.feature.astype(str).tolist();D=S.D_full_batch1.to_numpy(float);P=len(D);ix={f:i for i,f in enumerate(fs)}
HP=pd.read_csv(R/"09_BATCH1_DOWNSTREAM_V1_9/SELECTED_DEVELOPMENT_HYPERPARAMS.csv")
SETS=pd.read_csv(R/"09_BATCH1_DOWNSTREAM_V1_9/FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv")
E=pd.read_csv(R/"09_BATCH1_DOWNSTREAM_V1_9/selective_PAIRK_WEIGHTS_FROZEN.csv")
def anch(x):
 r=rankdata(-x,method="average");u=1-(r-.5)/len(x);return ndtri(np.clip(u,1e-6,1-1e-6))
def tk(z,k):return np.lexsort((np.arange(P),-D,-z))[:k]
def one(k,lam):
 q=E[(E.k==k)&(E.evidence_gate==1)].reset_index(drop=True);ii=np.array([ix[x] for x in q.feature_A]);jj=np.array([ix[x] for x in q.feature_B])
 yy=(q.p_selective_A.to_numpy(float)>.5).astype(float);w=q.actionable_boundary_score.to_numpy(float)*q.c_pair.to_numpy(float);W=w.sum();a=anch(D)
 act=np.unique(np.r_[ii,jj]);pos={v:t for t,v in enumerate(act)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj]);b=a[act].copy()
 def fg(v):
  d=v[li]-v[lj];df=v-b;ce=np.logaddexp(0,d)-yy*d;f=.5*np.dot(df,df)/P+lam*np.dot(w,ce)/W
  g=df/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
 rr=minimize(lambda v:fg(v)[0],b,jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":2000,"ftol":1e-13,"gtol":1e-9});assert rr.success
 z=a.copy();z[act]=rr.x;ew=P*lam*w/W;rho=np.zeros(P)
 for i,j,v in zip(ii,jj,ew):rho[i]+=v;rho[j]+=v
 d=np.abs(z-a);rat=np.divide(d,rho,out=np.zeros(P),where=rho>0);ref=tk(a,k);cor=tk(z,k);outside=np.array([x for x in range(P) if x not in set(ref)])
 prot=den=0
 for i in range(P-1):
  da=a[i]-a[i+1:];rb=rho[i]+rho[i+1:];nz=da!=0;den+=nz.sum();prot+=(np.abs(da[nz])>rb[nz]).sum()
 cross=sum(((a[i]-a[outside])>(rho[i]+rho[outside])).sum() for i in ref);ct=k*(P-k)
 obs=set(str(SETS[(SETS.method=="selective")&(SETS.k==k)].iloc[0].selected_set).split("|"));got=set(fs[i] for i in cor);assert got==obs,(k,got,obs)
 q.assign(w0=w,theorem_edge_weight=ew).to_csv(O/f"K{k}_THEOREM_SCALE_EDGE_WEIGHTS.csv",index=False)
 pd.DataFrame({"feature":fs,"anchor_a":a,"corrected_s":z,"rho":rho,"abs_displacement":d,"ratio":np.where(rho>0,rat,np.nan)}).to_csv(O/f"K{k}_FEATURE_PROTECTION_BUDGETS.csv",index=False)
 return {"dataset":"Hospital Osteoporosis","selector":"L1 logistic rank","k":k,"lam":lam,"n_constraints":len(q),"rho_positive_n":int((rho>0).sum()),"rho_positive_fraction":float((rho>0).mean()),"bound_violation_n":int((d>rho+1e-9).sum()),"max_bound_violation":float(np.maximum(d-rho,0).max()),"max_displacement_to_budget_ratio":float(rat[rho>0].max()),"pairwise_protected_order_fraction":float(prot/den),"protected_pairs":int(prot),"pairwise_order_denominator":int(den),"cross_boundary_protected_fraction":float(cross/ct),"protected_cross_boundary_pairs":int(cross),"cross_boundary_pairs_total":int(ct),"full_topk_certificate":bool(cross==ct),"topk_jaccard":float(len(set(ref)&set(cor))/len(set(ref)|set(cor))),"topk_changed_n":int(k-len(set(ref)&set(cor))),"rho_sum_check":float(rho.sum()),"expected_rho_sum":float(2*P*lam),"rho_sum_check_pass":bool(np.isclose(rho.sum(),2*P*lam))}
rows=[one(k,float(HP[(HP.method=="selective")&(HP.k==k)].iloc[0].hyperparam)) for k in [5,10]]
pd.DataFrame(rows).to_csv(O/"PROTECTION_SUMMARY.csv",index=False);(O/"PROTECTION_SUMMARY.json").write_text(json.dumps(rows,indent=2));print(pd.DataFrame(rows).to_string(index=False))
