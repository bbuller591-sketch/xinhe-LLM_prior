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
from scipy.special import expit
from scipy.stats import rankdata,norm
PKG=Path(str(REPRO_ROOT / 'GBM_REPRO_PACKAGE_V2_9_20260919'))
B=PKG/"PROJECT/GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6";DWN=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/07_DARMANIS/downstream'));CONF=B/"DATA_CONFUSION_V2_7";PILOT=B/"DATA_ONLY_PILOT"
O=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q4'));O.mkdir(parents=True,exist_ok=True)
M=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/07_DARMANIS/selective_COMPLETE_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B.csv')));mm={(int(r.node_i),int(r.node_j),str(r.arm)):(float(r.y),float(r.c)) for r in M.itertuples()};arms=sorted(M.arm.unique());P=2000
Dfull={"ELASTICNET":np.abs(np.load(PILOT/"ELASTICNET_FULL_COEF_1SE_V2_7.npy")),"SIS":np.load(PILOT/"SIS_ABS_CORR_SCORE_PILOT_V2_7.npy")}
lams=pd.read_csv(DWN/"selective_SELECTED_ETA_V2_9.csv");expected=pd.read_csv(DWN/"selective_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv")
def anch(D):
 r=rankdata(-D,method="average");u=1-(r-.5)/len(D);return norm.ppf(np.clip(u,1e-6,1-1e-6))
def one(sel,k):
 D=Dfull[sel];a=anch(D);z0=pd.read_csv(CONF/f"{sel}_K{k}_PAIR_CONFUSION_V2_7.csv");z0=z0[z0.pair_type=="ACTIONABLE_BOUNDARY_CONFUSION"]
 rows=[]
 for r in z0.itertuples():
  i,j=int(r.node_i),int(r.node_j);av=[a0 for a0 in arms if (i,j,a0) in mm];m=len(av)
  for arm in av:
   y,c=mm[(i,j,arm)];rows.append((i,j,y,float(r.actionable_boundary_score)*c/m,arm))
 lam=float(lams[(lams.selector==sel)&(lams.k==k)].iloc[0].lam);ii=np.array([x[0] for x in rows]);jj=np.array([x[1] for x in rows]);yy=np.array([x[2] for x in rows]);w=np.array([x[3] for x in rows]);W=w.sum()
 def fg(v):
  d=v[ii]-v[jj];df=v-a;ce=np.logaddexp(0,d)-yy*d;f=.5*np.dot(df,df)/P+lam*np.dot(w,ce)/W
  g=df/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(g,ii,rr);np.add.at(g,jj,-rr);return float(f),g
 rr=minimize(lambda v:fg(v)[0],a.copy(),jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":1000,"ftol":1e-12,"gtol":1e-8});assert np.all(np.isfinite(rr.x));s=rr.x
 order=np.lexsort((np.arange(P),-D,-s));got=order[:k];exp=expected[(expected.selector==sel)&(expected.k==k)].sort_values("rank").node.to_numpy(int);assert np.array_equal(got,exp),(sel,k,got,exp)
 ew=P*lam*w/W;rho=np.zeros(P)
 for i,j,v in zip(ii,jj,ew):rho[i]+=v;rho[j]+=v
 disp=np.abs(s-a);rat=np.divide(disp,rho,out=np.zeros(P),where=rho>0);ref=np.lexsort((np.arange(P),-D,-a))[:k];outside=np.array([x for x in range(P) if x not in set(ref)])
 prot=den=0
 for i in range(P-1):
  da=a[i]-a[i+1:];rb=rho[i]+rho[i+1:];nz=da!=0;den+=int(nz.sum());prot+=int((np.abs(da[nz])>rb[nz]).sum())
 cross=sum(int(((a[i]-a[outside])>(rho[i]+rho[outside])).sum()) for i in ref);ct=k*(P-k)
 pd.DataFrame(rows,columns=["node_i","node_j","y","w0","arm"]).assign(theorem_edge_weight=ew).to_csv(O/f"{sel}_K{k}_THEOREM_SCALE_EDGE_WEIGHTS.csv",index=False)
 pd.DataFrame({"node":np.arange(P),"anchor_a":a,"corrected_s":s,"rho":rho,"abs_displacement":disp,"ratio":np.where(rho>0,rat,np.nan)}).to_csv(O/f"{sel}_K{k}_FEATURE_PROTECTION_BUDGETS.csv",index=False)
 return {"dataset":"Darmanis GBM","selector":sel,"k":k,"lam":lam,"n_constraints":len(rows),"rho_positive_n":int((rho>0).sum()),"rho_positive_fraction":float((rho>0).mean()),"bound_violation_n":int((disp>rho+1e-9).sum()),"max_bound_violation":float(np.maximum(disp-rho,0).max()),"max_displacement_to_budget_ratio":float(rat[rho>0].max()),"pairwise_protected_order_fraction":float(prot/den),"protected_pairs":prot,"pairwise_order_denominator":den,"cross_boundary_protected_fraction":float(cross/ct),"protected_cross_boundary_pairs":cross,"cross_boundary_pairs_total":ct,"full_topk_certificate":bool(cross==ct),"topk_jaccard":float(len(set(ref)&set(got))/len(set(ref)|set(got))),"topk_changed_n":int(k-len(set(ref)&set(got))),"rho_sum_check":float(rho.sum()),"expected_rho_sum":float(2*P*lam),"rho_sum_check_pass":bool(np.isclose(rho.sum(),2*P*lam)),"objective_compatibility":"EXACT_NORMALIZED_CE; full-development routing internal diagnostic"}
rows=[one("SIS",10),one("SIS",20),one("ELASTICNET",10)];pd.DataFrame(rows).to_csv(O/"PROTECTION_SUMMARY.csv",index=False);(O/"PROTECTION_SUMMARY.json").write_text(json.dumps(rows,indent=2));print(pd.DataFrame(rows).to_string(index=False))
