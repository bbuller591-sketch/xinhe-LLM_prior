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
import json,hashlib,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
R=Path(str(REPRO_ROOT / '04_credit_g'));O=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion/artifacts/CREDIT_Q2_Q4_MIGRATION'));O.mkdir(parents=True,exist_ok=True)
FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"];CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"];NUM=[x for x in FEATURES if x not in CAT];P=len(FEATURES);K=10;LAMS=[0,.025,.05,.075,.10,.15,.20,.30]
X=pd.read_csv(R/"01_DATA_AND_SPLITS/X.csv");y=pd.read_csv(R/"01_DATA_AND_SPLITS/y.csv").label.to_numpy();dev=np.load(R/"01_DATA_AND_SPLITS/modern_dev_indices_seed20260918.npy");hold=np.load(R/"01_DATA_AND_SPLITS/modern_holdout_indices_seed20260918.npy");Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev];Xh=X.iloc[hold].reset_index(drop=True);yh=y[hold]
ranks=pd.read_csv(R/"06_RESULTS/reference_RESAMPLE_RANKS.csv");m0res=pd.read_csv(R/"06_RESULTS/reference_RESAMPLE_RESULTS.csv");fullr=pd.read_csv(R/"06_RESULTS/FINAL_DATA_ONLY_RANKINGS_FREEZE.csv")
G=pd.read_csv(R/"02_PROTOCOLS/selective_SELECTIVE_GRAPH_FREEZE.csv");M=pd.read_csv(R/"06_RESULTS/selective_PAIR_MEASUREMENT.csv");M=M.set_index("pair_id");idx={f:i for i,f in enumerate(FEATURES)}
def prep(cols):
 num=[c for c in NUM if c in cols];cat=[c for c in CAT if c in cols];tr=[]
 if num:tr.append(("num",StandardScaler(),num))
 if cat:tr.append(("cat",Pipeline([("ohe",OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False)),("scale",StandardScaler())]),cat))
 return ColumnTransformer(tr,remainder="drop",sparse_threshold=0.)
def evalset(sel,Xtr,ytr,Xev,yev):
 p=Pipeline([("prep",prep(sel)),("clf",LogisticRegression(penalty="l2",solver="lbfgs",C=.01,max_iter=5000,tol=1e-5))]);p.fit(Xtr[sel],ytr);pr=p.predict_proba(Xev[sel])[:,1];return float(roc_auc_score(yev,pr)),float(average_precision_score(yev,pr))
def anchor_from_rank(rmap):return np.array([(P-rmap[f])/(P-1) for f in FEATURES],float)
def solve(a,rows,lam):
 if lam==0 or not rows:return a.copy()
 ii=np.array([idx[x[0]] for x in rows]);jj=np.array([idx[x[1]] for x in rows]);yy=np.array([x[2] for x in rows]);w=np.array([x[3] for x in rows]);W=w.sum();act=np.unique(np.r_[ii,jj]);pos={v:t for t,v in enumerate(act)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj]);b=a[act].copy()
 def fg(v):
  d=v[li]-v[lj];df=v-b;ce=np.logaddexp(0,d)-yy*d;f=.5*np.dot(df,df)/P+lam*np.dot(w,ce)/W;g=df/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
 rr=minimize(lambda v:fg(v)[0],b,jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":2000,"ftol":1e-13,"gtol":1e-9});z=a.copy();z[act]=rr.x;return z
def top(z):return [FEATURES[i] for i in np.lexsort((np.arange(P),-z))[:K]]
eligible=G[G.formal_selective_measurement_eligible==True].copy()
def rows_for(selector,mode):
 req="requested_gbm_perm" if selector=="GBM_PERM" else "requested_elastic_net";uc="u_data_gbm_perm" if selector=="GBM_PERM" else "u_data_elastic_net";sel=eligible[eligible[req]==True]
 n=len(sel);weights=sorted(sel[uc].to_numpy(float),reverse=True)
 if mode=="selective":chosen=sel.sort_values("pair_id").copy()
 else:chosen=eligible.sort_values("pair_id").head(n).copy()
 rows=[]
 for (_,r),u in zip(chosen.iterrows(),weights):
  m=M.loc[r.pair_id];p=float(m.p_canonical_a_order_neutral);c=float(m.c_edge_entropy);rows.append((str(r.feature_a),str(r.feature_b),1. if p>.5 else 0.,u*c,str(r.pair_id)))
 return rows
outer=list(StratifiedShuffleSplit(n_splits=300,train_size=.8,test_size=.2,random_state=20260918).split(Xd,yd))
out=[];q4=[]
for s in ["GBM_PERM","ELASTIC_NET"]:
 for mode in ["selective","global"]:
  rows=rows_for(s,mode);cur=[];cache={}
  for lam in LAMS:
   aa=[]
   for rr,(itr,iev) in enumerate(outer):
    q=ranks[(ranks["resample"]==rr)&(ranks.selector==s)].set_index("feature");rmap=q["rank"].to_dict();a=anchor_from_rank(rmap);ss=tuple(top(solve(a,rows,lam)));key=(rr,ss)
    if key not in cache:cache[key]=evalset(list(ss),Xd.iloc[itr].reset_index(drop=True),yd[itr],Xd.iloc[iev].reset_index(drop=True),yd[iev])[0]
    aa.append(cache[key])
   cur.append((lam,float(np.mean(aa))))
  best=max(v for _,v in cur);lam=min(e for e,v in cur if v>=best-1e-4)
  q=fullr[fullr.selector==s].set_index("feature");a=anchor_from_rank(q.data_rank.to_dict());z=solve(a,rows,lam);ss=top(z);hau,hap=evalset(ss,Xd,yd,Xh,yh)
  ref=pd.read_csv(R/"06_RESULTS/FINAL_HOLDOUT_RESULTS.csv");r0=ref[(ref.selector==s)&(ref.method=="reference")].iloc[0];legacy=ref[(ref.selector==s)&(ref.method=="selective")].iloc[0]
  out.append({"dataset":"CREDIT-G","selector":"GBM-permutation" if s=="GBM_PERM" else "Elastic Net","k":10,"arm":mode,"n_pairs":len(rows),"selected_lam":lam,"dev_mean_auroc":best,"reference_holdout_auroc":float(r0.holdout_auroc),"migrated_holdout_auroc":hau,"migrated_delta_vs_reference":hau-float(r0.holdout_auroc),"legacy_selective_holdout_auroc":float(legacy.holdout_auroc),"pair_ids":"|".join(x[4] for x in rows),"chronology":"theorem-objective migration and matched-Q2 added after original holdout had been inspected; development selects lam; holdout number is post-hoc diagnostic only","new_llm_calls":0})
  if mode=="selective":
   ii=np.array([idx[x[0]] for x in rows]);jj=np.array([idx[x[1]] for x in rows]);w=np.array([x[3] for x in rows]);W=w.sum();ew=P*lam*w/W if W>0 else np.zeros(len(w));rho=np.zeros(P)
   for i,j,v in zip(ii,jj,ew):rho[i]+=v;rho[j]+=v
   d=np.abs(z-a);rat=np.divide(d,rho,out=np.zeros(P),where=rho>0);refids=np.lexsort((np.arange(P),-a))[:K];corids=np.lexsort((np.arange(P),-z))[:K];outside=np.array([x for x in range(P) if x not in set(refids)]);prot=den=0
   for i in range(P-1):
    da=a[i]-a[i+1:];rb=rho[i]+rho[i+1:];nz=da!=0;den+=int(nz.sum());prot+=int((np.abs(da[nz])>rb[nz]).sum())
   cross=sum(int(((a[i]-a[outside])>(rho[i]+rho[outside])).sum()) for i in refids);ct=K*(P-K)
   q4.append({"dataset":"CREDIT-G","selector":"GBM-permutation" if s=="GBM_PERM" else "Elastic Net","k":10,"lam":lam,"n_constraints":len(rows),"rho_positive_fraction":float((rho>0).mean()),"bound_violation_n":int((d>rho+1e-9).sum()),"max_displacement_to_budget_ratio":float(rat[rho>0].max()) if (rho>0).any() else 0.,"pairwise_protected_order_fraction":float(prot/den),"cross_boundary_protected_fraction":float(cross/ct),"full_topk_certificate":bool(cross==ct),"topk_jaccard":float(len(set(refids)&set(corids))/len(set(refids)|set(corids))),"topk_changed_n":int(K-len(set(refids)&set(corids))),"rho_sum_check_pass":bool(np.isclose(rho.sum(),2*P*lam)),"objective_compatibility":"MIGRATED_TO_NORMALIZED_CE"})
 print(out[-1],flush=True)
pd.DataFrame(out).to_csv(O/"Q2_MIGRATED_MATCHED_RESULTS.csv",index=False);pd.DataFrame(q4).to_csv(O/"Q4_MIGRATED_PROTECTION_RESULTS.csv",index=False);(O/"Q2_Q4_MIGRATION.json").write_text(json.dumps({"q2":out,"q4":q4},indent=2))
