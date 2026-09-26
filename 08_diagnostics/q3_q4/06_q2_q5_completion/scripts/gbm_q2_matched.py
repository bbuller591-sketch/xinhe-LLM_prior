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
import hashlib,json,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata,norm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

PKG=Path(str(REPRO_ROOT / 'GBM_REPRO_PACKAGE_V2_9_20260919'))
PR=PKG/"PROJECT/GBM_EXTERNAL_EVIDENCE_20260917"
BASE=PR/"10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6"
DWN=BASE/"DOWNSTREAM_V2_9";CONF=BASE/"DATA_CONFUSION_V2_7";RUN=PR/"10_LLM_MEASUREMENT/SCALEUP_V2_8";PP=RUN/"POSTPROCESS_V2_8"
ROOT=Path(str(REPRO_ROOT / 'gbm_canonical_recovery_20260917'))
OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion/artifacts/DARMANIS_GBM_Q2_MATCHED'));OUT.mkdir(parents=True,exist_ok=True)
SEED=2026091817;LAMS=[0,.03,.1,.3,1,3,10];CELLS=[("SIS",10),("SIS",20),("ELASTICNET",10)]
srcfiles={"ARM_IVY":PP/"IVY_BROAD_PAIR_NEUTRALIZED_V2_8.csv","ARM_G116":PP/"G116_BROAD_PAIR_NEUTRALIZED_V2_8.csv","ARM_G132":PP/"G132_BROAD_PAIR_NEUTRALIZED_V2_8.csv"}
broad={}
for arm,fp in srcfiles.items():
 d=pd.read_csv(fp); broad[arm]=d.copy()
X=np.load(ROOT/"canonical/X_candidate_aligned_log1p.npy").astype(float);y=pd.read_csv(ROOT/"canonical/y_canonical.csv")["y"].to_numpy(int)
mp=pd.read_csv(ROOT/"canonical/cell_sample_mapping_canonical.csv");ann=pd.read_csv(ROOT/"metadata/darmanis_cell_annotation.csv")[["cell_key","plate"]];groups=mp.merge(ann,left_on="cell_id",right_on="cell_key",how="left",validate="one_to_one")["plate"].astype(str).to_numpy()
splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups));P=X.shape[1]
selpar={"ELASTICNET":{"C":.1,"l1_ratio":.8}}
def score(selector,tr,fold):
 if selector=="SIS":
  xx=X[tr];yy=y[tr];yc=yy-yy.mean();xc=xx-xx.mean(0);den=np.sqrt((xc*xc).sum(0)*(yc*yc).sum());return np.nan_to_num(np.abs((xc*yc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0.)
 sc=StandardScaler().fit(X[tr]);m=LogisticRegression(C=.1,solver="saga",class_weight="balanced",penalty="elasticnet",l1_ratio=.8,max_iter=6000,tol=2e-4,random_state=SEED+fold,n_jobs=1).fit(sc.transform(X[tr]),y[tr]);return np.abs(m.coef_[0])
def anch(D):
 r=rankdata(-D,method="average");return norm.ppf(np.clip(1-(r-.5)/len(D),1e-6,1-1e-6))
def top(z,D,k):return np.lexsort((np.arange(P),-D,-z))[:k]
def ev(ids,tr,va):
 sc=StandardScaler().fit(X[tr][:,ids]);m=LogisticRegression(C=1.,penalty="l2",solver="lbfgs",class_weight="balanced",max_iter=5000,random_state=SEED).fit(sc.transform(X[tr][:,ids]),y[tr]);p=m.predict_proba(sc.transform(X[va][:,ids]))[:,1];ap=average_precision_score(y[va],p);apn=average_precision_score(1-y[va],1-p);return float(roc_auc_score(y[va],p)),float(.5*(ap+apn))
M=pd.read_csv(DWN/"selective_COMPLETE_PAIR_SOURCE_MEASUREMENTS_V2_9.csv");mmap={(int(r.node_i),int(r.node_j),str(r.arm)):(float(r.y),float(r.c)) for r in M.itertuples()}
def selective_template(sel,k):
 z=pd.read_csv(CONF/f"{sel}_K{k}_PAIR_CONFUSION_V2_7.csv");z=z[z.pair_type=="ACTIONABLE_BOUNDARY_CONFUSION"];rows=[]
 for r in z.itertuples():
  i,j=int(r.node_i),int(r.node_j);avs=[arm for arm in srcfiles if (i,j,arm) in mmap];m=len(avs)
  for arm in avs:
   yy,c=mmap[(i,j,arm)];rows.append({"i":i,"j":j,"arm":arm,"base":float(r.actionable_boundary_score)/m,"y":yy,"c":c})
 return pd.DataFrame(rows)
def global_rows(sel,k):
 t=selective_template(sel,k);rows=[]
 for arm,g in t.groupby("arm"):
  d=broad[arm].copy();d["hh"]=[hashlib.sha256((f"{sel}|{k}|{arm}|"+str(p)).encode()).hexdigest() for p in d["pair"]];d=d.sort_values("hh").head(len(g)).copy()
  bases=sorted(g["base"].to_numpy(float),reverse=True);d["ha"]=[hashlib.sha256(("assign|"+str(p)).encode()).hexdigest() for p in d["pair"]];d=d.sort_values("ha")
  for rr,b in zip(d.itertuples(),bases):rows.append((int(rr.i),int(rr.j),float(rr.hard_y),float(b)*float(rr.certainty),arm,str(rr.pair)))
 return rows,t
def solve(a,rows,lam):
 if lam==0:return a.copy()
 ii=np.array([r[0] for r in rows]);jj=np.array([r[1] for r in rows]);yy=np.array([r[2] for r in rows]);w=np.array([r[3] for r in rows]);W=w.sum()
 act=np.unique(np.r_[ii,jj]);pos={v:t for t,v in enumerate(act)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj]);b=a[act].copy()
 def fg(v):
  d=v[li]-v[lj];df=v-b;ce=np.logaddexp(0,d)-yy*d;f=.5*np.dot(df,df)/P+lam*np.dot(w,ce)/W;gr=df/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(gr,li,rr);np.add.at(gr,lj,-rr);return float(f),gr
 rr=minimize(lambda v:fg(v)[0],b,jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":1500,"ftol":1e-12,"gtol":1e-8});z=a.copy();z[act]=rr.x;return z
real=pd.read_csv(DWN/"reference_global_selective_SELECTED_COMPARISON_V2_9.csv")
out=[]
for sel,k in CELLS:
 rows,t=global_rows(sel,k);curve=[]
 for lam in LAMS:
  aa=[];mm=[]
  for fi,(tr,va) in enumerate(splits,1):
   D=score(sel,tr,fi);z=solve(anch(D),rows,lam);ids=top(z,D,k);a,m=ev(ids,tr,va);aa.append(a);mm.append(m)
  curve.append((lam,float(np.mean(aa)),float(np.mean(mm))))
 best=max(x[1] for x in curve);lam=min(x[0] for x in curve if np.isclose(x[1],best,atol=1e-12,rtol=0))
 rr0=real[(real.selector==sel)&(real.method=="reference")&(real.k==k)].iloc[0];rr3=real[(real.selector==sel)&(real.method=="selective")&(real.k==k)].iloc[0]
 armcounts=t.arm.value_counts().to_dict()
 out.append({"dataset":"Darmanis GBM","selector":sel,"k":k,"global_selected_lam":lam,"global_cv_mean_auroc":best,"reference_cv_auroc":float(rr0.mean_auroc),"selective_selective_cv_auroc":float(rr3.mean_auroc),"global_delta_vs_reference":best-float(rr0.mean_auroc),"selective_delta_vs_reference":float(rr3.mean_auroc)-float(rr0.mean_auroc),"selective_minus_global":float(rr3.mean_auroc)-best,"n_source_constraints":len(rows),"source_constraint_counts":json.dumps(armcounts,sort_keys=True),"global_pair_rule":"within each frozen source arm, deterministic SHA256 pair sampling; source-specific query counts matched to Selective and Selective routing-weight multiset reassigned within arm; same CE objective and lam grid","evaluation_scope":"internal 5-fold plate-grouped diagnostic; current selective routing is full-development/leakage-sensitive and not patient-level generalization","new_llm_calls":0})
 print(out[-1],flush=True)
pd.DataFrame(out).to_csv(OUT/"Q2_MATCHED_RESULTS.csv",index=False);(OUT/"Q2_MATCHED_RESULTS.json").write_text(json.dumps(out,indent=2))
