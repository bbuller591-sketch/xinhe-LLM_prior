

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
import json, math, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, norm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / 'gbm_canonical_recovery_20260917'))
BASE=Path(str(REPRO_ROOT / 'GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6'))
RUN=Path(str(REPRO_ROOT / 'GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/SCALEUP_V2_8'))
PP=RUN/"POSTPROCESS_V2_8"
OUT=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/07_DARMANIS/downstream')); OUT.mkdir(parents=True,exist_ok=True)
CONF=BASE/"DATA_CONFUSION_V2_7"

# ---------- use GPT-4o-mini complete selective pair-source measurement map ----------
source_files={"ARM_IVY":None,"ARM_G116":None,"ARM_G132":None}
M=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/07_DARMANIS/selective_COMPLETE_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B.csv')))
meas={}
for r in M.itertuples():
    meas[(int(r.node_i),int(r.node_j),str(r.arm))]={"y":float(r.y),"c":float(r.c),"H":float(r.H),"origin":str(r.origin),"pair":str(r.pair)}
print("GPT selective pair-source map ready",len(M),"records",M.origin.value_counts().to_dict(),flush=True)
# ---------- load data ----------
X=np.load(ROOT/"canonical/X_candidate_aligned_log1p.npy").astype(float)
y=pd.read_csv(ROOT/"canonical/y_canonical.csv")["y"].to_numpy(int)
mp=pd.read_csv(ROOT/"canonical/cell_sample_mapping_canonical.csv")
ann=pd.read_csv(ROOT/"metadata/darmanis_cell_annotation.csv")[["cell_key","plate"]]
mm=mp.merge(ann,left_on="cell_id",right_on="cell_key",how="left",validate="one_to_one")
groups=mm["plate"].astype(str).to_numpy()
SEED=2026091817
splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))
LAMS=[0,0.03,0.1,0.3,1,3,10]; KGRID=[10,20,30]
selpar={"LASSO":{"C":0.1,"l1_ratio":1.0},"ELASTICNET":{"C":0.1,"l1_ratio":0.8}}
selectors=["LASSO","ELASTICNET","SIS"]

def selector_score(selector,tr,fold):
    if selector=="SIS":
        xx=X[tr]; yy=y[tr]; yc=yy-yy.mean(); xc=xx-xx.mean(0)
        den=np.sqrt((xc*xc).sum(0)*(yc*yc).sum())
        return np.nan_to_num(np.abs((xc*yc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0.0)
    sc=StandardScaler().fit(X[tr]); Xt=sc.transform(X[tr]); par=selpar[selector]
    mod=LogisticRegression(C=par["C"],solver="saga",class_weight="balanced",
      penalty="l1" if selector=="LASSO" else "elasticnet",
      l1_ratio=None if selector=="LASSO" else par["l1_ratio"],
      max_iter=6000,tol=2e-4,random_state=SEED+fold,fit_intercept=True,n_jobs=1)
    mod.fit(Xt,y[tr]); return np.abs(mod.coef_[0])

def data_anchor(D):
    r=rankdata(-D,method="average")
    u=1-(r-0.5)/len(D)
    return norm.ppf(np.clip(u,1e-6,1-1e-6))

def stable_topk(z,D,k):
    # descending optimized z; ties -> original D; then column index
    return np.lexsort((np.arange(len(z)),-D,-z))[:k]

def eval_selected(cols,tr,va):
    sc=StandardScaler().fit(X[tr][:,cols]); Xt=sc.transform(X[tr][:,cols]); Xv=sc.transform(X[va][:,cols])
    mod=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",class_weight="balanced",
                           max_iter=5000,random_state=SEED)
    mod.fit(Xt,y[tr]); p=mod.predict_proba(Xv)[:,1]
    return {"auroc":roc_auc_score(y[va],p),
            "ap_core":average_precision_score(y[va],p),
            "ap_periphery":average_precision_score(1-y[va],1-p),
            "macro_ap":0.5*(average_precision_score(y[va],p)+average_precision_score(1-y[va],1-p)),
            "balanced_accuracy":balanced_accuracy_score(y[va],p>=0.5)}

def constraints_for(selector,k):
    z=pd.read_csv(CONF/f"{selector}_K{k}_PAIR_CONFUSION_V2_7.csv")
    z=z[z.pair_type=="ACTIONABLE_BOUNDARY_CONFUSION"].copy()
    rows=[]
    for r in z.itertuples():
        i,j=int(r.node_i),int(r.node_j)
        avail=[]
        for arm in source_files:
            key=(i,j,arm)
            if key in meas: avail.append((arm,meas[key]))
        m=len(avail)
        if m==0: continue
        for arm,q in avail:
            rows.append((i,j,float(q["y"]),float(r.actionable_boundary_score)*float(q["c"])/m,arm,float(q["c"])))
    return rows

def optimize_selective(sD,constraints,lam):
    if lam==0 or not constraints: return sD.copy(),True,0
    ii=np.array([x[0] for x in constraints],int); jj=np.array([x[1] for x in constraints],int)
    yy=np.array([x[2] for x in constraints],float); w=np.array([x[3] for x in constraints],float)
    W=w.sum()
    if W<=0:return sD.copy(),True,0
    p=len(sD)
    def fg(z):
        dz=z[ii]-z[jj]
        ce=np.logaddexp(0,dz)-yy*dz
        f=0.5*np.mean((z-sD)**2)+lam*np.dot(w,ce)/W
        grad=(z-sD)/p
        rr=lam*(w/W)*(expit(dz)-yy)
        np.add.at(grad,ii,rr); np.add.at(grad,jj,-rr)
        return float(f),grad
    res=minimize(lambda z:fg(z),sD.copy(),jac=True,method="L-BFGS-B",
                 options={"maxiter":1000,"ftol":1e-12,"gtol":1e-8})
    return res.x,bool(res.success),int(res.nit)

cons={(s,k):constraints_for(s,k) for s in selectors for k in KGRID}
cons_audit=[]
for (s,k),cc in cons.items():
    cons_audit.append({"selector":s,"k":k,"n_source_constraints":len(cc),
        "n_unique_pairs":len(set((x[0],x[1]) for x in cc)),
        "total_weight":float(sum(x[3] for x in cc))})
pd.DataFrame(cons_audit).to_csv(OUT/"selective_CONSTRAINT_AUDIT_V2_9.csv",index=False)
print(pd.DataFrame(cons_audit).to_string(index=False),flush=True)

rows=[]; t0=time.time()
for fi,(tr,va) in enumerate(splits,1):
    for selector in selectors:
        D=selector_score(selector,tr,fi); sD=data_anchor(D)
        for k in KGRID:
            cc=cons[(selector,k)]
            for lam in LAMS:
                z,ok,nit=optimize_selective(sD,cc,lam)
                cols=stable_topk(z,D,k)
                met=eval_selected(cols,tr,va)
                rows.append({"fold":fi,"selector":selector,"method":"selective","k":k,"lam":lam,
                             "opt_success":ok,"opt_nit":nit,**met,
                             "selected_nodes":"|".join(map(str,cols.tolist()))})
        print(f"done fold={fi} selector={selector}",flush=True)
fold=pd.DataFrame(rows); fold.to_csv(OUT/"selective_FOLD_RESULTS_V2_9.csv",index=False)
agg=(fold.groupby(["selector","method","lam","k"],as_index=False)
     .agg(mean_auroc=("auroc","mean"),sd_auroc=("auroc","std"),
          mean_macro_ap=("macro_ap","mean"),mean_ap_periphery=("ap_periphery","mean"),
          mean_balanced_accuracy=("balanced_accuracy","mean")))
agg.to_csv(OUT/"selective_ETA_CURVES_V2_9.csv",index=False)
sel=[]
for (selector,method,k),g in agg.groupby(["selector","method","k"]):
    best=float(g.mean_auroc.max())
    ch=g[np.isclose(g.mean_auroc,best,atol=1e-12,rtol=0)].sort_values("lam").iloc[0]
    sel.append(ch.to_dict())
sel=pd.DataFrame(sel); sel.to_csv(OUT/"selective_SELECTED_ETA_V2_9.csv",index=False)

# full-development selected sets using frozen full D and CV-selected lam
PILOT=BASE/"DATA_ONLY_PILOT"
Dfull={"LASSO":np.abs(np.load(PILOT/"LASSO_FULL_COEF_1SE_V2_7.npy")),
       "ELASTICNET":np.abs(np.load(PILOT/"ELASTICNET_FULL_COEF_1SE_V2_7.npy")),
       "SIS":np.load(PILOT/"SIS_ABS_CORR_SCORE_PILOT_V2_7.npy")}
cf=pd.read_csv(ROOT/"canonical/candidate_features_canonical.csv")
setrows=[]
for r in sel.itertuples():
    D=Dfull[r.selector]; sD=data_anchor(D)
    z,ok,nit=optimize_selective(sD,cons[(r.selector,int(r.k))],float(r.lam))
    cols=stable_topk(z,D,int(r.k))
    for rank,node in enumerate(cols,1):
        setrows.append({"selector":r.selector,"method":"selective","k":int(r.k),"lam":float(r.lam),
                        "rank":rank,"node":int(node),
                        "ensembl":cf.loc[node,"ensembl_gene_id_stable"],
                        "gene_symbol":"" if pd.isna(cf.loc[node,"gene_symbol"]) else cf.loc[node,"gene_symbol"],
                        "data_score":float(D[node]),"optimized_z":float(z[node])})
pd.DataFrame(setrows).to_csv(OUT/"selective_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv",index=False)
status={"status":"PASS_selective_INTERNAL_CV_COMPLETE","n_pair_source_measurements":len(M),
        "lambda_grid":LAMS,"k_grid":KGRID,"n_fold_rows":len(fold),
        "runtime_sec":round(time.time()-t0,1),"uses_test":False,
        "routing_note":"frozen full-development B=200 routing; internal diagnostic only"}
(OUT/"selective_STATUS_V2_9.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
print(json.dumps(status,indent=2),flush=True)
print("\nSELECTED ETA SUMMARY",flush=True)
print(sel[["selector","k","lam","mean_auroc","mean_macro_ap","mean_ap_periphery"]].to_string(index=False),flush=True)
