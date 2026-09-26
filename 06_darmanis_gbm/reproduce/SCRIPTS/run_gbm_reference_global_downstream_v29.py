

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
import json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / 'gbm_canonical_recovery_20260917'))
BASE=Path(str(REPRO_ROOT / 'GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6'))
PP=Path(str(REPRO_ROOT / 'GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/SCALEUP_V2_8/POSTPROCESS_V2_8'))
OUT=BASE/"DOWNSTREAM_V2_9"; OUT.mkdir(parents=True,exist_ok=True)

X=np.load(ROOT/"canonical/X_candidate_aligned_log1p.npy").astype(float)
y=pd.read_csv(ROOT/"canonical/y_canonical.csv")["y"].to_numpy(int)
cf=pd.read_csv(ROOT/"canonical/candidate_features_canonical.csv")
mp=pd.read_csv(ROOT/"canonical/cell_sample_mapping_canonical.csv")
ann=pd.read_csv(ROOT/"metadata/darmanis_cell_annotation.csv")[["cell_key","plate"]]
m=mp.merge(ann,left_on="cell_id",right_on="cell_key",how="left",validate="one_to_one")
groups=m["plate"].astype(str).to_numpy()
assert X.shape==(632,2000) and len(y)==632

H=pd.read_csv(PP/"SOURCE_AGGREGATED_H_global_V2_8.csv").sort_values("node")
assert np.array_equal(H.node.to_numpy(),np.arange(2000))
h1=H.h_global.to_numpy(float); h2=H.h_global_certainty.to_numpy(float)

SEED=2026091817
splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))
LAMBDAS=[0,0.03,0.1,0.3,1,3,10]
KGRID=[10,20,30]
selpar={"LASSO":{"C":0.1,"l1_ratio":1.0},"ELASTICNET":{"C":0.1,"l1_ratio":0.8}}

def stable_order(score):
    return np.lexsort((np.arange(len(score)),-np.asarray(score)))
def rank_normalize(score):
    o=stable_order(score)
    r=np.empty(len(score),int); r[o]=np.arange(1,len(score)+1)
    return 2*(1-(r-1)/(len(score)-1))-1

def selector_score(selector,tr,fold):
    if selector=="SIS":
        xx=X[tr]; yy=y[tr]
        yc=yy-yy.mean(); xc=xx-xx.mean(0)
        den=np.sqrt((xc*xc).sum(0)*(yc*yc).sum())
        return np.nan_to_num(np.abs((xc*yc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0.0)
    sc=StandardScaler().fit(X[tr]); Xt=sc.transform(X[tr])
    par=selpar[selector]
    mod=LogisticRegression(C=par["C"],solver="saga",class_weight="balanced",
        penalty="l1" if selector=="LASSO" else "elasticnet",
        l1_ratio=None if selector=="LASSO" else par["l1_ratio"],
        max_iter=6000,tol=2e-4,random_state=SEED+fold,fit_intercept=True,n_jobs=1)
    mod.fit(Xt,y[tr])
    return np.abs(mod.coef_[0])

def eval_selected(cols,tr,va):
    sc=StandardScaler().fit(X[tr][:,cols])
    Xt=sc.transform(X[tr][:,cols]); Xv=sc.transform(X[va][:,cols])
    mod=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",class_weight="balanced",
                           max_iter=5000,random_state=SEED)
    mod.fit(Xt,y[tr]); p=mod.predict_proba(Xv)[:,1]
    return {
        "auroc":roc_auc_score(y[va],p),
        "ap_core":average_precision_score(y[va],p),
        "ap_periphery":average_precision_score(1-y[va],1-p),
        "macro_ap":0.5*(average_precision_score(y[va],p)+average_precision_score(1-y[va],1-p)),
        "balanced_accuracy":balanced_accuracy_score(y[va],p>=0.5)
    }

rows=[]; score_rows=[]
t0=time.time()
selectors=["LASSO","ELASTICNET","SIS"]
for fi,(tr,va) in enumerate(splits,1):
    for selector in selectors:
        D=selector_score(selector,tr,fi)
        d=rank_normalize(D)
        score_rows.append({"fold":fi,"selector":selector,"n_positive_D":int(np.sum(D>0)),
                           "D_max":float(D.max()),"D_median":float(np.median(D))})
        for method,h in [("reference",np.zeros(2000)),("global",h1),("global_certainty",h2)]:
            lams=[0] if method=="reference" else LAMBDAS
            for lam in lams:
                F=d+lam*h
                order=stable_order(F)
                for k in KGRID:
                    cols=order[:k]
                    met=eval_selected(cols,tr,va)
                    rows.append({"fold":fi,"selector":selector,"method":method,"lam":lam,"k":k,
                                 **met,"selected_nodes":"|".join(map(str,cols.tolist()))})
        print(f"done fold={fi} selector={selector}",flush=True)
fold=pd.DataFrame(rows)
fold.to_csv(OUT/"reference_global_FOLD_RESULTS_V2_9.csv",index=False)
pd.DataFrame(score_rows).to_csv(OUT/"reference_global_FOLD_SELECTOR_AUDIT_V2_9.csv",index=False)

agg=(fold.groupby(["selector","method","lam","k"],as_index=False)
     .agg(mean_auroc=("auroc","mean"),sd_auroc=("auroc","std"),
          mean_macro_ap=("macro_ap","mean"),mean_ap_periphery=("ap_periphery","mean"),
          mean_balanced_accuracy=("balanced_accuracy","mean")))
agg.to_csv(OUT/"reference_global_GAMMA_CURVES_V2_9.csv",index=False)

# choose lam by mean AUROC, deterministic smallest lam on ties
selected=[]
for (sel,method,k),g in agg.groupby(["selector","method","k"]):
    g=g.sort_values(["mean_auroc","lam"],ascending=[False,True],kind="mergesort")
    best=float(g.mean_auroc.max())
    ch=g[np.isclose(g.mean_auroc,best,atol=1e-12,rtol=0)].sort_values("lam").iloc[0]
    selected.append(ch.to_dict())
sel_df=pd.DataFrame(selected)
sel_df.to_csv(OUT/"reference_global_SELECTED_GAMMA_V2_9.csv",index=False)

# full-development selected sets using frozen full-data D and CV-selected lam
PILOT=BASE/"DATA_ONLY_PILOT"
Dfull={
"LASSO":np.abs(np.load(PILOT/"LASSO_FULL_COEF_1SE_V2_7.npy")),
"ELASTICNET":np.abs(np.load(PILOT/"ELASTICNET_FULL_COEF_1SE_V2_7.npy")),
"SIS":np.load(PILOT/"SIS_ABS_CORR_SCORE_PILOT_V2_7.npy")
}
setrows=[]
for r in sel_df.itertuples():
    d=rank_normalize(Dfull[r.selector])
    h=np.zeros(2000) if r.method=="reference" else (h1 if r.method=="global" else h2)
    F=d+float(r.lam)*h
    cols=stable_order(F)[:int(r.k)]
    for rank,node in enumerate(cols,1):
        setrows.append({"selector":r.selector,"method":r.method,"k":int(r.k),"lam":float(r.lam),
                        "rank":rank,"node":int(node),
                        "ensembl":cf.loc[node,"ensembl_gene_id_stable"],
                        "gene_symbol":"" if pd.isna(cf.loc[node,"gene_symbol"]) else cf.loc[node,"gene_symbol"],
                        "data_score":float(Dfull[r.selector][node]),
                        "h":float(h[node]),"combined_score":float(F[node])})
pd.DataFrame(setrows).to_csv(OUT/"reference_global_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv",index=False)

status={"status":"PASS_reference_global_INTERNAL_CV_COMPLETE","n_fold_rows":len(fold),
        "n_curve_rows":len(agg),"n_selected_configs":len(sel_df),
        "lambda_grid":LAMBDAS,"k_grid":KGRID,"selectors":selectors,
        "runtime_sec":round(time.time()-t0,1),
        "uses_test":False,"claim_scope":"internal plate-grouped diagnostic/stress-test utility"}
(OUT/"reference_global_STATUS_V2_9.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
print(json.dumps(status,indent=2))
print("\nSELECTED GAMMA SUMMARY")
print(sel_df[["selector","method","k","lam","mean_auroc","mean_macro_ap","mean_ap_periphery"]].to_string(index=False))
