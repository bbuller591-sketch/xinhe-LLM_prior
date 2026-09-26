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
import json, time, warnings, platform
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
FROZEN=ROOT/"03_FROZEN_DATA"
OUTROOT=ROOT/"04_reference"
OUTROOT.mkdir(parents=True,exist_ok=True)

SEED=2026091901
OUTER_N=5
INNER_N=4
CGRID=[0.01,0.03,0.1,0.3,1.0,3.0]
ALPHAS=[0.2,0.5,0.8]
KGRID=[10,20,30]
MAXK=max(KGRID)

TASKS=["BREAST_GSE25055_GSE25065","SEPSIS_GSE65682"]

def stable_order(score):
    score=np.asarray(score,float)
    return np.lexsort((np.arange(len(score)),-score))

def fit_sparse(X,y,C,l1_ratio,seed):
    sc=StandardScaler().fit(X)
    Xt=sc.transform(X)
    penalty="l1" if l1_ratio==1.0 else "elasticnet"
    mod=LogisticRegression(C=C,solver="saga",class_weight="balanced",penalty=penalty,
        l1_ratio=None if penalty=="l1" else l1_ratio,max_iter=5000,tol=1e-4,
        random_state=seed,fit_intercept=True,n_jobs=1)
    mod.fit(Xt,y)
    return sc,mod

def inner_tune(X,y,selector,outer_fold):
    # Same frozen grid/solver as the protocol, but use warm starts along C paths
    # within each inner split. This changes only computation, not the candidate
    # hyperparameters or the CV decision rule.
    inner=StratifiedKFold(n_splits=INNER_N,shuffle=True,random_state=SEED+100*outer_fold)
    alphas=[1.0] if selector=="LASSO" else ALPHAS
    rec={(C,a):{"auc":[],"nnz":[],"converged":[]} for a in alphas for C in CGRID}
    for ii,(tr,va) in enumerate(inner.split(X,y),1):
        sc=StandardScaler().fit(X[tr])
        Xt=sc.transform(X[tr]); Xv=sc.transform(X[va])
        for a in alphas:
            penalty="l1" if selector=="LASSO" else "elasticnet"
            mod=LogisticRegression(C=CGRID[0],solver="saga",class_weight="balanced",
                penalty=penalty,l1_ratio=None if penalty=="l1" else a,
                max_iter=5000,tol=1e-4,
                random_state=SEED+10000*outer_fold+100*ii+int(a*10),
                fit_intercept=True,n_jobs=1,warm_start=True)
            for C in CGRID:
                mod.set_params(C=C)
                mod.fit(Xt,y[tr])
                pred=mod.predict_proba(Xv)[:,1]
                q=rec[(C,a)]
                q["auc"].append(float(roc_auc_score(y[va],pred)))
                q["nnz"].append(int(np.sum(np.abs(mod.coef_[0])>1e-12)))
                q["converged"].append(bool(int(mod.n_iter_[0])<mod.max_iter))
    rows=[]
    for (C,a),q in rec.items():
        rows.append({"C":C,"l1_ratio":a,"mean_inner_auroc":float(np.mean(q["auc"])),
                     "sd_inner_auroc":float(np.std(q["auc"],ddof=1)),
                     "min_nnz":int(min(q["nnz"])),"median_nnz":float(np.median(q["nnz"])),
                     "max_nnz":int(max(q["nnz"])),"all_converged":bool(all(q["converged"]))})
    tab=pd.DataFrame(rows)
    admiss=tab[tab.min_nnz>=MAXK].copy()
    fallback=False
    if len(admiss)==0:
        admiss=tab.copy(); fallback=True
        admiss=admiss.sort_values(["min_nnz","mean_inner_auroc","C","l1_ratio"],
                                  ascending=[False,False,True,False],kind="mergesort")
    else:
        admiss=admiss.sort_values(["mean_inner_auroc","C","l1_ratio"],
                                  ascending=[False,True,False],kind="mergesort")
    ch=admiss.iloc[0]
    return tab,dict(C=float(ch.C),l1_ratio=float(ch.l1_ratio),
                    inner_auroc=float(ch.mean_inner_auroc),min_nnz=int(ch.min_nnz),
                    admissibility_fallback=bool(fallback))

def selector_score(X,y,selector,params,fold):
    if selector=="SIS":
        yc=y-y.mean(); xc=X-X.mean(axis=0)
        den=np.sqrt((xc*xc).sum(axis=0)*(yc*yc).sum())
        return np.nan_to_num(np.abs((xc*yc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0),None,None
    sc,mod=fit_sparse(X,y,params["C"],params["l1_ratio"],SEED+fold)
    D=np.abs(mod.coef_[0])
    return D,sc,mod

def eval_selected(X,y,tr,va,cols):
    sc=StandardScaler().fit(X[tr][:,cols])
    Xt=sc.transform(X[tr][:,cols]); Xv=sc.transform(X[va][:,cols])
    mod=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",class_weight="balanced",
                           max_iter=5000,random_state=SEED)
    mod.fit(Xt,y[tr])
    p=mod.predict_proba(Xv)[:,1]
    ap_pos=average_precision_score(y[va],p)
    ap_neg=average_precision_score(1-y[va],1-p)
    return {"auroc":float(roc_auc_score(y[va],p)),
            "ap_positive":float(ap_pos),"ap_negative":float(ap_neg),
            "macro_ap":float(0.5*(ap_pos+ap_neg)),
            "balanced_accuracy":float(balanced_accuracy_score(y[va],p>=0.5)),
            "validation_prevalence":float(y[va].mean())}

def pairwise_jaccard(sets):
    rows=[]
    keys=sorted(sets)
    for a in range(len(keys)):
        for b in range(a+1,len(keys)):
            ka,kb=keys[a],keys[b]
            A,B=sets[ka],sets[kb]
            rows.append({"fold_a":ka,"fold_b":kb,"jaccard":len(A&B)/len(A|B)})
    return rows

def run_task(task):
    t0=time.time()
    src=FROZEN/task
    out=OUTROOT/task; out.mkdir(parents=True,exist_ok=True)
    # IMPORTANT: intentionally load DEVELOPMENT files only. Sealed validation is not referenced below.
    X=np.load(src/"X_development.npy").astype(float)
    y=np.load(src/"y_development.npy").astype(int)
    samples=pd.read_csv(src/"development_samples.csv")
    feat=pd.read_csv(src/"features_p2000.csv")
    assert X.shape==(len(y),2000) and len(samples)==len(y) and len(feat)==2000
    assert np.array_equal(samples.y.to_numpy(int),y)
    assert np.isfinite(X).all() and len(np.unique(y))==2

    outer=StratifiedKFold(n_splits=OUTER_N,shuffle=True,random_state=SEED)
    splits=list(outer.split(X,y))
    split_rows=[]
    for fi,(tr,va) in enumerate(splits,1):
        for ix in tr: split_rows.append({"fold":fi,"role":"outer_train","sample_index":int(ix),"sample_id":samples.iloc[ix].sample_id,"y":int(y[ix])})
        for ix in va: split_rows.append({"fold":fi,"role":"outer_validation","sample_index":int(ix),"sample_id":samples.iloc[ix].sample_id,"y":int(y[ix])})
    pd.DataFrame(split_rows).to_csv(out/"OUTER_SPLITS.csv",index=False)

    fold_rows=[]; tune_rows=[]; sel_rows=[]; sparse_rows=[]
    sets={}
    for fi,(tr,va) in enumerate(splits,1):
        for selector in ["LASSO","ELASTICNET","SIS"]:
            params=None
            if selector!="SIS":
                tab,params=inner_tune(X[tr],y[tr],selector,fi)
                tab.insert(0,"fold",fi); tab.insert(1,"selector",selector)
                tune_rows.extend(tab.to_dict("records"))
            D,sc,mod=selector_score(X[tr],y[tr],selector,params,fi)
            nnz=int(np.sum(D>1e-12))
            sparse_rows.append({"fold":fi,"selector":selector,"n_positive_scores":nnz,
                                "D_max":float(D.max()),"D_median":float(np.median(D)),
                                "chosen_C":None if params is None else params["C"],
                                "chosen_l1_ratio":None if params is None else params["l1_ratio"],
                                "chosen_inner_auroc":None if params is None else params["inner_auroc"],
                                "admissibility_fallback":False if params is None else params["admissibility_fallback"]})
            order=stable_order(D)
            for k in KGRID:
                cols=order[:k]
                met=eval_selected(X,y,tr,va,cols)
                fold_rows.append({"task":task,"fold":fi,"selector":selector,"method":"reference","k":k,
                                  **met,"selected_indices":"|".join(map(str,cols.tolist())),
                                  "selected_genes":"|".join(feat.iloc[cols].gene_symbol.astype(str).tolist())})
                sets[(selector,k,fi)]=set(cols.tolist())
                for rank,j in enumerate(cols,1):
                    sel_rows.append({"fold":fi,"selector":selector,"k":k,"rank":rank,
                                     "feature_index":int(j),"gene_symbol":str(feat.iloc[j].gene_symbol),
                                     "data_score":float(D[j])})
        print(task,"done outer fold",fi,flush=True)

    fold=pd.DataFrame(fold_rows)
    fold.to_csv(out/"reference_FOLD_RESULTS.csv",index=False)
    pd.DataFrame(tune_rows).to_csv(out/"reference_INNER_TUNING.csv",index=False)
    pd.DataFrame(sel_rows).to_csv(out/"reference_SELECTED_FEATURES.csv",index=False)
    pd.DataFrame(sparse_rows).to_csv(out/"reference_SELECTOR_AUDIT.csv",index=False)

    agg=(fold.groupby(["selector","k"],as_index=False)
         .agg(mean_auroc=("auroc","mean"),sd_auroc=("auroc","std"),
              mean_ap_positive=("ap_positive","mean"),mean_macro_ap=("macro_ap","mean"),
              mean_balanced_accuracy=("balanced_accuracy","mean"),
              mean_validation_prevalence=("validation_prevalence","mean")))
    agg.to_csv(out/"reference_AGGREGATE.csv",index=False)

    jac=[]
    for selector in ["LASSO","ELASTICNET","SIS"]:
        for k in KGRID:
            ss={fi:sets[(selector,k,fi)] for fi in range(1,OUTER_N+1)}
            rr=pairwise_jaccard(ss)
            for z in rr: z.update({"selector":selector,"k":k})
            jac.extend(rr)
    jac=pd.DataFrame(jac)
    jac.to_csv(out/"reference_SELECTION_JACCARD.csv",index=False)
    js=(jac.groupby(["selector","k"],as_index=False).agg(mean_jaccard=("jaccard","mean"),sd_jaccard=("jaccard","std")))
    js.to_csv(out/"reference_SELECTION_JACCARD_SUMMARY.csv",index=False)

    best=agg.sort_values(["mean_auroc","selector","k"],ascending=[False,True,True],kind="mergesort").iloc[0]
    best_auc=float(best.mean_auroc)
    n_high=int((agg.mean_auroc>=0.93).sum())
    if best_auc<0.58: gate="NO_SIGNAL_CONCERN"
    elif best_auc>=0.93 and n_high>=2: gate="STRONG_SATURATION_CONCERN"
    elif best_auc>=0.90: gate="SATURATION_CONCERN"
    elif best_auc>=0.60: gate="PREFERRED_USABLE_ZONE"
    else: gate="BORDERLINE_SIGNAL"
    status={"task":task,"status":"reference_COMPLETE","gate":gate,"best_selector":str(best.selector),
            "best_k":int(best.k),"best_mean_auroc":best_auc,
            "development_n":int(len(y)),"positive_n":int(y.sum()),"candidate_p":2000,
            "outer_folds":OUTER_N,"inner_folds":INNER_N,"seed":SEED,
            "C_grid":CGRID,"elasticnet_l1_ratio_grid":ALPHAS,"k_grid":KGRID,
            "uses_sealed_validation":False,"uses_external_evidence":False,"uses_llm":False,
            "runtime_sec":round(time.time()-t0,2),
            "python":platform.python_version(),"sklearn":sklearn.__version__}
    (out/"reference_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print("\n",task,json.dumps(status,indent=2),flush=True)
    print(agg.to_string(index=False),flush=True)
    return status,agg

all_status={}
for task in TASKS:
    s,a=run_task(task); all_status[task]=s
(OUTROOT/"reference_ALL_TASKS_STATUS.json").write_text(json.dumps(all_status,indent=2),encoding="utf-8")
print(json.dumps(all_status,indent=2))
