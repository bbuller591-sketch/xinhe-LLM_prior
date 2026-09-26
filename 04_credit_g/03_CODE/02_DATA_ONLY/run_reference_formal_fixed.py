#!/usr/bin/env python
from __future__ import annotations

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
import argparse, json, math, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
DATA=WS/"legacy_archive_snapshot/positive_results_package_20260918/credit_g/data"
TASK=WS/"01_TASK_FREEZE"
OUT=WS/"02_DATA_ONLY"
HP=json.loads((OUT/"reference_GLOBAL_HYPERPARAMETER_FREEZE.json").read_text())

FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[x for x in FEATURES if x not in CAT]
K_GRID=[3,5,7,10]
MASTER_SEED=20260918
SCORE_EPS=1e-8

def make_preprocessor(cols=None):
    cols=FEATURES if cols is None else list(cols)
    num=[c for c in NUM if c in cols]
    cat=[c for c in CAT if c in cols]
    tr=[]
    if num:
        tr.append(("num",StandardScaler(),num))
    if cat:
        tr.append(("cat",Pipeline([
            ("ohe",OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False)),
            ("scale",StandardScaler()),
        ]),cat))
    return ColumnTransformer(tr,remainder="drop",sparse_threshold=0.0)

def transformed_groups(prep, cols=None):
    cols=FEATURES if cols is None else list(cols)
    names=list(prep.get_feature_names_out())
    groups=defaultdict(list)
    for i,n in enumerate(names):
        if n.startswith("num__"):
            f=n[len("num__"):]
            groups[f].append(i)
        elif n.startswith("cat__"):
            rest=n[len("cat__"):]
            hits=[f for f in CAT if rest==f or rest.startswith(f+"_")]
            if not hits:
                raise RuntimeError(f"Cannot map transformed column: {n}")
            groups[max(hits,key=len)].append(i)
        else:
            raise RuntimeError(n)
    for f in cols:
        if f not in groups:
            raise RuntimeError(f"No transformed columns for {f}")
    return groups

def group_coef_scores(pipe):
    prep=pipe.named_steps["prep"]
    coef=np.asarray(pipe.named_steps["clf"].coef_).reshape(-1)
    groups=transformed_groups(prep)
    return {f:float(np.sqrt(np.mean(np.square(coef[groups[f]])))) for f in FEATURES}

def rank_features(scores):
    order=sorted(FEATURES,key=lambda f:(-scores[f],FEATURES.index(f)))
    ranks={f:i+1 for i,f in enumerate(order)}
    return order,ranks

def fit_l1(Xtr,ytr,seed):
    clf=LogisticRegression(
        penalty="l1",solver="saga",C=float(HP["L1"]["C"]),fit_intercept=True,
        class_weight=None,max_iter=5000,tol=1e-4,random_state=seed
    )
    pipe=Pipeline([("prep",make_preprocessor()),("clf",clf)])
    pipe.fit(Xtr,ytr)
    return pipe

def fit_enet(Xtr,ytr,seed):
    clf=LogisticRegression(
        penalty="elasticnet",solver="saga",C=float(HP["ELASTIC_NET"]["C"]),
        l1_ratio=float(HP["ELASTIC_NET"]["l1_ratio"]),fit_intercept=True,
        class_weight=None,max_iter=5000,tol=1e-4,random_state=seed
    )
    pipe=Pipeline([("prep",make_preprocessor()),("clf",clf)])
    pipe.fit(Xtr,ytr)
    return pipe

def fit_gbm_perm(Xtr,ytr,split_seed,model_seed,perm_seed,perm_repeats):
    inner=StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=split_seed)
    i_fit,i_val=next(inner.split(Xtr,ytr))
    params=HP["GBM"]
    clf=GradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        n_estimators=int(params["n_estimators"]),
        subsample=float(params["subsample"]),
        random_state=model_seed,
    )
    pipe=Pipeline([("prep",make_preprocessor()),("clf",clf)])
    pipe.fit(Xtr.iloc[i_fit],ytr[i_fit])
    pi=permutation_importance(
        pipe,Xtr.iloc[i_val],ytr[i_val],scoring="roc_auc",
        n_repeats=perm_repeats,random_state=perm_seed,n_jobs=1
    )
    return {f:float(pi.importances_mean[i]) for i,f in enumerate(FEATURES)}

def eval_common(selected,Xtr,ytr,Xev,yev):
    selected=list(selected)
    clf=LogisticRegression(
        penalty="l2",solver="lbfgs",C=float(HP["COMMON_L2"]["C"]),
        fit_intercept=True,class_weight=None,max_iter=5000,tol=1e-5
    )
    pipe=Pipeline([("prep",make_preprocessor(selected)),("clf",clf)])
    pipe.fit(Xtr[selected],ytr)
    p=pipe.predict_proba(Xev[selected])[:,1]
    return float(roc_auc_score(yev,p))

def structural_association(Xdev):
    prep=make_preprocessor()
    Z=np.asarray(prep.fit_transform(Xdev),dtype=float)
    groups=transformed_groups(prep)
    C=np.corrcoef(Z,rowvar=False)
    out={}
    for i,a in enumerate(FEATURES):
        for b in FEATURES[i+1:]:
            block=np.abs(C[np.ix_(groups[a],groups[b])])
            out[(a,b)]=float(np.nanmax(block))
    return out

def choose_k(summary_df,selector):
    d=summary_df[(summary_df.selector==selector)&(summary_df.validity_rate>=0.90)&summary_df.mean_auc.notna()].copy()
    if d.empty:
        return None,"NO_K_WITH_VALIDITY_GE_0.90"
    best=d.sort_values(["mean_auc","k"],ascending=[False,True]).iloc[0]
    threshold=float(best.mean_auc-best.se_auc)
    chosen=int(d[d.mean_auc>=threshold].sort_values("k").iloc[0].k)
    return chosen,f"ONE_SE(best_k={int(best.k)},best_mean={best.mean_auc:.6f},se_best={best.se_auc:.6f},threshold={threshold:.6f})"

def feature_stability(ranks,selector,k,valid_res):
    d=ranks[(ranks.selector==selector)&ranks.resample.isin(valid_res)]
    rows=[]
    for f in FEATURES:
        q=d[d.feature==f]
        rr=q["rank"].to_numpy()
        sc=q["score"].to_numpy()
        inc=(rr<=k).astype(int)
        rows.append({
            "selector":selector,"chosen_k":k,"feature":f,"n_valid_resamples":len(q),
            "inclusion_probability":float(np.mean(inc)),
            "mean_rank":float(np.mean(rr)),"median_rank":float(np.median(rr)),
            "sd_rank":float(np.std(rr,ddof=1)) if len(rr)>1 else 0.0,
            "iqr_rank":float(np.quantile(rr,.75)-np.quantile(rr,.25)),
            "mean_score":float(np.mean(sc)),
            "sd_score":float(np.std(sc,ddof=1)) if len(sc)>1 else 0.0,
        })
    return pd.DataFrame(rows)

def confusion_table(ranks,selector,k,valid_res,assoc):
    d=ranks[(ranks.selector==selector)&ranks.resample.isin(valid_res)]
    P=d.pivot(index="resample",columns="feature",values="rank").sort_index()
    rows=[]
    for i,a in enumerate(FEATURES):
        for b in FEATURES[i+1:]:
            ra=P[a].to_numpy(); rb=P[b].to_numpy()
            ain=ra<=k; bin_=rb<=k
            pab=float(np.mean(ain & ~bin_)); pba=float(np.mean(bin_ & ~ain))
            par=float(np.mean(ra<rb)); pbr=float(np.mean(rb<ra))
            actionable=pab>=0.10 and pba>=0.10
            rankflip=par>=0.10 and pbr>=0.10
            cls="ACTIONABLE_BOUNDARY_CONFUSION" if actionable else ("RANK_ORDER_ONLY" if rankflip else "STABLE")
            ia=ain.astype(float); ib=bin_.astype(float)
            icorr=float(np.corrcoef(ia,ib)[0,1]) if ia.std()>0 and ib.std()>0 else np.nan
            pexact=float(np.mean(ain^bin_)); inca=float(np.mean(ain)); incb=float(np.mean(bin_))
            sa=float(assoc[(a,b)])
            csub=bool(sa>=0.50 and 0.10<=inca<=0.90 and 0.10<=incb<=0.90 and np.isfinite(icorr) and icorr<=-0.25 and pexact>=0.50)
            rows.append({
                "selector":selector,"chosen_k":k,"feature_a":a,"feature_b":b,
                "p_a_in_b_out":pab,"p_b_in_a_out":pba,"swap_probability":pab+pba,
                "p_a_ranks_above_b":par,"p_b_ranks_above_a":pbr,
                "pair_class":cls,"structural_max_abs_encoded_corr":sa,
                "topk_indicator_corr":icorr,"p_exactly_one_selected":pexact,
                "inclusion_a":inca,"inclusion_b":incb,
                "correlated_substitution_candidate":csub,"n_valid_resamples":len(P)
            })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-resamples",type=int,default=300)
    ap.add_argument("--tag",default="formal")
    ap.add_argument("--perm-repeats",type=int,default=5)
    args=ap.parse_args()

    X=pd.read_csv(DATA/"X.csv")
    y=pd.read_csv(DATA/"y.csv")["label"].to_numpy()
    dev=np.load(TASK/"modern_dev_indices_seed20260918.npy")
    Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]
    assoc=structural_association(Xd)

    outer=StratifiedShuffleSplit(
        n_splits=args.n_resamples,train_size=0.80,test_size=0.20,random_state=MASTER_SEED
    )
    metric_rows=[]; rank_rows=[]; seed_rows=[]
    t0=time.time()

    for r,(itr,iev) in enumerate(outer.split(Xd,yd)):
        Xtr=Xd.iloc[itr].reset_index(drop=True); ytr=yd[itr]
        Xev=Xd.iloc[iev].reset_index(drop=True); yev=yd[iev]
        base=MASTER_SEED+100000*r
        seeds={
            "outer_resample_index":r,
            "l1_model_seed":base+11,
            "enet_model_seed":base+21,
            "gbm_inner_split_seed":base+31,
            "gbm_model_seed":base+32,
            "gbm_permutation_seed":base+33,
        }
        seed_rows.append(seeds)

        l1=fit_l1(Xtr,ytr,seeds["l1_model_seed"])
        l1s=group_coef_scores(l1); l1o,l1r=rank_features(l1s)

        en=fit_enet(Xtr,ytr,seeds["enet_model_seed"])
        ens=group_coef_scores(en); eno,enr=rank_features(ens)

        gs=fit_gbm_perm(
            Xtr,ytr,seeds["gbm_inner_split_seed"],seeds["gbm_model_seed"],
            seeds["gbm_permutation_seed"],args.perm_repeats
        )
        go,gr=rank_features(gs)

        info={"L1":(l1s,l1o,l1r),"GBM_PERM":(gs,go,gr),"ELASTIC_NET":(ens,eno,enr)}
        for s,(scores,order,ranks) in info.items():
            for f in FEATURES:
                rank_rows.append({"resample":r,"selector":s,"feature":f,"rank":int(ranks[f]),"score":float(scores[f])})

        ref_auc=eval_common(FEATURES,Xtr,ytr,Xev,yev)
        metric_rows.append({
            "resample":r,"selector":"ALL20_REFERENCE","k":20,"valid_topk":True,
            "kth_score":np.nan,"outer_auc":ref_auc,"selected_features":"|".join(FEATURES)
        })
        cache={}
        for s,(scores,order,ranks) in info.items():
            for k in K_GRID:
                chosen=tuple(order[:k]); kth=float(scores[order[k-1]])
                valid=bool(kth>SCORE_EPS) if s in ("L1","ELASTIC_NET") else bool(kth>0)
                if chosen not in cache:
                    cache[chosen]=eval_common(chosen,Xtr,ytr,Xev,yev)
                metric_rows.append({
                    "resample":r,"selector":s,"k":k,"valid_topk":valid,
                    "kth_score":kth,"outer_auc":cache[chosen],"selected_features":"|".join(chosen)
                })
        if r==0 or (r+1)%25==0 or r+1==args.n_resamples:
            print(f"[{args.tag}] {r+1}/{args.n_resamples}; elapsed={time.time()-t0:.1f}s",flush=True)

    metrics=pd.DataFrame(metric_rows); ranks=pd.DataFrame(rank_rows); seeds=pd.DataFrame(seed_rows)
    prefix="" if args.tag=="formal" else args.tag.upper()+"_"
    metrics.to_csv(OUT/f"{prefix}reference_RESAMPLE_RESULTS.csv",index=False)
    ranks.to_csv(OUT/f"{prefix}reference_RESAMPLE_RANKS.csv",index=False)
    seeds.to_csv(OUT/f"{prefix}reference_SEED_LOG.csv",index=False)

    if args.tag!="formal":
        print("Nonformal smoke complete; formal k/artifacts not written.")
        return

    summary=[]
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        for k in K_GRID:
            d=metrics[(metrics.selector==s)&(metrics.k==k)]
            dv=d[d.valid_topk.astype(bool)]
            summary.append({
                "selector":s,"k":k,"n_resamples":len(d),"n_valid":len(dv),
                "validity_rate":float(len(dv)/len(d)),
                "mean_auc":float(dv.outer_auc.mean()) if len(dv) else np.nan,
                "sd_auc":float(dv.outer_auc.std(ddof=1)) if len(dv)>1 else np.nan,
                "se_auc":float(dv.outer_auc.std(ddof=1)/math.sqrt(len(dv))) if len(dv)>1 else np.nan,
                "mean_kth_score":float(dv.kth_score.mean()) if len(dv) else np.nan,
            })
    sdf=pd.DataFrame(summary)
    chosen={}; reason={}
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        chosen[s],reason[s]=choose_k(sdf,s)
    sdf["chosen_k"]=sdf.apply(lambda z:bool(chosen.get(z.selector)==z.k) if chosen.get(z.selector) is not None else False,axis=1)

    ref=metrics[metrics.selector=="ALL20_REFERENCE"]
    refrow=pd.DataFrame([{
        "selector":"ALL20_REFERENCE","k":20,"n_resamples":len(ref),"n_valid":len(ref),"validity_rate":1.0,
        "mean_auc":float(ref.outer_auc.mean()),"sd_auc":float(ref.outer_auc.std(ddof=1)),
        "se_auc":float(ref.outer_auc.std(ddof=1)/math.sqrt(len(ref))),"mean_kth_score":np.nan,"chosen_k":False
    }])
    reference=pd.concat([refrow,sdf],ignore_index=True)
    reference.to_csv(OUT/"reference_BASELINE_RESULTS.csv",index=False)

    pd.DataFrame([
        {"feature_a":a,"feature_b":b,"structural_max_abs_encoded_corr":v}
        for (a,b),v in assoc.items()
    ]).to_csv(OUT/"STRUCTURAL_ASSOCIATION_DIAGNOSTIC.csv",index=False)

    stabfiles={"L1":"reference_L1_FEATURE_STABILITY.csv","GBM_PERM":"reference_GBM_PERM_FEATURE_STABILITY.csv","ELASTIC_NET":"reference_ELASTIC_NET_FEATURE_STABILITY.csv"}
    conffiles={"L1":"ACTIONABLE_CONFUSION_L1.csv","GBM_PERM":"ACTIONABLE_CONFUSION_GBM_PERM.csv","ELASTIC_NET":"ACTIONABLE_CONFUSION_ELASTIC_NET.csv"}
    stabs={}; confs={}
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        k=chosen[s]
        if k is None:
            pd.DataFrame(columns=["selector","chosen_k","feature","n_valid_resamples","inclusion_probability","mean_rank","median_rank","sd_rank","iqr_rank","mean_score","sd_score"]).to_csv(OUT/stabfiles[s],index=False)
            pd.DataFrame(columns=["selector","chosen_k","feature_a","feature_b","pair_class"]).to_csv(OUT/conffiles[s],index=False)
            continue
        valid_res=metrics[(metrics.selector==s)&(metrics.k==k)&metrics.valid_topk.astype(bool)].resample.to_numpy()
        st=feature_stability(ranks,s,k,valid_res)
        cf=confusion_table(ranks,s,k,valid_res,assoc)
        st.to_csv(OUT/stabfiles[s],index=False); cf.to_csv(OUT/conffiles[s],index=False)
        stabs[s]=st; confs[s]=cf

    pairs=set()
    for s,cf in confs.items():
        for _,z in cf[cf.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION"].iterrows():
            pairs.add((z.feature_a,z.feature_b))
    overlap=[]
    for a,b in sorted(pairs,key=lambda x:(FEATURES.index(x[0]),FEATURES.index(x[1]))):
        row={"feature_a":a,"feature_b":b}; n=0
        for s in ["L1","GBM_PERM","ELASTIC_NET"]:
            q=confs.get(s)
            flag=False
            if q is not None:
                t=q[(q.feature_a==a)&(q.feature_b==b)]
                flag=bool(len(t) and t.iloc[0].pair_class=="ACTIONABLE_BOUNDARY_CONFUSION")
            row[f"actionable_{s.lower()}"]=flag; n+=int(flag)
        row["n_selectors_actionable"]=n
        overlap.append(row)
    pd.DataFrame(overlap,columns=["feature_a","feature_b","actionable_l1","actionable_gbm_perm","actionable_elastic_net","n_selectors_actionable"]).to_csv(OUT/"CROSS_SELECTOR_CONFUSION_OVERLAP.csv",index=False)

    freeze={"formal_resamples":args.n_resamples,"master_seed":MASTER_SEED,"chosen_k":chosen,"choice_reason":reason,"modern_final_holdout_metrics_inspected":False}
    (OUT/"reference_K_FREEZE.json").write_text(json.dumps(freeze,indent=2)+"\n")

    lines=["# CREDIT-G Data-Only Confusion Report","",
      "Status: formal reference complete on development data only.","",
      "The modern predeclared holdout was not loaded by the formal reference program and no holdout predictive metric was computed.","",
      "## Frozen hyperparameters","",json.dumps(HP,indent=2),"","## Frozen k decisions",""]
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        lines.append(f"- {s}: k={chosen[s]} ; {reason[s]}")
    lines+=["","## reference development-resample table","",reference.to_markdown(index=False),"","## Stability / confusion headline",""]
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        if s not in stabs:
            lines.append(f"- {s}: no valid primary k.")
            continue
        st=stabs[s].sort_values(["inclusion_probability","mean_rank"],ascending=[False,True]); cf=confs[s]
        nact=int((cf.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION").sum())
        nrank=int((cf.pair_class=="RANK_ORDER_ONLY").sum())
        ncorr=int(cf.correlated_substitution_candidate.sum())
        top=", ".join(f"{z.feature} ({z.inclusion_probability:.2f})" for _,z in st.head(8).iterrows())
        lines.append(f"- {s}: actionable pairs={nact}, rank-order-only pairs={nrank}, correlated-substitution candidates={ncorr}; top inclusion: {top}")
    lines+=["","## Interpretation boundary","",
      "- These are development-only data diagnostics, not final-holdout results.",
      "- Selector stability is conditional on the frozen development-tuned hyperparameters.",
      "- Cross-selector overlap is diagnostic only; it does not redefine selector-specific selective routing.",
      "- Correlated-substitution flags are heuristic supplementary diagnostics, not causal claims.",
      "- No LLM/literature signal was used to construct reference, choose the 20-feature universe, or choose the modern holdout.",
    ]
    (OUT/"DATA_CONFUSION_REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(freeze,indent=2))

if __name__=="__main__":
    main()
