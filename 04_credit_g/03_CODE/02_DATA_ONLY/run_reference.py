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

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

WS = Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
DATA = WS / "legacy_archive_snapshot/positive_results_package_20260918/credit_g/data"
TASK = WS / "01_TASK_FREEZE"
OUT = WS / "02_DATA_ONLY"

FEATURES = [
    "checking_status","duration","credit_history","purpose","credit_amount",
    "savings_status","employment","installment_commitment","personal_status",
    "other_parties","residence_since","property_magnitude","age",
    "other_payment_plans","housing","existing_credits","job",
    "num_dependents","own_telephone","foreign_worker"
]
CAT = [
    "checking_status","credit_history","purpose","savings_status","employment",
    "personal_status","other_parties","property_magnitude","other_payment_plans",
    "housing","job","own_telephone","foreign_worker"
]
NUM = [x for x in FEATURES if x not in CAT]
K_GRID = [3,5,7,10]
C_GRID = np.logspace(-3, 3, 13)
MASTER_SEED = 20260918
SCORE_EPS = 1e-8


def make_preprocessor(cols=None):
    cols = FEATURES if cols is None else list(cols)
    num = [c for c in NUM if c in cols]
    cat = [c for c in CAT if c in cols]
    transformers = []
    if num:
        transformers.append(("num", StandardScaler(), num))
    if cat:
        cat_pipe = Pipeline([
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("cat", cat_pipe, cat))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def transformed_groups(prep, cols=None):
    cols = FEATURES if cols is None else list(cols)
    names = list(prep.get_feature_names_out())
    groups = defaultdict(list)
    for i,n in enumerate(names):
        if n.startswith("num__"):
            f = n[len("num__"):]
            groups[f].append(i)
        elif n.startswith("cat__"):
            rest = n[len("cat__"):]
            # Exact semantic feature prefix, longest first avoids prefix ambiguity.
            hits = [f for f in CAT if rest == f or rest.startswith(f + "_")]
            if not hits:
                raise RuntimeError(f"Cannot map transformed column: {n}")
            f = max(hits, key=len)
            groups[f].append(i)
        else:
            raise RuntimeError(f"Unexpected transformed name: {n}")
    for f in cols:
        if f not in groups:
            raise RuntimeError(f"No transformed columns for {f}")
    return groups, names


def group_coef_scores(fitted_pipe, cols=None):
    cols = FEATURES if cols is None else list(cols)
    prep = fitted_pipe.named_steps["prep"]
    clf = fitted_pipe.named_steps["clf"]
    coef = np.asarray(clf.coef_).reshape(-1)
    groups,_ = transformed_groups(prep, cols)
    scores = {}
    for f in cols:
        b = coef[groups[f]]
        scores[f] = float(np.sqrt(np.mean(np.square(b))))
    return scores


def rank_features(scores):
    # Stable deterministic tie-break by frozen original feature order.
    order = sorted(FEATURES, key=lambda f: (-scores[f], FEATURES.index(f)))
    ranks = {f:i+1 for i,f in enumerate(order)}
    return order, ranks


def fit_l1(Xtr,ytr,seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegressionCV(
        Cs=C_GRID,
        cv=cv,
        scoring="roc_auc",
        solver="saga",
        penalty="l1",
        fit_intercept=True,
        class_weight=None,
        max_iter=5000,
        tol=1e-4,
        random_state=seed+1,
        n_jobs=1,
        refit=True,
    )
    pipe = Pipeline([("prep",make_preprocessor()),("clf",clf)])
    pipe.fit(Xtr,ytr)
    return pipe


def fit_enet(Xtr,ytr,seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegressionCV(
        Cs=C_GRID,
        l1_ratios=[0.25,0.50,0.75],
        cv=cv,
        scoring="roc_auc",
        solver="saga",
        penalty="elasticnet",
        fit_intercept=True,
        class_weight=None,
        max_iter=5000,
        tol=1e-4,
        random_state=seed+1,
        n_jobs=1,
        refit=True,
    )
    pipe = Pipeline([("prep",make_preprocessor()),("clf",clf)])
    pipe.fit(Xtr,ytr)
    return pipe


def gbm_param_grid():
    return {
        "clf__n_estimators":[100,200],
        "clf__learning_rate":[0.03,0.07],
        "clf__max_depth":[1,2],
        "clf__min_samples_leaf":[10],
        "clf__subsample":[0.8],
    }


def fit_gbm_and_perm_scores(Xtr,ytr,seed,perm_repeats=5):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    base = Pipeline([
        ("prep",make_preprocessor()),
        ("clf",GradientBoostingClassifier(random_state=seed+11)),
    ])
    gs = GridSearchCV(
        base,
        gbm_param_grid(),
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=1,
        return_train_score=False,
    )
    gs.fit(Xtr,ytr)
    best_params = dict(gs.best_params_)

    fold_imps = []
    for fold,(itr,iva) in enumerate(cv.split(Xtr,ytr)):
        est = clone(base).set_params(**best_params)
        est.set_params(clf__random_state=seed+100+fold)
        est.fit(Xtr.iloc[itr], ytr[itr])
        pseed = seed + 10000 + fold
        pi = permutation_importance(
            est,
            Xtr.iloc[iva],
            ytr[iva],
            scoring="roc_auc",
            n_repeats=perm_repeats,
            random_state=pseed,
            n_jobs=1,
        )
        fold_imps.append(np.asarray(pi.importances_mean))
    mean_imp = np.mean(np.vstack(fold_imps),axis=0)
    scores = {f:float(mean_imp[i]) for i,f in enumerate(FEATURES)}
    return scores, best_params


def eval_common(selected, Xtr,ytr,Xev,yev,seed):
    selected = list(selected)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegressionCV(
        Cs=C_GRID,
        cv=cv,
        scoring="roc_auc",
        solver="lbfgs",
        penalty="l2",
        fit_intercept=True,
        class_weight=None,
        max_iter=5000,
        tol=1e-5,
        n_jobs=1,
        refit=True,
    )
    pipe = Pipeline([
        ("prep",make_preprocessor(selected)),
        ("clf",clf),
    ])
    pipe.fit(Xtr[selected],ytr)
    p = pipe.predict_proba(Xev[selected])[:,1]
    return float(roc_auc_score(yev,p))


def structural_association(Xdev):
    prep=make_preprocessor()
    Z=np.asarray(prep.fit_transform(Xdev),dtype=float)
    groups,_=transformed_groups(prep)
    # transformed columns are already standardized; np.corrcoef is still robust.
    C=np.corrcoef(Z,rowvar=False)
    assoc={}
    for i,a in enumerate(FEATURES):
        ia=groups[a]
        for b in FEATURES[i+1:]:
            ib=groups[b]
            block=np.abs(C[np.ix_(ia,ib)])
            val=float(np.nanmax(block)) if block.size else float("nan")
            assoc[(a,b)]=val
    return assoc


def choose_k(summary_rows):
    df=pd.DataFrame(summary_rows)
    elig=df[(df["validity_rate"]>=0.90) & df["mean_auc"].notna()].copy()
    if elig.empty:
        return None, "NO_K_WITH_VALIDITY_GE_0.90"
    best=elig.sort_values(["mean_auc","k"],ascending=[False,True]).iloc[0]
    threshold=float(best["mean_auc"]-best["se_auc"])
    acceptable=elig[elig["mean_auc"]>=threshold].sort_values("k")
    chosen=int(acceptable.iloc[0]["k"])
    return chosen, f"ONE_SE(best_k={int(best['k'])},best_mean={best['mean_auc']:.6f},se_best={best['se_auc']:.6f},threshold={threshold:.6f})"


def summarize_feature_stability(rank_df, chosen_k, selector, valid_mask):
    d=rank_df[(rank_df.selector==selector) & valid_mask].copy()
    rows=[]
    for f in FEATURES:
        rr=d[d.feature==f]
        ranks=rr["rank"].to_numpy()
        scores=rr["score"].to_numpy()
        inc=(ranks<=chosen_k).astype(int)
        rows.append({
            "selector":selector,
            "chosen_k":chosen_k,
            "feature":f,
            "n_valid_resamples":len(rr),
            "inclusion_probability":float(np.mean(inc)),
            "mean_rank":float(np.mean(ranks)),
            "median_rank":float(np.median(ranks)),
            "sd_rank":float(np.std(ranks,ddof=1)) if len(ranks)>1 else 0.0,
            "iqr_rank":float(np.quantile(ranks,.75)-np.quantile(ranks,.25)),
            "mean_score":float(np.mean(scores)),
            "sd_score":float(np.std(scores,ddof=1)) if len(scores)>1 else 0.0,
        })
    return pd.DataFrame(rows)


def pair_confusion(rank_df, chosen_k, selector, valid_resamples, assoc):
    d=rank_df[(rank_df.selector==selector) & (rank_df.resample.isin(valid_resamples))].copy()
    piv_rank=d.pivot(index="resample",columns="feature",values="rank").sort_index()
    rows=[]
    for i,a in enumerate(FEATURES):
        for b in FEATURES[i+1:]:
            ra=piv_rank[a].to_numpy()
            rb=piv_rank[b].to_numpy()
            ain=ra<=chosen_k; bin_=rb<=chosen_k
            p_ab=float(np.mean(ain & ~bin_))
            p_ba=float(np.mean(bin_ & ~ain))
            swap=p_ab+p_ba
            p_a_rank=float(np.mean(ra<rb))
            p_b_rank=float(np.mean(rb<ra))
            actionable=(p_ab>=0.10 and p_ba>=0.10)
            rank_flip=(p_a_rank>=0.10 and p_b_rank>=0.10)
            if actionable:
                cls="ACTIONABLE_BOUNDARY_CONFUSION"
            elif rank_flip:
                cls="RANK_ORDER_ONLY"
            else:
                cls="STABLE"

            ia=ain.astype(float); ib=bin_.astype(float)
            if ia.std()>0 and ib.std()>0:
                ind_corr=float(np.corrcoef(ia,ib)[0,1])
            else:
                ind_corr=float("nan")
            p_exact_one=float(np.mean(ain ^ bin_))
            inc_a=float(np.mean(ain)); inc_b=float(np.mean(bin_))
            key=(a,b) if (a,b) in assoc else (b,a)
            sa=float(assoc[key])
            corr_sub=(
                sa>=0.50 and 0.10<=inc_a<=0.90 and 0.10<=inc_b<=0.90
                and np.isfinite(ind_corr) and ind_corr<=-0.25 and p_exact_one>=0.50
            )
            rows.append({
                "selector":selector,"chosen_k":chosen_k,
                "feature_a":a,"feature_b":b,
                "p_a_in_b_out":p_ab,"p_b_in_a_out":p_ba,
                "swap_probability":swap,
                "p_a_ranks_above_b":p_a_rank,
                "p_b_ranks_above_a":p_b_rank,
                "pair_class":cls,
                "structural_max_abs_encoded_corr":sa,
                "topk_indicator_corr":ind_corr,
                "p_exactly_one_selected":p_exact_one,
                "inclusion_a":inc_a,"inclusion_b":inc_b,
                "correlated_substitution_candidate":bool(corr_sub),
                "n_valid_resamples":len(piv_rank),
            })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-resamples",type=int,default=300)
    ap.add_argument("--tag",default="formal")
    ap.add_argument("--perm-repeats",type=int,default=5)
    args=ap.parse_args()

    OUT.mkdir(parents=True,exist_ok=True)
    X=pd.read_csv(DATA/"X.csv")
    y=pd.read_csv(DATA/"y.csv")["label"].to_numpy()
    dev_idx=np.load(TASK/"modern_dev_indices_seed20260918.npy")
    Xd=X.iloc[dev_idx].reset_index(drop=True)
    yd=y[dev_idx]

    assoc=structural_association(Xd)

    outer=StratifiedShuffleSplit(
        n_splits=args.n_resamples,
        train_size=0.80,
        test_size=0.20,
        random_state=MASTER_SEED,
    )

    metric_rows=[]
    rank_rows=[]
    hp_rows=[]
    start=time.time()

    for r,(itr,iev) in enumerate(outer.split(Xd,yd)):
        Xtr=Xd.iloc[itr].reset_index(drop=True)
        ytr=yd[itr]
        Xev=Xd.iloc[iev].reset_index(drop=True)
        yev=yd[iev]
        seed=MASTER_SEED + 1000*r

        # Data-only selectors.
        l1=fit_l1(Xtr,ytr,seed+10)
        l1_scores=group_coef_scores(l1)
        l1_order,l1_ranks=rank_features(l1_scores)
        hp_rows.append({
            "resample":r,"selector":"L1",
            "C":float(np.asarray(l1.named_steps["clf"].C_).reshape(-1)[0]),
            "l1_ratio":1.0,"gbm_params":""
        })

        en=fit_enet(Xtr,ytr,seed+20)
        en_scores=group_coef_scores(en)
        en_order,en_ranks=rank_features(en_scores)
        en_clf=en.named_steps["clf"]
        hp_rows.append({
            "resample":r,"selector":"ELASTIC_NET",
            "C":float(np.asarray(en_clf.C_).reshape(-1)[0]),
            "l1_ratio":float(np.asarray(en_clf.l1_ratio_).reshape(-1)[0]),
            "gbm_params":""
        })

        gbm_scores,gbm_params=fit_gbm_and_perm_scores(
            Xtr,ytr,seed+30,perm_repeats=args.perm_repeats
        )
        gbm_order,gbm_ranks=rank_features(gbm_scores)
        hp_rows.append({
            "resample":r,"selector":"GBM_PERM",
            "C":np.nan,"l1_ratio":np.nan,
            "gbm_params":json.dumps(gbm_params,sort_keys=True)
        })

        sel_info={
            "L1":(l1_scores,l1_order,l1_ranks),
            "GBM_PERM":(gbm_scores,gbm_order,gbm_ranks),
            "ELASTIC_NET":(en_scores,en_order,en_ranks),
        }
        for s,(scores,order,ranks) in sel_info.items():
            for f in FEATURES:
                rank_rows.append({
                    "resample":r,"selector":s,"feature":f,
                    "rank":int(ranks[f]),"score":float(scores[f])
                })

        # Common all-20 downstream comparator, development-only.
        all20_auc=eval_common(FEATURES,Xtr,ytr,Xev,yev,seed+500)
        metric_rows.append({
            "resample":r,"selector":"ALL20_REFERENCE","k":20,
            "valid_topk":True,"kth_score":np.nan,
            "outer_auc":all20_auc,
            "selected_features":"|".join(FEATURES),
        })

        # Cache identical selected sets within the resample.
        cache={}
        for s,(scores,order,ranks) in sel_info.items():
            for k in K_GRID:
                chosen=tuple(order[:k])
                kth=float(scores[order[k-1]])
                if s in ("L1","ELASTIC_NET"):
                    valid=bool(kth>SCORE_EPS)
                else:
                    valid=bool(kth>0.0)
                if chosen not in cache:
                    cache[chosen]=eval_common(
                        list(chosen),Xtr,ytr,Xev,yev,seed+600+17*k+len(cache)
                    )
                metric_rows.append({
                    "resample":r,"selector":s,"k":k,
                    "valid_topk":valid,"kth_score":kth,
                    "outer_auc":cache[chosen],
                    "selected_features":"|".join(chosen),
                })

        if (r+1)==1 or (r+1)%10==0 or (r+1)==args.n_resamples:
            elapsed=time.time()-start
            print(f"[{args.tag}] {r+1}/{args.n_resamples} resamples; elapsed={elapsed:.1f}s",flush=True)

    metrics=pd.DataFrame(metric_rows)
    ranks=pd.DataFrame(rank_rows)
    hps=pd.DataFrame(hp_rows)

    raw_prefix="" if args.tag=="formal" else f"{args.tag.upper()}_"
    metrics.to_csv(OUT/f"{raw_prefix}reference_RESAMPLE_RESULTS.csv",index=False)
    ranks.to_csv(OUT/f"{raw_prefix}reference_RESAMPLE_RANKS.csv",index=False)
    hps.to_csv(OUT/f"{raw_prefix}reference_HYPERPARAMETERS.csv",index=False)

    if args.tag!="formal":
        print("Smoke/nonformal run complete; no formal k freeze/artifacts written.")
        return

    summary=[]
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        for k in K_GRID:
            d=metrics[(metrics.selector==s)&(metrics.k==k)]
            valid=d.valid_topk.astype(bool)
            dv=d[valid]
            summary.append({
                "selector":s,"k":k,
                "n_resamples":len(d),
                "n_valid":int(valid.sum()),
                "validity_rate":float(valid.mean()),
                "mean_auc":float(dv.outer_auc.mean()) if len(dv) else np.nan,
                "sd_auc":float(dv.outer_auc.std(ddof=1)) if len(dv)>1 else np.nan,
                "se_auc":float(dv.outer_auc.std(ddof=1)/math.sqrt(len(dv))) if len(dv)>1 else np.nan,
                "mean_kth_score":float(dv.kth_score.mean()) if len(dv) else np.nan,
            })
    summary_df=pd.DataFrame(summary)
    chosen={}
    reasons={}
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        k,why=choose_k(summary_df[summary_df.selector==s].to_dict("records"))
        chosen[s]=k; reasons[s]=why
    summary_df["chosen_k"]=summary_df.apply(
        lambda z: bool(chosen.get(z.selector)==z.k) if chosen.get(z.selector) is not None else False,
        axis=1
    )

    ref=metrics[metrics.selector=="ALL20_REFERENCE"]
    baseline_rows=[{
        "selector":"ALL20_REFERENCE","k":20,
        "n_resamples":len(ref),"n_valid":len(ref),"validity_rate":1.0,
        "mean_auc":float(ref.outer_auc.mean()),
        "sd_auc":float(ref.outer_auc.std(ddof=1)),
        "se_auc":float(ref.outer_auc.std(ddof=1)/math.sqrt(len(ref))),
        "mean_kth_score":np.nan,"chosen_k":False
    }]
    reference_table=pd.concat([pd.DataFrame(baseline_rows),summary_df],ignore_index=True)
    reference_table.to_csv(OUT/"reference_BASELINE_RESULTS.csv",index=False)

    assoc_rows=[
        {"feature_a":a,"feature_b":b,"structural_max_abs_encoded_corr":v}
        for (a,b),v in assoc.items()
    ]
    pd.DataFrame(assoc_rows).to_csv(OUT/"STRUCTURAL_ASSOCIATION_DIAGNOSTIC.csv",index=False)

    stability_files={
        "L1":"reference_L1_FEATURE_STABILITY.csv",
        "GBM_PERM":"reference_GBM_PERM_FEATURE_STABILITY.csv",
        "ELASTIC_NET":"reference_ELASTIC_NET_FEATURE_STABILITY.csv",
    }
    confusion_files={
        "L1":"ACTIONABLE_CONFUSION_L1.csv",
        "GBM_PERM":"ACTIONABLE_CONFUSION_GBM_PERM.csv",
        "ELASTIC_NET":"ACTIONABLE_CONFUSION_ELASTIC_NET.csv",
    }
    conf_tables={}
    stab_tables={}
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        k=chosen[s]
        if k is None:
            pd.DataFrame(columns=[
                "selector","chosen_k","feature","n_valid_resamples","inclusion_probability",
                "mean_rank","median_rank","sd_rank","iqr_rank","mean_score","sd_score"
            ]).to_csv(OUT/stability_files[s],index=False)
            pd.DataFrame().to_csv(OUT/confusion_files[s],index=False)
            continue
        valid_res=metrics[
            (metrics.selector==s)&(metrics.k==k)&(metrics.valid_topk.astype(bool))
        ].resample.to_numpy()
        valid_mask=(ranks.resample.isin(valid_res))
        stab=summarize_feature_stability(ranks,k,s,valid_mask)
        conf=pair_confusion(ranks,k,s,valid_res,assoc)
        stab.to_csv(OUT/stability_files[s],index=False)
        conf.to_csv(OUT/confusion_files[s],index=False)
        stab_tables[s]=stab; conf_tables[s]=conf

    # Cross-selector actionable overlap.
    pair_keys=set()
    for s,c in conf_tables.items():
        for _,row in c[c.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION"].iterrows():
            pair_keys.add((row.feature_a,row.feature_b))
    overlap=[]
    for a,b in sorted(pair_keys,key=lambda z:(FEATURES.index(z[0]),FEATURES.index(z[1]))):
        row={"feature_a":a,"feature_b":b}
        count=0
        for s in ["L1","GBM_PERM","ELASTIC_NET"]:
            c=conf_tables.get(s)
            flag=False
            if c is not None:
                q=c[(c.feature_a==a)&(c.feature_b==b)]
                flag=bool(len(q) and q.iloc[0].pair_class=="ACTIONABLE_BOUNDARY_CONFUSION")
            row[f"actionable_{s.lower()}"]=flag
            count+=int(flag)
        row["n_selectors_actionable"]=count
        overlap.append(row)
    pd.DataFrame(overlap,columns=[
        "feature_a","feature_b","actionable_l1","actionable_gbm_perm",
        "actionable_elastic_net","n_selectors_actionable"
    ]).to_csv(OUT/"CROSS_SELECTOR_CONFUSION_OVERLAP.csv",index=False)

    freeze={
        "formal_resamples":args.n_resamples,
        "master_seed":MASTER_SEED,
        "chosen_k":chosen,
        "choice_reason":reasons,
        "modern_final_holdout_metrics_inspected":False,
    }
    (OUT/"reference_K_FREEZE.json").write_text(json.dumps(freeze,indent=2)+"\n")

    # Human-readable report.
    lines=[
        "# CREDIT-G Data-Only Confusion Report",
        "",
        "Status: formal reference complete on development data only.",
        "",
        "The modern predeclared holdout was not loaded by the reference program and no holdout predictive metric was computed.",
        "",
        "## Frozen k decisions",
        "",
    ]
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        lines.append(f"- {s}: k={chosen[s]} ; {reasons[s]}")
    lines += ["","## reference development-resample table","",reference_table.to_markdown(index=False),"","## Stability / confusion headline",""]
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        if s not in stab_tables:
            lines.append(f"- {s}: no valid primary k.")
            continue
        st=stab_tables[s].sort_values(["inclusion_probability","mean_rank"],ascending=[False,True])
        cf=conf_tables[s]
        nact=int((cf.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION").sum())
        nrank=int((cf.pair_class=="RANK_ORDER_ONLY").sum())
        ncorr=int(cf.correlated_substitution_candidate.sum())
        top=", ".join(f"{r.feature} ({r.inclusion_probability:.2f})" for _,r in st.head(8).iterrows())
        lines.append(f"- {s}: actionable pairs={nact}, rank-order-only pairs={nrank}, correlated-substitution candidates={ncorr}; top inclusion: {top}")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- These are data-only development diagnostics, not final-test results.",
        "- Cross-selector overlap is diagnostic only; it does not redefine any selector-specific selective route.",
        "- Correlated-substitution flags are heuristic supplementary diagnostics, not causal claims.",
        "- No LLM or literature signal was used to construct reference, choose the 20-feature universe, or choose the modern holdout.",
    ]
    (OUT/"DATA_CONFUSION_REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(freeze,indent=2))


if __name__=="__main__":
    main()
