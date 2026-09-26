

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
import numpy as np
import pandas as pd

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
OUT=WS/"02_DATA_ONLY"
FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]

metrics=pd.read_csv(OUT/"reference_RESAMPLE_RESULTS.csv")
ranks=pd.read_csv(OUT/"reference_RESAMPLE_RANKS.csv")
reference=pd.read_csv(OUT/"reference_BASELINE_RESULTS.csv")
assoc_df=pd.read_csv(OUT/"STRUCTURAL_ASSOCIATION_DIAGNOSTIC.csv")
assoc={(r.feature_a,r.feature_b):float(r.structural_max_abs_encoded_corr) for _,r in assoc_df.iterrows()}

chosen={}
for s in ["L1","GBM_PERM","ELASTIC_NET"]:
    q=reference[(reference.selector==s)&(reference.chosen_k.astype(bool))]
    if len(q)!=1:
        raise RuntimeError(f"Expected one chosen k for {s}, got {len(q)}")
    chosen[s]=int(q.iloc[0].k)

def feature_stability(selector,k,valid_res):
    d=ranks[(ranks.selector==selector)&(ranks["resample"].isin(valid_res))]
    rows=[]
    for f in FEATURES:
        q=d[d.feature==f]
        rr=q["rank"].to_numpy(dtype=float)
        sc=q["score"].to_numpy(dtype=float)
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

def confusion(selector,k,valid_res):
    d=ranks[(ranks.selector==selector)&(ranks["resample"].isin(valid_res))]
    P=d.pivot(index="resample",columns="feature",values="rank").sort_index()
    rows=[]
    for i,a in enumerate(FEATURES):
        for b in FEATURES[i+1:]:
            ra=P[a].to_numpy(dtype=float); rb=P[b].to_numpy(dtype=float)
            ain=ra<=k; bin_=rb<=k
            pab=float(np.mean(ain & ~bin_)); pba=float(np.mean(bin_ & ~ain))
            par=float(np.mean(ra<rb)); pbr=float(np.mean(rb<ra))
            actionable=(pab>=0.10 and pba>=0.10)
            rankflip=(par>=0.10 and pbr>=0.10)
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
                "correlated_substitution_candidate":csub,
                "n_valid_resamples":len(P),
            })
    return pd.DataFrame(rows)

stabfiles={"L1":"reference_L1_FEATURE_STABILITY.csv","GBM_PERM":"reference_GBM_PERM_FEATURE_STABILITY.csv","ELASTIC_NET":"reference_ELASTIC_NET_FEATURE_STABILITY.csv"}
conffiles={"L1":"ACTIONABLE_CONFUSION_L1.csv","GBM_PERM":"ACTIONABLE_CONFUSION_GBM_PERM.csv","ELASTIC_NET":"ACTIONABLE_CONFUSION_ELASTIC_NET.csv"}
stabs={}; confs={}
for s in chosen:
    k=chosen[s]
    valid_res=metrics[(metrics.selector==s)&(metrics.k==k)&(metrics.valid_topk.astype(bool))]["resample"].to_numpy(dtype=int)
    st=feature_stability(s,k,valid_res)
    cf=confusion(s,k,valid_res)
    st.to_csv(OUT/stabfiles[s],index=False)
    cf.to_csv(OUT/conffiles[s],index=False)
    stabs[s]=st; confs[s]=cf

pairs=set()
for s,cf in confs.items():
    for _,z in cf[cf.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION"].iterrows():
        pairs.add((z.feature_a,z.feature_b))
rows=[]
for a,b in sorted(pairs,key=lambda x:(FEATURES.index(x[0]),FEATURES.index(x[1]))):
    row={"feature_a":a,"feature_b":b}; n=0
    for s in ["L1","GBM_PERM","ELASTIC_NET"]:
        q=confs[s]
        t=q[(q.feature_a==a)&(q.feature_b==b)]
        flag=bool(len(t) and t.iloc[0].pair_class=="ACTIONABLE_BOUNDARY_CONFUSION")
        row[f"actionable_{s.lower()}"]=flag; n+=int(flag)
    row["n_selectors_actionable"]=n
    rows.append(row)
pd.DataFrame(rows,columns=["feature_a","feature_b","actionable_l1","actionable_gbm_perm","actionable_elastic_net","n_selectors_actionable"]).to_csv(OUT/"CROSS_SELECTOR_CONFUSION_OVERLAP.csv",index=False)

freeze={
    "formal_resamples":300,
    "master_seed":20260918,
    "chosen_k":chosen,
    "choice_reason":{
        s:"chosen by frozen one-standard-error rule; see reference_BASELINE_RESULTS.csv"
        for s in chosen
    },
    "modern_final_holdout_metrics_inspected":False,
}
(OUT/"reference_K_FREEZE.json").write_text(json.dumps(freeze,indent=2)+"\n")

lines=[
"# CREDIT-G Data-Only Confusion Report","",
"Status: formal reference complete on development data only.","",
"The modern predeclared holdout was not loaded by the formal reference program and no holdout predictive metric was computed.","",
"## Frozen k","",
]
for s,k in chosen.items():
    lines.append(f"- {s}: k={k}")
lines+=["","## Development resample AUROC summary",""]
for _,r in reference.iterrows():
    mark=" [CHOSEN]" if bool(r.chosen_k) else ""
    lines.append(f"- {r.selector}, k={int(r.k)}: mean AUROC={r.mean_auc:.6f}, SD={r.sd_auc:.6f}, validity={r.validity_rate:.3f}{mark}")
lines+=["","## Stability / confusion headline",""]
for s in ["L1","GBM_PERM","ELASTIC_NET"]:
    st=stabs[s].sort_values(["inclusion_probability","mean_rank"],ascending=[False,True])
    cf=confs[s]
    nact=int((cf.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION").sum())
    nrank=int((cf.pair_class=="RANK_ORDER_ONLY").sum())
    ncorr=int(cf.correlated_substitution_candidate.sum())
    top=", ".join(f"{z.feature} ({z.inclusion_probability:.2f})" for _,z in st.head(10).iterrows())
    lines.append(f"- {s}: actionable pairs={nact}; rank-order-only pairs={nrank}; correlated-substitution candidates={ncorr}; top inclusion: {top}")
lines+=["","## Interpretation boundary","",
"- Development-only data diagnostics; not final-holdout results.",
"- Stability is conditional on frozen development-tuned hyperparameters.",
"- Cross-selector overlap is diagnostic only and does not redefine selector-specific selective routing.",
"- Correlated-substitution flags are heuristic supplementary diagnostics, not causal claims.",
"- No LLM or literature signal was used in reference.",
]
(OUT/"DATA_CONFUSION_REPORT.md").write_text("\n".join(lines)+"\n")
print(json.dumps(freeze,indent=2))
for s in ["L1","GBM_PERM","ELASTIC_NET"]:
    cf=confs[s]
    print(s,"actionable",int((cf.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION").sum()),"rank_only",int((cf.pair_class=="RANK_ORDER_ONLY").sum()),"corr_sub",int(cf.correlated_substitution_candidate.sum()))
