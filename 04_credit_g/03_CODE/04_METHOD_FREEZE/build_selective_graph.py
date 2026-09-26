

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
import pandas as pd

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
D=WS/"02_DATA_ONLY"
O=WS/"04_METHOD_FREEZE"
O.mkdir(parents=True,exist_ok=True)

files={
 "L1":D/"ACTIONABLE_CONFUSION_L1.csv",
 "GBM_PERM":D/"ACTIONABLE_CONFUSION_GBM_PERM.csv",
 "ELASTIC_NET":D/"ACTIONABLE_CONFUSION_ELASTIC_NET.csv",
}
tabs={s:pd.read_csv(p) for s,p in files.items()}
pairs=set()
for s,t in tabs.items():
    a=t[t["pair_class"]=="ACTIONABLE_BOUNDARY_CONFUSION"]
    pairs.update(zip(a.feature_a,a.feature_b))

features=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
pairs=sorted(pairs,key=lambda x:(features.index(x[0]),features.index(x[1])))
rows=[]
for i,(a,b) in enumerate(pairs,1):
    row={"pair_id":f"M3U_{i:03d}","feature_a":a,"feature_b":b}
    n=0
    for s,t in tabs.items():
        q=t[(t.feature_a==a)&(t.feature_b==b)]
        if len(q):
            z=q.iloc[0]
            flag=z.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION"
            pab=float(z.p_a_in_b_out); pba=float(z.p_b_in_a_out)
            u=2*min(pab,pba) if flag else 0.0
        else:
            flag=False; pab=pba=u=0.0
        key=s.lower()
        row[f"requested_{key}"]=bool(flag)
        row[f"p_a_in_b_out_{key}"]=pab
        row[f"p_b_in_a_out_{key}"]=pba
        row[f"u_data_{key}"]=u
        n+=int(flag)
    row["n_selectors_requesting"]=n
    row["evidence_gate_state"]="NO_RETRIEVED_EVIDENCE"
    row["formal_selective_measurement_eligible"]=False
    rows.append(row)

out=pd.DataFrame(rows)
out.to_csv(O/"selective_SELECTIVE_GRAPH_FREEZE.csv",index=False)
summary={
 "selector_actionable_counts":{s:int((t.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION").sum()) for s,t in tabs.items()},
 "total_selector_requests":sum(int((t.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION").sum()) for t in tabs.values()),
 "unique_union_pairs":len(out),
 "pair_measurements_saved_by_reuse_before_evidence_gating":sum(int((t.pair_class=="ACTIONABLE_BOUNDARY_CONFUSION").sum()) for t in tabs.values())-len(out),
 "features_in_union":sorted(set(out.feature_a)|set(out.feature_b),key=features.index),
 "selective_graph_derived_only_from_reference":True,
 "cross_selector_consensus_used_to_modify_primary_routes":False,
 "formal_llm_calls_started":False,
}
(O/"selective_GRAPH_AUDIT.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
