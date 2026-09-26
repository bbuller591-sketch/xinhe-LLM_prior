#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_DATA_AND_SPLITS"
RES = ROOT / "06_RESULTS"
PROT = ROOT / "02_PROTOCOLS"
EVID = ROOT / "05_EXTERNAL_EVIDENCE"

FEATURES = [
    "checking_status","duration","credit_history","purpose","credit_amount",
    "savings_status","employment","installment_commitment","personal_status",
    "other_parties","residence_since","property_magnitude","age",
    "other_payment_plans","housing","existing_credits","job",
    "num_dependents","own_telephone","foreign_worker"
]
CAT = [
    "checking_status","credit_history","purpose","savings_status","employment",
    "installment_commitment","personal_status","other_parties","residence_since",
    "property_magnitude","other_payment_plans","housing","existing_credits","job",
    "num_dependents","own_telephone","foreign_worker"
]
NUM = [x for x in FEATURES if x not in CAT]

def make_prep(cols):
    cols=list(cols)
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

def fail(msg):
    print("FAIL:",msg)
    sys.exit(2)

# Basic provenance / split checks.
X=pd.read_csv(DATA/"X.csv")
y=pd.read_csv(DATA/"y.csv")["label"].to_numpy()
dev=np.load(DATA/"modern_dev_indices_seed20260918.npy")
hold=np.load(DATA/"modern_holdout_indices_seed20260918.npy")
if X.shape!=(1000,20): fail(f"X shape {X.shape}")
if len(dev)!=800 or len(hold)!=200 or len(np.intersect1d(dev,hold))!=0: fail("modern split integrity")
if set(np.unique(y))!={0,1}: fail("labels")

# reference k freeze.
m0k=json.loads((RES/"reference_K_FREEZE.json").read_text())
expected_k={"L1":10,"GBM_PERM":10,"ELASTIC_NET":10}
if m0k["chosen_k"]!=expected_k: fail(f"reference k mismatch: {m0k['chosen_k']}")

# selective graph/evidence coverage.
selective=pd.read_csv(PROT/"selective_SELECTIVE_GRAPH_FREEZE.csv")
eligible=selective[selective.formal_selective_measurement_eligible.astype(bool)]
pairs=set(zip(eligible.feature_a,eligible.feature_b))
expected_pairs={
    ("purpose","installment_commitment"),
    ("purpose","housing"),
    ("installment_commitment","housing"),
}
if pairs!=expected_pairs: fail(f"selective eligible-pair mismatch: {pairs}")

status=pd.read_csv(EVID/"EVIDENCE_STATUS.csv")
usable=set(status.loc[status.evidence_state=="USABLE_EVIDENCE","feature"])
if usable!={"purpose","installment_commitment","housing"}: fail(f"usable evidence mismatch: {usable}")

# Actual selector-specific routing among the 3 eligible pairs.
expected_route={"L1":0,"GBM_PERM":1,"ELASTIC_NET":3}
route={
    "L1":int(eligible.requested_l1.astype(bool).sum()),
    "GBM_PERM":int(eligible.requested_gbm_perm.astype(bool).sum()),
    "ELASTIC_NET":int(eligible.requested_elastic_net.astype(bool).sum()),
}
if route!=expected_route: fail(f"selective selector routing mismatch: {route}")

# LLM measurement counts.
m12=pd.read_csv(ROOT/"04_LLM_RAW_AND_MANIFESTS/global_DEEPSEEK_MEASUREMENT.csv")
m3llm=pd.read_csv(ROOT/"04_LLM_RAW_AND_MANIFESTS/selective_DEEPSEEK_MEASUREMENT.csv")
if len(m12)!=144 or int(m12.primary_measurement.astype(bool).sum())!=120: fail("global/global_certainty call counts")
if len(m3llm)!=18 or int(m3llm.primary_measurement.astype(bool).sum())!=6: fail("selective call counts")

# Recompute final holdout metrics from frozen selected sets.
Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]
Xh=X.iloc[hold].reset_index(drop=True); yh=y[hold]
sets=pd.read_csv(RES/"FINAL_SELECTED_SETS_FREEZE.csv")
reported=pd.read_csv(RES/"FINAL_HOLDOUT_RESULTS.csv")
recomputed=[]
for _,z in sets.iterrows():
    sel=str(z.selected_features).split("|")
    pipe=Pipeline([
        ("prep",make_prep(sel)),
        ("clf",LogisticRegression(
            penalty="l2",solver="lbfgs",C=0.01,fit_intercept=True,
            class_weight=None,max_iter=5000,tol=1e-5
        )),
    ])
    pipe.fit(Xd[sel],yd)
    p=pipe.predict_proba(Xh[sel])[:,1]
    recomputed.append({
        "selector":z.selector,
        "method":z.method,
        "holdout_auroc":float(roc_auc_score(yh,p)),
        "holdout_average_precision":float(average_precision_score(yh,p)),
        "holdout_log_loss":float(log_loss(yh,p,labels=[0,1])),
    })
rr=pd.DataFrame(recomputed)
merged=reported.merge(rr,on=["selector","method"],suffixes=("_reported","_recomputed"))
for metric in ["holdout_auroc","holdout_average_precision","holdout_log_loss"]:
    diff=np.max(np.abs(merged[f"{metric}_reported"]-merged[f"{metric}_recomputed"]))
    if diff>1e-10: fail(f"{metric} max abs diff={diff}")

print("PASS")
print("X/y and modern split integrity: PASS")
print("reference chosen k:", expected_k)
print("selective eligible semantic pairs:", sorted(expected_pairs))
print("selective actual selector routing:", route)
print("Formal LLM calls: global/global_certainty=144 total, selective=18 total")
print("Final holdout metrics reproduced to <=1e-10 absolute error")
