

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
DATA=WS/"legacy_archive_snapshot/positive_results_package_20260918/credit_g/data"
TASK=WS/"01_TASK_FREEZE"
OUT=WS/"02_DATA_ONLY"
FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[x for x in FEATURES if x not in CAT]
C_GRID=np.logspace(-3,3,13)
SEED=20260918

def prep(cols=FEATURES):
    cols=list(cols); num=[c for c in NUM if c in cols]; cat=[c for c in CAT if c in cols]
    tr=[]
    if num: tr.append(("num",StandardScaler(),num))
    if cat:
        tr.append(("cat",Pipeline([
            ("ohe",OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False)),
            ("scale",StandardScaler()),
        ]),cat))
    return ColumnTransformer(tr,remainder="drop",sparse_threshold=0.0)

X=pd.read_csv(DATA/"X.csv")
y=pd.read_csv(DATA/"y.csv")["label"].to_numpy()
idx=np.load(TASK/"modern_dev_indices_seed20260918.npy")
Xd=X.iloc[idx].reset_index(drop=True); yd=y[idx]
cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=SEED)

l1=Pipeline([("prep",prep()),("clf",LogisticRegressionCV(
    Cs=C_GRID,cv=cv,scoring="roc_auc",solver="saga",penalty="l1",max_iter=5000,
    tol=1e-4,random_state=SEED+1,n_jobs=1,refit=True))])
l1.fit(Xd,yd)
l1_C=float(np.asarray(l1.named_steps["clf"].C_).reshape(-1)[0])

en=Pipeline([("prep",prep()),("clf",LogisticRegressionCV(
    Cs=C_GRID,l1_ratios=[0.25,0.5,0.75],cv=cv,scoring="roc_auc",solver="saga",
    penalty="elasticnet",max_iter=5000,tol=1e-4,random_state=SEED+2,n_jobs=1,refit=True))])
en.fit(Xd,yd)
en_clf=en.named_steps["clf"]
en_C=float(np.asarray(en_clf.C_).reshape(-1)[0])
en_l1=float(np.asarray(en_clf.l1_ratio_).reshape(-1)[0])

base=Pipeline([("prep",prep()),("clf",GradientBoostingClassifier(random_state=SEED+3))])
pg={
  "clf__n_estimators":[100,200],
  "clf__learning_rate":[0.03,0.07],
  "clf__max_depth":[1,2],
  "clf__min_samples_leaf":[10],
  "clf__subsample":[0.8],
}
gs=GridSearchCV(base,pg,scoring="roc_auc",cv=cv,n_jobs=1,refit=True)
gs.fit(Xd,yd)
gbm={k.replace("clf__",""):v for k,v in gs.best_params_.items()}

l2=Pipeline([("prep",prep()),("clf",LogisticRegressionCV(
    Cs=C_GRID,cv=cv,scoring="roc_auc",solver="lbfgs",penalty="l2",max_iter=5000,
    tol=1e-5,n_jobs=1,refit=True))])
l2.fit(Xd,yd)
l2_C=float(np.asarray(l2.named_steps["clf"].C_).reshape(-1)[0])

out={
 "tuning_scope":"development_800_only",
 "cv":"StratifiedKFold(n_splits=10,shuffle=True,random_state=20260918)",
 "C_grid":[float(x) for x in C_GRID],
 "L1":{"C":l1_C},
 "ELASTIC_NET":{"C":en_C,"l1_ratio":en_l1},
 "GBM":gbm,
 "COMMON_L2":{"C":l2_C},
 "holdout_loaded":False,
 "note":"These development-wide hyperparameters are frozen before formal stability resampling; formal resampling measures conditional data instability at this fixed pipeline."
}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"reference_GLOBAL_HYPERPARAMETER_FREEZE.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
