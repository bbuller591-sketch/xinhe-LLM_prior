

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
from sklearn.model_selection import StratifiedShuffleSplit

WS = Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
SRC = WS / "legacy_archive_snapshot/positive_results_package_20260918/credit_g/data"
OUT = WS / "01_TASK_FREEZE"
OUT.mkdir(parents=True, exist_ok=True)

X = pd.read_csv(SRC / "X.csv")
y_df = pd.read_csv(SRC / "y.csv")
y = y_df["label"].to_numpy()

expected_features = [
    "checking_status","duration","credit_history","purpose","credit_amount",
    "savings_status","employment","installment_commitment","personal_status",
    "other_parties","residence_since","property_magnitude","age",
    "other_payment_plans","housing","existing_credits","job",
    "num_dependents","own_telephone","foreign_worker"
]
assert list(X.columns) == expected_features
assert X.shape == (1000, 20)
assert set(np.unique(y)) == {0, 1}
assert not X.isna().any().any()

# Variable-type freeze uses the corrected South German Credit documentation
# (UCI id 522 / Groemping 2019) to avoid treating discretized group indexes as
# truly quantitative measurements. The underlying OpenML-31 archive is NOT
# replaced; this only fixes preprocessing semantics for the same 20 variables.
categorical = {
    "checking_status","credit_history","purpose","savings_status","employment",
    "installment_commitment","personal_status","other_parties","residence_since",
    "property_magnitude","other_payment_plans","housing","existing_credits","job",
    "num_dependents","own_telephone","foreign_worker"
}
numeric = set(expected_features) - categorical
assert numeric == {"duration","credit_amount","age"}

rows = []
for j, f in enumerate(expected_features, start=1):
    vals = X[f]
    rows.append({
        "feature_index_1based": j,
        "feature_name": f,
        "semantic_unit": f,
        "primary_type": "categorical" if f in categorical else "numeric",
        "n_unique_full_dataset": int(vals.nunique(dropna=False)),
        "missing_count_full_dataset": int(vals.isna().sum()),
        "preprocessing": (
            "one_hot_drop_first_then_standardize_dummy_columns"
            if f in categorical else
            "standardize_mean0_sd1"
        ),
        "selector_unit": "original_semantic_feature",
        "llm_evidence_may_change_universe": False,
    })
pd.DataFrame(rows).to_csv(OUT / "FEATURE_UNIVERSE_FREEZE.csv", index=False)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=20260918)
dev_idx, holdout_idx = next(sss.split(X, y))
dev_idx = np.asarray(dev_idx, dtype=np.int64)
holdout_idx = np.asarray(holdout_idx, dtype=np.int64)
np.save(OUT / "modern_dev_indices_seed20260918.npy", dev_idx)
np.save(OUT / "modern_holdout_indices_seed20260918.npy", holdout_idx)

assert len(dev_idx) == 800 and len(holdout_idx) == 200
assert len(np.intersect1d(dev_idx, holdout_idx)) == 0
assert len(np.union1d(dev_idx, holdout_idx)) == 1000

audit = {
    "split_seed": 20260918,
    "split_method": "StratifiedShuffleSplit(n_splits=1,test_size=0.20,random_state=20260918)",
    "development_n": int(len(dev_idx)),
    "holdout_n": int(len(holdout_idx)),
    "development_class_counts": {
        str(k): int(v) for k, v in zip(*np.unique(y[dev_idx], return_counts=True))
    },
    "holdout_class_counts": {
        str(k): int(v) for k, v in zip(*np.unique(y[holdout_idx], return_counts=True))
    },
    "train_holdout_overlap": int(len(np.intersect1d(dev_idx, holdout_idx))),
    "coverage": int(len(np.union1d(dev_idx, holdout_idx))),
    "holdout_outcome_metric_inspected": False,
}
(OUT / "MODERN_SPLIT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
