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
"""GBM s5b - STRICT X-only candidate-universe builder + equivalence proof.

The original s4 audit script loaded `real_y` (for reporting class balance) although
y never entered the ranking. This file provides a builder whose signature accepts
only (X, feature_ids) - there is no y argument and no y is read anywhere.

It re-derives the universe and asserts that it is IDENTICAL to the canonical one
(2000/2000 identities, ranks, variance values, column order). Nothing is rewritten.
"""
import json
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
K = 2000


def build_candidate_universe(X, feature_ids, k=K, transform="log1p"):
    """X-only candidate universe. No label argument exists by construction.

    Returns (kept_original_indices_in_rank_order, rank_of_every_feature, statistic).
    """
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2 and X.shape[1] == len(feature_ids)
    assert np.isfinite(X).all()
    Xr = np.log1p(X) if transform == "log1p" else X
    v = Xr.var(axis=0, ddof=0)
    idx = np.arange(len(feature_ids))
    order = np.lexsort((idx, -v))                  # variance desc, tie-break index asc
    keep = order[:k]
    rank = np.empty(len(feature_ids), dtype=int)
    rank[order] = np.arange(1, len(feature_ids) + 1)
    return keep, rank, v


raw = pd.read_csv(f"{B}/raw/rna2.csv")
gene_ids = [c for c in raw.columns if c.startswith("ENSG")]
X = raw[gene_ids].to_numpy(np.float64)             # y is never extracted here
print(f"X {X.shape}  feature_ids {len(gene_ids)}  (no label column is ever read)")
keep, rank, var = build_candidate_universe(X, gene_ids)

cf = pd.read_csv(f"{B}/canonical/candidate_features_canonical.csv", keep_default_na=False)
assert len(cf) == K
canon_idx = cf.original_uci_feature_index.astype(int).to_numpy()
canon_ids = cf.original_gene_id.to_numpy()
canon_rank = cf.x_only_rank.astype(int).to_numpy()
canon_var = cf.variance_log1p.to_numpy()

same_set = bool(np.array_equal(keep, canon_idx))
same_ids = bool(np.array_equal(np.array(gene_ids)[keep], canon_ids))
same_rank = bool(np.array_equal(rank[keep], canon_rank))
max_var_diff = float(np.abs(var[keep] - canon_var).max())
Xref = np.load(f"{B}/canonical/X_candidate_aligned.npy")
same_X = bool(np.array_equal(X[:, keep].astype(np.float32), Xref))

print(f"identities 2000/2000 identical : {same_ids}")
print(f"column order identical         : {same_set}")
print(f"variance ranks identical       : {same_rank}")
print(f"variance values max abs diff   : {max_var_diff:.3e}")
print(f"canonical X bitwise identical  : {same_X}")
assert same_ids and same_set and same_rank and max_var_diff < 1e-12 and same_X

out = {
    "what": "strict X-only re-derivation of the canonical candidate universe",
    "builder_signature": "build_candidate_universe(X, feature_ids, k=2000, transform='log1p')",
    "x_only_statement": ("y was not used in candidate ranking or membership, although the "
                         "original audit script (s4) loaded y for reporting"),
    "inputs": ["X (632 x 23257)", "feature_ids (23257 versioned Ensembl ids)"],
    "rule": ("log1p -> per-gene variance ddof=0 -> descending, tie-break ascending original "
             "column index -> top 2000"),
    "equivalence_to_canonical": {
        "gene_identities_identical": "2000/2000",
        "column_order_identical": same_set,
        "variance_ranks_identical": same_rank,
        "variance_values_max_abs_diff": max_var_diff,
        "canonical_X_bitwise_identical": same_X,
    },
    "candidate_universe_changed": False,
    "script": "scripts/s5b_x_only_builder.py",
}
json.dump(out, open(f"{B}/audit/X_ONLY_BUILDER_AUDIT.json", "w"), indent=2)
print("written audit/X_ONLY_BUILDER_AUDIT.json (candidate universe unchanged)")
