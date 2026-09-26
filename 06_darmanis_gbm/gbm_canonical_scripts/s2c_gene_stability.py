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
"""GBM s2c - fixed-seed gene-subset stability audit of the 632-cell matching.

For k in {500, 1000, 2000, 5000} shared genes, 5 fixed-seed random subsets are drawn
and the FULL 632 x 3589 matching is repeated. Per cell we record the number of
subsets that reproduce the all-12,387-gene best match.

Wording constraint (deliberate): the result is reported as
"632/632 identity stable under the audited random gene-subset perturbations".
This is NOT claimed to be a proof of identity.

Writes only NEW files.
"""
import json
import time
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
SIZES = [500, 1000, 2000, 5000]
NREP = 5
SEED = 0
t0 = time.time()

dl = pd.read_csv(f"{B}/raw/rna2.csv")
Xdl = dl.iloc[:, :-1].to_numpy(np.float64)
y = dl.iloc[:, -1].to_numpy(int)
al = np.load(f"{B}/metadata/s2_gene_alignment.npz", allow_pickle=True)
dl_col, dm_row = al["dl_col"], al["dm_row"]
n_shared = len(dl_col)
D = np.load(f"{B}/processed/darmanis_counts_genes_by_cells.npy", mmap_mode="r")
A = Xdl[:, dl_col]
Dm = np.asarray(D[dm_row, :], dtype=np.float64).T


def rz(M):
    M = np.log1p(M)
    M = M - M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return M / s


gr = pd.read_csv(f"{B}/metadata/s2b_global_rematch.csv")
cells = json.load(open(f"{B}/metadata/darmanis_cell_names.json"))
ref = np.array([cells.index(k) for k in gr.best_cell_all12387])   # all-gene reference
print(f"reference = all-{n_shared}-gene best match; seeds fixed (default_rng({SEED}))")

rng = np.random.default_rng(SEED)
stab = {}
for k in SIZES:
    hits = np.zeros(632, dtype=int)
    for _ in range(NREP):
        idx = rng.choice(n_shared, k, replace=False)
        b = (rz(A[:, idx]) @ rz(Dm[:, idx]).T).argmax(axis=1)
        hits += (b == ref)
    stab[k] = hits / NREP
    print(f"  k={k:5d}: stable {int((hits == NREP).sum())}/632  "
          f"less-than-full {int((hits < NREP).sum())}")

res = pd.DataFrame({
    "deeplink_row_index": np.arange(632),
    "best_cell_all12387": gr.best_cell_all12387.to_numpy(),
    "patient_id": gr.patient_id.to_numpy(),
    "tissue": gr.tissue.to_numpy(),
    "label": y,
    "allgene_best_corr": gr.allgene_best_corr.to_numpy(),
    "allgene_margin": gr.allgene_margin.to_numpy(),
})
for k in SIZES:
    res[f"stability_k{k}"] = stab[k]
stack = np.vstack([stab[k] for k in SIZES])
res["stability_min_over_k"] = stack.min(axis=0)
res["stability_mean_over_k"] = stack.mean(axis=0)
res["stable_under_all_audited_subsets"] = (stack.min(axis=0) == 1.0)
res.to_csv(f"{B}/metadata/s2c_gene_subset_stability.csv", index=False)

n_stable = int(res.stable_under_all_audited_subsets.sum())
out = {
    "what": "fixed-seed random gene-subset rematching stability of the 632 cell identities",
    "sizes_genes": SIZES,
    "n_replicates_per_size": NREP,
    "seed": SEED,
    "rng": "numpy.random.default_rng(0), drawn for k in the order listed",
    "reference_identity": f"best match from the full {n_shared}-gene global rematch (s2b)",
    "n_shared_genes": int(n_shared),
    "n_cells": 632,
    "n_stable_under_all_audited_subsets": n_stable,
    "n_unstable": 632 - n_stable,
    "statement": (f"{n_stable}/632 identity stable under the audited random gene-subset "
                  f"perturbations (k in {SIZES}, {NREP} fixed-seed replicates each)"),
    "not_claimed": ("this is NOT a proof of identity; it quantifies robustness of the "
                    "matching to gene-subset selection only"),
    "script": "scripts/s2c_gene_stability.py",
    "runtime_s": round(time.time() - t0, 1),
}
json.dump(out, open(f"{B}/audit/GENE_SUBSET_STABILITY_AUDIT.json", "w"), indent=2)
print(out["statement"])
print(f"written metadata/s2c_gene_subset_stability.csv + "
      f"audit/GENE_SUBSET_STABILITY_AUDIT.json ({time.time()-t0:.0f}s)")
