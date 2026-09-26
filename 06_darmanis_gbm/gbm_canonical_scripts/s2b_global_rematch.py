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
"""GBM s2b - the full 632 x 3589 GLOBAL REMATCH on all shared genes.

Why this file exists
--------------------
The original fingerprint (s2) matched cells on the 2000 highest-variance shared
genes and then *verified the already-selected pairs* on all 12,387 shared genes.
That is a selected-pair verification, NOT a global rematch.

This script performs the strictly stronger test: it recomputes the entire
632 x 3589 correlation matrix on all 12,387 shared genes and re-derives
best / second-best / margin from scratch.

Writes only NEW files (metadata/s2b_global_rematch.csv,
audit/GLOBAL_REMATCH_AUDIT.json); no existing artifact is modified.
"""
import json
import time
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
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
assert A.shape == (632, n_shared), A.shape
assert Dm.shape == (3589, n_shared), Dm.shape
print(f"query {A.shape}  reference {Dm.shape}  shared genes {n_shared}")


def rz(M):
    M = np.log1p(M)
    M = M - M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return M / s


t1 = time.time()
C = rz(A) @ rz(Dm).T                     # entries = n_shared * Pearson r
print(f"global correlation matrix {C.shape} in {time.time()-t1:.1f}s")
assert C.shape == (632, 3589)

order = np.argsort(-C, axis=1)
best, second = order[:, 0], order[:, 1]
best_r = C[np.arange(632), best] / n_shared
second_r = C[np.arange(632), second] / n_shared
margin = best_r - second_r
assert np.allclose(C[np.arange(632), best], C.max(axis=1))

cells = json.load(open(f"{B}/metadata/darmanis_cell_names.json"))
ann = pd.read_csv(f"{B}/metadata/darmanis_cell_annotation.csv", dtype=str).set_index("cell_key")
fp = pd.read_csv(f"{B}/metadata/s2_cell_fingerprint.csv")
cm = pd.read_csv(f"{B}/canonical/cell_sample_mapping_canonical.csv")

rows = []
for i in range(632):
    bc, sc = cells[best[i]], cells[second[i]]
    rows.append({
        "deeplink_row_index": i,
        "deeplink_cell_id": cm.darmanis_cell_title.iloc[i],
        "label": int(y[i]),
        "patient_id": cm.patient_id.iloc[i],
        "tissue": cm.tissue.iloc[i],
        "best_cell_2000": fp.matched_cell_key.iloc[i],
        "best_cell_all12387": bc,
        "same_best_match": bool(fp.matched_cell_key.iloc[i] == bc),
        "allgene_best_corr": float(best_r[i]),
        "allgene_second_corr": float(second_r[i]),
        "allgene_margin": float(margin[i]),
        "best_patient": ann.loc[bc, "patient"],
        "best_tissue": ann.loc[bc, "tissue"],
        "second_best_cell": sc,
        "second_best_patient": ann.loc[sc, "patient"],
        "second_best_tissue": ann.loc[sc, "tissue"],
    })
res = pd.DataFrame(rows)
assert res.same_best_match.all(), "global rematch changed at least one match"
res.to_csv(f"{B}/metadata/s2b_global_rematch.csv", index=False)

old_idx = np.array([cells.index(k) for k in fp.matched_cell_key])
rank_of_old = (C > C[np.arange(632), old_idx][:, None]).sum(axis=1)
second_same_tissue = int((res.second_best_tissue == res.best_tissue).sum())
second_same_patient = int((res.second_best_patient == res.best_patient).sum())

out = {
    "what": ("full 632 x 3589 global rematch on all shared genes "
             "(re-derives best/second/margin from scratch)"),
    "why": ("the original s2 all-gene step verified the already-selected pairs only; "
            "this is the strictly stronger test"),
    "shared_gene_count": int(n_shared),
    "n_query_cells": 632,
    "n_reference_cells": 3589,
    "agreement": "632/632",
    "n_agreement": int(res.same_best_match.sum()),
    "changed": int((~res.same_best_match).sum()),
    "rank_of_2000_gene_answer_in_allgene_ranking": {
        "median": int(np.median(rank_of_old)),
        "max": int(rank_of_old.max()),
        "n_rank1": int((rank_of_old == 0).sum()),
    },
    "allgene_best_corr": {"min": float(best_r.min()), "median": float(np.median(best_r)),
                          "max": float(best_r.max())},
    "allgene_margin": {"min": float(margin.min()), "median": float(np.median(margin))},
    "second_best_same_tissue": second_same_tissue,
    "second_best_same_patient": second_same_patient,
    "transform": "log1p -> row(centre) -> row(scale), applied identically to both blocks",
    "query_rows": "DeepLINK rna2.csv rows in file order (rna2.csv carries no cell ids)",
    "deeplink_cell_id_note": ("rna2.csv has no cell identifiers; deeplink_cell_id holds the "
                              "RECOVERED GEO sample title, not an upstream id"),
    "gene_alignment": "metadata/s2_gene_alignment.npz (dl_col, dm_row, shared)",
    "script": "scripts/s2b_global_rematch.py",
    "seed": None,
    "deterministic": True,
    "runtime_s": round(time.time() - t0, 1),
}
json.dump(out, open(f"{B}/audit/GLOBAL_REMATCH_AUDIT.json", "w"), indent=2)
print(f"agreement {out['n_agreement']}/632  changed {out['changed']}")
print(f"all-gene best corr min {best_r.min():.4f} median {np.median(best_r):.4f} | "
      f"margin min {margin.min():.4f} median {np.median(margin):.4f}")
print(f"second-best is same tissue for {second_same_tissue}/632, same patient for "
      f"{second_same_patient}/632")
print(f"written metadata/s2b_global_rematch.csv + audit/GLOBAL_REMATCH_AUDIT.json "
      f"({time.time()-t0:.0f}s)")
