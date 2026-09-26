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
"""GBM s2 - parse the Darmanis matrix and fingerprint DeepLINK's 632 cells against
the 3,589 Darmanis cells to recover cell identity, patient and tissue.
"""
import gzip, json, os, time
from collections import Counter
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
REC = str(REPRO_ROOT / 'tcga_gene_mapping_recovery_20260916')
DL_RAW = str(REPRO_ROOT / 'high_dim_dataset_recovery_20260916/gbm_scrna/raw/rna2.csv')
DMAT = f"{B}/processed/darmanis_counts_genes_by_cells.npy"
DMETA = f"{B}/processed/darmanis_matrix_meta.json"
CELLS = f"{B}/metadata/darmanis_cell_names.json"
GENES = f"{B}/metadata/darmanis_gene_symbols.json"
t0 = time.time()

if not os.path.exists(DMAT):
    print("parsing the space-delimited Darmanis matrix ...", flush=True)
    df = pd.read_csv(f"{B}/raw/GSE84465_GBM_All_data.csv.gz", sep=" ", index_col=0,
                     quotechar='"', low_memory=False)
    cells = [str(c).strip('"') for c in df.columns]
    genes = [str(i).strip('"') for i in df.index]
    json.dump(cells, open(CELLS, "w"))
    json.dump(genes, open(GENES, "w"))
    np.save(DMAT, df.to_numpy(dtype=np.float32))
    json.dump({"cells": len(cells), "genes": len(genes)}, open(DMETA, "w"))
    del df
    print(f"  parsed in {time.time()-t0:.0f}s", flush=True)

ann = pd.read_csv(f"{B}/metadata/darmanis_cell_annotation.csv", dtype=str)
D = np.load(DMAT, mmap_mode="r")                 # (genes, cells)
cells = json.load(open(CELLS))
genes = json.load(open(GENES))
print(f"Darmanis {D.shape} (genes x cells)   annotation {ann.shape}   ({time.time()-t0:.0f}s)")
assert len(cells) == len(ann) == D.shape[1], (len(cells), len(ann), D.shape)
keyed = ann.set_index("cell_key").loc[cells]
print(f"  tissue of the matrix columns: {dict(Counter(keyed.tissue))}")

# ---------------------------------------------------- DeepLINK matrix + y
dl = pd.read_csv(DL_RAW)
y = dl.iloc[:, -1].to_numpy().astype(int)
Xdl = dl.iloc[:, :-1].to_numpy(dtype=np.float64)
dl_ids = [str(c).strip('"') for c in dl.columns[:-1]]
print(f"\nDeepLINK X {Xdl.shape}, y {dict(Counter(y))}  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------- symbol -> Ensembl
ens_map = json.load(open(f"{REC}/metadata/gdc_symbol_to_ensembl.json"))
sym_of_stable = {}
for s, e in ens_map.items():
    if isinstance(e, str) and e:
        for part in e.split(";"):
            sym_of_stable.setdefault(part, []).append(s)
stable_of_dl = [g.split(".")[0] for g in dl_ids]
dl_pos_of_stable = {}
for i, st in enumerate(stable_of_dl):
    dl_pos_of_stable.setdefault(st, []).append(i)
# symbol of each Darmanis gene (upper-cased match)
dm_pos_of_stable = {}
unmapped = 0
for j, s in enumerate(genes):
    hits = sym_of_stable.get("", [])
    # direct Ensembl-stable search via known symbols
    cand = ens_map.get(s)
    if not cand:
        continue
    for st in str(cand).split(";"):
        if st in dl_pos_of_stable:
            dm_pos_of_stable.setdefault(st, []).append(j)
print(f"  DeepLINK genes with a known symbol: "
      f"{sum(1 for st in stable_of_dl if st in sym_of_stable)} / {len(stable_of_dl)}")
shared = sorted(set(dl_pos_of_stable) & set(dm_pos_of_stable))
print(f"  matched genes (Ensembl stable, symbol-mediated): {len(shared)}")
json.dump({"n_shared": len(shared)}, open(f"{B}/metadata/s2_gene_overlap.json", "w"), indent=2)

dl_col = np.array([dl_pos_of_stable[st][0] for st in shared])
dm_row = np.array([dm_pos_of_stable[st][0] for st in shared])
np.savez(f"{B}/metadata/s2_gene_alignment.npz", dl_col=dl_col, dm_row=dm_row,
         shared=np.array(shared))
json.dump({"n_shared": len(shared), "n_dl_genes": len(dl_ids), "n_dm_genes": len(genes)},
          open(f"{B}/metadata/s2_gene_overlap.json", "w"), indent=2)
A = Xdl[:, dl_col]                                     # (632, g)
Dm = np.asarray(D[dm_row, :], dtype=np.float64).T      # (3589, g)
print(f"  aligned blocks: deeplink {A.shape}, darmanis {Dm.shape}")

# ---------------------------------------------------- fingerprint
def robust_z(M):
    M = np.asarray(M, dtype=np.float64)
    M = np.log1p(M)
    M = M - M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return M / s

var = Dm.var(axis=0)
top = np.argsort(-var)[:2000]
Az, Dz = robust_z(A[:, top]), robust_z(Dm[:, top])
C = Az @ Dz.T                                          # (632, 3589) correlations
print(f"  correlation matrix {C.shape} computed ({time.time()-t0:.0f}s)")
best = C.argmax(axis=1)
bestr = C.max(axis=1)
C2 = C.copy()
C2[np.arange(C.shape[0]), best] = -np.inf
second = C2.max(axis=1)
print(f"  best corr: min {bestr.min():.4f} median {np.median(bestr):.4f}")
print(f"  margin   : min {(bestr-second).min():.4f} median {np.median(bestr-second):.4f}")
print(f"  distinct best matches: {len(set(best.tolist()))} / 632")

# verification over ALL shared genes
zg = robust_z(A)
ind = np.unique(best)
sub = robust_z(Dm[ind, :])
Cfull = zg @ sub.T
rows = np.searchsorted(ind, best)
full_r = Cfull[np.arange(632), rows]
print(f"  verification on all {len(shared)} shared genes: "
      f"min {full_r.min():.6f} median {np.median(full_r):.6f}")

res = pd.DataFrame({
    "dl_row": np.arange(632),
    "y": y,
    "matched_cell_key": [cells[b] for b in best],
    "matched_cell_title": keyed.iloc[best]["title"].to_numpy(),
    "matched_patient": keyed.iloc[best]["patient"].to_numpy(),
    "matched_tissue": keyed.iloc[best]["tissue"].to_numpy(),
    "matched_cell_type": keyed.iloc[best]["cell_type"].to_numpy(),
    "corr_2000_genes": bestr,
    "corr_all_shared_genes": full_r,
    "corr_second_best": second,
})
res.to_csv(f"{B}/metadata/s2_cell_fingerprint.csv", index=False)
print("\ny x tissue (matched):")
print(pd.crosstab(res.y, res.matched_tissue))
print("\ny x cell_type (matched):")
print(pd.crosstab(res.y, res.matched_cell_type))
print("\npatients:", dict(Counter(res.matched_patient)))
print("duplicate matched cells:", int(res.matched_cell_key.duplicated().sum()))
print(f"\ndone {time.time()-t0:.0f}s")
