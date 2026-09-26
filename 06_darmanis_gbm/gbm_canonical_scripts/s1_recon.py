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
"""GBM s1 - reconnaissance: Darmanis annotation, value scales, cell counts."""
import gzip, json, os, re, sys, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
DL_RAW = str(REPRO_ROOT / 'high_dim_dataset_recovery_20260916/gbm_scrna/raw/rna2.csv')
ANN = f"{B}/metadata/darmanis_cell_annotation.csv"
CACHE = f"{B}/processed/darmanis_genes_by_cells.npy"
GR = f"{B}/processed/darmanis_gene_ids.json"
CELLS = f"{B}/processed/darmanis_cell_names.json"
t0 = time.time()

# ---------------------------------------------------------------- annotation
if not os.path.exists(ANN):
    txt = open(f"{B}/metadata/GSE84465_gsm_brief.txt", encoding="utf-8", errors="replace").read()
    recs = txt.split("^SAMPLE = ")[1:]
    rows = []
    for r in recs:
        f = {}
        for line in r.split("\n"):
            if line.startswith("!Sample_title"):
                f["title"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                k, _, v = line.split("=", 1)[1].strip().partition(": ")
                f[k.strip()] = v.strip()
            elif line.startswith("!Sample_source_name_ch1"):
                f["source"] = line.split("=", 1)[1].strip()
        rows.append(f)
    ann = pd.DataFrame(rows)
    ann["cell_key"] = ann["plate id"].astype(str) + "." + ann["well"].astype(str)
    ann = ann.rename(columns={"patient id": "patient", "tissue": "tissue",
                              "cell type": "cell_type", "tsne cluster": "tsne_cluster",
                              "plate id": "plate", "selection": "selection"})
    ann.to_csv(ANN, index=False)
print("annotation:", ann.shape if 'ann' in dir() else "loaded")
ann = pd.read_csv(ANN, dtype=str)
print(f"  columns: {list(ann.columns)}")
print(f"  patients: {dict(Counter(ann.patient))}")
print(f"  tissue  : {dict(Counter(ann.tissue))}")
print(f"  cell_type: {dict(Counter(ann.cell_type))}")
neo = ann[ann.cell_type.str.lower().str.startswith("neoplastic")]
print(f"\nneoplastic cells: {len(neo)}")
print("  by patient x tissue:")
print(pd.crosstab(neo.patient, neo.tissue))
print(f"  unique cell_key: {ann.cell_key.nunique()} / {len(ann)}")

# ---------------------------------------------------------------- darmani matrix
if not os.path.exists(CACHE):
    print(f"\nparsing GSE84465_GBM_All_data.csv.gz ...", flush=True)
    df = pd.read_csv(f"{B}/raw/GSE84465_GBM_All_data.csv.gz", index_col=0)
    json.dump([str(c) for c in df.columns], open(CELLS, "w"))
    json.dump([str(i) for i in df.index], open(GR, "w"))
    np.save(CACHE, df.to_numpy(dtype=np.float32))
    del df
    print(f"  parsed in {time.time()-t0:.0f}s", flush=True)
D = np.load(CACHE, mmap_mode="r")
cells = json.load(open(CELLS))
genes = json.load(open(GR))
print(f"\nDarmanis matrix {D.shape} (genes x cells), {len(cells)} cell names, {len(genes)} gene ids")
print(f"  first cells: {cells[:3]}   last: {cells[-1]}")
print(f"  first genes: {genes[:3]}")
sub = np.asarray(D[::500, ::37], dtype=np.float64)
print(f"  value scales: min {np.nanmin(sub):.4f} max {np.nanmax(sub):.4f} "
      f"mean {np.nanmean(sub):.4f}   NaN frac {np.isnan(sub).mean():.5f}")
print(f"  column sums (sampled cells): "
      f"{np.round(np.nansum(np.asarray(D[::500, ::37], dtype=np.float64), axis=0)[:5], 2)}")

# ---------------------------------------------------------------- deeplink matrix
dl = pd.read_csv(DL_RAW, nrows=3)
print(f"\nDeepLINK rna2.csv: header {len(dl.columns)} fields "
      f"(last column = {dl.columns[-1]!r}); first gene {dl.columns[0]!r}")
print(f"  first row values: {np.round(dl.iloc[0, :6].to_numpy(dtype=float), 4)}")
full = pd.read_csv(DL_RAW)
y = full.iloc[:, -1].to_numpy()
Xdl = full.iloc[:, :-1].to_numpy(dtype=np.float64)
print(f"  full shape {full.shape} -> X {Xdl.shape}, y classes {dict(Counter(y.astype(int)))}")
print(f"  DeepLINK X: row sums min {Xdl.sum(1).min():.1f} max {Xdl.sum(1).max():.1f}; "
      f"max value {Xdl.max():.2f}; zero fraction {(Xdl==0).mean():.4f}")
shared = sorted(set(str(c).strip('\"') for c in dl.columns[:-1]) & set(genes))
print(f"  gene-id overlap with Darmanis: {len(shared)} / {Xdl.shape[1]} (deeplink) "
      f"and {len(shared)} / {len(genes)} (darmanis)")
json.dump({"n_deeplink_genes": int(Xdl.shape[1]), "n_darmanis_genes": len(genes),
           "n_shared_genes": len(shared), "n_deeplink_cells": int(Xdl.shape[0]),
           "n_darmanis_cells": len(cells),
           "darmanis_annotation": {"n_cells": int(len(ann)), "n_patients": int(ann.patient.nunique()),
                                   "n_neoplastic": int(len(neo))}},
          open(f"{B}/metadata/s1_recon.json", "w"), indent=2)
print(f"\ndone {time.time()-t0:.0f}s")
