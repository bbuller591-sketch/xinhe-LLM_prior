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
"""GBM s3 - verify the cell mapping, recover the 632-cell subset rule, and build
the cell/patient canonical tables.
"""
import json, time
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
t0 = time.time()
ann = pd.read_csv(f"{B}/metadata/darmanis_cell_annotation.csv", dtype=str)
cells = json.load(open(f"{B}/metadata/darmanis_cell_names.json"))
genes = json.load(open(f"{B}/metadata/darmanis_gene_symbols.json"))
D = np.load(f"{B}/processed/darmanis_counts_genes_by_cells.npy", mmap_mode="r")
fp = pd.read_csv(f"{B}/metadata/s2_cell_fingerprint.csv")
dl = pd.read_csv(str(REPRO_ROOT / 'high_dim_dataset_recovery_20260916/gbm_scrna/raw/rna2.csv'))
Xdl = dl.iloc[:, :-1].to_numpy(dtype=np.float64)
y = dl.iloc[:, -1].to_numpy().astype(int)
keyed = ann.set_index("cell_key").loc[cells]

print("=== 1. fingerprint robustness ===")
al = np.load(f"{B}/metadata/s2_gene_alignment.npz", allow_pickle=True)
dl_col, dm_row = al["dl_col"], al["dm_row"]
n_shared = len(dl_col)
print(f"  shared genes used for the fingerprint: {n_shared}")
print(f"  unique matches: {fp.matched_cell_key.nunique()} / 632")
print(f"  label vs tissue agreement: "
      f"{int(((fp.y == 1) == (fp.matched_tissue == 'Tumor')).sum())} / 632")
print(f"  all matched cells Neoplastic: {bool((fp.matched_cell_type == 'Neoplastic').all())}")
import random
random.seed(0)
rows = random.sample(range(632), 25)
sp = []
for r in rows:
    j = fp.matched_cell_key.iloc[r]
    ci = cells.index(j)
    d = np.asarray(D[dm_row, ci], dtype=np.float64)
    a = Xdl[r, dl_col]
    m = (d > 0) | (a > 0)
    if m.sum() > 200:
        sp.append(spearmanr(a[m], d[m]).statistic)
print(f"  Spearman (25 random matched pairs, shared genes): "
      f"min {min(sp):.4f} median {np.median(sp):.4f}")

print("\n=== 2. which neoplastic cells form the 632-cell subset? ===")
nz = np.array([int((np.asarray(D[:, c]) > 0).sum()) for c in range(D.shape[1])])
tot = np.array([float(np.asarray(D[:, c], dtype=np.float64).sum()) for c in range(D.shape[1])])
info = pd.DataFrame({"cell_key": cells, "n_detected_genes": nz, "library_size": tot})
info = info.merge(ann[["cell_key", "patient", "tissue", "cell_type"]], on="cell_key", how="left")
info["in_deeplink"] = info.cell_key.isin(set(fp.matched_cell_key))
cand = info[(info.cell_type == "Neoplastic") & (info.patient.isin(["BT_S2", "BT_S4"]))]
print(f"  neoplastic cells of BT_S2/BT_S4: {len(cand)}  (DeepLINK subset {int(cand.in_deeplink.sum())})")
print("  detection-count quantiles by membership:")
print(cand.groupby(["patient", "in_deeplink"]).n_detected_genes.describe()[["count", "min", "50%", "max"]])
inc = cand[cand.in_deeplink]
exc = cand[~cand.in_deeplink]
print(f"  included: n_detected min {inc.n_detected_genes.min()} | library min {inc.library_size.min():.0f}")
print(f"  excluded: n_detected max {exc.n_detected_genes.max()} | library max {exc.library_size.max():.0f}")
thr = None
for t in range(0, 4000, 10):
    if (inc.n_detected_genes >= t).all() and (exc.n_detected_genes < t).all():
        thr = t
        break
print(f"  clean detection-count separation threshold: {thr}")
print(f"  excluded cells by patient/tissue: "
      f"{dict(Counter(zip(exc.patient, exc.tissue)))}")
print(f"  included cells by patient/tissue: "
      f"{dict(Counter(zip(inc.patient, inc.tissue)))}")

print("\n=== 3. canonical cell mapping ===")
cm = fp.copy()
cm["dl_row_index"] = cm.dl_row
cm["cell_id"] = cm.matched_cell_key
cm["darmanis_cell_title"] = cm.matched_cell_title
cm["patient_id"] = cm.matched_patient
cm["tissue"] = cm.matched_tissue
cm["label_core_vs_periphery"] = y
cm["label_meaning"] = np.where(y == 1, "tumour_core", "tumour_periphery")
cm["cell_type"] = cm.matched_cell_type
cm["n_detected_genes"] = [int(info.loc[info.cell_key == k, "n_detected_genes"].iloc[0])
                          for k in cm.cell_id]
cm["library_size"] = [float(info.loc[info.cell_key == k, "library_size"].iloc[0])
                      for k in cm.cell_id]
cm["match_metric"] = "pearson_r_on_log1p_2000_highest_variance_shared_genes"
cm["match_metric_value"] = cm.corr_2000_genes / 2000.0
cm["match_metric_value_all_genes"] = cm.corr_all_shared_genes / n_shared
cm["match_status"] = "VERIFIED_EXACT_CELL_ID"
out = cm[["dl_row_index", "cell_id", "darmanis_cell_title", "patient_id", "tissue",
          "label_core_vs_periphery", "label_meaning", "cell_type", "n_detected_genes",
          "library_size", "match_metric", "match_metric_value",
          "match_metric_value_all_genes", "match_status"]]
out.to_csv(f"{B}/canonical/cell_sample_mapping_canonical.csv", index=False)
print(f"  cell_sample_mapping_canonical.csv {out.shape}")
print(out.head(3).to_string(index=False))

ps = out.groupby(["patient_id", "tissue"]).size().unstack(fill_value=0)
ps["total"] = ps.sum(axis=1)
ps.to_csv(f"{B}/canonical/patient_summary.csv")
print("\n  patient_summary.csv:")
print(ps.to_string())
lab = pd.crosstab(out.patient_id, out.label_core_vs_periphery)
print("\n  label distribution per patient:")
print(lab.to_string())
print(f"\ndone {time.time()-t0:.0f}s")
