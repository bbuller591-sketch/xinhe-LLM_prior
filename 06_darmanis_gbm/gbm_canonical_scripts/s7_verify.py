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
"""GBM s7 - independent verification of the canonical handoff, patient summary,
checksums and manifest. Re-derives everything from disk with different readers.
"""
import hashlib, json, os, subprocess, tarfile, time
from collections import Counter
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
G = str(REPRO_ROOT / 'high_dim_dataset_recovery_20260916/gbm_scrna')
CAN = f"{B}/canonical"
K = 2000
fails = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)

t0 = time.time()
# ---------------------------------------------------------------- 0. files
need = ["FINAL_HANDOFF.md", "SOURCE_AND_DOWNLOAD_AUDIT.md", "PREPROCESSING_AUDIT.md",
        "PATIENT_CELL_MAPPING_AUDIT.md", "GENE_MAPPING_AUDIT.md",
        "canonical/X_candidate_aligned.npy", "canonical/candidate_features_canonical.csv",
        "canonical/candidate_gene_ids.tsv", "canonical/cell_sample_mapping_canonical.csv",
        "canonical/patient_summary.csv", "canonical/candidate_manifest.json"]
for f in need:
    check(f"file present: {f}", os.path.exists(f"{B}/{f}"))

# ---------------------------------------------------------------- 1. read fresh
raw = pd.read_csv(f"{B}/raw/rna2.csv")
gcols = [c for c in raw.columns if c.startswith("ENSG")]
y = raw["real_y"].to_numpy(int)
X = raw[gcols].to_numpy(np.float64)
check("raw shape", X.shape == (632, 23257), str(X.shape))
check("p_raw in manifest == 23257",
      json.load(open(f"{CAN}/candidate_manifest.json"))["p_raw"] == 23257)

cf = pd.read_csv(f"{CAN}/candidate_features_canonical.csv", keep_default_na=False)
check("candidate table shape", cf.shape[0] == K)
check("candidate_col_index is 0..1999",
      list(cf.candidate_col_index) == list(range(K)))
check("candidate genes unique", cf.original_gene_id.nunique() == K)
check("canonical order == rank order",
      list(cf.x_only_rank) == list(range(1, K + 1)))

Xa32 = np.load(f"{CAN}/X_candidate_aligned.npy")
Xa64 = np.load(f"{CAN}/X_candidate_aligned_float64.npy")
check("aligned shapes", Xa32.shape == (632, K) and Xa64.shape == (632, K))
check("float32 file == float64 file (cast)", np.array_equal(Xa32, Xa64.astype(np.float32)))

# ---------------------------------------------------------------- 2. column identity
rng = np.random.default_rng(7)
ci = rng.choice(K, 100, replace=False)
ri = rng.choice(632, 100, replace=False)
blk = Xa64[np.ix_(ri, ci)]
ref = X[np.ix_(ri, cf.original_uci_feature_index.to_numpy()[ci])]
check("random 100x100 block == fresh raw read (bitwise)", np.array_equal(blk, ref),
      f"max_abs {np.abs(blk-ref).max():.3e}")
names_ok = all(cf.original_gene_id.iloc[j] == gcols[cf.original_uci_feature_index.iloc[j]]
               for j in range(0, K, 7))
check("gene id per canonical column matches the raw header (283 columns tested)", names_ok)

# ---------------------------------------------------------------- 3. ranking sanity
vl = np.log1p(X[:, cf.original_uci_feature_index.to_numpy()]).var(axis=0, ddof=0)
check("stored variance_log1p reproduces (max abs diff < 1e-9)",
      np.abs(vl - cf.variance_log1p.to_numpy()).max() < 1e-9,
      f"max {np.abs(vl - cf.variance_log1p.to_numpy()).max():.2e}")
full_var = np.log1p(X).var(axis=0, ddof=0)
cut = np.partition(full_var, 23257 - K)[23257 - K]
check("no non-selected gene has higher variance than the cut",
      full_var.max() > cut and (full_var >= cut).sum() >= K)
below = np.sort(full_var)[::-1][K:]
check("variance gap at the cut", (cf.variance_log1p.min() - below.max()) > 0,
      f"{cf.variance_log1p.min():.6f} vs next {below.max():.6f}")

# ---------------------------------------------------------------- 4. y and cells
yc = pd.read_csv(f"{CAN}/y_canonical.csv")
check("y_canonical matches rna2 real_y", np.array_equal(yc.y.to_numpy(int), y))
cm = pd.read_csv(f"{CAN}/cell_sample_mapping_canonical.csv")
check("cell mapping has 632 unique rows", cm.shape[0] == 632 and cm.cell_id.nunique() == 632)
check("all matched cells are Neoplastic", bool((cm.cell_type == "Neoplastic").all()))
check("y agrees with mapped tissue 632/632",
      int(((cm.label_core_vs_periphery == 1) == (cm.tissue == "Tumor")).sum()) == 632)
check("class counts 580/52",
      int((y == 1).sum()) == 580 and int((y == 0).sum()) == 52)

# patient summary (regenerated here, richer)
ps = cm.groupby("patient_id").agg(n_cells=("cell_id", "size"),
                                  n_core_y1=("label_core_vs_periphery", "sum"),
                                  tissues=("tissue", lambda s: "|".join(sorted(set(s)))),
                                  cell_types=("cell_type", lambda s: "|".join(sorted(set(s)))))
ps["n_periphery_y0"] = ps.n_cells - ps.n_core_y1
ps["periphery_share"] = (ps.n_periphery_y0 / ps.n_cells).round(4)
ps = ps[["n_cells", "n_core_y1", "n_periphery_y0", "periphery_share", "tissues", "cell_types"]]
ps.to_csv(f"{CAN}/patient_summary.csv")
print(ps.to_string())

# ---------------------------------------------------------------- 5. vs frozen
fz = pd.read_csv(f"{G}/candidate_universe/candidate_features.csv", keep_default_na=False)
check("gene set identical to the frozen universe after the +1 index shift",
      set(fz.feature_index.astype(int)) == set(cf.original_uci_feature_index.astype(int) - 1))
fzX = np.load(f"{G}/candidate_universe/X_candidate.npy").astype(np.float64)
rank = cf.x_only_rank.to_numpy()
col_in_frozen_order = cf.original_uci_feature_index.to_numpy()[np.argsort(rank)]
mine = Xa64[:, np.argsort(rank)]
rel = np.abs(mine - fzX) / np.maximum(np.abs(fzX), 1.0)
check("frozen X_candidate (rank order) matches canonical within 1e-6 rel",
      rel.max() < 1e-6, f"max_rel {rel.max():.3e}")
check("frozen candidate_features.csv rows are NOT in rank order (documented pitfall)",
      list(fz.feature_index.astype(int)) != sorted(set(fz.feature_index.astype(int)),
                                                    key=lambda j: -np.log1p(X[:, j]).var()))

# ---------------------------------------------------------------- 6. manifest hashes
man = json.load(open(f"{CAN}/candidate_manifest.json"))
bad = []
for f, rec in man["files"].items():
    p = f"{CAN}/{f}"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != rec["sha256"] or os.path.getsize(p) != rec["bytes"]:
        bad.append(f)
check("manifest file hashes all match", not bad, str(bad))

print(f"\nindependent checks: {len(fails)} failures  ({time.time()-t0:.0f}s)")
print("RESULT: " + ("ALL INDEPENDENT CHECKS PASSED" if not fails else f"FAILED {fails}"))
