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
"""GBM s5 - canonical alignment + hard assertions + manifest.

canonical/X_candidate_aligned.npy[:, j]  <->  canonical/candidate_features_canonical.csv row j
"""
import hashlib, json, os, time
from collections import Counter
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
G = str(REPRO_ROOT / 'high_dim_dataset_recovery_20260916/gbm_scrna')
CAN = f"{B}/canonical"
K = 2000
t0 = time.time()
os.makedirs(CAN, exist_ok=True)

# ------------------------------------------------------------------ load
raw = pd.read_csv(f"{B}/raw/rna2.csv")
gcols = [c for c in raw.columns if c.startswith("ENSG")]
y = raw.iloc[:, -1].to_numpy().astype(int)
X = raw[gcols].to_numpy(dtype=np.float64)
assert X.shape == (632, 23257)
assert raw.columns[-1] == "real_y"
print(f"raw rna2: X {X.shape}  gene columns {len(gcols)}  y {dict(Counter(y))}")

cand = pd.read_csv(f"{B}/metadata/candidate_features_all2000.csv", keep_default_na=False)
assert len(cand) == K and list(cand.candidate_col_index) == list(range(K))
idx = cand.original_uci_feature_index.astype(int).to_numpy()
assert len(set(idx.tolist())) == K
Xa = X[:, idx].copy()                                    # (632, 2000) - direct slice
print(f"X_candidate_aligned {Xa.shape} built as a direct column slice")

# ------------------------------------------------------------------ A. column identity
fid_col = np.array(gcols)
okA = all(cand.original_gene_id.iloc[j] == fid_col[idx[j]] for j in range(K))
assert okA, "A FAILED: canonical gene id != raw column id"
print(f"A. per-column gene identity: 2000/2000 OK")

# ------------------------------------------------------------------ B. rank order == frozen
rank = np.empty(23257, dtype=int)
rank[idx] = cand.x_only_rank.to_numpy()                  # rank 1..2000 for the kept set
ord_keys = np.argsort(rank[idx])
# frozen X_candidate is stored in descending-variance rank order
X_rankord = X[:, idx[ord_keys]]
frozen = np.load(f"{G}/candidate_universe/X_candidate.npy")
assert frozen.shape == X_rankord.shape
diff = np.abs(frozen.astype(np.float64) - X_rankord)
scale = np.maximum(np.abs(frozen.astype(np.float64)), 1.0)
rel = diff / scale
print(f"B. vs frozen X_candidate (rank order): max_abs {diff.max():.3e} "
      f"max_rel {rel.max():.3e} entries_rel>1e-6 {int((rel > 1e-6).sum())}")
assert rel.max() < 1e-6
# gene identity of the frozen table is correct while its feature_index is off by one
fz = pd.read_csv(f"{G}/candidate_universe/candidate_features.csv", keep_default_na=False)
fv = fz.set_index(fz.feature_index.astype(int))
offby = 0
for j in range(K):
    jj = idx[j] - 1
    if jj in fv.index and fv.loc[jj, "feature_id"] == cand.original_gene_id.iloc[j]:
        offby += 1
print(f"   frozen feature_id matches after the -1 index shift: {offby}/2000")
assert offby == K

# ------------------------------------------------------------------ C. y / order
assert Xa.shape[0] == 632
np.save(f"{CAN}/X_candidate_aligned.npy", Xa.astype(np.float32))
np.save(f"{CAN}/X_candidate_aligned_float64.npy", Xa)
pd.DataFrame(Xa).to_parquet(f"{CAN}/X_candidate_aligned.parquet", index=False)
np.save(f"{CAN}/X_candidate_aligned_log1p.npy", np.log1p(Xa).astype(np.float32))
pd.DataFrame({"row": np.arange(632), "y": y}).to_csv(f"{CAN}/y_canonical.csv", index=False)
print(f"C. wrote X_candidate_aligned.npy/.parquet/.float64/.log1p + y_canonical.csv")

# ------------------------------------------------------------------ D. independent re-read
dcols = sorted(set(idx[0::20].tolist()))
sub = pd.read_csv(f"{B}/raw/rna2.csv", usecols=[gcols[j] for j in dcols]).to_numpy(np.float64)
ref = X[:, dcols]
print(f"D. independent re-read (usecols by name) of {len(dcols)} columns in "
      f"{len(set(idx[0::20].tolist()))} distinct positions: "
      f"max_abs {np.abs(sub - ref).max():.3e} bitwise {np.array_equal(sub, ref)}")
assert np.array_equal(sub, ref)

# ------------------------------------------------------------------ E. random 100x100 block
rng = np.random.default_rng(0)
ri = rng.choice(632, 100, replace=False)
ci = rng.choice(K, 100, replace=False)
blk = Xa[np.ix_(ri, ci)]
blk2 = X[np.ix_(ri, idx[ci])]
print(f"E. random 100x100 block: max_abs {np.abs(blk - blk2).max():.3e} "
      f"bitwise {np.array_equal(blk, blk2)}")
assert np.array_equal(blk, blk2)

# ------------------------------------------------------------------ write tables
cf = cand.copy()
cf["in_canonical_X"] = True
cf["canonical_X_column"] = cf.candidate_col_index
cf.to_csv(f"{CAN}/candidate_features_canonical.csv", index=False)
gid = cf[["candidate_col_index", "original_uci_feature_index", "original_gene_id",
          "ensembl_gene_id_versioned", "ensembl_gene_id_stable", "gene_symbol",
          "entrez_id", "biotype", "variance_log1p", "variance_raw", "nonzero_fraction",
          "x_only_rank", "mapping_status"]]
gid.to_csv(f"{CAN}/candidate_gene_ids.tsv", sep="\t", index=False)
open(f"{CAN}/candidate_gene_symbols.txt", "w").write(
    "\n".join(cf.gene_symbol.tolist()) + "\n")
print(f"candidate_features_canonical.csv {cf.shape}, candidate_gene_ids.tsv {gid.shape}")

# ------------------------------------------------------------------ manifest
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

s4 = json.load(open(f"{B}/metadata/s4_candidate_rule.json"))
s2 = json.load(open(f"{B}/metadata/s2_gene_overlap.json"))
cmap = pd.read_csv(f"{CAN}/cell_sample_mapping_canonical.csv")
man = {
    "dataset": "gbm_scrna_darmanis_deeplink",
    "task": "binary classification: neoplastic tumour core (1) vs tumour periphery (0)",
    "source": {
        "expression_matrix": "DeepLINK PNAS 2021 repository, Real_data_analyses/human_sc_RNAseq/data/rna2.csv",
        "deeplink_repo": "https://github.com/zifanzhu/DeepLINK",
        "deeplink_commit": "ab45db7ae7293f1d82c12a954af12f690263c88b",
        "deeplink_doi": "10.1073/pnas.2104683118",
        "original_data": "Darmanis et al. 2017 GBM scRNA-seq, GEO GSE84465 (GSE84465_GBM_All_data.csv.gz, GSE84465_gsm_brief.txt)",
        "fdfi_reference": "FDFI repo paper/Experiment/Real_data/Human/Human.ipynb: executed output (632, 2000), alpha=0.05/2000",
    },
    "raw_shape": [632, 23257],
    "n_samples": 632,
    "p_raw": 23257,
    "p_raw_correction": ("The earlier frozen manifest recorded p_raw=23256 because it read rna2.csv "
                         "with pandas index_col=0, which silently consumed the first gene column. "
                         "The CSV has 23258 header fields = 23257 genes + real_y, and the DeepLINK "
                         "pipeline uses X0[:, 0:23257]; the correct p_raw is 23257."),
    "candidate_shape": [632, K],
    "k": K,
    "candidate_rule": ("X-only: library-size normalised values (upstream, technical) -> log1p "
                       "(variance-stabilising, X-only) -> per-gene variance ddof=0 -> descending "
                       "order, tie-break ascending original column index -> top 2000"),
    "whether_y_used": False,
    "exact_reproduction_of_reference_preprocessing": False,
    "reference_selection_rule_status": "UNPUBLISHED",
    "forbidden_rule_excluded": ("DeepLINK screening_FORBIDDEN_as_preprocessing.R uses distance "
                                "correlation with real_y to pick top500_p50.csv; excluded."),
    "class_counts": {"0_periphery": int((y == 0).sum()), "1_core": int((y == 1).sum())},
    "majority_class_baseline_accuracy": float(max((y == 0).mean(), (y == 1).mean())),
    "patients": {"n_patients": int(cmap.patient_id.nunique()),
                 "counts": {k: int(v) for k, v in Counter(cmap.patient_id).items()},
                 "core_periphery_same_patient": True,
                 "group_unit_for_splits": "patient_id"},
    "candidate_set_identical_to_frozen_universe": True,
    "frozen_universe_index_offset": 1,
    "canonical_column_order": "descending variance rank (candidate_col_index 0 == rank 1)",
    "alignment_assertions": {
        "A_gene_identity_per_column": "PASS 2000/2000",
        "B_matches_frozen_X_candidate_in_rank_order": f"PASS max_rel {rel.max():.3e}",
        "C_shape_and_float32_output": f"PASS {list(Xa.shape)}",
        "D_independent_reread_bitwise_equal": "PASS",
        "E_random_100x100_block_bitwise_equal": "PASS",
    },
    "files": {f: {"bytes": os.path.getsize(f"{CAN}/{f}"), "sha256": sha(f"{CAN}/{f}")}
              for f in sorted(os.listdir(CAN)) if os.path.isfile(f"{CAN}/{f}")},
    "gene_mapping": {"n_genes": 23257,
                     "with_symbol": int((cf.gene_symbol != "").sum()),
                     "with_entrez": int((cf.entrez_id != "").sum()),
                     "versioned_ensembl_kept": True,
                     "stable_ensembl_generated": True,
                     "shared_genes_used_for_cell_fingerprint": s2["n_shared"]},
    "var_rank1": s4["var_rank1"], "var_rank2000": s4["var_rank2000"],
    "var_rank2001": s4["var_rank2001"],
}
json.dump(man, open(f"{CAN}/candidate_manifest.json", "w"), indent=2)
print(f"candidate_manifest.json written; {time.time()-t0:.0f}s")
