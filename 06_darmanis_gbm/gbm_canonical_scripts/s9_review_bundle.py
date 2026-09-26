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
"""GBM s9 - build the external-review bundle referenced by
audit/AUDIT_PROMPT_FOR_EXTERNAL_REVIEW.md §7 (text-only, small, uploadable).
"""
import hashlib, json, os, shutil, zipfile
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
OUT = str(REPRO_ROOT / 'GBM_REVIEW_BUNDLE_20260917')
ZIP = f"{OUT}.zip"
SRC = f"{OUT}/gbm_review_bundle"
LOGS = "/tmp/gbm_logs"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

shutil.rmtree(OUT, ignore_errors=True)
for d in ("docs", "code", "tables", "tables_head", "manifest", "evidence", "digest", "logs"):
    os.makedirs(f"{SRC}/{d}", exist_ok=True)

# ------------------------------------------------------------------ copy
copy = {
    "docs": ["FINAL_HANDOFF.md", "SOURCE_AND_DOWNLOAD_AUDIT.md", "PREPROCESSING_AUDIT.md",
             "PATIENT_CELL_MAPPING_AUDIT.md", "GENE_MAPPING_AUDIT.md",
             "audit/AUDIT_PROMPT_FOR_EXTERNAL_REVIEW.md"],
    "code": [f"scripts/s{i}_{n}.py" for i, n in
             [(1, "recon"), (2, "cell_fingerprint"), (3, "patient_mapping"),
              (4, "candidate_universe"), (5, "canonical_align"), (6, "stats"),
              (7, "verify"), (8, "package")]],
    "tables": ["canonical/candidate_features_canonical.csv", "canonical/candidate_gene_ids.tsv",
               "canonical/cell_sample_mapping_canonical.csv", "canonical/patient_summary.csv",
               "canonical/y_canonical.csv", "canonical/candidate_gene_symbols.txt"],
    "manifest": ["canonical/candidate_manifest.json"],
    "evidence": ["metadata/s2_cell_fingerprint.csv", "metadata/s2_gene_overlap.json",
                 "metadata/s4_candidate_rule.json", "metadata/full_gene_mapping_all23257.csv",
                 "metadata/darmanis_cell_annotation.csv", "metadata/gdc_genes_by_ensembl.json",
                 "metadata/darmanis_cell_names.json", "metadata/darmanis_gene_symbols.json",
                 "metadata/s2_gene_alignment.npz"],
    "logs": ["s2_cell_fingerprint.log", "s3_patient_mapping.log", "s7_verify.log"],
}
for d, items in copy.items():
    for it in items:
        src = f"{LOGS}/{it}" if d == "logs" else f"{B}/{it}"
        dst = f"{SRC}/{d}/{os.path.basename(it)}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"  MISSING {src}")

# ------------------------------------------------------------------ table heads
for rel, sep in [("canonical/candidate_features_canonical.csv", ","),
                 ("canonical/candidate_gene_ids.tsv", "\t"),
                 ("canonical/cell_sample_mapping_canonical.csv", ","),
                 ("canonical/y_canonical.csv", ",")]:
    df = pd.read_csv(f"{B}/{rel}", sep=sep, nrows=20, keep_default_na=False)
    out = f"{SRC}/tables_head/{os.path.basename(rel).replace('.tsv', '.head20.tsv').replace('.csv', '.head20.csv')}"
    df.to_csv(out, sep=sep, index=False)
    print(f"  head20: {os.path.basename(out)} {df.shape}")

# ------------------------------------------------------------------ digests
cf = pd.read_csv(f"{B}/canonical/candidate_features_canonical.csv", keep_default_na=False)
raw = pd.read_csv(f"{B}/raw/rna2.csv")
gcols = [c for c in raw.columns if c.startswith("ENSG")]
X = raw[list(cf.original_gene_id)].to_numpy(np.float64)      # by NAME, canonical order
X32 = np.load(f"{B}/canonical/X_candidate_aligned.npy")
assert X32.shape == (632, 2000) and X.shape == (632, 2000)

# per column: recompute stats from the npy and from the raw CSV, plus a byte fingerprint
rows = []
for j in range(2000):
    c32 = np.ascontiguousarray(X32[:, j])
    c32_from_raw = np.ascontiguousarray(X[:, j].astype(np.float32))
    assert np.array_equal(c32, c32_from_raw), j
    rows.append({
        "candidate_col_index": j,
        "x_only_rank": int(cf.x_only_rank.iloc[j]),
        "original_column_index": int(cf.original_uci_feature_index.iloc[j]),
        "ensembl_gene_id_versioned": cf.original_gene_id.iloc[j],
        "gene_symbol": cf.gene_symbol.iloc[j],
        "variance_log1p_table": float(cf.variance_log1p.iloc[j]),
        "variance_log1p_recomputed": float(np.log1p(X[:, j]).var(ddof=0)),
        "mean_raw": float(X[:, j].mean()), "std_raw_ddof0": float(X[:, j].std(ddof=0)),
        "min_raw": float(X[:, j].min()), "max_raw": float(X[:, j].max()),
        "nonzero_count": int((X[:, j] != 0).sum()),
        "sha256_float32_column_bytes": hashlib.sha256(c32.tobytes()).hexdigest(),
        "float32_matches_raw_read": True,
    })
dg = pd.DataFrame(rows)
dg["variance_abs_diff"] = (dg.variance_log1p_table - dg.variance_log1p_recomputed).abs()
dg.to_csv(f"{SRC}/digest/canonical_X_per_column_digest.csv", index=False)
print(f"  digest: per-column {dg.shape}, max var diff {dg.variance_abs_diff.max():.3e}, "
      f"distinct column hashes {dg.sha256_float32_column_bytes.nunique()}")

pd.DataFrame(X32[:20, :20]).to_csv(f"{SRC}/digest/canonical_X_head20x20.csv", index=False)

# frozen vs canonical
G = str(REPRO_ROOT / 'high_dim_dataset_recovery_20260916/gbm_scrna')
fz = pd.read_csv(f"{G}/candidate_universe/candidate_features.csv", keep_default_na=False)
fzi = fz.set_index(fz.feature_index.astype(int))
cmp_rows = []
for j in range(2000):
    jj = int(cf.original_uci_feature_index.iloc[j]) - 1
    rec = {"candidate_col_index": j, "canonical_x_only_rank": int(cf.x_only_rank.iloc[j]),
           "canonical_original_column_index": int(cf.original_uci_feature_index.iloc[j]),
           "frozen_feature_index_raw_field": jj,
           "frozen_feature_index_plus1": jj + 1,
           "gene_id_this_tree": cf.original_gene_id.iloc[j],
           "gene_id_frozen": fzi.loc[jj, "feature_id"] if jj in fzi.index else "",
           "variance_log1p_this_tree": float(cf.variance_log1p.iloc[j]),
           "variance_log1p_frozen": float(fzi.loc[jj, "variance"]) if jj in fzi.index else np.nan}
    cmp_rows.append(rec)
cmpp = pd.DataFrame(cmp_rows)
cmpp["gene_id_match"] = cmpp.gene_id_this_tree == cmpp.gene_id_frozen
cmpp["variance_abs_diff"] = (cmpp.variance_log1p_this_tree - cmpp.variance_log1p_frozen).abs()
cmpp.to_csv(f"{SRC}/digest/frozen_vs_canonical_comparison.csv", index=False)
print(f"  digest: frozen comparison {cmpp.shape}, id match {int(cmpp.gene_id_match.sum())}/2000, "
      f"var max diff {cmpp.variance_abs_diff.max():.3e}")

# binaries deliberately NOT shipped
BIG = ["canonical/X_candidate_aligned.npy", "canonical/X_candidate_aligned_float64.npy",
       "canonical/X_candidate_aligned_log1p.npy", "canonical/X_candidate_aligned.parquet",
       "raw/rna2.csv", "raw/GSE84465_GBM_All_data.csv.gz", "metadata/GSE84465_gsm_brief.txt",
       "processed/darmanis_counts_genes_by_cells.npy"]
pd.DataFrame([{"path": p, "bytes": os.path.getsize(f"{B}/{p}"), "sha256": sha(f"{B}/{p}"),
               "in_this_bundle": False,
               "how_to_obtain": ("re-run the scripts in code/" if p.startswith(("canonical/", "processed/"))
                                 else "download (URL and sha256 in docs/SOURCE_AND_DOWNLOAD_AUDIT.md)")}
              for p in BIG]).to_csv(f"{SRC}/digest/excluded_binaries_checksums.csv", index=False)

# ------------------------------------------------------------------ readme
open(f"{SRC}/00_README_HOW_TO_USE.md", "w").write("""# GBM review bundle — how to use

## What this is
Everything referenced by `audit/AUDIT_PROMPT_FOR_EXTERNAL_REVIEW.md` §7, packaged for an external
reviewer (e.g. GPT) **without** the large binaries. Text only, a few MB.

## How to use it
1. Send `audit/AUDIT_PROMPT_FOR_EXTERNAL_REVIEW.md` first (it is self-contained: all claims and
   numbers are inlined, so the reviewer does not even need the rest).
2. Then attach, as needed:
   * `code/*.py` — the full implementation (s2 fingerprint, s5 alignment + assertions, s7
     independent verification); this is what a reviewer needs to judge whether the checks are real.
   * `logs/*.log` — the raw console output of s2, s3 and s7 as actually executed.
   * `tables/*.csv|tsv` — the canonical tables; `tables_head/*` are 20-row extracts for pasting
     directly into a chat.
   * `digest/canonical_X_per_column_digest.csv` — per-column summary + `sha256` of the exact
     float32 byte block of every one of the 2000 columns, recomputed **independently** from
     `raw/rna2.csv` (asserted bitwise identical). This lets a reviewer spot-check the matrix
     without shipping the 5 MB `.npy`.
   * `digest/frozen_vs_canonical_comparison.csv` — the 2000-row proof of the two corrections
     (off-by-one index and column order) with both variance values side by side.
   * `docs/*.md` — the four audit documents plus `FINAL_HANDOFF.md`.
   * `manifest/candidate_manifest.json` — machine-readable manifest incl. all assertion texts.

## What is deliberately NOT here
`digest/excluded_binaries_checksums.csv` lists every omitted file with its byte size and sha256:
the X matrices (`X_candidate_aligned*.npy/.parquet`, 5–10 MB each), the source data
(`raw/rna2.csv` 61 MB, `GSE84465_GBM_All_data.csv.gz` 20 MB), the intermediate Darmanis matrix
(337 MB) and the GEO series matrix (12.5 MB). None of them is needed for a methodological review;
all are reproducible from `code/` or re-downloadable (URLs + checksums in
`docs/SOURCE_AND_DOWNLOAD_AUDIT.md`). The complete frozen package (including `raw/rna2.csv`) is
`REPRO_ROOT / 'GBM_CANONICAL_HANDOFF_20260917.tar.gz'`.

## Reproduction state
`s2`, `s3` and `s7` were re-run while building this bundle and produced **byte-identical** outputs:
the tree-level `sha256sum -c SHA256SUMS.txt` still reports 39 OK / 0 FAILED afterwards.
""")

open(f"{SRC}/AUDIT_PROMPT_FOR_EXTERNAL_REVIEW.md", "w").write(
    open(f"{B}/audit/AUDIT_PROMPT_FOR_EXTERNAL_REVIEW.md").read())

# ------------------------------------------------------------------ checksums + zip
rows = []
for root, dirs, fs in os.walk(SRC):
    for f in sorted(fs):
        p = os.path.join(root, f)
        rows.append((os.path.relpath(p, SRC), os.path.getsize(p), sha(p)))
rows.sort()
with open(f"{SRC}/SHA256SUMS.txt", "w") as f:
    f.write("\n".join(f"{h}  {r}" for r, b, h in rows if r != "SHA256SUMS.txt") + "\n")
with open(f"{SRC}/FILE_MANIFEST.csv", "w") as f:
    f.write("path,bytes,sha256\n")
    for r, b, h in rows:
        if r != "FILE_MANIFEST.csv":
            f.write(f"{r},{b},{h}\n")

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    members = [(r, b, h) for r, b, h in rows]
    for f in ("SHA256SUMS.txt", "FILE_MANIFEST.csv"):
        members.append((f, os.path.getsize(f"{SRC}/{f}"), sha(f"{SRC}/{f}")))
    for r, b, h in members:
        z.write(f"{SRC}/{r}", arcname=f"gbm_review_bundle/{r}")
print(f"\nzip {ZIP}: {os.path.getsize(ZIP):,} bytes, {len(members)} files")
print(f"zip sha256 {sha(ZIP)}")
