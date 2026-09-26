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
"""GBM s10 - build the v2 handoff package (TASK A / A9).

Creates REPRO_ROOT / 'GBM_CANONICAL_HANDOFF_V2_20260917/' (stage tree) and
GBM_CANONICAL_HANDOFF_V2_20260917.tar.gz, then verifies the tar round trip with sha256sum -c.

Does NOT touch the v1 packages (their hashes are recorded in audit/LEGACY_PACKAGE_HASHES.txt).
Also refreshes the tree-level SHA256SUMS.txt / FILE_MANIFEST.csv for the v2 revision.
"""
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import time

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
OUT = str(REPRO_ROOT)
STAGE = f"{OUT}/GBM_CANONICAL_HANDOFF_V2_20260917"
TAR = f"{STAGE}.tar.gz"
TAR_ROOT = "GBM_CANONICAL_HANDOFF_V2_20260917"
SELF = {"SHA256SUMS.txt", "FILE_MANIFEST.csv"}
t0 = time.time()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- A9 file list
DOCS = ["FINAL_HANDOFF_V2.md", "FINAL_HANDOFF.md", "SOURCE_AND_DOWNLOAD_AUDIT.md",
        "PREPROCESSING_AUDIT.md", "PATIENT_CELL_MAPPING_AUDIT.md", "GENE_MAPPING_AUDIT.md"]
CANONICAL = ["candidate_features_canonical.csv", "candidate_gene_ids.tsv",
             "candidate_gene_symbols.txt", "cell_sample_mapping_canonical.csv",
             "patient_summary.csv", "y_canonical.csv", "candidate_manifest.json",
             "X_candidate_aligned.npy", "X_candidate_aligned_log1p.npy",
             "X_candidate_aligned_float64.npy", "X_candidate_aligned.parquet"]
METADATA = ["plate_summary.csv", "s2b_global_rematch.csv", "s2c_gene_subset_stability.csv",
            "s2_cell_fingerprint.csv", "s2_gene_alignment.npz", "s2_gene_overlap.json",
            "s4_candidate_rule.json", "candidate_features_all2000.csv",
            "full_gene_mapping_all23257.csv", "darmanis_cell_annotation.csv",
            "gdc_genes_by_ensembl.json", "darmanis_cell_names.json",
            "darmanis_gene_symbols.json"]
AUDIT = ["GLOBAL_REMATCH_AUDIT.json", "GENE_SUBSET_STABILITY_AUDIT.json", "PLATE_BATCH_AUDIT.json",
         "X_ONLY_BUILDER_AUDIT.json", "LEGACY_PACKAGE_HASHES.txt", "candidate_manifest_v1.json",
         "AUDIT_PROMPT_FOR_EXTERNAL_REVIEW.md", "stats_extra.json"]
SCRIPTS = sorted(f for f in os.listdir(f"{B}/scripts") if f.endswith(".py"))
RAWS = ["rna2.csv"]

MEMBERS = ([(f, f) for f in DOCS] + [(f"canonical/{f}", f) for f in CANONICAL]
           + [(f"metadata/{f}", f) for f in METADATA] + [(f"audit/{f}", f) for f in AUDIT]
           + [(f"scripts/{f}", f) for f in SCRIPTS] + [(f"raw/{f}", f) for f in RAWS])

# files deliberately NOT shipped (hash recorded in FILE_MANIFEST.csv)
EXCLUDED = ["raw/GSE84465_GBM_All_data.csv.gz", "metadata/GSE84465_gsm_brief.txt",
            "processed/darmanis_counts_genes_by_cells.npy"]

# ---------------------------------------------------------------- stage
shutil.rmtree(STAGE, ignore_errors=True)
for d in ("canonical", "metadata", "audit", "scripts", "raw"):
    os.makedirs(f"{STAGE}/{d}", exist_ok=True)
missing = []
for src, name in MEMBERS:
    if not os.path.exists(f"{B}/{src}"):
        missing.append(src)
        continue
    shutil.copy2(f"{B}/{src}", f"{STAGE}/{src}")
if missing:
    raise SystemExit(f"missing inputs: {missing}")
print(f"staged {len(MEMBERS)} files")


def write_checksums(root, excluded_rel, excluded_base=None):
    rows = []
    for r, _ in MEMBERS:
        p = f"{root}/{r}"
        rows.append((r, os.path.getsize(p), sha(p), True))
    for r in excluded_rel:
        p = f"{excluded_base or root}/{r}"
        rows.append((r, os.path.getsize(p), sha(p), False))
    rows += [(f, os.path.getsize(f"{root}/{f}"), sha(f"{root}/{f}"), True) for f in sorted(SELF)
             if os.path.exists(f"{root}/{f}")]
    rows = sorted(set(rows))
    sums = [f"{h}  {r}" for r, b, h, intar in rows if r not in SELF and intar]
    open(f"{root}/SHA256SUMS.txt", "w").write("\n".join(sums) + "\n")
    with open(f"{root}/FILE_MANIFEST.csv", "w") as f:
        f.write("path,bytes,sha256,in_tarball\n")
        for r, b, h, intar in rows:
            if r != "FILE_MANIFEST.csv":
                f.write(f"{r},{b},{h},{int(intar)}\n")
    return len(sums), len(rows)


n_sums, n_rows = write_checksums(STAGE, EXCLUDED, excluded_base=B)
print(f"stage SHA256SUMS.txt {n_sums} entries, FILE_MANIFEST.csv {n_rows} rows")

# ---------------------------------------------------------------- tarball
if os.path.exists(TAR):
    os.remove(TAR)
with tarfile.open(TAR, "w:gz", compresslevel=6) as tf:
    for src, _ in MEMBERS:
        tf.add(f"{B}/{src}", arcname=f"{TAR_ROOT}/{src}")
    for f in sorted(SELF):
        tf.add(f"{STAGE}/{f}", arcname=f"{TAR_ROOT}/{f}")
print(f"tarball {TAR}: {os.path.getsize(TAR):,} bytes  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- round trip
tmp = tempfile.mkdtemp(prefix="gbm_v2_verify_")
try:
    with tarfile.open(TAR) as tf:
        names = tf.getnames()
        tf.extractall(tmp)
    res = subprocess.run(["sha256sum", "-c", "SHA256SUMS.txt"], cwd=f"{tmp}/{TAR_ROOT}",
                         capture_output=True, text=True)
    ok = res.stdout.count(": OK")
    bad = res.stdout.count("FAILED") + res.stderr.count("FAILED")
    print(f"round trip: extracted {len(names)} members -> {ok} OK / {bad} FAILED")
    for k in ["FINAL_HANDOFF_V2.md", "canonical/candidate_manifest.json",
              "metadata/s2b_global_rematch.csv", "metadata/s2c_gene_subset_stability.csv",
              "metadata/plate_summary.csv", "audit/GLOBAL_REMATCH_AUDIT.json",
              "audit/PLATE_BATCH_AUDIT.json", "canonical/X_candidate_aligned.npy"]:
        print(f"   {'ok     ' if os.path.exists(f'{tmp}/{TAR_ROOT}/{k}') else 'MISSING'} {k}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- tree-level refresh
tree_rows = []
for root, dirs, fs in os.walk(B):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for f in fs:
        r = os.path.relpath(os.path.join(root, f), B)
        tree_rows.append(r)
tree_rows = sorted(set(tree_rows))
BIG = {"raw/GSE84465_GBM_All_data.csv.gz", "processed/darmanis_counts_genes_by_cells.npy"}
lines, man_lines = [], ["path,bytes,sha256,in_tarball"]
for r in tree_rows:
    p = f"{B}/{r}"
    h, b = sha(p), os.path.getsize(p)
    if r not in BIG and r not in SELF:
        lines.append(f"{h}  {r}")
    if r not in SELF:
        man_lines.append(f"{r},{b},{h},{int(r not in BIG)}")
open(f"{B}/SHA256SUMS.txt", "w").write("\n".join(lines) + "\n")
open(f"{B}/FILE_MANIFEST.csv", "w").write("\n".join(man_lines) + "\n")
print(f"tree SHA256SUMS.txt refreshed: {len(lines)} entries, FILE_MANIFEST {len(man_lines)-1} rows")
print(f"tarball sha256 {sha(TAR)}")
print(f"done {time.time()-t0:.0f}s")
