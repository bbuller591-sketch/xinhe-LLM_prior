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
"""GBM s11 - TASK A hard checks (the 10 PASS conditions for handoff v2).

Exits non-zero if any condition fails. Read-only apart from stdout.
"""
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
STAGE = str(REPRO_ROOT / 'GBM_CANONICAL_HANDOFF_V2_20260917')
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    return open(p, encoding="utf-8", errors="replace").read()


v1 = json.load(open(f"{B}/audit/candidate_manifest_v1.json"))
man = json.load(open(f"{B}/canonical/candidate_manifest.json"))
cf = pd.read_csv(f"{B}/canonical/candidate_features_canonical.csv", keep_default_na=False)

# 1. canonical candidate p = 2000 unchanged
check("canonical candidate p = 2000 unchanged",
      len(cf) == 2000 and man.get("k") == 2000 and man["candidate_shape"] == [632, 2000],
      f"rows {len(cf)}, manifest k {man.get('k')}")

# 2. candidate identity 2000/2000 unchanged (vs the v1 frozen record)
same_hash = all(v1["files"][f]["sha256"] == sha(f"{B}/canonical/{f}") for f in v1["files"])
check("candidate identity / canonical tables unchanged vs v1", same_hash,
      f"{len(v1['files'])} canonical files hash-compared")

# 3. canonical X alignment unchanged + invariant re-asserted
raw = pd.read_csv(f"{B}/raw/rna2.csv")
gene_ids = [c for c in raw.columns if c.startswith("ENSG")]
X = raw[gene_ids].to_numpy(np.float64)
Xa = np.load(f"{B}/canonical/X_candidate_aligned.npy")
idx = cf.original_uci_feature_index.astype(int).to_numpy()
rng = np.random.default_rng(11)
ri, ci = rng.choice(632, 100, replace=False), rng.choice(2000, 100, replace=False)
align = bool(np.array_equal(Xa[np.ix_(ri, ci)], X[np.ix_(ri, idx[ci])].astype(np.float32)))
align &= all(cf.original_gene_id.iloc[j] == gene_ids[idx[j]] for j in range(0, 2000, 7))
check("canonical X alignment unchanged (column j <-> table row j)", align,
      "random 100x100 block bitwise + 286 gene ids")

# 4. global rematch 632/632
gr = json.load(open(f"{B}/audit/GLOBAL_REMATCH_AUDIT.json"))
r2 = pd.read_csv(f"{B}/metadata/s2b_global_rematch.csv")
check("global rematch 632/632", gr["agreement"] == "632/632" and gr["changed"] == 0
      and bool(r2.same_best_match.all()) and len(r2) == 632,
      f"{gr['n_agreement']}/632, changed {gr['changed']}")

# 5-6. documentation: wrong numbers gone, provenance explicitly unresolved
docs = ["SOURCE_AND_DOWNLOAD_AUDIT.md", "PREPROCESSING_AUDIT.md", "FINAL_HANDOFF_V2.md",
        "FINAL_HANDOFF.md", "PATIENT_CELL_MAPPING_AUDIT.md", "GENE_MAPPING_AUDIT.md",
        "audit/AUDIT_PROMPT_FOR_EXTERNAL_REVIEW.md"]
txt = {d: read(f"{B}/{d}") for d in docs if os.path.exists(f"{B}/{d}")}
bad = [f"{d}:{s}" for d, t in txt.items() for s in
       ("20.78", "0.744", "library-size normalised", "library-size-normalised",
        "library-size normalized", "按文库大小归一化") if s in t]
check("documentation normalization error fixed", not bad, f"offending: {bad}")
need = ["upstream normalization provenance unresolved", "processed expression matrix"]
missing = [s for s in need if not any(s in t for t in txt.values())]
check("upstream normalization explicitly marked unresolved", not missing, f"missing {missing}")

# 7. plate / tissue confounding documented
pb = json.load(open(f"{B}/audit/PLATE_BATCH_AUDIT.json"))
ps = pd.read_csv(f"{B}/metadata/plate_summary.csv")
v2 = txt.get("FINAL_HANDOFF_V2.md", "")
conf_ok = (pb["number_of_cross_tissue_plates"] == 0 and pb["plate_nested_within_tissue"]
           and len(ps) == 24
           and "DOES NOT IDENTIFY BIOLOGICAL TISSUE EFFECT SEPARATELY FROM PLATE/BATCH" in v2
           and "structurally confounded" in v2)
check("plate/tissue confounding explicitly documented", conf_ok,
      f"plates {pb['n_plates']}, cross-tissue {pb['number_of_cross_tissue_plates']}")

# 8. dataset role
check("dataset role = secondary/diagnostic",
      "SECONDARY / DIAGNOSTIC" in v2.upper() and "PRIMARY GENERALIZATION BENCHMARK" in v2.upper(),
      "v2 role statement present")

dep_ok = ("DEPRECATED FOR DIRECT DOWNSTREAM USE" in v2
          and "DEPRECATED" in json.dumps(man)
          and os.path.exists(f"{B}/audit/LEGACY_PACKAGE_HASHES.txt"))
check("old frozen artifact marked deprecated", dep_ok, "v2 + manifest + legacy hash record")

# 10. v2 checksums
sums = f"{STAGE}/SHA256SUMS.txt"
if os.path.exists(sums):
    r = subprocess.run(["sha256sum", "-c", "SHA256SUMS.txt"], cwd=STAGE,
                       capture_output=True, text=True)
    n_ok = r.stdout.count(": OK")
    n_bad = r.stdout.count("FAILED") + r.stderr.count("FAILED")
    check("v2 package checksums all pass", n_bad == 0 and n_ok > 0, f"{n_ok} OK / {n_bad} FAILED")
else:
    check("v2 package checksums all pass", False, f"{sums} missing")

nf = sum(1 for _, ok, _ in CHECKS if not ok)
print(f"\nTASK A hard checks: {len(CHECKS) - nf}/{len(CHECKS)} PASS, {nf} FAIL")
print("RESULT: " + ("ALL TASK A HARD CHECKS PASSED" if nf == 0 else "TASK A BLOCKED"))
sys.exit(1 if nf else 0)
