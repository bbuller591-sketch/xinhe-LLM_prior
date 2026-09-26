#!/usr/bin/env python3
"""p0 - input reference: prove the frozen canonical recovery is the untouched input.

Checks
  1. every canonical CSV matches the sha256 recorded in the frozen manifest;
  2. the source data_clean files are hashed and snapshotted (so any later mutation is detectable);
  3. structural invariants of the analysis table hold (rows / patients / label identity);
  4. no batch2 label is reachable from the pilot's modelling table.
"""
import os
import pandas as pd
from common import (CANON, D, CANON_FILES, canon_hash_snapshot, json_dump, sha256,
                    batch1_raw, design_table, label_check, semantic_universe)

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


snap = canon_hash_snapshot()
json_dump(snap, f"{D['00_INPUT_REFERENCE']}/canonical_input_sha256.json")

import json
man = json.load(open(f"{CANON}/canonical/canonical_manifest.json"))
for name, rec in man["files"].items():
    key = f"canonical/{name}"
    got = sha256(f"{CANON}/{key}")
    snap[key] = {"bytes": os.path.getsize(f"{CANON}/{key}"), "sha256": got}
    check(f"canonical file {name} unchanged", got == rec["sha256"],
          f"{got[:16]}… vs manifest {rec['sha256'][:16]}…")
json_dump(snap, f"{D['00_INPUT_REFERENCE']}/canonical_input_sha256.json")

# ------------------------------------------------------------------ analysis table
tab = design_table()
n_rows, n_pat = len(tab), tab.patient_uid.nunique()
print(f"\nbatch1 analysis table: {n_rows} rows / {n_pat} patients / sites "
      f"{tab.site.value_counts().to_dict()}")
check("batch1 rows = 1038", n_rows == 1038, f"{n_rows}")
check("batch1 patients = 691", n_pat == 691, f"{n_pat}")
check("label == 1[T_used <= -2.5] for all batch1 rows", label_check(tab) == 0,
      f"mismatches {label_check(tab)}")
check("(patient, site) unique in batch1",
      not tab.duplicated(["patient_uid", "site"]).any())
check("positive rows = 346 (0.3333)", int(tab.y.sum()) == 346,
      f"{int(tab.y.sum())} / {n_rows} = {tab.y.mean():.4f}")

cols, _ = semantic_universe()
check("semantic universe = 40", len(cols) == 40, f"{len(cols)}")
missing_from_table = [c for c in cols if c not in tab.columns and c not in
                      ("C-反应蛋白",)]
check("all semantic features present in the analysis table", not missing_from_table,
      str(missing_from_table))

forbidden = {"T_used", "T_source", "subregion", "label_osteoporosis"}
check("no forbidden DXA-derived column is used as a feature",
      not (forbidden & set(cols)), str(sorted(forbidden & set(cols))))
check("batch2 label not present in the analysis table",
      "label_osteoporosis" not in tab.columns)

# batch2 must be untouched by the pilot: prove it exists, WITHOUT reading its label column.
b2 = pd.read_csv(f"{CANON}/source/data_clean/batch2_full.csv", dtype=str, usecols=["site"])
check("batch2 exists but is SEALED (only its row count is touched here, never a label value)",
      len(b2) == 828, f"{len(b2)} batch2 rows sealed")

mtime = {k: os.path.getmtime(f"{CANON}/{k}") for k in CANON_FILES}
json_dump({"canonical_input_sha256": snap,
           "canonical_mtime": mtime,
           "batch1_rows": n_rows, "batch1_patients": n_pat,
           "batch1_positive_rows": int(tab.y.sum())},
          f"{D['00_INPUT_REFERENCE']}/input_verification.json")

print(f"\nP0: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
