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
"""GBM s3b - plate / batch confounding audit.

The 632 cells were captured on 24 10x plates. This script establishes, from the
recovered cell identity, whether plate is nested inside tissue (= the label).

Writes only NEW files: metadata/plate_summary.csv, audit/PLATE_BATCH_AUDIT.json.
"""
import json
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
cm = pd.read_csv(f"{B}/canonical/cell_sample_mapping_canonical.csv")
cm["plate_id"] = [c.split(".")[0] for c in cm.cell_id]
cm["well"] = [c.split(".")[1] for c in cm.cell_id]

g = (cm.groupby(["plate_id", "patient_id", "tissue", "label_core_vs_periphery"])
       .agg(n_cells=("cell_id", "size"),
            median_library_size=("library_size", "median"),
            median_n_detected_genes=("n_detected_genes", "median"),
            first_cell=("cell_id", "min"))
       .reset_index()
       .rename(columns={"label_core_vs_periphery": "label"})
       .sort_values(["patient_id", "tissue", "plate_id"]))
g.to_csv(f"{B}/metadata/plate_summary.csv", index=False)
print(g.to_string(index=False))

n_plates = int(g.plate_id.nunique())
cross_tissue = int((cm.groupby("plate_id").tissue.nunique() > 1).sum())
cross_patient = int((cm.groupby("plate_id").patient_id.nunique() > 1).sum())
per_class = cm.groupby("tissue").plate_id.nunique().to_dict()
n_plates_strict = g.groupby("tissue").size().to_dict()

out = {
    "what": "plate / batch confounding audit of the 632-cell task",
    "n_plates": n_plates,
    "plates_per_class": {k: int(v) for k, v in n_plates_strict.items()},
    "plates_per_class_distinct_cells": {k: int(v) for k, v in per_class.items()},
    "plates_per_patient_tissue": {
        f"{p}|{t}": int(n) for (p, t), n in
        cm.groupby(["patient_id", "tissue"]).plate_id.nunique().items()},
    "cells_per_plate": {"min": int(g.n_cells.min()), "median": float(g.n_cells.median()),
                        "max": int(g.n_cells.max()),
                        "n_plates_with_1_cell": int((g.n_cells == 1).sum())},
    "number_of_cross_tissue_plates": cross_tissue,
    "number_of_cross_patient_plates": cross_patient,
    "plate_nested_within_tissue": bool(cross_tissue == 0),
    "tsne_cluster_vs_tissue": {
        str(k): {str(kk): int(vv) for kk, vv in v.items()}
        for k, v in pd.crosstab(cm.tissue, cm.cell_type).to_dict().items()},
    "selection_field_is_not_cell_type": True,
    "limitations": {
        "cell_level_random_split": "NOT VALID as an independent generalization estimate",
        "patient_disjoint_generalization": ("NOT adequately supported: only 2 patients exist "
                                            "(BT_S2, BT_S4)"),
        "leave_plate_out": ("may prevent exact same-plate train/test leakage, BUT DOES NOT "
                            "IDENTIFY BIOLOGICAL TISSUE EFFECT SEPARATELY FROM PLATE/BATCH "
                            "EFFECT, because plate and tissue are structurally confounded"),
        "forbidden_claims": [
            "leave-plate-out removes batch confounding",
            "leave-plate-out is an unbiased evaluation",
            "generalization to new GBM patients",
            "clean biological discrimination of tumour core vs periphery",
        ],
    },
    "role": "SECONDARY / DIAGNOSTIC / STRESS-TEST DATASET",
    "script": "scripts/s3b_plate_audit.py",
}
json.dump(out, open(f"{B}/audit/PLATE_BATCH_AUDIT.json", "w"), indent=2)
print(f"\nplates {n_plates} | cross-tissue plates {cross_tissue} | cross-patient plates "
      f"{cross_patient} | plates per class {out['plates_per_class']}")
print(f"cells per plate: min {g.n_cells.min()} median {g.n_cells.median()} max {g.n_cells.max()}")
print("written metadata/plate_summary.csv + audit/PLATE_BATCH_AUDIT.json")
