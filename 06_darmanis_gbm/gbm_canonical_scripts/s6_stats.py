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
"""GBM s6 - compact statistics dump used by the audit documents."""
import json
from collections import Counter
import numpy as np
import pandas as pd

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
full = pd.read_csv(f"{B}/metadata/full_gene_mapping_all23257.csv", keep_default_na=False)
cf = pd.read_csv(f"{B}/canonical/candidate_features_canonical.csv", keep_default_na=False)
cm = pd.read_csv(f"{B}/canonical/cell_sample_mapping_canonical.csv")
ne = lambda s: int((s != "").sum())
print("== all 23257 genes ==")
print(f" stable unique {full.ensembl_gene_id_stable.nunique()} | versioned unique {full.original_gene_id.nunique()}")
print(f" symbol non-empty {ne(full.gene_symbol)} | duplicated symbol values {int(full[full.gene_symbol != ''].gene_symbol.duplicated().sum())}")
v = full.ensembl_gene_id_versioned.str.split(".").str[1].astype(int)
print(f" version range {v.min()}..{v.max()}")
print(f" biotype top {dict(Counter(full.biotype).most_common(6))}")
print("== 2000 candidates ==")
print(f" symbol non-empty {ne(cf.gene_symbol)} | entrez non-empty {ne(cf.entrez_id)}")
print(f" biotype {dict(Counter(cf.biotype).most_common(8))}")
print(f" mapping_status {dict(cf.mapping_status.value_counts())}")
print(f" variance_log1p {cf.variance_log1p.min():.4f}..{cf.variance_log1p.max():.4f}")
print(cf.head(12)[["candidate_col_index", "original_uci_feature_index",
                   "ensembl_gene_id_stable", "gene_symbol", "entrez_id",
                   "variance_log1p"]].to_string(index=False))
print("== patients ==")
t = cm.groupby(["patient_id", "tissue"]).size().unstack(fill_value=0)
print(t.to_string())
lab = pd.crosstab(cm.patient_id, cm.label_core_vs_periphery)
print(lab.to_string())
maj = float(max(cm.label_core_vs_periphery.mean(), 1 - cm.label_core_vs_periphery.mean()))
print(f" majority baseline {maj:.4f}")
print(f" match corr 2000g min {cm.match_metric_value.min():.4f} median {cm.match_metric_value.median():.4f}"
      f" | all shared g min {cm.match_metric_value_all_genes.min():.4f} median {cm.match_metric_value_all_genes.median():.4f}")
json.dump({"patient_tissue": t.to_dict(), "majority": maj,
           "n_symbol": ne(cf.gene_symbol), "n_entrez": ne(cf.entrez_id),
           "biotype": dict(Counter(cf.biotype))},
          open(f"{B}/audit/stats_extra.json", "w"), indent=2)
print("written audit/stats_extra.json")
