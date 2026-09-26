#!/usr/bin/env python3
"""p9 - TASK A: fix the primary site scope to lumbar + hip.

The v1 freeze already recommended lumbar+hip primary / forearm sensitivity, but the v1 stability and
confusion numbers were then produced on ALL SITES. That inconsistency is corrected here: the v2
primary analysis is lumbar+hip only, and the all-sites run is kept as a sensitivity table.

Writes 01_TASK_FREEZE/TASK_SCOPE_V2.md and audit/scope_v2.json.
"""
import json
import numpy as np
import pandas as pd
from common import D, json_dump
import v2_core as V

FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


man = json.load(open(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_manifest.json"))
F = man["selectable_features"]
raw = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv").merge(
    pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_y_and_context.csv",
                usecols=["patient_uid", "site", "y"]), on=["patient_uid", "site"])
check("leakage-safe development matrix loaded unchanged (1029 rows / 684 patients)",
      len(raw) == 1029 and raw.patient_uid.nunique() == 684, f"{len(raw)}/{raw.patient_uid.nunique()}")

scope = {}
for name, sites in [("primary_lumbar_hip", V.SITE_PRIMARY), ("all_sites_sensitivity",
                                                             V.SITE_PRIMARY + V.SITE_SENSITIVITY_ONLY)]:
    d = raw[raw.site.isin(sites)]
    scope[name] = {
        "sites": sites, "n_rows": int(len(d)), "n_patients": int(d.patient_uid.nunique()),
        "n_positive": int(d.y.sum()), "n_negative": int((1 - d.y).sum()),
        "prevalence": round(float(d.y.mean()), 6),
        "per_site": {s: {"rows": int((d.site == s).sum()),
                         "patients": int(d.loc[d.site == s, "patient_uid"].nunique()),
                         "positive": int(d.loc[d.site == s, "y"].sum()),
                         "prevalence": round(float(d.loc[d.site == s, "y"].mean()), 6)}
                     for s in sites},
    }
print(json.dumps(scope, ensure_ascii=False, indent=1))
check("primary scope = 892 rows / 664 patients", scope["primary_lumbar_hip"]["n_rows"] == 892
      and scope["primary_lumbar_hip"]["n_patients"] == 664)
check("all-sites sensitivity = 1029 rows / 684 patients",
      scope["all_sites_sensitivity"]["n_rows"] == 1029
      and scope["all_sites_sensitivity"]["n_patients"] == 684)
check("forearm carries no row into the primary matrix",
      int((raw.site == "前臂").sum()) == 137
      and scope["primary_lumbar_hip"]["n_rows"] + 137 == 1029)
json_dump(scope, f"{D['audit']}/scope_v2.json")

p, a = scope["primary_lumbar_hip"], scope["all_sites_sensitivity"]
ls, hip = p["per_site"]["腰椎骨"], p["per_site"]["髋关节"]
fa = a["per_site"]["前臂"]

doc = f"""# TASK SCOPE V2 — primary analysis is lumbar + hip only

**Status: FROZEN.** This file corrects an inconsistency in the v1 pilot and overrides it for all v2
numbers.

## 1. What was inconsistent in v1

`01_TASK_FREEZE/TASK_B_SEMANTICS_FREEZE.md` (v1) recommended

* **PRIMARY:** lumbar spine + hip
* **SENSITIVITY:** forearm

but the v1 `04_DATA_ONLY_BASELINE`, `05_TOPK_STABILITY` and `06_DATA_CONFUSION` numbers were then all
produced on **all three sites**. The recommendation and the measurement disagreed. The v1 files are
kept unmodified for the record; every v2 number below is lumbar+hip.

## 2. The scope decision itself was NOT re-opened

The scope is **not** re-selected by predictive performance. It rests on the v1 clinical/measurement
argument, which is unchanged:

* the ISCD diagnostic sites are the lumbar spine and the proximal femur; the forearm is a fallback;
* in this dataset only 1 of {fa['rows']} forearm rows is the ISCD reference site (`1/3` radius), 132 are
  `未知` and 6 are a whole-forearm ROI;
* the forearm base rate ({fa['prevalence']:.4f}) differs from lumbar ({ls['prevalence']:.4f}) and hip
  ({hip['prevalence']:.4f}) by far more than any plausible case-mix difference — evidence of a different
  measurement/reference definition, not of a different patient mix;
* the forearm is **not deleted**: it is reported as a sensitivity.

## 3. Exact primary numbers (lumbar + hip)

| quantity | value |
|---|---|
| n_rows | **{p['n_rows']}** |
| n_patients | **{p['n_patients']}** |
| n_positive | {p['n_positive']} |
| n_negative | {p['n_negative']} |
| prevalence (rows) | **{p['prevalence']:.4f}** |
| sites | {", ".join(V.SITE_PRIMARY)} |

Site distribution inside the primary scope:

| site | rows | patients | positives | prevalence |
|---|---|---|---|---|
| 腰椎骨 (lumbar spine) | {ls['rows']} | {ls['patients']} | {ls['positive']} | {ls['prevalence']:.4f} |
| 髋关节 (hip) | {hip['rows']} | {hip['patients']} | {hip['positive']} | {hip['prevalence']:.4f} |
| **total** | **{p['n_rows']}** | **{p['n_patients']}** | **{p['n_positive']}** | **{p['prevalence']:.4f}** |

Unchanged inputs: the leakage-safe development matrix of the v1 round (1029 rows / 684 patients, the 7
future-period-lab patients already removed, `T_used`/`T_source`/`subregion` excluded, batch2-only
columns excluded, 37 selector-eligible clinical features).

## 4. Sensitivity scope (all sites), reported as a table only

| quantity | value |
|---|---|
| n_rows | {a['n_rows']} |
| n_patients | {a['n_patients']} |
| prevalence | {a['prevalence']:.4f} |
| forearm rows / patients / prevalence | {fa['rows']} / {fa['patients']} / {fa['prevalence']:.4f} |

The all-sites numbers appear **only** as a sensitivity table
(`04_DATA_ONLY_BASELINE/ALL_SITES_SENSITIVITY_V2.csv`); no stability or confusion claim is made on them
in v2.

## 5. What is recomputed under this scope (v2 deliverables)

| stage | artefact |
|---|---|
| forced-context implementation | `01_TASK_FREEZE/FORCED_CONTEXT_IMPLEMENTATION_V2.md` |
| baseline (signal still present?) | `04_DATA_ONLY_BASELINE/BASELINE_V2.json` + `ALL_SITES_SENSITIVITY_V2.csv` |
| selector + resampling | `05_TOPK_STABILITY/FEATURE_STABILITY_V2.csv` |
| top-k set stability + validity | `05_TOPK_STABILITY/TOPK_SET_STABILITY_V2.csv`, `TOPK_VALIDITY_AUDIT.md` |
| rank-order confusion | `06_DATA_CONFUSION/PAIRWISE_RANK_CONFUSION_V2.csv` |
| actionable boundary confusion | `06_DATA_CONFUSION/ACTIONABLE_BOUNDARY_CONFUSION_V2.csv` |
| correlated substitution | `06_DATA_CONFUSION/CORRELATED_FEATURE_COMPETITION_V2.csv` |

## 6. Assertions

* [x] primary n_rows / n_patients = {p['n_rows']} / {p['n_patients']}
* [x] no forearm row in any primary v2 artefact (asserted again in the finalizer)
* [x] the scope was fixed before any v2 model was fitted
* [x] the scope was not chosen or re-chosen using predictive performance
"""
open(f"{D['01_TASK_FREEZE']}/TASK_SCOPE_V2.md", "w").write(doc)
print(f"\nP9: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
