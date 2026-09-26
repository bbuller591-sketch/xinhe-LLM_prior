#!/usr/bin/env python3
"""p3 - TEMPORAL LEAKAGE RESOLUTION + LEAKAGE-SAFE TASK B DEVELOPMENT MATRIX.

§8  Resolve the batch1 future-period laboratory contamination by PROVENANCE, not by outcome.
§9  Build the leakage-safe site-level development matrix (batch1 only) and its manifest.

The decision rule is fixed before any model is fitted:
  * the affected CELLS are identified at column granularity -> they are hard-masked/served outside the
    schema (layer 1);
  * the period of the remaining laboratory values of the affected PATIENTS cannot be *verified*
    because the source lab tables carry no collection timestamps, so the conservative branch is
    applied -> those patients are excluded from the PRIMARY development matrix (layer 2);
  * an inclusion sensitivity is recorded; nothing here is chosen by looking at model accuracy,
    feature importance or y.
"""
import json
import numpy as np
import pandas as pd
from common import (CANON, D, C_GRID, FORCED_NAMES, K_GRID, N_FOLDS, N_RESAMPLES,
                    RESAMPLE_FRACTION, SEEDS, batch1_raw, design_table, json_dump,
                    semantic_universe, SITES)

MARKER = "batch2_export_matched_batch1_patient"
FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


# ------------------------------------------------------------------ provenance: locate the contamination
shared_cols = pd.read_csv(f"{CANON}/source/data_clean/cohort_all.csv", nrows=1).columns.tolist()
full_cols = pd.read_csv(f"{CANON}/source/data_clean/cohort_all_full.csv", nrows=1).columns.tolist()
extra_cols = [c for c in full_cols if c not in shared_cols]
print(f"shared schema = {len(shared_cols)} columns; full = {len(full_cols)}; batch2-only = {len(extra_cols)}")

keep = ["住院唯一号", "cohort", "site", "lab_source_period"] + extra_cols
cfull = pd.read_csv(f"{CANON}/source/data_clean/cohort_all_full.csv", dtype={"住院唯一号": str},
                    usecols=keep, low_memory=False)
cfull = cfull[cfull["cohort"] == "batch1"].reset_index(drop=True)   # batch1 rows only, immediately
print(f"batch1 rows in cohort_all_full: {len(cfull)}")
flagged = cfull[cfull["lab_source_period"].notna()]
af_rows = flagged.index.tolist()
uid = pd.read_csv(f"{CANON}/private_only/patient_uid_map.csv", dtype=str) \
    .set_index("住院唯一号")["patient_uid"]
af_uids = sorted(uid[flagged["住院唯一号"]].unique())
n_af_rows, n_af_pat = len(flagged), len(af_uids)

# per-column cell counts of future-period-sourced values
cell_counts = {c: int(flagged[c].notna().sum()) for c in extra_cols}
af_cols = {c: v for c, v in cell_counts.items() if v > 0 and c != "lab_source_period"}
MARKER_COL = "lab_source_period"
n_af_cells = int(sum(af_cols.values()))
n_af_cells_with_marker = n_af_cells + int(flagged[MARKER_COL].notna().sum())
print(f"\naffected rows={n_af_rows} patients={n_af_pat} "
      f"lab columns={len(af_cols)} lab cells={n_af_cells}; "
      f"including the marker column: {len(af_cols) + 1} columns / {n_af_cells_with_marker} cells")
print("affected laboratory columns:", sorted(af_cols))
check("affected rows = 9", n_af_rows == 9, f"{n_af_rows}")
check("affected patients = 7", n_af_pat == 7, f"{n_af_pat}")
check("independently recomputed affected volume reconciles with the source README "
      "(23 columns / 177 cells = 22 laboratory columns + the marker column)",
      len(af_cols) == 22 and n_af_cells == 168 and len(af_cols) + 1 == 23
      and n_af_cells_with_marker == 177,
      f"lab {len(af_cols)}/{n_af_cells}; with marker {len(af_cols)+1}/{n_af_cells_with_marker}")
check("every affected column is a batch2-only column (outside the shared analysis schema)",
      set(af_cols) <= set(extra_cols) and not (set(af_cols) & set(shared_cols)),
      f"{len(set(af_cols) & set(shared_cols))} shared columns affected")
check("no non-laboratory batch2-only column is affected",
      not (set(af_cols) & {"科室", "民族", "入院时间", "出院时间", "身高_raw", "体重_raw",
                           "个人史_raw", "吸烟", "饮酒", "吸烟_cat", "饮酒_cat"}),
      str(sorted(set(af_cols) & {"科室", "民族", "入院时间", "出院时间", "吸烟", "饮酒"})))

# does the flagged set of shared cells agree with batch1_full.csv?  (cross-file consistency)
b1f = pd.read_csv(f"{CANON}/source/data_clean/batch1_full.csv", dtype={"住院唯一号": str},
                  low_memory=False)
shared_only = pd.read_csv(f"{CANON}/source/data_clean/cohort_all_full.csv",
                          dtype={"住院唯一号": str}, usecols=shared_cols + [MARKER_COL],
                          low_memory=False)
shared_only = shared_only[shared_only["cohort"] == "batch1"].reset_index(drop=True)
mrg = shared_only.merge(b1f, on=["住院唯一号", "site"], suffixes=("_full", "_b1"))
sc = [c for c in shared_cols if c not in ("住院唯一号", "site")]
diff_cells = 0
for c in sc:
    a, b = mrg[f"{c}_full"].astype(str), mrg[f"{c}_b1"].astype(str)
    diff_cells += int((a.fillna("<NA>") != b.fillna("<NA>")).sum())
check("the flagged batch1 rows' shared 48-column cells are identical in cohort_all_full and "
      "batch1_full", diff_cells == 0, f"{diff_cells} differing cells out of {len(mrg) * len(sc)}")

# ------------------------------------------------------------------ chosen rule
tab_all = design_table(batch1_raw())
tab_all["patient_uid"] = tab_all["patient_uid"].astype(str)
af_mask = tab_all["patient_uid"].isin(af_uids)
tab = tab_all.loc[~af_mask].reset_index(drop=True)
n_excl_rows, n_excl_pat = int(af_mask.sum()), int(tab_all.loc[af_mask, "patient_uid"].nunique())
print(f"\nexcluded from PRIMARY development: {n_excl_rows} rows / {n_excl_pat} patients")
check("excluded rows/patients = 9/7", (n_excl_rows, n_excl_pat) == (9, 7),
      f"{n_excl_rows}/{n_excl_pat}")
check("primary development matrix = 1029 rows / 684 patients",
      len(tab) == 1029 and tab.patient_uid.nunique() == 684,
      f"{len(tab)} / {tab.patient_uid.nunique()}")
check("no marker-bearing row survives into the primary matrix",
      not tab.patient_uid.isin(af_uids).any())
check("no future-period column exists in the primary matrix by construction",
      not (set(af_cols) & set(tab.columns)), str(sorted(set(af_cols) & set(tab.columns))))

semantic, man = semantic_universe()
elig = pd.read_csv(f"{D['02_FEATURE_ELIGIBILITY']}/DEVELOPMENT_FEATURE_ELIGIBILITY.csv")
eligible = elig.loc[elig.selector_eligibility_status == "ELIGIBLE", "feature_name"].tolist()
inelig_obs = elig.loc[elig.selector_eligibility_status == "INSUFFICIENT_DEV_OBSERVATION",
                      "feature_name"].tolist()
inelig_cens = elig.loc[elig.selector_eligibility_status == "CENSORING_RULE_UNRESOLVED",
                       "feature_name"].tolist()
check("selector-eligible set = 37 from the eligibility audit",
      len(eligible) == 37, f"{len(eligible)}")

# ------------------------------------------------------------------ write the matrices
sel = tab[["patient_uid", "site"] + eligible].copy()
sel.to_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv", index=False,
           encoding="utf-8-sig")
tab[["patient_uid", "site", "y", "T_used"]].to_csv(
    f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_y_and_context.csv", index=False,
    encoding="utf-8-sig")
# site-scope variants for the frozen recommendation A (primary) / sensitivity
scope_rows = {}
for s in SITES:
    scope_rows[s] = int((tab.site == s).sum())
scope_pat = {s: int(tab.loc[tab.site == s, "patient_uid"].nunique()) for s in SITES}
prim = tab[tab.site.isin(["腰椎骨", "髋关节"])]
print("site scope counts:", scope_rows, "| primary (lumbar+hip) rows:", len(prim))

manifest = {
    "artifact": "leakage-safe Task B development matrix",
    "task": "SITE-LEVEL LOW T-SCORE CLASSIFICATION",
    "cohort_used": "batch1 ONLY (batch2 sealed; never opened by this script)",
    "grain": "patient x skeletal site",
    "n_rows": int(len(tab)), "n_patients": int(tab.patient_uid.nunique()),
    "n_positive": int(tab.y.sum()), "n_negative": int((1 - tab.y).sum()),
    "prevalence": round(float(tab.y.mean()), 6),
    "group_key": "patient_uid",
    "forced_context_features": FORCED_NAMES + ["(baseline level: 腰椎骨)"],
    "forced_context_implementation": f"one-hot drop-first of site, multiplied by {1000.0}; never penalised in effect; never selectable",
    "n_forced_context_columns": 2,
    "p_semantic": len(semantic),
    "p_selector_eligible": len(eligible),
    "selectable_features": eligible,
    "site_scope": {
        "primary": ["腰椎骨", "髋关节"],
        "sensitivity_only": ["前臂"],
        "primary_rows": int(len(prim)),
        "primary_patients": int(prim.patient_uid.nunique()),
        "per_site_rows": scope_rows, "per_site_patients": scope_pat,
        "rationale": "see 01_TASK_FREEZE/TASK_B_SEMANTICS_FREEZE.md section 5 (recommendation A)",
    },
    "excluded_due_to_direct_leakage": ["T_used", "T_source", "subregion"],
    "excluded_due_to_insufficient_dev_observation": inelig_obs,
    "excluded_due_to_unresolved_censoring": inelig_cens,
    "excluded_due_to_future_period_lab_contamination": {
        "rule": "EXCLUDE_PATIENTS_CONFIRMED_AT_ROW_LEVEL__MASK_CELLS_IDENTIFIED_AT_COLUMN_LEVEL",
        "n_rows": n_excl_rows, "n_patients": n_excl_pat,
        "marker_value": MARKER,
        "affected_columns_batch2_only": sorted(af_cols),
        "n_affected_columns": len(af_cols), "n_affected_cells": n_af_cells,
        "sensitivity_variant": "development_with_future_period_patients_included (n=1038/691)",
        "detail": "03_LEAKAGE_SAFE_DEVELOPMENT/TEMPORAL_LEAKAGE_RESOLUTION.md",
    },
    "batch2_only_columns_excluded": len(extra_cols),
    "preprocessing": {
        "numeric": "training-fold median imputation + training-fold standardisation",
        "categorical": "fixed documented map for DXA_性别 (女=1, 男=0); no target encoding",
        "missingness_indicators": "NOT used in the primary pilot",
        "fit_scope": "inside each training fold only; never on the full batch1",
    },
    "preregistered_design": {
        "k_grid": K_GRID, "C_grid": C_GRID, "seeds": SEEDS, "n_folds": N_FOLDS,
        "cv": f"StratifiedGroupKFold({N_FOLDS}) grouped by patient_uid, fixed seeds",
        "stability": {"n_resamples": N_RESAMPLES, "patient_fraction": RESAMPLE_FRACTION,
                      "draw": "patient-level 80% subsample, both classes required"},
        "frozen_before_any_result": True,
    },
    "privacy": {"identifiers": "patient_uid only; no raw hospital ID anywhere in this tree"},
}
json_dump(manifest, f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_manifest.json")

# ------------------------------------------------------------------ leakage resolution doc
doc = f"""# TEMPORAL LEAKAGE RESOLUTION — batch1 future-period laboratory values

## 1. The problem, restated

Previous phase recorded: **9 rows / 7 batch1 patients** carry `lab_source_period =
`{MARKER}``. A flag alone is not a resolution: the whole point of the marker is that the
*period* of those laboratory values is not established, and the batch1 cohort is the
**development** cohort of the temporal design. If such values are used for development, the
development/validation boundary is pierced.

## 2. Establishing the provenance granularity (empirical, recomputed here)

Source of truth: `source/data_clean/README.md` §"一处刻意的取舍：批次 1 文件里剔除了 177 个格"
plus `FEATURE_COVERAGE.md`. Everything below was **recomputed independently** from
`cohort_all_full.csv` (batch1 rows only) and cross-checked against `batch1_full.csv`.

| question | answer |
|---|---|
| Can the affected **columns** be identified? | **YES — {len(af_cols)} laboratory columns** (all **batch2-only**), plus the marker column itself = **{len(af_cols)+1}** columns as the source states |
| Can the affected **cells** be identified? | **YES at column×row granularity** — {n_af_cells} laboratory cells (+{n_af_rows} marker cells = {n_af_cells_with_marker}) |
| Can the affected cells be identified **inside the shared 48-column schema**? | **NO — {len(set(af_cols) & set(shared_cols))} shared columns are affected** |
| Can the **period** of the remaining laboratory values of those 7 patients be verified? | **NO** — the source states that the laboratory export carries **no collection timestamp** (`化验文件没有采集时间戳`), so only an argument, not a verification, is possible |

Affected laboratory columns ({len(af_cols)}): `{", ".join(sorted(af_cols))}`

Cell counts per affected column: {json.dumps(af_cols, ensure_ascii=False)}

**Reconciliation with the source's own wording.** The README says "23 个化验列上带来 177 个非空格".
Independently recomputed here: **{len(af_cols)} laboratory columns carry {n_af_cells} cells**, and adding the
`{MARKER_COL}` column itself ({n_af_rows} cells) gives **{len(af_cols)+1} columns / {n_af_cells_with_marker} cells**.
So the source's "23 / 177" counts the marker column as one of its columns; the laboratory-only volume is
22 / 168. Both are recorded, and the exact reconciliation is: 22+1 = 23, 168+9 = 177.

## 3. The chosen rule (two layers, fixed before any model was fitted)

### Layer 1 — cell/column masking (Case A is attainable, at column granularity)

The {n_af_cells} future-period laboratory cells all live in **{len(af_cols)} columns that are not part of the
shared 48-column analysis schema** (`cohort_all.csv` / `batch1_full.csv`). The hospital removed those
columns from the batch1 file *entirely* precisely so that no batch1 row would carry them.

> **Action:** those {len(af_cols)} columns are hard-excluded from the primary matrix by schema, and the
> exclusion is asserted (`p7_assertions.py`: 0 of them present in the primary development matrix).
> Their {n_af_cells} values are never read, never imputed, never used as a missingness indicator.

This alone is **not** sufficient, which is why layer 2 exists.

### Layer 2 — conservative patient exclusion (the conservative branch of the brief)

The residual uncertainty is *not* about which column a value sits in; it is about **whether the
laboratory record of those 7 patients is contemporaneous with their DXA**. The source explicitly
cannot answer this (no timestamps in the laboratory export), and the hospital itself warns that
"任何将来使用这些化验列的分析必须注意：批次 1 患者身上出现的化验值来自后一时期的导出".

Given unverifiable period, the brief's stated priority applies — *do not pretend the values are
usable*:

> **Action (PRIMARY RULE): the 7 affected patients are excluded from the primary development matrix.**

| quantity | value |
|---|---|
| affected patients (pseudonymous `patient_uid`) | {", ".join("`" + u + "`" for u in af_uids)} |
| affected rows | {n_af_rows} |
| affected columns (identifiable) | {len(af_cols)} laboratory (all batch2-only) + the marker column |
| affected cells (identifiable) | {n_af_cells} laboratory cells (+{n_af_rows} marker cells) |
| affected site(s) | {", ".join(sorted(set(flagged['site'].astype(str))))} |
| **resulting development n** | **{len(tab)} rows / {tab.patient_uid.nunique()} patients** (from 1038 / 691) |
| positives | {int(tab.y.sum())} ({tab.y.mean():.4f}) |

### Why this is provenance-determined, not performance-determined

* The affected rows are found by the `lab_source_period` marker and by column membership — **no
  outcome, no model, no importance, no accuracy** was consulted.
* Both layers are applied *before* any model is fitted; the sensitivity variant below is reported,
  never used to pick the rule.
* The alternative "substitute 4.0/2.0/0" style decisions cannot arise here at all: the affected
  columns are not even part of the schema.

## 4. Zero-future-period-value assertion

Asserted (and re-asserted in `p7_assertions.py`):

* [x] 0 marker-bearing rows in the primary development matrix;
* [x] 0 of the {len(af_cols)} affected columns present in the primary development matrix;
* [x] 0 cells of the primary development matrix sourced from the batch2 export (the batch2-only
      schema is excluded wholesale — {len(extra_cols)} columns);
* [x] `lab_source_period` itself is not a modelling column.

## 5. Recorded sensitivity (not a rule change)

`development_with_future_period_patients_included`: identical pipeline, n = {len(tab_all)} rows /
{tab_all.patient_uid.nunique()} patients, i.e. the 7 patients kept with layer 1 still enforced.
Reported side-by-side in `04_DATA_ONLY_BASELINE/DATA_ONLY_SUITABILITY_REPORT.md`. If the two
variants disagree materially, the conservative one is the one to trust; if they agree, that is
evidence the decision is immaterial — either way it is *reported*, not *used to choose*.

## 6. Cross-file consistency check performed here

For the {n_af_rows} flagged batch1 rows, the **shared 48-column** cells are byte-identical between
`cohort_all_full.csv` and `batch1_full.csv` ({diff_cells} differing cells) — i.e. the hospital's
removal of the batch2-only columns did not perturb anything else in those rows.
"""
open(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/TEMPORAL_LEAKAGE_RESOLUTION.md", "w").write(doc)

print(f"\nP3: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
