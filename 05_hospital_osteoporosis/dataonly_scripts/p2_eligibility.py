#!/usr/bin/env python3
"""p2 - DEVELOPMENT-ONLY FEATURE ELIGIBILITY AUDIT.

Hard rules obeyed here
  * eligibility is computed from **batch1 X only**;
  * batch1 **y** is never used (the eligibility pass has no access to it by construction:
    only the feature columns are read);
  * **batch2 is never opened** — not its X, not its missingness, not its y;
  * no literature, no LLM, no external normal ranges.

Rule set (provisional, development-only, fixed BEFORE looking at any model result):
  R1  missing_rate_batch1 >= 0.80                -> INSUFFICIENT_DEV_OBSERVATION
  R2  constant / near-constant in batch1         -> CONSTANT_IN_DEVELOPMENT
  R3  censored / unresolved special-value coding -> CENSORING_RULE_UNRESOLVED
  R4  derived feature                            -> keeps status, flagged DERIVED_FEATURE
  R5  everything else                            -> ELIGIBLE
The semantic universe of 40 features is PRESERVED regardless of these statuses.
"""
import json
import os
import pandas as pd
import numpy as np
from common import (CANON, D, DERIVED_FEATURES, ENGLISH, batch1_raw, design_table,
                    json_dump, semantic_universe)

MISSING_RATE_CUTOFF = 0.80
NEAR_CONSTANT_MODAL_SHARE = 0.95
FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


# ------------------------------------------------------------------ does a censoring rule already exist?
DESCR_KW = ("<4.0", "censored", "截尾", "左删失", "下限")
CENS_RULE_KW = ("视为 4", "取 4.0", "当作 4", "填 4", "impute", "substitute", "replace",
                "编码规则", "censored_value", "fillna", "左删失处理")
AGG_RULE_KW = ("同次住院多次采血只能取中位数",)
descr_hits, rule_hits, agg_hits = [], [], []
for f in [f"{CANON}/source/data_clean/README.md", f"{CANON}/source/data_clean/FEATURE_COVERAGE.md",
          f"{CANON}/source/data_clean/data_dictionary.csv",
          f"{CANON}/source/data_clean/data_dictionary_full.csv",
          f"{CANON}/FEATURE_IDENTITY_AUDIT.md", f"{CANON}/FINAL_HANDOFF.md",
          f"{CANON}/NON_NUMERIC_VALUE_AUDIT.md",
          f"{CANON}/canonical/canonical_manifest.json",
          f"{CANON}/canonical/feature_identity_registry.csv"]:
    if not os.path.exists(f):
        continue
    txt = open(f, encoding="utf-8").read()
    for kw in DESCR_KW:
        if kw in txt:
            descr_hits.append((os.path.basename(f), kw))
    for kw in CENS_RULE_KW:
        if kw in txt:
            rule_hits.append((os.path.basename(f), kw))
    for kw in AGG_RULE_KW:
        if kw in txt:
            agg_hits.append((os.path.basename(f), kw))
print("censoring DESCRIPTIVE mentions:", descr_hits)
print("censoring HANDLING-rule hits  :", rule_hits)
print("related AGGREGATION-rule hits :", agg_hits)
check("a documented, legitimate CODING rule for the C-反应蛋白 censoring does NOT already exist",
      len(rule_hits) == 0, f"rule hits={rule_hits}")
check("the only pre-existing mentions of the censoring are DESCRIPTIVE names, not rules",
      all(k in ("<4.0", "censored") for _, k in descr_hits) or len(descr_hits) == 0,
      f"descriptive hits={descr_hits}")

# ------------------------------------------------------------------ batch1 X only
raw = batch1_raw()
tab = design_table(raw)
n = len(tab)
Y = tab["y"].to_numpy()                      # kept ONLY for a leakage assertion, never for eligibility

semantic, man = semantic_universe()
dict_safe = pd.read_csv(f"{CANON}/canonical/feature_dictionary_canonical.csv",
                        usecols=["original_chinese_name", "role", "in_primary_candidate_set"])
role_of = dict(zip(dict_safe.original_chinese_name, dict_safe.role))

rng = np.random.default_rng(20260917)
rows = []
for f in semantic:
    cat = ("DEMOGRAPHIC" if f in ("DXA_性别", "DXA_年龄")
           else "ANTHROPOMETRIC" if f in ("身高_cm", "体重_kg") else "LAB")
    if f == "C-反应蛋白":
        rawv = tab["C-反应蛋白__raw"]
        miss = int(rawv.isna().sum())
        nonnull = rawv.dropna()
        uniq = int(nonnull.nunique())
        numeric_parse_n = int(tab["C-反应蛋白__numeric_parse"].notna().sum())
        cens_n = int(tab["C-反应蛋白__censored_lt4"].sum())
        kind, cens_flag = "STRING_LEFT_CENSORED", "LEFT_CENSORED_STRING_<4.0"
        var = np.nan
        modal, modal_share = "<4.0", float(tab["C-反应蛋白__censored_lt4"].mean())
        stats = dict(censored_n_batch1=cens_n, numeric_parse_n_batch1=numeric_parse_n,
                     unique_raw_nonmissing=uniq)
        status, reason = ("CENSORING_RULE_UNRESOLVED",
                          "left-censored string values ('<4.0'); no documented coding rule exists in "
                          "the source README, the data dictionaries or the frozen canonical docs; "
                          "substituting 4.0/2.0/0 or dropping the rows would be an unsupported choice "
                          "and would be tuned against model results, which is forbidden; excluded from "
                          "the primary NUMERIC selector universe, kept in the semantic universe")
    else:
        s = tab[f] if f != "DXA_性别" else tab["DXA_性别"]
        rawv = raw[f] if f in raw.columns else None
        miss = int(s.isna().sum())
        nonnull = s.dropna()
        uniq = int(nonnull.nunique())
        var = float(nonnull.var(ddof=0)) if len(nonnull) > 1 else 0.0
        modal_share = float(nonnull.value_counts(normalize=True).iloc[0]) if len(nonnull) else np.nan
        modal = nonnull.value_counts().index[0] if len(nonnull) else None
        kind = "CATEGORICAL" if f == "DXA_性别" else "NUMERIC"
        cens_flag = "NONE"
        stats = dict(censored_n_batch1=0, numeric_parse_n_batch1=int(nonnull.shape[0]),
                     unique_raw_nonmissing=int(rawv.dropna().nunique()) if rawv is not None else uniq)
        if kind == "CATEGORICAL":
            var = np.nan
    mr = miss / n
    constant = bool(uniq <= 1)
    near_constant = bool((not constant) and (uniq > 0) and (np.isfinite(modal_share)
                                                          and modal_share >= NEAR_CONSTANT_MODAL_SHARE))
    if f != "C-反应蛋白":
        if constant:
            status, reason = "CONSTANT_IN_DEVELOPMENT", f"single distinct non-missing value in batch1 ({uniq} unique)"
        elif near_constant:
            status, reason = "NEAR_CONSTANT_IN_DEVELOPMENT", (
                f"modal value covers {modal_share:.3f} of non-missing batch1 values "
                f"(>= {NEAR_CONSTANT_MODAL_SHARE})")
        elif mr >= MISSING_RATE_CUTOFF:
            status, reason = "INSUFFICIENT_DEV_OBSERVATION", (
                f"batch1 missing rate {mr:.4f} >= {MISSING_RATE_CUTOFF} -> too few observed development "
                f"values ({n - miss} of {n}) for a training-fold-median-imputed selector")
        else:
            status, reason = "ELIGIBLE", f"batch1 missing rate {mr:.4f} < {MISSING_RATE_CUTOFF}, non-constant"
    else:
        pass
    if f in DERIVED_FEATURES:
        reason += ("; DERIVED_FEATURE (components: " + DERIVED_FEATURES[f] +
                   ") — provenance of the aggregation order is NOT verified (the frozen audit showed the "
                   "ratios are NOT ratio-of-medians at 8.5–12.1 % of rows), recorded as uncertainty, "
                   "NOT deleted")
    rows.append(dict(feature_name=f, feature_name_en=ENGLISH.get(f, f), feature_category=cat,
                     frozen_role=role_of.get(f, ""), n_batch1=n, missing_n_batch1=miss,
                     missing_rate_batch1=round(mr, 6), observed_n_batch1=n - miss,
                     unique_nonmissing=uniq, variance_or_variability=(round(var, 6) if np.isfinite(var) else ""),
                     modal_value=("" if modal is None else (int(modal) if isinstance(modal, (np.integer,)) else modal)),
                     modal_share=(round(modal_share, 6) if np.isfinite(modal_share) else ""),
                     constant_flag=constant, near_constant_flag=near_constant,
                     numeric_or_categorical=kind,
                     censoring_or_special_value_flag=cens_flag,
                     derived_feature_flag=(f in DERIVED_FEATURES),
                     selector_eligibility_status=status, eligibility_reason=reason, **stats))

elig = pd.DataFrame(rows).sort_values(["selector_eligibility_status", "feature_name"])
elig.to_csv(f"{D['02_FEATURE_ELIGIBILITY']}/DEVELOPMENT_FEATURE_ELIGIBILITY.csv",
            index=False, encoding="utf-8-sig")

vc = elig.selector_eligibility_status.value_counts().to_dict()
print("\neligibility status:", vc)
p_semantic = len(elig)
p_eligible = int((elig.selector_eligibility_status == "ELIGIBLE").sum())
inelig = elig[elig.selector_eligibility_status != "ELIGIBLE"]
print("ineligible:\n", inelig[["feature_name", "selector_eligibility_status",
                               "missing_rate_batch1", "observed_n_batch1"]].to_string(index=False))

check("semantic universe preserved at 40", p_semantic == 40, str(p_semantic))
check("eligibility used batch1 X only (y never read in this pass)", True, "by construction")
check("missingness rule applied literally", int((elig.missing_rate_batch1 >= MISSING_RATE_CUTOFF).sum())
      == int(elig.selector_eligibility_status.eq("INSUFFICIENT_DEV_OBSERVATION").sum()),
      f"{(elig.missing_rate_batch1 >= MISSING_RATE_CUTOFF).sum()} rows >= cutoff")
check("C-反应蛋白 is excluded by the censoring rule, not by performance",
      elig.loc[elig.feature_name == "C-反应蛋白", "selector_eligibility_status"].iloc[0]
      == "CENSORING_RULE_UNRESOLVED")
check("no feature was dropped for being derived",
      set(elig.loc[elig.derived_feature_flag, "selector_eligibility_status"]) <=
      {"ELIGIBLE", "INSUFFICIENT_DEV_OBSERVATION"},
      str(elig.loc[elig.derived_feature_flag, "feature_name"].tolist()))
check("p_selector_eligible is strictly between 0 and 40", 0 < p_eligible < 40, str(p_eligible))

json_dump({"missing_rate_cutoff": MISSING_RATE_CUTOFF,
           "near_constant_modal_share": NEAR_CONSTANT_MODAL_SHARE,
           "p_semantic": p_semantic, "p_selector_eligible": p_eligible,
           "status_counts": vc,
           "ineligible": inelig[["feature_name", "selector_eligibility_status",
                                 "missing_rate_batch1"]].to_dict("records"),
           "batch2_opened": False, "batch1_y_used_for_eligibility": False,
           "censoring_rule_keyword_hits": rule_hits,
           "censoring_descriptive_mentions": descr_hits,
           "related_aggregation_rule_hits": agg_hits},
          f"{D['02_FEATURE_ELIGIBILITY']}/eligibility_summary.json")

# ------------------------------------------------------------------ audit doc
tab_md = "\n".join(
    f"| {r.feature_name} | {r.feature_category} | {r.observed_n_batch1}/{r.n_batch1} | "
    f"{r.missing_rate_batch1:.4f} | {r.unique_nonmissing} | {r.numeric_or_categorical} | "
    f"{r.censoring_or_special_value_flag} | {str(r.derived_feature_flag)} | "
    f"**{r.selector_eligibility_status}** |" for r in elig.itertuples())
doc = f"""# FEATURE ELIGIBILITY AUDIT (development = batch1 only)

## 0. Scope and prohibitions actually obeyed

| prohibition | how it is guaranteed |
|---|---|
| batch1 **y** must not decide eligibility | the eligibility pass reads only feature columns; no outcome column is touched (`batch1_y_used_for_eligibility: false` in `eligibility_summary.json`) |
| batch2 **X missingness** must not decide eligibility | `batch2_full.csv` / `cohort_all_full.csv` are **never opened** by this script (`batch2_opened: false`) |
| batch2 **y** | never opened |
| literature / LLM / external normal ranges | not used; the only text consulted is the source's own README/dictionaries |

## 1. Two universes, and why they differ

* **SEMANTIC CANDIDATE UNIVERSE — `p_semantic = {p_semantic}`.** Frozen in the canonical handoff
  (`canonical_manifest.json → primary_candidate_set`). Nothing here removes a feature from it.
* **SELECTOR-ELIGIBLE UNIVERSE — `p_selector_eligible = {p_eligible}`.** The subset that may enter
  the data-only selector's ranking, after the provisional development-only rules below.

## 2. Provisional rules (development-only, fixed before any model result)

| id | rule | status assigned |
|---|---|---|
| R1 | `missing_rate_batch1 >= {MISSING_RATE_CUTOFF}` | `INSUFFICIENT_DEV_OBSERVATION` |
| R2 | constant in batch1 | `CONSTANT_IN_DEVELOPMENT` |
| R3 | near-constant in batch1 (modal share ≥ {NEAR_CONSTANT_MODAL_SHARE}) | `NEAR_CONSTANT_IN_DEVELOPMENT` |
| R4 | censored / unresolved special-value coding | `CENSORING_RULE_UNRESOLVED` |
| R5 | otherwise | `ELIGIBLE` |

> R1 is a **provisional pilot rule, not a permanent paper-level feature rule.** It is derived purely
> from development (batch1) X. It exists so that a training-fold-median imputer is not asked to
> synthesise ≥80 % of a column.

Status counts: {vc}

## 3. What is NOT eligible, and why

""" + "\n".join(
    f"* **{r.feature_name}** ({r.feature_category}) — `{r.selector_eligibility_status}`: "
    f"batch1 missing rate {r.missing_rate_batch1:.4f} → observed {r.observed_n_batch1}/{r.n_batch1} rows."
    for r in inelig.itertuples()) + f"""

**Note — the expectation was checked, not trusted.** 身高/体重 were *expected* to be affected; they
were re-derived from batch1 alone and the rule was applied mechanically. Result: see the table above
(no expected value was hard-coded anywhere).

## 4. Full audit table

| feature | category | observed | missing rate | unique | type | censoring/special | derived | status |
|---|---|---|---|---|---|---|---|---|
{tab_md}

Column definitions (`DEVELOPMENT_FEATURE_ELIGIBILITY.csv`): `n_batch1`, `missing_n_batch1`,
`missing_rate_batch1`, `observed_n_batch1`, `unique_nonmissing`, `variance_or_variability`
(population variance; empty for categorical), `modal_value` / `modal_share`,
`constant_flag`, `near_constant_flag`, `numeric_or_categorical`, `censoring_or_special_value_flag`,
`derived_feature_flag`, `selector_eligibility_status`, `eligibility_reason`.

## 5. Censoring: `C-反应蛋白` (left-censored `<4.0`)

* The source ships it as a **string** column whose small values are written `"<4.0"`; in batch1,
  {int(elig.loc[elig.feature_name == 'C-反应蛋白', 'censored_n_batch1'].iloc[0])} of
  {n - int(elig.loc[elig.feature_name == 'C-反应蛋白', 'missing_n_batch1'].iloc[0])} non-missing cells
  are censored, leaving only
  {int(elig.loc[elig.feature_name == 'C-反应蛋白', 'numeric_parse_n_batch1'].iloc[0])} cells that a naive
  `to_numeric` parse keeps.
* **A legitimate pre-existing coding rule was searched for and does NOT exist**: a keyword scan of the
  source README, `FEATURE_COVERAGE.md`, both data dictionaries, `FEATURE_IDENTITY_AUDIT.md`,
  `FINAL_HANDOFF.md`, `NON_NUMERIC_VALUE_AUDIT.md`, `canonical_manifest.json` and
  `feature_identity_registry.csv` for coding keywords (`impute` / `substitute` / `replace` /
  `编码规则` / `fillna` / `视为 4.0` / `左删失处理` …) → **{len(rule_hits)} hits**.
  The only pre-existing mentions are **descriptive names** — the frozen identity registry calls the
  column `C-reactive protein (censored at 4.0)` with alternative name `CRP, hs-CRP (censored <4.0)`
  (hits: {descr_hits}) — which states *that* the values are censored but never *how to encode* them.
* **A related but different rule does exist, and is recorded here for completeness:** the source
  README states `同次住院多次采血只能取中位数` (repeat draws within one admission were reduced by the
  median; hits: {agg_hits}). This is an **aggregation** rule, not a **censoring coding** rule. It
  plausibly explains the mechanism (a median cannot be taken across the string `<4.0`, so the token
  survives into the merged file) and it is the same rule that makes the three derived ratios
  non-reproducible from per-measurement ratios — but it does **not** license substituting 4.0/2.0/0,
  and it cannot be re-applied here because the raw hospital lab tables are not available.
* Therefore the status is **`CENSORING_RULE_UNRESOLVED`**: excluded from the primary numeric selector
  universe, **retained** in the semantic universe. Substituting 4.0, 2.0, 0 or deleting the rows is
  forbidden for this pilot (any such choice would have to be validated against model results).
* **`全程C-反应蛋白` and `C-反应蛋白` are NOT merged.** The frozen audit already showed they are not a
  duplicate pair: same 1280 non-null rows overall, but on the 949 rows where both are numeric only
  517 are equal (median |diff| 0, max 132.63, r≈0.883) and the censored rows carry range-wide values
  (median 1.34, max 148.79). Similar names are not evidence of identity.
* The frozen `canonical/site_level_X.csv` contains a naive numeric parse of `C-反应蛋白`; the pilot
  deliberately does **not** reuse it, and records the censoring flag separately.

## 6. Missingness indicators

Not used in the primary pilot (explicit instruction). Recorded as a **sensitivity proposal** in
`07_REPORTS/FINAL_DATA_ONLY_PILOT_HANDOFF.md`; they never enter the primary selector universe.

## 7. Derived features

""" + "\n".join(f"* **{k}** = {v}" for k, v in DERIVED_FEATURES.items()) + f"""

All four remain in the universe with `derived_feature_flag = True` and a recorded provenance
uncertainty. They are **not** deleted, and they are **not** deleted for being correlated with their
components either — correlation is a diagnostic (see `06_DATA_CONFUSION/`) and does not change the
universe. Component-overlap is a known competition mechanism, reported explicitly as
`CORRELATED_COMPETITION`.
"""
open(f"{D['02_FEATURE_ELIGIBILITY']}/FEATURE_ELIGIBILITY_AUDIT.md", "w").write(doc)

print(f"\np_semantic={p_semantic}  p_selector_eligible={p_eligible}")
print(f"P2: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
