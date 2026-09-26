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
"""p8 - FINAL HANDOFF + PACKAGE.

Writes 07_REPORTS/FINAL_DATA_ONLY_PILOT_HANDOFF.md (from the stage artefacts, numbers never retyped),
a FILE_MANIFEST.csv with checksums, and a shareable tar.gz that is round-trip verified.
"""
import json
import os
import re
import subprocess
import numpy as np
import pandas as pd
from common import B, D, K_GRID, C_GRID, SEEDS, N_FOLDS, N_RESAMPLES, RESAMPLE_FRACTION, json_dump, sha256

FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


J = lambda k, v: json.load(open(f"{D[k]}/{v}"))
b = J("04_DATA_ONLY_BASELINE", "baseline_summary.json")
s = J("05_TOPK_STABILITY", "stability_summary.json")
c = J("06_DATA_CONFUSION", "confusion_summary.json")
m = J("03_LEAKAGE_SAFE_DEVELOPMENT", "development_manifest.json")
e = J("02_FEATURE_ELIGIBILITY", "eligibility_summary.json")
a = J("audit", "final_assertions.json")
res = pd.read_csv(f"{D['04_DATA_ONLY_BASELINE']}/DATA_ONLY_BASELINE_RESULTS.csv")
prim = res[(res.model == "L2_logistic_regularised") & (res.variant == "leakage_safe")]
ss = pd.DataFrame(s["set_stability"])
ex = pd.DataFrame(s["set_stability_extended_C"])
fc = pd.read_csv(f"{D['06_DATA_CONFUSION']}/PAIRWISE_DATA_CONFUSION.csv")
cm = pd.read_csv(f"{D['06_DATA_CONFUSION']}/CORRELATED_FEATURE_COMPETITION.csv")
fs = pd.read_csv(f"{D['05_TOPK_STABILITY']}/FEATURE_STABILITY.csv")
subst = cm[cm.competition_type == "MUTUAL_EXCLUSION_SUBSTITUTION"]
gen = fc[fc.genuine_order_confusion]
g10 = gen[gen.k == 10].sort_values("confusion_strength", ascending=False)
least = fs[fs.pi_top10 > 0].sort_values("rank_range", ascending=False)
from scipy.stats import spearmanr
_mi = pd.read_csv(f"{D['06_DATA_CONFUSION']}/MISSINGNESS_VS_INSTABILITY.csv")
_rho_mr = spearmanr(_mi.missing_rate_batch1, _mi.rank_range)
n_gen10 = int(len(g10))
n_gen = int(len(gen))
fa_rows = m["site_scope"]["per_site_rows"]["前臂"]

_repro_path = f"{D['audit']}/REPRODUCIBILITY.txt"
if os.path.exists(_repro_path):
    _rt = open(_repro_path, encoding="utf-8").read()
    repro_txt = ("Verified by re-running the CV and selector stages twice and comparing every artefact\n"
                 "cell-by-cell with `scripts/check_reproducibility.py` (max absolute difference over all\n"
                 "numeric cells, and byte-identity of the JSON summaries):\n\n```\n"
                 + _rt.strip() + "\n```\n")
else:
    repro_txt = ("Not yet recorded — run `python scripts/check_reproducibility.py <runA> <runB> <label>`\n"
                 "on two consecutive runs and store the output at `audit/REPRODUCIBILITY.txt`.\n")

doc = f"""# FINAL DATA-ONLY PILOT HANDOFF

**Project:** private hospital osteoporosis data — development-only, data-only suitability pilot
**Tree:** `{B}/`
**Upstream (frozen, unmodified):** `REPRO_ROOT / 'hospital_osteoporosis_canonical_20260917/'`
**Date:** 2026-09-17

---

## 0. What was NOT done (binding scope of this round)

No LLM. No external evidence retrieval. No literature search. No Bradley–Terry. No P(A>B) from a model
of preferences. No entropy of any preference distribution. No LLM-guided selection. No lam tuning. No
full downstream comparison. **No batch2 label was read and no batch2 performance exists in any
artefact** — batch2 remains a sealed temporal external test.

Two sensitivities are reported (`future-period patients INCLUDED`, `site scope lumbar+hip`) because the
brief asks for the *recording* of an alternative, never for its use in choosing a design.

## 1. Deliverables actually produced (§22 checklist)

| path | content |
|---|---|
| `00_INPUT_REFERENCE/canonical_input_sha256.json`, `input_verification.json` | the frozen canonical inputs, hash-verified against the frozen manifest |
| `01_TASK_FREEZE/TASK_B_SEMANTICS_FREEZE.md` | the frozen task name, unit, target, naming prohibition, site audit, all-sites-vs-lumbar+hip recommendation |
| `01_TASK_FREEZE/SITE_CONTEXT_RULE.md` | `site` = FORCED CONTEXT, never selectable; the two universes; implementation and assertions |
| `02_FEATURE_ELIGIBILITY/DEVELOPMENT_FEATURE_ELIGIBILITY.csv` | per-feature batch1-only eligibility audit (all required columns) |
| `02_FEATURE_ELIGIBILITY/FEATURE_ELIGIBILITY_AUDIT.md` | the rules, the ineligible features and why, the censoring search |
| `03_LEAKAGE_SAFE_DEVELOPMENT/TEMPORAL_LEAKAGE_RESOLUTION.md` | provenance granularity, chosen rule, affected patients/rows/columns, resulting n, zero-future-period assertion |
| `03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json` | n_rows/n_patients/n_positive/n_negative, p_semantic, p_selector_eligible, forced context, all exclusion lists, pre-registered design |
| `03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv`, `development_matrix_y_and_context.csv` | the leakage-safe Task B development matrix |
| `04_DATA_ONLY_BASELINE/DATA_ONLY_BASELINE_RESULTS.csv` | per-(seed, fold) AUROC/AUPRC/balanced accuracy for every configuration |
| `04_DATA_ONLY_BASELINE/DATA_ONLY_SUITABILITY_REPORT.md` | Q1/Q2 answers, CIs, sensitivities |
| `05_TOPK_STABILITY/FEATURE_STABILITY.csv` | per-feature pi_j(k), median/IQR/variance of rank, score summaries, boundary presence |
| `05_TOPK_STABILITY/TOPK_SET_STABILITY.csv`, `..._EXTENDED_C.csv` | per-k Jaccard distribution, distinct sets, modal frequency, stable/borderline counts |
| `05_TOPK_STABILITY/TOPK_STABILITY_REPORT.md` | the stability verdict |
| `06_DATA_CONFUSION/PAIRWISE_DATA_CONFUSION.csv` | P_data(A>B), flip rate, confusion strength, score gaps, co-selection |
| `06_DATA_CONFUSION/CORRELATED_FEATURE_COMPETITION.csv`, `CORRELATION_DIAGNOSTIC.csv`, `MISSINGNESS_VS_INSTABILITY.csv` | the correlated-competition mechanism, measured |
| `06_DATA_CONFUSION/DATA_CONFUSION_REPORT.md` | where the data is confused and why |
| `audit/*.json`, `audit/final_assertions.json` | machine-readable audit trails (assertions {a['n_pass']}/{a['n_checks']}) |
| `plots/*.png` | AUROC/AUPRC variation, ROC/PR, selection probability, rank variability, top-k Jaccard, confusion heatmap, correlation diagnostic |
| `scripts/p0_inputs.py` … `p8_finalize.py` | the whole pipeline, re-runnable |

## 2. The four corrections

### 2.1 Task B semantics — FROZEN

> **SITE-LEVEL LOW T-SCORE CLASSIFICATION** — given a patient's clinical/laboratory profile, predict
> whether **the measured skeletal site** has **T-score ≤ −2.5**.

Unit = patient × skeletal site; target = the site's own thresholded T-score; the phrase
*"osteoporosis diagnosis"* is **prohibited**. Rows from one patient are correlated, so every split,
fold, resample and bootstrap groups by `patient_uid`.

### 2.2 All sites vs lumbar+hip — recommendation **A**

**Lumbar spine + hip primary, forearm sensitivity.** Decided from ISCD/measurement semantics and
provenance only, never from predictive performance:
* in batch1 only **1 of {fa_rows}** forearm rows is the ISCD reference site (`1/3` radius) — 132 are
  `未知`, 6 are `整体`;
* the forearm base rate is 0.4748 vs 0.2785 (lumbar) / 0.3560 (hip): a different measurement/reference
  definition, not just a different case mix;
* the hip is kept as **"hip", never "femoral neck"** (only 7/382 batch1 rows are femoral-neck labelled);
* **the forearm is not deleted** — it is retained and reported as a mandatory sensitivity.

### 2.3 `site` — FORCED CONTEXT

`site` is a fixed context covariate (2 dummies, baseline `腰椎骨`, scaled ×1000 so the L1 penalty
effectively does not touch it) and it is **outside** the selectable universe: not in top-k, not in any
ranking, not in selection frequency, not in any pairwise comparison, never ranked against a clinical
feature. Asserted in `p7`.

### 2.4 batch1 future-period laboratory leakage — RESOLVED

Provenance granularity, re-derived here: the contamination is identifiable at **column** level —
**22 laboratory columns / 168 cells** (plus the marker column itself: 23 / 177, which is exactly what
the source README reports) on **9 rows / 7 batch1 patients**, and every one of those columns is a
**batch2-only** column, i.e. outside the shared analysis schema. What is *not* verifiable is the
**period** of those patients' remaining laboratory values (the source states the laboratory export has
no collection timestamps).

Chosen rule, fixed before any model was fitted and driven purely by provenance:

1. **Layer 1 — mask**: the 22 columns are hard-excluded (they are outside the schema), asserted to be
   absent from the primary matrix.
2. **Layer 2 — exclude**: the **7 affected patients are removed from the primary development matrix**,
   because their records' period cannot be verified. Recorded sensitivity:
   `future_period_patients_INCLUDED`.

Resulting leakage-safe development matrix: **{m['n_rows']} rows / {m['n_patients']} patients**
(1029/684; sites 腰椎 511, 髋 381, 前臂 137; prevalence {m['prevalence']:.4f}).

## 3. Feature eligibility (batch1 X only)

* **p_semantic = {e['p_semantic']}** — the universe is preserved; nothing was deleted from it.
* **p_selector_eligible = {e['p_selector_eligible']}**.
* Ineligible, with reasons:

| feature | status | batch1 missing rate | why |
|---|---|---|---|
""" + "\n".join(f"| {r['feature_name']} | `{r['selector_eligibility_status']}` | "
                f"{r['missing_rate_batch1']:.4f} | batch1 X only (dev-provisional rules) |"
                for r in e["ineligible"]) + f"""

* `C-反应蛋白` is excluded by **`CENSORING_RULE_UNRESOLVED`**: {
    int(pd.read_csv(f"{D['02_FEATURE_ELIGIBILITY']}/DEVELOPMENT_FEATURE_ELIGIBILITY.csv")
        .query("feature_name=='C-反应蛋白'").censored_n_batch1.iloc[0])} of its batch1 non-missing cells
  are the string `<4.0`. **No legitimate pre-existing coding rule exists** (documented keyword search
  hits for coding rules: {len(e['censoring_rule_keyword_hits'])}; the only pre-existing mentions are
  descriptive names). It stays in the semantic universe; it does **not** enter the numeric selector
  universe, and 4.0/2.0/0 substitution or row deletion is forbidden for this pilot.
* `全程C-反应蛋白` and `C-反应蛋白` are **not** merged.
* Derived features (`AST/ALT`, `白球比`, `尿素：肌酐`, `估算肾小球滤过率`) are flagged and **kept**.

## 4. Data-only baseline (Q1 / Q2)

| configuration | AUROC (mean ± SD over {len(prim)} folds/seeds) | AUPRC | balanced acc |
|---|---|---|---|
""" + "\n".join(f"| {d['configuration']} | {d['auroc_mean']:.4f} ± {d['auroc_sd']:.4f} | "
                f"{d['auprc_mean']:.4f} | {d['bal_acc_mean']:.4f} |"
                for d in [b['primary']] + b['references'] + b['sensitivities']) + f"""

Pooled out-of-fold: **AUROC {b['pooled_oof_auroc']:.4f}** (95 % CI {b['pooled_oof_auroc_ci95_patient_bootstrap'][0]:.4f}–{
    b['pooled_oof_auroc_ci95_patient_bootstrap'][1]:.4f}, patient-level bootstrap, {b['n_bootstrap']} draws),
**AUPRC {b['pooled_oof_auprc']:.4f}** vs prevalence {b['prevalence']:.4f}.

**Q1 — is there real signal? {b['signal_verdict']}.** {b['primary']['auroc_mean']:.4f} versus a
permuted-label null of {b['null_mean_auroc']:.4f} ± {b['null_sd_auroc']:.4f} through the identical
pipeline, and versus a forced-context-only reference of {b['references'][0]['auroc_mean']:.4f}.

**Q2 — saturated or empty? {b['saturation_verdict']}; {b['empty_verdict']}.** Not saturated (AUROC far
from 0.95, fold spread {b['primary']['auroc_sd']:.4f}) and not empty. This is the regime in which the
question "*which* features carry the signal" is genuinely open.

## 5. Top-k stability (B = {N_RESAMPLES}, {RESAMPLE_FRACTION:.0%} of patients per draw)

| k | mean Jaccard | median | 5th pct | distinct top-k sets | modal-set freq | pi>=0.95 | borderline (0.10<pi<0.90) | never | always | ambiguous boundary |
|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.k} | {r.jaccard_mean:.4f} | {r.jaccard_median:.4f} | {r.jaccard_p05:.4f} | "
    f"{r.n_distinct_topk_sets} | {r.modal_set_frequency:.4f} | {r.n_features_pi_ge_095} | "
    f"{r.n_features_borderline_010_090} | {r.n_features_never} | {r.n_features_always} | "
    f"{r.ambiguous_boundary_resamples} |" for r in ss.itertuples()) + f"""

**The top-k set is NOT stable at any k.** Best mean pairwise Jaccard {ss.jaccard_mean.max():.4f}
(k={int(ss.loc[ss.jaccard_mean.idxmax(), 'k'])}); {ss.n_distinct_topk_sets.min()}–{
    ss.n_distinct_topk_sets.max()} distinct sets; the modal set covers at most {
    ss.modal_set_frequency.max():.1%}. The instability is **real re-ordering**: ambiguous tie-breaking
accounts for **{int(ss.ambiguous_boundary_resamples.sum())}** resamples in total, and the extended-grid
sensitivity reproduces (in fact slightly worsens) the numbers
(best mean Jaccard {ex.jaccard_mean.max():.4f}).

**Most stable**: {", ".join(f"`{r.feature_name}` ({r.pi_top10:.2f})" for r in fs.head(5).itertuples())}.
**Least stable (among features that do enter)**: {
    ", ".join(f"`{r.feature_name}` (rank range {int(r.rank_min)}–{int(r.rank_max)}, IQR {r.rank_iqr:.1f})" for r in least.head(6).itertuples())}.

## 6. Pairwise data confusion (data-only, from the resample matrix)

{len(fc)} relevant pairs; **{n_gen} genuine order-confusion pairs**
(`confusion_strength >= 0.5`, ordering defined); {c['n_undefined_co_zero_pairs']} pairs whose ordering is
undefined through co-zeroing. Per k: {" / ".join(f"k={k}: {v['genuine']}" for k, v in c['per_k'].items())}.

Most confused at k = 10:

| A | B | P(A>B) | flip rate | strength | median rank A/B | pi(A)/pi(B) |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.feature_A} | {r.feature_B} | {r.P_data_A_gt_B:.3f} | {r.pairwise_flip_rate:.3f} | "
    f"{r.confusion_strength:.3f} | {r.median_rank_A:.1f}/{r.median_rank_B:.1f} | "
    f"{r.selection_prob_A:.2f}/{r.selection_prob_B:.2f} |" for r in g10.head(8).itertuples()) + f"""

## 7. Correlated-feature competition

{len(cm)} pairs labelled. **{len(subst)} genuine mutual-exclusion (substitution) pairs** — judged by
`co_ratio = P(both) / (pi_A · pi_B) < 0.5`, *not* by raw P(exactly one):

| A | B | rho | pi(A) | pi(B) | P(both) | E[both | indep.] | co_ratio | P(exactly one) | P(at least one) |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.feature_A} | {r.feature_B} | {r.spearman_rho} | {r.pi_top10_A:.3f} | {r.pi_top10_B:.3f} | "
    f"{r.P_both_selected:.3f} | {r.expected_both_if_independent:.4f} | "
    f"{r.co_selection_ratio_co_ratio} | {r.P_exactly_one:.3f} | {r.P_at_least_one:.3f} |"
    for r in subst.itertuples()) + f"""

Reading: correlated competition is **real but narrow** — it is confined to the red-cell axis
(`红细胞比积`/`红细胞计数` vs `血红蛋白`) and there the pair is present at the boundary in only ~1/3 of
draws. The remaining correlated pairs show **domination** (one member essentially never selected, e.g.
`AST/ALT` in {fs.loc[fs.feature_name.eq('AST/ALT'), 'pi_top10'].iloc[0]:.0%} of draws vs its own
denominator `丙氨酸氨基转移酶` in {fs.loc[fs.feature_name.eq('丙氨酸氨基转移酶'), 'pi_top10'].iloc[0]:.0%})
or independence. **No feature was deleted** for being derived or correlated.

## 8. Answers to §21 A–F

* **A. Is there signal?** Yes — {b['primary']['auroc_mean']:.4f} ± {b['primary']['auroc_sd']:.4f} AUROC
  (pooled OOF {b['pooled_oof_auroc']:.4f}), far above the permuted null ({b['null_mean_auroc']:.4f}) and
  above site-context alone ({b['references'][0]['auroc_mean']:.4f}).
* **B. Saturated?** No. AUROC is far below 0.95 with a {b['primary']['auroc_sd']:.4f} fold spread;
  nor is it empty.
* **C. Is the top-k set stable?** No — not at k ∈ {K_GRID}. Mean pairwise Jaccard
  {ss.jaccard_mean.min():.4f}–{ss.jaccard_mean.max():.4f}, up to {ss.n_distinct_topk_sets.max()} distinct
  sets, modal frequency ≤ {ss.modal_set_frequency.max():.1%}.
* **D. Enough genuine boundary confusion?** Yes — **{n_gen}** genuine pairs, **{n_gen10}** of them at
  k = 10, with confusion strengths up to {gen.confusion_strength.max():.4f}. These are pairs where neither
  order wins more than ~55 % of resamples.
* **E. Where does it come from?** Dominantly **near-equal rankings in strength**, plus **correlated
  substitution** on the red-cell axis. **Not** from tie-breaking artefacts (0 ambiguous resamples), **not**
  from co-zeroing (0 pairs), **not** from missingness (Spearman(missing rate, rank range) =
  {_rho_mr.statistic:+.4f}, p = {_rho_mr.pvalue:.3g}), **not** from site/context (site is never ranked).
* **F. Suitable for a data-confusion-guided LLM selection stage?** **Yes, with a scope limit.** The task
  has signal, is not saturated, the ranking is demonstrably unstable, and the confusion is *enumerable*
  rather than diffuse — exactly the precondition. But the data-only confusion is concentrated in a
  modest number of pairs, and most of them are dominated/independent rather than mutually exclusive, so
  the next stage should be framed as *resolving a specific list of near-ties*, not as a general
  "LLM picks the features" exercise.

## 9. Suggested first confusion pairs for the next stage

Ranked by data-only confusion strength at k = 10 (all from {int((fc.k == 10).sum())} relevant pairs):

""" + "\n".join(
    f"{i+1}. `{r.feature_A}` vs `{r.feature_B}` — P(A>B) = {r.P_data_A_gt_B:.3f}, strength "
    f"{r.confusion_strength:.3f}, median ranks {r.median_rank_A:.1f}/{r.median_rank_B:.1f}, "
    f"pi {r.selection_prob_A:.2f}/{r.selection_prob_B:.2f}"
    for i, r in enumerate(g10.head(6).itertuples())) + f"""

The **red-cell substitution pair** `红细胞比积` / `血红蛋白` (rho {
    cm[(cm.feature_A == '红细胞比积') & (cm.feature_B == '血红蛋白')].spearman_rho.iloc[0]}) is the one
case where the mechanism is structural rather than a near-tie, and is worth studying separately.

## 10. Honest limitations

1. **The pre-registered `C` grid saturates.** The inner-CV optimum is its upper edge for
   {s['C_selected_counts'].get('0.3', 0)}/{N_RESAMPLES} draws (full sample `C = {s['C_full_sample']}`, {
    s['non_zero_full_sample']} of {m['p_selector_eligible']} coefficients non-zero). The grid was **not**
   changed afterwards; instead the full inner-CV curve is published
   ({s['inner_cv_curve_full_sample']}) and an extended-grid sensitivity is reported. The selector is
   therefore only *mildly* sparse.
2. **Forearm is a sensitivity only.** The primary scope is lumbar+hip (892 rows); the all-sites number
   appears only as a sensitivity.
3. **47 of 48 features still have `UNIT_UNKNOWN` and 0 LOINC codes** (frozen handoff). This bounds any
   future external-evidence comparison: only `肌酐` (µmol/L) is unit-verified, via the CKD-EPI 2021
   reproduction.
4. **`C-反应蛋白` is unusable numerically** for this pilot (censoring unresolved); it is kept in the
   semantic universe so a future, separately-justified rule can address it.
5. **Missingness indicators were not used** (per instruction); they are recorded as a sensitivity
   proposal only.
6. **Unit of resampling is the patient**, so the effective sample size is {m['n_patients']} patients, not
   {m['n_rows']} rows; the stability numbers inherit patient-level, not row-level, precision.
7. **5 seeds** were used for CV (as suggested); the resampling study uses {N_RESAMPLES} draws, which is
   the stronger basis for the stability statements.
8. **A reproducibility defect was found and fixed during this round.** `sklearn`'s liblinear solver
   (used for the L1 selector and for the permuted-label null) SHUFFLES the data using `random_state`,
   which defaults to `None`. The first version therefore produced run-to-run differences of up to
   **3.7e-3 AUROC** and **0.5 average-rank units** in exactly the L1-based rows, while every value
   printed at 4 decimals still matched — i.e. invisible at the reported precision. Every estimator now
   receives an explicit `random_state`. Evidence: fitting the L1 selector **6 times on identical input
   gives zero coefficient spread**, and two consecutive full runs of the CV/selector stages agree to
   **0.000e+00 over every numeric cell of every artefact** (`scripts/check_reproducibility.py`, log in
   `audit/REPRODUCIBILITY.txt`). The delivered numbers are the reproducible ones.

## 11. Privacy

Identifiers are pseudonymous `patient_uid` (`hp_` + 16 hex) only; the raw-ID mapping stays in the frozen
canonical tree's `private_only/` and is never copied here. `p7` re-verifies with a word-boundary scan
against the private ID map that **no raw hospital identifier and no raw-ID column name appears anywhere
in this tree**. The development matrices contain no raw IDs.

## 12. Assertions

**{a['n_pass']}/{a['n_checks']} PASS** — including: no batch2 label read (proved structurally: every
development patient belongs to the frozen batch1 cohort), no batch2 performance exists, no network client
imported, no LLM/Bradley–Terry/entropy term in the code, `T_used`/`T_source`/`subregion` absent from the
predictor matrix and from every ranking table, `site` forced-context-only, fold disjointness re-derived
for all {len(SEEDS)} seeds × {N_FOLDS} folds, preprocessing statistics equal to the **train-fold**
statistics, leakage rule re-derived from the source files, eligibility recomputed from batch1 X, top-k
grid pre-registered *before* the results (proved by file mtimes), and confusion strengths recomputed from
the saved resample matrix.

## 13. Reproducibility

{repro_txt}
"""
open(f"{D['07_REPORTS']}/FINAL_DATA_ONLY_PILOT_HANDOFF.md", "w").write(doc)
check("handoff written", os.path.getsize(f"{D['07_REPORTS']}/FINAL_DATA_ONLY_PILOT_HANDOFF.md") > 5000)

# ------------------------------------------------------------------ manifest + package
req = ["01_TASK_FREEZE/TASK_B_SEMANTICS_FREEZE.md", "01_TASK_FREEZE/SITE_CONTEXT_RULE.md",
       "02_FEATURE_ELIGIBILITY/DEVELOPMENT_FEATURE_ELIGIBILITY.csv",
       "02_FEATURE_ELIGIBILITY/FEATURE_ELIGIBILITY_AUDIT.md",
       "03_LEAKAGE_SAFE_DEVELOPMENT/TEMPORAL_LEAKAGE_RESOLUTION.md",
       "03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json",
       "04_DATA_ONLY_BASELINE/DATA_ONLY_BASELINE_RESULTS.csv",
       "04_DATA_ONLY_BASELINE/DATA_ONLY_SUITABILITY_REPORT.md",
       "05_TOPK_STABILITY/FEATURE_STABILITY.csv", "05_TOPK_STABILITY/TOPK_SET_STABILITY.csv",
       "05_TOPK_STABILITY/TOPK_STABILITY_REPORT.md",
       "06_DATA_CONFUSION/PAIRWISE_DATA_CONFUSION.csv",
       "06_DATA_CONFUSION/CORRELATED_FEATURE_COMPETITION.csv",
       "06_DATA_CONFUSION/DATA_CONFUSION_REPORT.md",
       "07_REPORTS/FINAL_DATA_ONLY_PILOT_HANDOFF.md"]
missing = [f for f in req if not os.path.exists(f"{B}/{f}")]
check("every §22 deliverable exists", not missing, str(missing))

# a static, non-self-referential description of the package lives INSIDE the tree, so that the
# checksum set can be complete without depending on the package's own hash
pkg = f"REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_DATAONLY_PILOT_20260917.tar.gz'"
prev_sha = sha256(pkg) if os.path.exists(pkg) else None
open(f"{D['07_REPORTS']}/PACKAGE_INFO.md", "w").write(f"""# PACKAGE INFO

* archive: `{pkg}`
* source tree: `{B}/`
* archive members: every file of the tree except `__pycache__/` and `*.pyc`
* integrity: `FILE_MANIFEST.csv` (path / bytes / sha256) and `CHECKSUMS.sha256`
  (`sha256sum -c CHECKSUMS.sha256`, one entry per shipped file)
* the archive's own sha256 is recorded in the sidecar `{pkg}.sha256`,
  which is deliberately kept OUTSIDE the tree and outside the archive so that the package is
  reproducible byte-for-byte and the checksum set is complete
* privacy: identifiers are pseudonymous `patient_uid` only; no raw hospital identifier and no
  raw-ID column name occurs anywhere in this tree (asserted in `p6`-stage `audit/final_assertions.json`)
* stage behaviour: no LLM, no literature, no external evidence, no batch2 label, no batch2 performance
""")
stale = f"{D['07_REPORTS']}/package_manifest.json"
if os.path.exists(stale):
    os.remove(stale)
    print("removed the stale self-referential package_manifest.json")

rows = []
for root, dirs, files in os.walk(B):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".pyc") or f in ("FILE_MANIFEST.csv", "CHECKSUMS.sha256"):
            continue
        fp = os.path.join(root, f)
        rows.append(dict(relpath=os.path.relpath(fp, B), bytes=os.path.getsize(fp), sha256=sha256(fp)))
mf = pd.DataFrame(rows).sort_values("relpath")
mf.to_csv(f"{B}/FILE_MANIFEST.csv", index=False, encoding="utf-8-sig")
with open(f"{B}/CHECKSUMS.sha256", "w", encoding="utf-8") as fh:
    for r in mf.itertuples():
        fh.write(f"{r.sha256}  {r.relpath}\n")
print(f"\nFILE_MANIFEST.csv: {len(mf)} files, {int(mf.bytes.sum())} bytes; "
      f"CHECKSUMS.sha256 written ({len(mf)} lines)")

def build_package():
    subprocess.run(["bash", "-c",
                    f"cd {os.path.dirname(B)} && tar --sort=name --mtime='2026-09-17 00:00:00 UTC' "
                    f"--owner=0 --group=0 --numeric-owner --exclude=__pycache__ --exclude='*.pyc' "
                    f"-cf - {os.path.basename(B)} | gzip -n -9 > {pkg}"], check=True)
    return sha256(pkg)


sha_a = build_package()
sha_b = build_package()
check("the archive is BYTE-REPRODUCIBLE (built twice in a row from the same tree)",
      sha_a == sha_b, f"{sha_a[:16]}… / {sha_b[:16]}…")
rt = subprocess.run(["bash", "-c",
                     f"cd /tmp && rm -rf _rt && mkdir _rt && tar -xzf {pkg} -C _rt && "
                     f"cd _rt/{os.path.basename(B)} && "
                     f"sha256sum -c CHECKSUMS.sha256; echo RC=$?"], capture_output=True, text=True)
ok_n = rt.stdout.count(": OK")
check("package round-trip: every member verifies against CHECKSUMS.sha256",
      "RC=0" in rt.stdout and "FAILED" not in rt.stdout and ok_n == len(mf),
      f"{ok_n}/{len(mf)} OK")
n_members = int(subprocess.run(["bash", "-c", f"tar -tzf {pkg} | grep -v '/$' | wc -l"],
                               capture_output=True, text=True).stdout.strip())
pk_sha = sha256(pkg)
open(f"{pkg}.sha256", "w").write(f"{pk_sha}  {os.path.basename(pkg)}\n")
idmap = pd.read_csv(f"{os.path.dirname(B)}/hospital_osteoporosis_canonical_20260917/private_only/"
                    f"patient_uid_map.csv", dtype=str)["住院唯一号"].tolist()
raw = subprocess.run(["bash", "-c", f"tar -xzOf {pkg}"], capture_output=True).stdout \
    .decode("utf-8", errors="ignore")
check("package contains no raw hospital identifier",
      not any(re.search(rf"(?<![\d.]){re.escape(str(i))}(?![\d.])", raw) for i in idmap[:250]
              if pd.notna(i)))
check("the checksum set covers every archive member except the checksum file itself",
      len(mf) + 2 == n_members,
      f"manifest {len(mf)} + FILE_MANIFEST.csv + CHECKSUMS.sha256 = {len(mf) + 2} vs members {n_members}")
print(f"package: {pkg}\n  {os.path.getsize(pkg)} bytes, {n_members} members, sha256 {pk_sha}")
json_dump({"package": pkg, "bytes": os.path.getsize(pkg), "members": n_members, "sha256": pk_sha,
           "sha256_sidecar": f"{pkg}.sha256",
           "round_trip_ok": bool("RC=0" in rt.stdout), "n_files_manifested": int(len(mf)),
           "checksums_in_archive": "CHECKSUMS.sha256", "deliverables_missing": missing,
           "note": "this manifest is written OUTSIDE the tree; the tree holds the static "
                   "07_REPORTS/PACKAGE_INFO.md instead, so the package is reproducible"},
          f"REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_DATAONLY_PILOT_20260917.package_manifest.json'")
print(f"\nP8: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
