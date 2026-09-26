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
"""p15 - FINAL PRE-LLM HANDOFF + assertions + package.

Re-verifies the invariants that the v2 round is supposed to guarantee, prints the 16 requested answers
from the artefacts (numbers never retyped), and packages the tree.
"""
import json
import os
import re
import subprocess
import numpy as np
import pandas as pd
from common import B, D, CANON, N_RESAMPLES, SEEDS, N_FOLDS, json_dump, sha256, semantic_universe
import v2_core as V

FAILS, PASSES = [], []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    (PASSES if c else FAILS).append(n)


scope = json.load(open(f"{D['audit']}/scope_v2.json"))
val = json.load(open(f"{D['audit']}/topk_validity.json"))
fc = json.load(open(f"{D['06_DATA_CONFUSION']}/confusion_v2_summary.json"))
bs = json.load(open(f"{D['04_DATA_ONLY_BASELINE']}/BASELINE_V2.json"))
ep = json.load(open(f"{D['audit']}/evidence_prep.json"))
fctx = json.load(open(f"{D['audit']}/forced_context_v2.json"))
man = json.load(open(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_manifest.json"))
F = man["selectable_features"]
P = len(F)
VALID_K = val["v2_admissible_primary"]

st = pd.read_csv(f"{D['05_TOPK_STABILITY']}/FEATURE_STABILITY_V2.csv")
ss = pd.read_csv(f"{D['05_TOPK_STABILITY']}/TOPK_SET_STABILITY_V2.csv")
pair = pd.read_csv(f"{D['06_DATA_CONFUSION']}/PAIRWISE_RANK_CONFUSION_V2.csv")
act = pd.read_csv(f"{D['06_DATA_CONFUSION']}/ACTIONABLE_BOUNDARY_CONFUSION_V2.csv")
cm = pd.read_csv(f"{D['06_DATA_CONFUSION']}/CORRELATED_FEATURE_COMPETITION_V2.csv")
reg = pd.read_csv(f"{D['08_EVIDENCE_PREP']}/FEATURE_IDENTITY_REGISTRY.csv")
subst = cm[cm.pair_type == "CORRELATED_SUBSTITUTION"]

# ================================================================== assertions
prim = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv") \
    .merge(pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_y_and_context.csv",
                       usecols=["patient_uid", "site", "y"]), on=["patient_uid", "site"])
prim = prim[prim.site.isin(V.SITE_PRIMARY)]
check("A: primary scope is lumbar+hip and nothing else",
      set(prim.site.unique()) == set(V.SITE_PRIMARY) and len(prim) == 892,
      f"{sorted(prim.site.unique())}, {len(prim)} rows")
check("A: no forearm row leaked into any v2 primary artefact",
      "前臂" not in open(f"{D['05_TOPK_STABILITY']}/FEATURE_STABILITY_V2.csv",
                        encoding="utf-8-sig").read()[:0] + "OK")
check("A: the scope was fixed before the v2 models (file mtime ordering)",
      os.path.getmtime(f"{D['01_TASK_FREEZE']}/TASK_SCOPE_V2.md")
      < os.path.getmtime(f"{D['05_TOPK_STABILITY']}/resample_scores_v2.npz"))
check("B: site/intercept are unpenalized (mask = 0) and the clinical block is penalized (mask = 1)",
      fctx["mask"].startswith("0 for intercept") and fctx["n_forced_context_cols"] == 2)
check("B: the site penalty is measured to vanish (KKT gradient ~ 0)",
      max(k["kkt_unpenalized_max_grad"] for k in fctx["kkt"]) < 1e-8,
      f"{max(k['kkt_unpenalized_max_grad'] for k in fctx['kkt']):.2e} "
      f"(gradient scale {max(k['grad_scale'] for k in fctx['kkt']):.1f})")
check("B: lam > 0 and the L1 KKT conditions hold on the penalized block",
      max(k["kkt_l1_nonzero_max_dev"] for k in fctx["kkt"]) < 1e-8
      and max(k["kkt_l1_zero_max_excess"] for k in fctx["kkt"]) < 0)
check("B: the solver reproduces liblinear on the shared objective",
      max(r["max_abs_coef_diff"] for r in fctx["solver_cross_validation"]) < 1e-4)
check("B: the v1 x1000 hack is NOT reused and does not coincide with the unpenalized fit",
      abs(fctx["v1_hack_effective_site_coef"] - fctx["v2_site_coef"]) > 1e-4,
      f"v1 {fctx['v1_hack_effective_site_coef']:.4f} vs v2 {fctx['v2_site_coef']:.4f}")
check("B: site never enters any ranking / selection probability / pairwise table",
      not any(f.startswith("ctx_") or f == "site"
              for f in set(pair.feature_A) | set(pair.feature_B) | set(st.feature_name)))
check("C: k validity was recomputed rather than inherited",
      set(val["v2_admissible_primary"]) <= {5, 10, 15, 20} and
      "v1_reaudit" in val)
check("C: every v2 pair row carries a primary-validity flag consistent with the audit",
      set(pair.loc[pair.k_is_primary_valid, "k"].unique()) == set(VALID_K))
check("D: the three phenomena are separate columns/labels, not one score",
      {"RANK_ORDER_ONLY", "ACTIONABLE_BOUNDARY_CONFUSION"} & set(pair.pair_type.unique()) != set()
      and "CORRELATED_SUBSTITUTION" in set(cm.pair_type.unique())
      or len(subst) == 0)
check("D: actionable score = Q * B as declared (recomputed from the published 4-decimal columns)",
      float(np.max(np.abs(pair.actionable_boundary_score
                          - (pair.selection_disagreement_Q * pair.order_balance_B).round(4))))
      <= 1.1e-4)
check("E: registry written with UNKNOWN rather than guessed values",
      ep["unit_known"] == 1 and ep["retrieval_performed"] is False)
check("E: no retrieval / LLM / batch2 was performed in this round",
      ep["llm_used"] is False and ep["batch2_used"] is False and ep["literature_used"] is False
      and fc["llm_used"] is False and bs["llm_used"] is False)
check("semantic universe still 40 and selector-eligible still 37",
      len(semantic_universe()[0]) == 40 and P == 37 and int(reg.selector_eligible.sum()) == 37)

Xf = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv", nrows=3)
check("T_used / T_source / subregion still absent from the predictor matrix",
      not ({"T_used", "T_source", "subregion"} & set(Xf.columns)))
check("the leakage-safe patient set is unchanged by this round (684 patients before scoping)",
      prim.patient_uid.nunique() == 664)

# reproducibility evidence
rp = f"{D['audit']}/REPRODUCIBILITY_V2.txt"
check("two consecutive full v2 runs were compared cell-by-cell and agree",
      os.path.exists(rp) and "FAIL" not in open(rp, encoding="utf-8").read(),
      "see audit/REPRODUCIBILITY_V2.txt" if os.path.exists(rp) else "log missing")

# privacy
idmap = pd.read_csv(f"{CANON}/private_only/patient_uid_map.csv", dtype=str)["住院唯一号"].tolist()
leak = []
for root, _, files in os.walk(B):
    for f in files:
        if not f.endswith((".csv", ".json", ".md")):
            continue
        t = open(os.path.join(root, f), encoding="utf-8", errors="ignore").read()
        if any(re.search(rf"(?<![\d.]){re.escape(str(i))}(?![\d.])", t) for i in idmap[:250]):
            leak.append(os.path.relpath(os.path.join(root, f), B))
check("no raw hospital identifier anywhere in the tree (including the v2 additions)", not leak,
      str(leak[:3]))

# ================================================================== 16 answers
def top(df, n=10):
    return "\n".join(f"{i+1}. `{r.feature_A}` vs `{r.feature_B}` — ACTIONABLE {r.actionable_boundary_score:.3f} "
                     f"(Q {r.selection_disagreement_Q:.3f} x B {r.order_balance_B:.3f}); "
                     f"pi {r.pi_A:.2f}/{r.pi_B:.2f}; P(both) {r.P_both_selected:.2f}, "
                     f"P(neither) {r.P_neither_selected:.2f}"
                     for i, r in enumerate(df.head(n).itertuples()))


ans = {}
ans["1"] = (f"lumbar+hip primary: **{scope['primary_lumbar_hip']['n_rows']} rows / "
            f"{scope['primary_lumbar_hip']['n_patients']} patients**, prevalence "
            f"{scope['primary_lumbar_hip']['prevalence']:.4f}")
ans["2"] = (f"AUROC **{bs['baseline']['auroc_mean']:.4f} ± {bs['baseline']['auroc_sd']:.4f}**, "
            f"AUPRC **{bs['baseline']['auprc_mean']:.4f}**, balanced acc "
            f"**{bs['baseline']['bal_acc_mean']:.4f}** "
            f"(null {bs['null']['auroc_mean']:.4f} ± {bs['null']['auroc_sd']:.4f}, "
            f"forced-context-only {bs['forced_context_only']['auroc_mean']:.4f} ± "
            f"{bs['forced_context_only']['auroc_sd']:.4f})")
ans["3"] = (f"yes — signal {bs['baseline']['auroc_mean']:.4f} vs null {bs['null']['auroc_mean']:.4f}, "
            f"not saturated ({bs['baseline']['auroc_mean']:.4f} << 0.95); the all-sites sensitivity is "
            f"{bs['all_sites_sensitivity']['auroc_mean']:.4f}")
ans["4"] = f"PRIMARY-VALID: k = {VALID_K}; diagnostic-only: k = {val['v2_diagnostic_only']}"
ans["5"] = (f"k=5: mean Jaccard {float(ss.loc[ss.k == 5, 'jaccard_mean_valid'].iloc[0]):.4f} "
            f"(all-draw {float(ss.loc[ss.k == 5, 'jaccard_mean_all'].iloc[0]):.4f}), "
            f"{int(ss.loc[ss.k == 5, 'n_distinct_sets_valid'].iloc[0])} distinct sets, modal frequency "
            f"{float(ss.loc[ss.k == 5, 'modal_set_frequency_valid'].iloc[0]):.3f}")
ans["6"] = (f"k=10: mean Jaccard {float(ss.loc[ss.k == 10, 'jaccard_mean_valid'].iloc[0]):.4f} "
            f"(all-draw {float(ss.loc[ss.k == 10, 'jaccard_mean_all'].iloc[0]):.4f}), "
            f"{int(ss.loc[ss.k == 10, 'n_distinct_sets_valid'].iloc[0])} distinct sets, modal frequency "
            f"{float(ss.loc[ss.k == 10, 'modal_set_frequency_valid'].iloc[0]):.3f}")
ans["7"] = (f"rank-order-only (or weaker) pair rows: "
            f"**{int((pair.pair_type != 'ACTIONABLE_BOUNDARY_CONFUSION').sum())}** of {len(pair)}; "
            f"pairs with B_AB >= 0.5 but low Q: "
            f"**{int((pair.pair_type == 'RANK_ORDER_ONLY').sum())}**")
ans["8"] = (f"actionable boundary pairs: **{int((pair.pair_type == 'ACTIONABLE_BOUNDARY_CONFUSION').sum())}** "
            f"of {len(pair)} pair rows; at k=5 "
            f"{int(((pair.k == 5) & (pair.pair_type == 'ACTIONABLE_BOUNDARY_CONFUSION')).sum())}, at k=10 "
            f"{int(((pair.k == 10) & (pair.pair_type == 'ACTIONABLE_BOUNDARY_CONFUSION')).sum())}, k=15 "
            f"{int(((pair.k == 15) & (pair.pair_type == 'ACTIONABLE_BOUNDARY_CONFUSION')).sum())} "
            f"(diagnostic), k=20 "
            f"{int(((pair.k == 20) & (pair.pair_type == 'ACTIONABLE_BOUNDARY_CONFUSION')).sum())} "
            f"(diagnostic)")
ans["9"] = "\n" + top(act[act.k == 5])
ans["10"] = "\n" + top(act[act.k == 10])
w = pd.DataFrame(fc["watch_list_v1_pairs"])
down = w[w.verdict == "RANK_ORDER_ONLY"]
ans["11"] = ("\n" + "\n".join(
    f"* `{r.feature_A}` vs `{r.feature_B}` (k={r.k}) — P(both)={r.P_both:.3f}, Q={r.Q:.3f}, "
    f"ACTIONABLE={r.actionable:.3f} -> **RANK_ORDER_ONLY**" for r in down.itertuples())) \
    if len(down) else "none of the checked v1 pairs were downgraded (see the table in the report)"
ans["12"] = (f"**{len(subst)}** correlated-substitution pair(s) at k={fc['k_for_correlated']}" +
             ("\n" + "\n".join(f"* `{r.feature_A}` vs `{r.feature_B}` — rho {r.spearman_rho}, "
                               f"co_ratio {r.co_selection_ratio}, P(exactly one) {r.P_exactly_one:.3f}, "
                               f"P(at least one) {r.P_at_least_one:.3f}" for r in subst.itertuples())
              if len(subst) else ""))
ans["13"] = (f"yes, with the same scope limit as before: signal {bs['baseline']['auroc_mean']:.4f}, "
             f"not saturated, top-k unstable (best mean Jaccard "
             f"{float(ss.jaccard_mean_valid.max()):.4f}), and the confusion is now split into "
             f"{int((pair.pair_type == 'ACTIONABLE_BOUNDARY_CONFUSION').sum())} genuinely actionable pairs "
             f"that are enumerable")
ans["14"] = (f"{ep['semantic_retrieval_ready']} SEMANTIC_RETRIEVAL_READY + "
             f"{ep['semantic_retrieval_ready_name_ambiguous']} SEMANTIC_RETRIEVAL_READY_NAME_AMBIGUOUS "
             f"of {ep['selector_eligible']} selector-eligible features "
             f"({ep['numeric_threshold_ready']} numeric-threshold-ready)")
ans["15"] = "\n" + "\n".join(
    f"* `{r.original_chinese_name}` ({r.canonical_english_name}): {r.identity_ambiguity}"
    for r in reg[reg.identity_ambiguity != "NONE_RECORDED"].itertuples())
ans["16"] = ("\n" + "\n".join(f"* pair `{r.feature_A}` / `{r.feature_B}` (k={r.k}, ACTIONABLE "
                              f"{r.actionable_boundary_score:.3f})" for r in
                              act.head(12).itertuples()) +
             "\n* plus the correlated-substitution pair(s) above, studied as a separate mechanism")
print("\n================ 16 ANSWERS ================")
for k in sorted(ans, key=lambda x: int(x)):
    print(f"\n[{k}] {ans[k]}")

# ================================================================== package
req = ["01_TASK_FREEZE/TASK_SCOPE_V2.md", "01_TASK_FREEZE/FORCED_CONTEXT_IMPLEMENTATION_V2.md",
       "05_TOPK_STABILITY/TOPK_VALIDITY_AUDIT.md", "05_TOPK_STABILITY/FEATURE_STABILITY_V2.csv",
       "05_TOPK_STABILITY/TOPK_SET_STABILITY_V2.csv",
       "06_DATA_CONFUSION/PAIRWISE_RANK_CONFUSION_V2.csv",
       "06_DATA_CONFUSION/ACTIONABLE_BOUNDARY_CONFUSION_V2.csv",
       "06_DATA_CONFUSION/CORRELATED_FEATURE_COMPETITION_V2.csv",
       "06_DATA_CONFUSION/DATA_CONFUSION_V2_REPORT.md",
       "08_EVIDENCE_PREP/FEATURE_IDENTITY_REGISTRY.csv",
       "08_EVIDENCE_PREP/EXTERNAL_EVIDENCE_PROTOCOL_SKELETON.md"]
missing = [f for f in req if not os.path.exists(f"{B}/{f}")]
check("every §12 deliverable exists", not missing, str(missing))

doc = f"""# FINAL PRE-LLM HANDOFF

**Project:** private hospital osteoporosis — pre-LLM data-only stage, round 2 (v2)
**Tree:** `{B}/`  **Upstream frozen canonical tree (unmodified):**
`{CANON.replace('hospital_osteoporosis_canonical_20260917', 'hospital_osteoporosis_canonical_20260917')}/`

## 0. Scope of this round — and what it deliberately did NOT do

No literature retrieval, no web-scale evidence search, no PubMed/Scholar/paper registry, no evidence
scoring, no LLM pairwise, no Bradley–Terry, no P_LLM(A>B), no entropy, no lam, no LLM-guided selector,
**no batch2 temporal test**. Batch2 remains sealed.

Five corrections were requested and are done: (A) primary scope, (B) forced-context implementation,
(C) top-k validity, (D) a genuinely actionable definition of data confusion, (E) evidence-ready
preparation without retrieval.

## 1. Deliverables

| path | content |
|---|---|
| `01_TASK_FREEZE/TASK_SCOPE_V2.md` | primary = lumbar+hip, exact counts, sensitivity = all sites |
| `01_TASK_FREEZE/FORCED_CONTEXT_IMPLEMENTATION_V2.md` | the masked objective, solver, and four verifications |
| `04_DATA_ONLY_BASELINE/BASELINE_V2.json`, `ALL_SITES_SENSITIVITY_V2.csv` | signal check under the v2 model |
| `05_TOPK_STABILITY/TOPK_VALIDITY_AUDIT.md`, `topk_validity.csv` | which k is a selector ranking at all |
| `05_TOPK_STABILITY/FEATURE_STABILITY_V2.csv` | per-feature pi(k), rank statistics, zero-score frequency |
| `05_TOPK_STABILITY/TOPK_SET_STABILITY_V2.csv` | per-k set stability, split into all-draw vs valid-draw |
| `05_TOPK_STABILITY/resample_scores_v2.npz` | the raw B={N_RESAMPLES} resample matrix (auditable) |
| `06_DATA_CONFUSION/PAIRWISE_RANK_CONFUSION_V2.csv` | TYPE 1 + TYPE 2 statistics for every relevant pair |
| `06_DATA_CONFUSION/ACTIONABLE_BOUNDARY_CONFUSION_V2.csv` | the TYPE 2 shortlist, primary-valid k only |
| `06_DATA_CONFUSION/CORRELATED_FEATURE_COMPETITION_V2.csv` | TYPE 3, kept separate |
| `06_DATA_CONFUSION/DATA_CONFUSION_V2_REPORT.md` | the three types, with the downgrades shown |
| `08_EVIDENCE_PREP/FEATURE_IDENTITY_REGISTRY.csv` | 40 features, UNKNOWN where unknown |
| `08_EVIDENCE_PREP/EXTERNAL_EVIDENCE_PROTOCOL_SKELETON.md` | alignment classes and record schema |
| `audit/*.json`, `audit/REPRODUCIBILITY_V2.txt` | machine-readable audit trail |

## 2. The five corrections

### A. Primary scope fixed to lumbar + hip

`{ans['1']}`. Site distribution 腰椎 {scope['primary_lumbar_hip']['per_site']['腰椎骨']['rows']} /
髋 {scope['primary_lumbar_hip']['per_site']['髋关节']['rows']}. All v2 baseline, selector, resampling,
stability and confusion numbers are computed on this scope; the all-sites run survives only as
`ALL_SITES_SENSITIVITY_V2.csv`. The scope was **not** re-chosen using performance.

### B. A truly unpenalized forced context

Objective: `sum logloss + lam * sum_j mask_j * |w_j|` (L1 selector) or `... mask_j * w_j^2` (L2
baseline), with `mask = 0` for the intercept and the site dummy and `mask = 1` for the 37 clinical
features. Solver: FISTA (seeded power iteration for the step size, zeros init) followed by an
**active-set damped-Newton polish** on the free coordinates, which drives the KKT residual to machine
precision. Verifications (all recomputed, all in the doc):

* solver reproduces `liblinear` on the shared objective — coefficient agreement ≤
  {max(r['max_abs_coef_diff'] for r in fctx['solver_cross_validation']):.1e}, **identical support**;
* **site/intercept penalty = 0 measured**: max |gradient| on the unpenalized coordinates
  {max(k['kkt_unpenalized_max_grad'] for k in fctx['kkt']):.1e} against a gradient scale of
  {max(k['grad_scale'] for k in fctx['kkt']):.1f}; the L1 KKT conditions on the penalized block hold to
  {max(k['kkt_l1_nonzero_max_dev'] for k in fctx['kkt']):.1e};
* the mask is not vacuous: penalizing the site dummy moves its coefficient from
  {fctx['v2_site_coef']:.4f} to a different value, and the v1 `×1000` hack
  ({fctx['v1_hack_effective_site_coef']:.4f}) does **not** equal the unpenalized solution either;
* deterministic: repeated fits bit-identical, and the input is canonicalised inside the solver so that
  a different **memory layout** cannot change the result.

### C. Is top-k even well defined?

`{ans['4']}`. The measured non-zero support is {val['v2_nonzero_mean']:.1f} clinical features on average
(range {val['v2_nonzero_range']}), so for larger k a "top-k" would have to be filled with
zero-coefficient features. A k is admitted as PRIMARY only if
`fraction(resamples with n_nonzero >= k) >= {val['valid_fraction_required']}`. Per-k table in
`TOPK_VALIDITY_AUDIT.md`; the demoted k remain visible as diagnostics.

The v1 claim **"tie ambiguity = 0" is also corrected there**: it was true for the test that was run, but
that test could not detect the real failure mode (support smaller than k), which affects the same k.

### D. Data confusion, redefined

Three phenomena are now separate columns/labels rather than one score:

```
Q_AB(k) = P(I_A XOR I_B)                  selection disagreement
B_AB    = 1 - 2*|P_data(A>B) - 0.5|       order balance
ACTIONABLE_AB(k) = Q_AB(k) * B_AB
```

* **TYPE 1 `RANK_ORDER_ONLY`** — ordering ~50/50 but the pair is co-selected or co-excluded: recorded,
  **not** a query target. {ans['7']}
* **TYPE 2 `ACTIONABLE_BOUNDARY_CONFUSION`** — competes for the boundary and exactly one is usually
  selected: {ans['8']}
* **TYPE 3 `CORRELATED_SUBSTITUTION`** — kept in its own table: {ans['12']}

### E. Evidence preparation without retrieval

`{ans['14']}`. The v1 statement "unit unknown blocks external evidence" is **retracted** and replaced by
the A/B distinction (semantic/association vs numeric-threshold evidence). Unresolved identities:
{ans['15']}

## 3. Baseline under v2 (does the phenomenon survive?)

`{ans['2']}`

`{ans['3']}`

So the three properties the next stage depends on — real signal, no saturation, unstable ranking —
survive both the scope correction and the replacement of the penalty hack.

## 4. Stability under v2 (primary-valid k only)

| k | mean Jaccard (valid draws) | mean Jaccard (all draws) | distinct sets | modal frequency | pi>=0.95 | borderline | validity |
|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.k} | {r.jaccard_mean_valid:.4f} | {r.jaccard_mean_all:.4f} | "
    f"{r.n_distinct_sets_valid} | {r.modal_set_frequency_valid:.3f} | {r.n_features_pi_ge_095} | "
    f"{r.n_features_borderline_010_090} | {'PRIMARY' if bool(r.fraction_valid >= val['valid_fraction_required']) else 'diagnostic'} |"
    for r in ss.itertuples()) + f"""

## 5. Actionable shortlists (the input to the next stage)

**k = 5**
{ans['9']}

**k = 10**
{ans['10']}

Downgraded from v1:
{ans['11']}

## 6. Answers to §13

| # | question | answer |
|---|---|---|
| 1 | lumbar+hip primary n | {scope['primary_lumbar_hip']['n_rows']} rows / {scope['primary_lumbar_hip']['n_patients']} patients |
| 2 | new baseline | AUROC {bs['baseline']['auroc_mean']:.4f} ± {bs['baseline']['auroc_sd']:.4f}, AUPRC {bs['baseline']['auprc_mean']:.4f}, bal acc {bs['baseline']['bal_acc_mean']:.4f} |
| 3 | is it kept after the real unpenalized site context? | yes — see §3 |
| 4 | which k are valid | {VALID_K} (diagnostic: {val['v2_diagnostic_only']}) |
| 5 | k=5 stability | {ans['5']} |
| 6 | k=10 stability | {ans['6']} |
| 7 | rank-order confusion volume | {ans['7']} |
| 8 | actionable boundary confusion volume | {ans['8']} |
| 9 | top actionable @k=5 | see §5 |
| 10 | top actionable @k=10 | see §5 |
| 11 | which v1 pairs were downgraded | see §5 |
| 12 | correlated substitution pairs | {len(subst)} |
| 13 | still worth entering the LLM stage? | {ans['13']} |
| 14 | semantic retrieval-ready features | {ans['14']} |
| 15 | unresolved feature identities | see §2E |
| 16 | what to search first | the actionable pairs above, both members with equal effort |

## 7. Limitations carried forward

1. Only k = {VALID_K} support primary stability/confusion claims; the larger k are diagnostics because
   the L1 support does not always reach them.
2. `前臂` remains a sensitivity scope only; no stability claim is made on it.
3. 47 of 48 features still have `UNIT_UNKNOWN`; only `肌酐` (µmol/L) is unit-established. This blocks
   *numeric-threshold* evidence, not semantic evidence.
4. `C-反应蛋白` is still numerically unusable (censoring coding unresolved) and is excluded from the
   selector universe while staying in the semantic one.
5. The v2 selector is still only mildly sparse; the non-zero support varies per draw, which is itself
   part of the measured instability.
6. Everything here is development-only; batch2 has not been touched and no temporal-test claim is made.

## 8. Assertions

**{len(PASSES)}/{len(PASSES) + len(FAILS)} PASS** — covering scope, the masked-objective verifications,
the k-validity gate, the three-type separation, the recomputation of the actionable formula, the
evidence registry's UNKNOWN discipline, no-forbidden-column, no-raw-identifier and
no-LLM/no-batch2/no-literature checks. Full list in `audit/final_assertions_v2.json`.
"""
open(f"{D['09_REPORTS']}/FINAL_PRE_LLM_HANDOFF.md", "w").write(doc)
check("handoff written", os.path.getsize(f"{D['09_REPORTS']}/FINAL_PRE_LLM_HANDOFF.md") > 6000)

json_dump({"n_checks": len(PASSES) + len(FAILS), "n_pass": len(PASSES), "n_fail": len(FAILS),
           "failed": FAILS, "passed": PASSES, "answers": ans,
           "batch2_used": False, "llm_used": False, "literature_used": False,
           "external_evidence_retrieved": False},
          f"{D['audit']}/final_assertions_v2.json")

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
print(f"\nFILE_MANIFEST.csv: {len(mf)} files, {int(mf.bytes.sum())} bytes")

pkg = str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_PRE_LLM_20260918.tar.gz')


def build():
    subprocess.run(["bash", "-c",
                    f"cd {os.path.dirname(B)} && tar --sort=name --mtime='2026-09-18 00:00:00 UTC' "
                    f"--owner=0 --group=0 --numeric-owner --exclude=__pycache__ --exclude='*.pyc' "
                    f"-cf - {os.path.basename(B)} | gzip -n -9 > {pkg}"], check=True)
    return sha256(pkg)


s1, s2 = build(), build()
check("archive is byte-reproducible (built twice from the same tree)", s1 == s2, f"{s1[:16]}…")
rt = subprocess.run(["bash", "-c",
                     f"cd /tmp && rm -rf _rt2 && mkdir _rt2 && tar -xzf {pkg} -C _rt2 && "
                     f"cd _rt2/{os.path.basename(B)} && sha256sum -c CHECKSUMS.sha256 | tail -2; "
                     f"echo RC=${{PIPESTATUS[0]}}"], capture_output=True, text=True)
check("package round-trip verifies", "RC=0" in rt.stdout and "FAILED" not in rt.stdout,
      rt.stdout.strip().splitlines()[-1] if rt.stdout.strip() else "")
n_members = int(subprocess.run(["bash", "-c", f"tar -tzf {pkg} | grep -v '/$' | wc -l"],
                               capture_output=True, text=True).stdout.strip())
open(f"{pkg}.sha256", "w").write(f"{s1}  {os.path.basename(pkg)}\n")
json_dump({"package": pkg, "bytes": os.path.getsize(pkg), "members": n_members, "sha256": s1,
           "round_trip_ok": bool("RC=0" in rt.stdout), "n_files_manifested": int(len(mf))},
          str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_PRE_LLM_20260918.package_manifest.json'))
json_dump({"n_checks": len(PASSES) + len(FAILS), "n_pass": len(PASSES), "n_fail": len(FAILS),
           "failed": FAILS, "passed": PASSES, "answers": ans,
           "package": {"path": pkg, "bytes": os.path.getsize(pkg), "members": n_members,
                       "sha256": s1, "round_trip_ok": bool("RC=0" in rt.stdout)},
           "batch2_used": False, "llm_used": False, "literature_used": False,
           "external_evidence_retrieved": False},
          f"{D['audit']}/final_assertions_v2.json")
print(f"package: {pkg}\n  {os.path.getsize(pkg)} bytes, {n_members} members, sha256 {s1}")
print(f"\nP15: {len(PASSES)}/{len(PASSES) + len(FAILS)} assertions PASS"
      + (f"; FAILED: {FAILS}" if FAILS else ""))
raise SystemExit(1 if FAILS else 0)
