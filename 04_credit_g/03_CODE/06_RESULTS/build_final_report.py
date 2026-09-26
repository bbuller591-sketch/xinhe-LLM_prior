

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
from pathlib import Path
import json
import pandas as pd

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
R=WS/'06_RESULTS'
res=pd.read_csv(R/'FINAL_HOLDOUT_RESULTS.csv')
sets=pd.read_csv(R/'FINAL_SELECTED_SETS_FREEZE.csv')
lam=json.loads((R/'GAMMA_FREEZE.json').read_text())
reference=pd.read_csv(WS/'02_DATA_ONLY/reference_BASELINE_RESULTS.csv')
audit=json.loads((R/'LLM_MEASUREMENT_AUDIT_SUMMARY.json').read_text())
evidence=json.loads((WS/'03_EVIDENCE/EVIDENCE_AUDIT_SUMMARY.json').read_text())

changes=[]
for s in ['L1','GBM_PERM','ELASTIC_NET']:
    q=sets[sets.selector==s].set_index('method')
    base=set(q.loc['reference','selected_features'].split('|'))
    for m in ['global','global_certainty','selective']:
        cur=set(q.loc[m,'selected_features'].split('|'))
        changes.append({
            'selector':s,'method':m,
            'added_vs_reference':'|'.join(sorted(cur-base)),
            'removed_vs_reference':'|'.join(sorted(base-cur)),
            'set_changed':cur!=base,
            'jaccard_vs_reference':len(cur&base)/len(cur|base),
        })
chg=pd.DataFrame(changes)
chg.to_csv(R/'FINAL_SELECTED_SET_CHANGES.csv',index=False)

lines=[]
lines += [
"# CREDIT-G Final Modern Experiment Report",
"",
"Date: 2026-09-18",
"",
"## 1. Scope and status",
"",
"- Legacy-exact reproduction remains blocked because the exact historical pair graph, placement mechanism, complete seed identities, downstream learner/hyperparameters, permutation arrays, and per-cell fits were not recoverable.",
"- The modern CREDIT-G reference-selective experiment is complete through one predeclared holdout evaluation.",
"- The final 200-row holdout was used only after task, preprocessing, selectors, k, reference confusion, evidence gates, LLM runtime/prompts, injection rules, lam grid, development-selected lam, and final selected sets were frozen.",
"- Because German Credit is a public and previously inspected benchmark, this set is described as a modern predeclared / historically non-sealed holdout, not a sealed test.",
"",
"## 2. Critical preprocessing audit",
"",
"- Frozen data remain the recovered OpenML-31 German Credit observations: n=1000, p=20.",
"- A documentation audit found that original Statlog coding descriptions are known to contain serious errors. Corrected South German Credit documentation was used only to determine construct/type semantics.",
"- Primary preprocessing therefore treats only duration, credit_amount, and age as continuous. The remaining 17 code-valued categorical/ordinal variables are one-hot encoded.",
"- No literature or LLM information changed the 20-variable universe.",
"",
"## 3. reference data-only experiment",
"",
"- Development set: 800 rows; final holdout: 200 rows.",
"- Formal reference: 300 paired development resamples.",
"- Frozen k: L1=10, GBM-permutation=10, Elastic Net=10.",
"",
"| Selector | k | Development mean AUROC | Actionable boundary pairs | Rank-order-only pairs |",
"|---|---:|---:|---:|---:|",
]
for s,name in [('L1','L1'),('GBM_PERM','GBM-permutation'),('ELASTIC_NET','Elastic Net')]:
    row=reference[(reference.selector==s)&(reference.chosen_k.astype(bool))].iloc[0]
    # counts from known files
    cf=pd.read_csv(WS/'02_DATA_ONLY'/({'L1':'ACTIONABLE_CONFUSION_L1.csv','GBM_PERM':'ACTIONABLE_CONFUSION_GBM_PERM.csv','ELASTIC_NET':'ACTIONABLE_CONFUSION_ELASTIC_NET.csv'}[s]))
    lines.append(f"| {name} | {int(row.k)} | {row.mean_auc:.6f} | {(cf.pair_class=='ACTIONABLE_BOUNDARY_CONFUSION').sum()} | {(cf.pair_class=='RANK_ORDER_ONLY').sum()} |")
lines += [
"",
"- Across selectors there were 115 selector-specific actionable requests but only 59 unique semantic pairs.",
"- The supplementary correlated-substitution heuristic found zero candidates at the predeclared thresholds; thresholds were not relaxed post hoc.",
"",
"## 4. External-evidence gate",
"",
f"- Targeted selective feature set: {evidence['targeted_features']} features.",
f"- USABLE_EVIDENCE: {', '.join(evidence['usable_features'])}.",
f"- CONFLICTING_EVIDENCE: {', '.join(evidence['conflicting_features'])}.",
"- The remaining targeted features were MISMATCH_OR_INSUFFICIENT.",
f"- The 59 unique selective pairs therefore shrank to {evidence['selective_pairs_after_primary_evidence_gate']} primary evidence-eligible pairs:",
]
for p in evidence['eligible_pairs']:
    lines.append(f"  - {p}")
lines += [
"- This shrinkage is intentional; the gate was not relaxed to increase selective coverage.",
"",
"## 5. LLM measurement audit",
"",
"- Runtime: DeepSeek deepseek-flash, thinking disabled, temperature 1.0, first-token logprobs, no tools/web/retrieval during judgment.",
"- Provider model and fingerprint were stable across all formal calls.",
"- A generic pre-study smoke test exposed whitespace-token variants (for example A and space+A) in top-logprobs; semantic token masses were therefore aggregated with log-sum-exp before any study-specific call.",
f"- global/global_certainty: {audit['m12_primary_pairs']} primary broad pairs, {audit['m12_primary_abstain_pairs']} primary abstentions, {audit['m12_bt_pairs_used']} pairs used in BT.",
f"- Mean global/global_certainty AB/BA order gap on the logit scale: {audit['m12_mean_order_gap_logit']:.3f}.",
f"- Median global/global_certainty edge entropy trust: {audit['m12_median_entropy_trust_edge']:.3f}.",
f"- Repeat diagnostic: {audit['m12_repeat_nonunanimous_cells']} of {audit['m12_repeat_order_cells']} repeated order-cells were non-unanimous.",
f"- selective: {audit['selective_primary_pairs']} primary pairs, {audit['selective_primary_abstain_pairs']} primary abstentions; {audit['selective_repeat_nonunanimous_cells']} of {audit['selective_repeat_order_cells']} repeated order-cells were non-unanimous.",
"- selective order sensitivity was not negligible: purpose-vs-housing had a very large AB/BA logit gap, so the external-evidence result must not be described as a perfectly stable oracle.",
"",
"## 6. Development-only lam selection",
"",
"| Selector | Method | lam | Mean dev delta-AUROC vs reference | Paired bootstrap 95% CI | Harm >0.01 fraction |",
"|---|---|---:|---:|---:|---:|",
]
for d in lam['diagnostics']:
    lines.append(f"| {d['selector']} | {d['method']} | {d['selected_lam']:.3f} | {d['mean_delta_auc_vs_gamma0']:+.6f} | [{d['bootstrap95_low']:+.6f}, {d['bootstrap95_high']:+.6f}] | {d['fraction_harmed_gt_0p01']:.3f} |")
lines += [
"",
"- All nonzero global/global_certainty development choices hit the upper frozen lam-grid boundary 0.30. The grid was not extended after seeing this result.",
"- L1-selective selected lam=0, i.e. development validation chose not to borrow.",
"",
"## 7. Final predeclared holdout point estimates",
"",
"| Selector | Method | lam | AUROC | Delta AUROC vs same-selector reference | AP | Log loss |",
"|---|---|---:|---:|---:|---:|---:|",
]
for _,z in res.iterrows():
    delta="" if pd.isna(z.delta_auroc_vs_selector_reference) else f"{z.delta_auroc_vs_selector_reference:+.6f}"
    lines.append(f"| {z.selector} | {z.method} | {z.lam:.3f} | {z.holdout_auroc:.6f} | {delta} | {z.holdout_average_precision:.6f} | {z.holdout_log_loss:.6f} |")
lines += [
"",
"Selected-set changes explain several exact ties:",
]
for _,z in chg.iterrows():
    if not z.set_changed:
        lines.append(f"- {z.selector}-{z.method}: same top-10 set as reference; any ranking changes inside the set do not change the common evaluator.")
    else:
        lines.append(f"- {z.selector}-{z.method}: added {{{z.added_vs_reference}}}, removed {{{z.removed_vs_reference}}}; Jaccard={z.jaccard_vs_reference:.3f}.")
lines += [
"",
"## 8. What the final result supports",
"",
"- The modern experiment does not show a universal gain from LLM guidance across selectors.",
"- Broad pretrained/no-retrieved-evidence guidance produced a positive holdout delta for L1-global (+0.0070), no set change for GBM global/global_certainty, and a small negative AUROC delta for Elastic-Net global/global_certainty (-0.0026).",
"- The strictly evidence-gated selective arm was sparse: only three semantic pairs were eligible. L1 correctly selected lam=0; GBM-selective changed one boundary variable and had +0.0132 AUROC versus GBM-reference; Elastic-Net-selective had +0.0075 versus its reference.",
"- These are single 200-row holdout point estimates. The small differences should not be treated as definitive proof of superiority, and no post-hoc holdout-driven retuning is permitted.",
"- The all-20-variable reference reached AUROC 0.7920. GBM-selective reached 0.7933, only about +0.0013 above that reference, so there is no basis for a strong claim that sparse guided selection dominates using all 20 variables.",
"",
"## 9. Benchmark-recall boundary",
"",
"- global/global_certainty are no-retrieved-evidence measurements, not independent external evidence.",
"- Historical diagnostics already showed that this public benchmark could be recognized under stripped/blinded conditions. Therefore any global/global_certainty utility may reflect generic credit knowledge, benchmark recall, memorized published importance, or a mixture; the experiment cannot identify which mechanism generated the gain.",
"- selective was specifically designed to avoid this attribution problem by excluding German-Credit-family studies from its independent-evidence packets.",
"",
"## 10. Strongest empirical takeaways",
"",
"1. Data-only uncertainty is strongly selector-specific; the actionable pair sets differ materially across L1, GBM-permutation, and Elastic Net.",
"2. A strict evidence-eligibility layer is consequential: 59 candidate semantic pairs collapsed to only 3 primary selective pairs rather than letting weak/mismatched evidence enter the judge.",
"3. Entropy attenuation changed selected-set movement and development harm rates, but it did not uniformly improve holdout AUROC; uncertainty calibration and usefulness remain distinct.",
"4. Sparse, evidence-gated selective guidance can change a selector only at its data-confused boundary; in this run the clearest final changes were removal of housing in favor of another boundary feature for GBM/Elastic Net.",
"5. Results remain a benchmark stress test, not evidence that the same gains will generalize to biomedical or other high-dimensional tasks.",
"",
"## 11. Claim restrictions",
"",
"Do not claim:",
"- LEGACY-EXACT reproduction.",
"- A sealed confirmatory test.",
"- global/global_certainty gains prove independent semantic reasoning.",
"- Entropy equals correctness.",
"- The three selective pairs establish general external-evidence benefit.",
"- Any universal selector winner.",
"",
"Permissible wording:",
"- modern predeclared holdout point estimates;",
"- selector-specific development and holdout behavior;",
"- strict evidence gating and measured order/repeat diagnostics;",
"- benchmark-recall-confounded broad guidance versus independently sourced, evidence-gated selective guidance.",
"",
]
(R/'FINAL_EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n')
print('wrote FINAL_EXPERIMENT_REPORT.md and FINAL_SELECTED_SET_CHANGES.csv')
