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
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'q3_q4_diagnostics_20260922/05_summary'
OUT.mkdir(parents=True,exist_ok=True)

cols=['dataset','selector','k','q2_status','q2_selective_auroc','q2_global_auroc','q2_selective_minus_global','q2_scope',
      'q3_semantic_p_auroc','q3_random_p_auroc','q3_semantic_p_secondary','q3_random_p_secondary','q3_secondary_metric','q3_scope',
      'q4_status','q4_rho_positive_fraction','q4_protected_order_fraction','q4_cross_boundary_protected_fraction','q4_bound_violations','q4_full_topk_certificate','q4_objective_status',
      'q5_reference_auroc','q5_selective_auroc','q5_delta_auroc','q5_delta_secondary','q5_secondary_metric','q5_scope','caveat']

def row(**kw):
    d={c:np.nan for c in cols}
    d.update(kw)
    return d

R=[]

R.append(row(dataset='Renal GSE36059→GSE48581',selector='ElasticNet+SIS Reference',k=50,
 q2_status='COMPLETE_FORMAL_ORIGINAL',q2_selective_auroc=0.8115671642,q2_global_auroc=0.8106343284,q2_selective_minus_global=0.0009328358,
 q2_scope='100 Selective vs 100 Global formal pairs; pre-external',
 q3_semantic_p_auroc=0.1408591409,q3_random_p_auroc=0.1428571429,q3_semantic_p_secondary=0.3636363636,q3_secondary_metric='AUPRC',
 q3_scope='post-hoc external evaluation of frozen development-tuned null supports',
 q4_status='COMPLETE_EXACT_RESCALED',q4_rho_positive_fraction=0.0215,q4_protected_order_fraction=0.9584307154,q4_cross_boundary_protected_fraction=0.3778564103,q4_bound_violations=0,q4_full_topk_certificate=False,q4_objective_status='THEOREM_COMPATIBLE_AFTER_DETERMINISTIC_RESCALING',
 q5_reference_auroc=0.7855643657,q5_selective_auroc=0.8115671642,q5_delta_auroc=0.0260027985,q5_delta_secondary=0.0201727676,q5_secondary_metric='AUPRC',q5_scope='independent external GSE48581',
 caveat='Guided-minus-Reference bootstrap interval includes zero; semantic/random nulls do not clearly separate.'))

R.append(row(dataset='GSE272769',selector='Elastic Net',k=50,
 q2_status='BLOCKED_NETWORK_QUERY_PACK_FROZEN',q2_scope='strict source-arm/budget/fold-slot matched Global; 804 AB/BA calls required; 0 completed',
 q3_semantic_p_auroc=0.0199004975,q3_random_p_auroc=0.0199800200,q3_semantic_p_secondary=0.0099502488,q3_random_p_secondary=0.0089910090,q3_secondary_metric='Macro-AP',
 q3_scope='strict nested outer-CV; lam selected inside each outer fold',
 q4_status='COMPLETE_ACTIVE_FOLDS',q4_rho_positive_fraction=0.0128888889,q4_protected_order_fraction=0.9347908450,q4_cross_boundary_protected_fraction=0.6198620690,q4_bound_violations=0,q4_full_topk_certificate=False,q4_objective_status='EXACT_NORMALIZED_CE_FOLD_LOCAL',
 q5_reference_auroc=0.6287301587,q5_selective_auroc=0.6553968254,q5_delta_auroc=0.0266666667,q5_delta_secondary=0.0217411454,q5_secondary_metric='Macro-AP',q5_scope='strict nested 5-fold outer CV',
 caveat='No independent external cohort. Strict matched-global Q2 could not execute because the remote server refused connection to api.deepseek.com.'))

breast=[
('LASSO',10,0.7238095238,0.7539115646,-0.0301020408,0.0549450549,0.0389610390,0.0075,0.8171646427,0.4974874372,0.6950680272,0.7238095238,0.0287414966,0.0150209907),
('LASSO',20,0.6663265306,0.6556122449, 0.0107142857,0.6473526474,0.6683316683,0.0145,0.6988340232,0.4459090909,0.6732993197,0.6663265306,-0.0069727891,-0.0069821444),
('Elastic Net',10,0.7362244898,0.7476190476,-0.0113945578,0.2177822178,0.1998001998,0.0135,0.5572734474,0.0,0.7292517007,0.7362244898,0.0069727891,0.0176678211),
('Elastic Net',20,0.6962585034,0.7275510204,-0.0312925170,0.9790209790,0.9590409590,0.0190,0.6242534291,0.3457070707,0.7105442177,0.6962585034,-0.0142857143,-0.0121000361),
('SIS',10,0.7392857143,0.7489795918,-0.0096938776,0.2417582418,0.2637362637,0.0115,0.9817588794,0.1002010050,0.7200680272,0.7392857143,0.0192176871,0.0267391336),
('SIS',20,0.7375850340,0.7175170068, 0.0200680272,0.0989010989,0.0989010989,0.0165,0.9672641321,0.1487878788,0.7227891156,0.7375850340,0.0147959184,0.0357121675)
]
for sel,k,sa,ga,q2,sp,rp,rho,prot,cross,reference,selective,dau,dma in breast:
    R.append(row(dataset='Breast GSE25055→GSE25065',selector=sel,k=k,
      q2_status='COMPLETE_CACHE_ONLY_SOURCE_STRATIFIED',q2_selective_auroc=sa,q2_global_auroc=ga,q2_selective_minus_global=q2,
      q2_scope='post-hoc cached broad comparator; source-specific pair-source budget and actionable-weight slots matched; lam development-only',
      q3_semantic_p_auroc=sp,q3_random_p_auroc=rp,q3_scope='sealed external null comparison; post-hoc control chronology',
      q4_status='COMPLETE_EXACT',q4_rho_positive_fraction=rho,q4_protected_order_fraction=prot,q4_cross_boundary_protected_fraction=cross,q4_bound_violations=0,q4_full_topk_certificate=False,q4_objective_status='EXACT_NORMALIZED_CE',
      q5_reference_auroc=reference,q5_selective_auroc=selective,q5_delta_auroc=dau,q5_delta_secondary=dma,q5_secondary_metric='Macro-AP',q5_scope='sealed external GSE25065',
      caveat='Formal Breast scope is k∈{10,20}; k=30 excluded. Matched-Q2 baseline is post-hoc but cache-only.'))

credit=[
('GBM-permutation',0.7933333333,0.7801190476,0.0132142857,1.0,0.3216783217,0.10,0.8052631579,0.80,0.7801190476,0.7933333333,0.0132142857,0.0020614295,1),
('Elastic Net',0.7770238095,0.7770238095,0.0,0.6873126873,0.4615384615,0.15,0.7157894737,0.72,0.7695238095,0.7770238095,0.0075,0.0216468726,3)
]
for sel,sa,ga,q2,sp,rp,rho,prot,cross,reference,selective,dau,dap,npair in credit:
    R.append(row(dataset='CREDIT-G',selector=sel,k=10,
      q2_status='POSTHOC_MIGRATED_MATCHED_DIAGNOSTIC',q2_selective_auroc=sa,q2_global_auroc=ga,q2_selective_minus_global=q2,
      q2_scope='same pair-count normalized-CE migration using frozen cached measurements; holdout diagnostic only',
      q3_semantic_p_auroc=sp,q3_random_p_auroc=rp,q3_scope='development control under original CREDIT-G selective',
      q4_status='POSTHOC_MIGRATED_DIAGNOSTIC',q4_rho_positive_fraction=rho,q4_protected_order_fraction=prot,q4_cross_boundary_protected_fraction=cross,q4_bound_violations=0,q4_full_topk_certificate=False,q4_objective_status='MIGRATED_TO_NORMALIZED_CE',
      q5_reference_auroc=reference,q5_selective_auroc=selective,q5_delta_auroc=dau,q5_delta_secondary=dap,q5_secondary_metric='Average Precision',q5_scope='locked 200-row holdout',
      caveat=f'Public historically exposed benchmark; Q2/Q4 migration is post-hoc. selective effective support uses {npair} eligible actionable pair(s).'))

hospital=[
(5,0.7337815693,0.7272026219,0.0065789474,np.nan,np.nan,0.2162162162,0.4931773879,0.175,0.7374445730,0.7337815693,-0.0036630037,-0.0065292009),
(10,0.7381723540,0.7381723540,0.0,0.0439560440,0.0619380619,0.1621621622,0.9376218324,0.9555555556,0.7294582610,0.7381723540,0.0087140929,0.0105350161)
]
for k,sa,ga,q2,sp,rp,rho,prot,cross,reference,selective,dau,dap in hospital:
    R.append(row(dataset='Hospital Osteoporosis',selector='L1 logistic rank',k=k,
      q2_status='COMPLETE_CACHE_MATCHED',q2_selective_auroc=sa,q2_global_auroc=ga,q2_selective_minus_global=q2,
      q2_scope='post-hoc cached complete broad graph; pair count and weight multiset matched; Batch1-only tuning',
      q3_semantic_p_auroc=sp,q3_random_p_auroc=rp,q3_semantic_p_secondary=0.0449550450 if k==10 else np.nan,q3_random_p_secondary=0.1208791209 if k==10 else np.nan,q3_secondary_metric='AUPRC' if k==10 else '',
      q3_scope='post-hoc temporal external null evaluation for k=10; k=5 not rerun in completion tranche',
      q4_status='COMPLETE_EXACT',q4_rho_positive_fraction=rho,q4_protected_order_fraction=prot,q4_cross_boundary_protected_fraction=cross,q4_bound_violations=0,q4_full_topk_certificate=False,q4_objective_status='EXACT_NORMALIZED_CE',
      q5_reference_auroc=reference,q5_selective_auroc=selective,q5_delta_auroc=dau,q5_delta_secondary=dap,q5_secondary_metric='AUPRC',q5_scope='temporal external Batch2',
      caveat='Private single-hospital cohort. k=5 is a retained negative result; k=10 positive point estimate has bootstrap interval crossing zero.'))

gbm=[
('SIS',10,0.9481583265,0.9141261236,0.0340322029,0.1998001998,0.1778221778,0.0060,0.9880330165,0.1995979899,0.9141261236,0.9481583265,0.0340322029),
('SIS',20,0.9703165443,0.9479888624,0.0223276819,0.2287712288,0.2127872128,0.0060,0.9880330165,0.5491666667,0.9448426027,0.9703165443,0.0254739416),
('Elastic Net',10,0.9759441710,0.9563784394,0.0195657317,0.2787212787,0.3206793207,0.0035,0.8890406607,0.5990954774,0.9563784394,0.9759441710,0.0195657317)
]
for sel,k,sa,ga,q2,sp,rp,rho,prot,cross,reference,selective,dau in gbm:
    R.append(row(dataset='Darmanis GBM',selector=sel,k=k,
      q2_status='COMPLETE_CACHE_MATCHED_INTERNAL',q2_selective_auroc=sa,q2_global_auroc=ga,q2_selective_minus_global=q2,
      q2_scope='source-stratified cached broad comparator; internal 5-fold plate-grouped diagnostic',
      q3_semantic_p_auroc=sp,q3_random_p_auroc=rp,q3_scope='internal plate-grouped CV; same folds used for trust selection and utility summary',
      q4_status='COMPLETE_INTERNAL',q4_rho_positive_fraction=rho,q4_protected_order_fraction=prot,q4_cross_boundary_protected_fraction=cross,q4_bound_violations=0,q4_full_topk_certificate=False,q4_objective_status='EXACT_NORMALIZED_CE_FULL_DEV_ROUTING',
      q5_reference_auroc=reference,q5_selective_auroc=selective,q5_delta_auroc=dau,q5_scope='internal 5-fold plate-grouped diagnostic',
      caveat='632 cells from 24 plates and only 2 patients; no patient-level generalization; full-development routing is leakage-sensitive.'))

DF=pd.DataFrame(R,columns=cols)
DF.to_csv(OUT/'FINAL_Q2_Q5_CONFIGURATION_MATRIX_20260922.csv',index=False)

coverage=[]
for ds,g in DF.groupby('dataset',sort=False):
    blocked=bool(g.q2_status.str.contains('BLOCKED').any())
    q2complete=bool((g.q2_status.str.startswith('COMPLETE')|g.q2_status.str.startswith('POSTHOC')).any())
    coverage.append({
      'dataset':ds,
      'Q2_routing':'BLOCKED_NETWORK' if blocked and not q2complete else ('COMPLETE' if q2complete else 'MISSING'),
      'Q3_semantics':'COMPLETE' if g.q3_semantic_p_auroc.notna().any() else 'PARTIAL',
      'Q4_protection':'COMPLETE' if (g.q4_status!='NOT_EVALUATED').any() else 'MISSING',
      'Q5_predictive_value':'COMPLETE' if g.q5_delta_auroc.notna().any() else 'MISSING',
      'Q5_positive_cells':int((g.q5_delta_auroc>0).sum()),
      'Q5_negative_cells':int((g.q5_delta_auroc<0).sum()),
      'Q4_bound_violations_total':int(g.q4_bound_violations.fillna(0).sum())
    })
CV=pd.DataFrame(coverage)
CV.to_csv(OUT/'FINAL_Q2_Q5_DATASET_COVERAGE_20260922.csv',index=False)

def f(x,n=4):
    return '—' if pd.isna(x) else f'{float(x):.{n}f}'
def p(x):
    return '—' if pd.isna(x) else f'{float(x):.4g}'

lines=[
'# Q2–Q5 Six-Dataset Completion Report','',
'Date: 2026-09-22  ',
'Method focus: selective / selective data-confusion-guided correction  ',
'Formal Breast scope: k in {10,20}; k=30 is excluded from formal conclusions.','',
'## 1. Completion status','',
'All cache-only and zero-cost completion analyses were finished. The only unresolved requested cell is GSE272769 strict matched-global Q2. Its matched query identities, source budgets, fold-slot mapping, and cost estimate were frozen before execution, but the remote server returned connection refused for api.deepseek.com, so zero new calls completed. The frozen tranche requires 804 AB/BA calls and has an estimated peak cost of about $0.162.','',
'| Dataset | Q2 | Q3 | Q4 | Q5 |',
'|---|---|---|---|---|'
]
for x in coverage:
    lines.append(f"| {x['dataset']} | {x['Q2_routing']} | {x['Q3_semantics']} | {x['Q4_protection']} | {x['Q5_predictive_value']} |")

lines += ['','## 2. Q2 — Routing value','',
'Routing value is heterogeneous rather than universal.','',
'| Dataset / cell | Selective AUROC | Matched Global AUROC | Selective - Global | Scope |',
'|---|---:|---:|---:|---|']
for _,x in DF.iterrows():
    lines.append(f"| {x.dataset} / {x.selector} k={int(x.k)} | {f(x.q2_selective_auroc)} | {f(x.q2_global_auroc)} | {('BLOCKED' if 'BLOCKED' in x.q2_status else f(x.q2_selective_minus_global))} | {x.q2_scope} |")
lines += ['',
'Renal shows only a very small Selective-over-Global AUROC advantage, though the original external AUPRC advantage is clearer. Breast Selective beats the matched Global comparator in 2 of 6 formal cells and loses in 4 of 6. Hospital gives one Selective win and one tie. CREDIT-G migrated diagnostics give one win and one tie. Darmanis GBM shows Selective above matched Global in all three audited cells, but only in the internal plate-grouped, leakage-sensitive setting.',
'',
'Therefore Q2 supports selective routing as useful in some regimes, not a claim that selective routing always beats broad guidance.']

lines += ['','## 3. Q3 — Semantic value','',
'| Dataset / cell | Semantic p (AUROC) | Random p (AUROC) | Secondary | Scope |',
'|---|---:|---:|---|---|']
for _,x in DF.iterrows():
    sec='—'
    if isinstance(x.q3_secondary_metric,str) and x.q3_secondary_metric:
        sec=f"{x.q3_secondary_metric}: semantic {p(x.q3_semantic_p_secondary)}, random {p(x.q3_random_p_secondary)}"
    lines.append(f"| {x.dataset} / {x.selector} k={int(x.k)} | {p(x.q3_semantic_p_auroc)} | {p(x.q3_random_p_auroc)} | {sec} | {x.q3_scope} |")
lines += ['',
'The clearest semantic-value result is GSE272769 Elastic Net k=50: semantic-shuffle AUROC p≈0.0199 and random-probability AUROC p≈0.0200; Macro-AP gives p≈0.0100 and p≈0.0090. Breast LASSO k=10 is weaker against semantic reassignment (p≈0.055) but separates from its final random-probability null (p≈0.039). Most other configurations do not distinguish the real semantic assignment from reassigned/random guidance.',
'',
'Renal is a useful caution: selective has a positive external point estimate, but both semantic-shuffle and random-probability nulls produce nontrivial gains (AUROC p around 0.14). Downstream gain alone therefore does not identify correct pair semantics as the cause.']

lines += ['','## 4. Q4 — Deterministic protection','',
'| Dataset / cell | rho>0 fraction | Protected orders | Cross-boundary protected | Violations | Full top-k certificate | Status |',
'|---|---:|---:|---:|---:|---|---|']
for _,x in DF.iterrows():
    lines.append(f"| {x.dataset} / {x.selector} k={int(x.k)} | {f(x.q4_rho_positive_fraction)} | {f(x.q4_protected_order_fraction)} | {f(x.q4_cross_boundary_protected_fraction)} | {int(x.q4_bound_violations) if not pd.isna(x.q4_bound_violations) else '—'} | {x.q4_full_topk_certificate} | {x.q4_objective_status} |")
lines += ['',
'Every audited exact or migrated normalized-CE configuration has zero observed displacement-bound violations. This is partial protection, not full top-k invariance: full top-k certificates are usually false. The supported statement is that the correction budget is localized and many data-only orderings remain deterministically protected.',
'',
'CREDIT-G is special: its original selective was additive rank-utility. The Q4 audit therefore uses a post-hoc normalized-CE migration and must remain labeled diagnostic.']

lines += ['','## 5. Q5 — Predictive value','',
'| Dataset / cell | Reference AUROC | selective AUROC | Delta AUROC | Secondary delta | Scope |',
'|---|---:|---:|---:|---:|---|']
for _,x in DF.iterrows():
    sec='—' if not isinstance(x.q5_secondary_metric,str) or not x.q5_secondary_metric else f"{x.q5_secondary_metric} {f(x.q5_delta_secondary)}"
    lines.append(f"| {x.dataset} / {x.selector} k={int(x.k)} | {f(x.q5_reference_auroc)} | {f(x.q5_selective_auroc)} | {f(x.q5_delta_auroc)} | {sec} | {x.q5_scope} |")
lines += ['',
'Positive examples include Renal external (+0.0260 AUROC), GSE272769 EN-50 nested CV (+0.0267), Breast LASSO-10 (+0.0287), Breast SIS-10 (+0.0192), Breast SIS-20 (+0.0148), CREDIT-G GBM-permutation (+0.0132), CREDIT-G Elastic Net (+0.0075), Hospital k=10 (+0.0087), and all three reported Darmanis internal cells. Negative results are retained: Breast LASSO-20 (-0.0070), Breast Elastic Net-20 (-0.0143), and Hospital k=5 (-0.0037).',
'',
'External/sealed/temporal results should be weighted above strict nested CV, and Darmanis plate-grouped internal results should be treated only as stress-test evidence.']

lines += ['','## 6. Safe-borrowing audit','',
'The audit does not justify changing selective. Existing trust grids already include zero borrowing, but positive development utility does not reliably imply positive final utility. Breast LASSO-20, Breast Elastic Net-20, and Hospital k=5 are direct counterexamples. Conversely, several positive final results do not separate from semantic/random nulls.',
'',
'No new threshold or gate was fitted from these already-observed final outcomes. A future minimum-gain rule, null-separation requirement, or trust cap should be predeclared and validated prospectively on new tasks.']

lines += ['','## 7. GSE272769 strict Q2 blocker','',
'- Frozen matched-global budget: 402 unique pair-source measurement slots / 409 fold occurrences.',
'- Calls required: 804 AB/BA.',
'- Source budgets: GSE95233 D01 187 pair-source cells, VANISH 186, GAinS 29.',
'- Estimated prompt tokens: about 537,970.',
'- Estimated cost: about $0.081 off-peak / $0.162 peak.',
'- Pair identities were chosen independently of data-confusion routing and frozen before execution.',
'- Execution completed zero calls because the remote server could not connect to api.deepseek.com.',
'',
'This cell should remain missing/network-blocked. A confusion-enriched cached proxy should not be relabeled as Global.']

lines += ['','## 8. Scientific summary','',
'1. Q2: selective routing can help, but the advantage is dataset- and selector-dependent.',
'2. Q3: GSE272769 gives the clearest evidence that pair-specific semantics matter; many other gains are compatible with randomized or reassigned guidance.',
'3. Q4: the normalized correction gives a non-vacuous deterministic protection audit with zero observed bound violations, but usually not a full top-k certificate.',
'4. Q5: selective has multiple positive external/sealed results and several real negative cells; predictive benefit is heterogeneous.',
'5. Method status: keep the current method unchanged. A safe-borrowing rule should be a prospective extension rather than a post-hoc fit to these outcomes.','',
'## 9. Reproducibility outputs','',
'- Configuration matrix: q3_q4_diagnostics_20260922/05_summary/FINAL_Q2_Q5_CONFIGURATION_MATRIX_20260922.csv',
'- Dataset coverage: q3_q4_diagnostics_20260922/05_summary/FINAL_Q2_Q5_DATASET_COVERAGE_20260922.csv',
'- Completion inventory: q3_q4_diagnostics_20260922/05_summary/Q2_Q5_COMPLETION_INVENTORY.csv',
'- Safe-borrowing audit: q3_q4_diagnostics_20260922/06_q2_q5_completion/artifacts/SAFE_BORROWING_AUDIT/SAFE_BORROWING_AUDIT.md',
'- GSE272769 frozen query pack: q3_q4_diagnostics_20260922/06_q2_q5_completion/artifacts/GSE272769_Q2_MATCHED/Q2_GLOBAL_MATCHED_QUERIES.csv',
'- GSE272769 blocker: q3_q4_diagnostics_20260922/06_q2_q5_completion/artifacts/GSE272769_Q2_MATCHED/NETWORK_BLOCKER.json'
]
(OUT/'Q2_Q5_COMPLETION_REPORT_20260922.md').write_text('\n'.join(lines))

print('WROTE',OUT/'FINAL_Q2_Q5_CONFIGURATION_MATRIX_20260922.csv')
print('WROTE',OUT/'FINAL_Q2_Q5_DATASET_COVERAGE_20260922.csv')
print('WROTE',OUT/'Q2_Q5_COMPLETION_REPORT_20260922.md')
print(CV.to_string(index=False))
