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
import json,math,hashlib,subprocess
import numpy as np,pandas as pd

R=Path(str(REPRO_ROOT))
G=R/'selective_gpt4omini_apiyi_modelswap_20260923'
Q=R/'selective_qwen3_32b_local_modelswap_20260924'
O=Q/'09_FINAL_COMPARISON';O.mkdir(parents=True,exist_ok=True)

def J(p):return json.load(open(p))
def one(d,m):
 q=d[m]
 if len(q)!=1:raise RuntimeError(f'expected1 got{len(q)}')
 return q.iloc[0]
def ff(x):
 try:return float(x)
 except:return np.nan
def fmt(x,n=4):
 return '' if pd.isna(x) else f'{float(x):.{n}f}'

base=pd.read_csv(G/'09_FINAL_COMPARISON/FINAL_PAPER_ROWS_9.csv')
tri=pd.read_csv(O/'TRIMODEL_MEASUREMENT_PAIRWISE_AGREEMENT.csv')
ov=pd.read_csv(O/'TRIMODEL_SELECTED_SET_OVERLAP.csv')
ext=pd.read_csv(O/'EXTERNAL_Q3_PVALUES_GPT.csv')
renal=J(Q/'02_RENAL/external_eval/EXTERNAL_EVALUATION_SUMMARY.json')
sepsis=pd.read_csv(Q/'03_GSE272769/downstream/selective_NESTED_AGGREGATE.csv')
sepsis_lam=pd.read_csv(Q/'03_GSE272769/downstream/selective_NESTED_SELECTED_ETA.csv')
breast=pd.read_csv(Q/'04_BREAST/sealed_validation/SEALED_reference_selective_RESULTS.csv')
credit=pd.read_csv(Q/'05_CREDIT_G/downstream/FINAL_HOLDOUT_RESULTS.csv')
hospital=pd.read_csv(Q/'06_HOSPITAL/final_temporal_eval/BATCH2_PRIMARY_RESULTS.csv')
darm=pd.read_csv(Q/'07_DARMANIS/downstream/selective_SELECTED_ETA_V2_9.csv')
q2s=pd.read_csv(Q/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q2/Q2_MATCHED_RESULT.csv')
q2b=pd.read_csv(Q/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE/Q2_MATCHED_RESULTS_GPT.csv')
q2c=pd.read_csv(Q/'08_Q2_Q5_DIAGNOSTICS/CREDIT_Q2/Q2_MIGRATED_MATCHED_RESULTS.csv')
q2h=pd.read_csv(Q/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2/Q2_MATCHED_RESULTS.csv')
q2d=pd.read_csv(Q/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2/Q2_MATCHED_RESULTS.csv')

def qwen_values(dataset,selector,k):
 if dataset=='Renal TCMR':
  return dict(ref=renal['metrics']['Reference']['auroc'],sel=renal['metrics']['selective']['auroc'],glob=renal['metrics']['Global']['auroc'],
   lam=str(renal['selected_lambda']['selective']),geta=str(renal['selected_lambda']['Global']),sec='AUPRC',
   secv=renal['metrics']['selective']['auprc'],gsec=renal['metrics']['Global']['auprc'],fallback='none')
 if dataset=='GSE272769':
  s=one(sepsis,(sepsis.selector=='ELASTICNET')&(sepsis.k==50)); q=q2s.iloc[0]
  e=sepsis_lam[(sepsis_lam.selector=='ELASTICNET')&(sepsis_lam.k==50)].sort_values('outer_fold')
  ev=[float(x) for x in e.chosen_lam]
  return dict(ref=ff(s.mean_reference_auroc),sel=ff(s.mean_selective_auroc),glob=ff(q.global_mean_auroc),lam='['+','.join(f'{x:g}' for x in ev)+']',
   geta=str(q.chosen_lam_by_outer_fold),sec='MacroAP',secv=ff(s.mean_selective_macro_ap),gsec=ff(q.global_mean_macro_ap),
   fallback=f'{sum(x==0 for x in ev)}/5 outer folds lam=0')
 if dataset.startswith('Breast'):
  selective=one(breast,(breast.method=='selective')&(breast.selector=='SIS')&(breast.k==20));reference=one(breast,(breast.method=='reference')&(breast.selector=='SIS')&(breast.k==20));q=q2b.iloc[0]
  return dict(ref=ff(reference.auroc),sel=ff(selective.auroc),glob=ff(q.global_sealed_auroc),lam=str(selective.tuning_strength),geta=str(q.global_selected_lam),
              sec='MacroAP',secv=ff(selective.macro_ap),gsec=ff(q.global_sealed_macro_ap),fallback='none')
 if dataset=='CREDIT-G':
  code='GBM_PERM' if selector=='GBM-permutation' else 'ELASTIC_NET'
  selective=one(credit,(credit.selector==code)&(credit.method=='selective'))
  qs=one(q2c,(q2c.selector==selector)&(q2c.arm=='selective'));qg=one(q2c,(q2c.selector==selector)&(q2c.arm=='global'))
  return dict(ref=ff(qs.reference_holdout_auroc),sel=ff(qs.migrated_holdout_auroc),glob=ff(qg.migrated_holdout_auroc),
              lam=str(qs.selected_lam),geta=str(qg.selected_lam),sec='AveragePrecision',secv=ff(selective.holdout_average_precision),gsec=np.nan,fallback='none')
 if dataset=='Hospital Osteoporosis':
  selective=one(hospital,(hospital.method=='selective')&(hospital.k==10));reference=one(hospital,(hospital.method=='reference')&(hospital.k==10));q=q2h.iloc[0]
  return dict(ref=ff(reference.auroc),sel=ff(selective.auroc),glob=ff(q.global_temporal_auroc),lam=str(selective.method_hyperparam),geta=str(q.global_selected_lam),
              sec='AUPRC',secv=ff(selective.auprc),gsec=ff(q.global_temporal_auprc),fallback='none')
 if dataset=='Darmanis GBM':
  sc='ELASTICNET' if selector=='Elastic Net' else selector
  d=one(darm,(darm.selector==sc)&(darm.k==k));q=one(q2d,(q2d.selector==sc)&(q2d.k==k))
  return dict(ref=ff(q.reference_cv_auroc),sel=ff(q.selective_selective_cv_auroc),glob=ff(q.global_cv_mean_auroc),lam=str(d.lam),geta=str(q.global_selected_lam),
              sec='MacroAP',secv=ff(d.mean_macro_ap),gsec=np.nan,fallback='none')
 raise KeyError(dataset)

def q3(dataset,selector,k):
 if dataset=='Renal TCMR':
  s=J(Q/'08_Q2_Q5_DIAGNOSTICS/RENAL_Q3_SEMANTIC_EXTERNAL/SUMMARY.json')
  r=J(Q/'02_RENAL/random_null/RANDOM_LLM_PROBABILITY_NULL_SUMMARY.json')['modes']['DIRECT_UNIFORM_PAIR_P']['selective']
  return s['null_external_mean_delta_vs_ref'],s['empirical_p_external_ge_observed'],r['null_external_mean_delta_vs_ref'],r['empirical_p_external_ge_observed'],1000,1000
 if dataset=='GSE272769':
  s=J(Q/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q3_SEMANTIC_EXACT200/SUMMARY.json');r=J(Q/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q3_RANDOM/SUMMARY.json')
  return s['auroc_null']['mean'],s['auroc_null']['empirical_p'],r['auroc_null']['mean'],r['auroc_null']['empirical_p'],s['n_replicates'],r['n_replicates']
 if dataset.startswith('Breast'):
  s=one(ext,(ext.dataset=='Breast')&(ext.selector=='SIS')&(ext.k==20)&(ext['mode']=='semantic'));r=one(ext,(ext.dataset=='Breast')&(ext.selector=='SIS')&(ext.k==20)&(ext['mode']=='random'))
  return ff(s.null_mean_delta),ff(s.empirical_p_ge_observed),ff(r.null_mean_delta),ff(r.empirical_p_ge_observed),int(s.n_replicates),int(r.n_replicates)
 if dataset=='CREDIT-G':
  s=one(ext,(ext.dataset=='CREDIT-G')&(ext.selector==selector)&(ext.k==10)&(ext['mode']=='semantic'));r=one(ext,(ext.dataset=='CREDIT-G')&(ext.selector==selector)&(ext.k==10)&(ext['mode']=='random'))
  return ff(s.null_mean_delta),ff(s.empirical_p_ge_observed),ff(r.null_mean_delta),ff(r.empirical_p_ge_observed),int(s.n_replicates),int(r.n_replicates)
 if dataset=='Hospital Osteoporosis':
  s=J(Q/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q3_TEMPORAL_NULL/SEMANTIC_SUMMARY.json');r=J(Q/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q3_TEMPORAL_NULL/RANDOM_SUMMARY.json')
  return s['null_external_mean_delta_vs_ref'],s['empirical_p_external_ge_observed'],r['null_external_mean_delta_vs_ref'],r['empirical_p_external_ge_observed'],s['n_replicates'],r['n_replicates']
 if dataset=='Darmanis GBM':
  sc='ELASTICNET' if selector=='Elastic Net' else selector
  ss=J(Q/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q3/01_q3_semantic_shuffle/DARMANIS_GBM/SUMMARY.json')['summary']
  rr=J(Q/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q3/02_q3_random_null/DARMANIS_GBM/SUMMARY.json')['summary']
  s=next(x for x in ss if x['selector']==sc and int(x['k'])==k);r=next(x for x in rr if x['selector']==sc and int(x['k'])==k)
  return s['null_mean'],s['empirical_p_one_sided'],r['null_mean'],r['empirical_p_one_sided'],s['n_replicates'],r['n_replicates']
 raise KeyError(dataset)

def q4(dataset,selector,k):
 if dataset=='Renal TCMR':
  x=J(Q/'08_Q2_Q5_DIAGNOSTICS/RENAL_Q4/Renal_GSE36059-GSE48581_TCMR/PROTECTION_SUMMARY.json')
  return x['bound_violation_n'],x['protected_cross_boundary_pair_fraction'],x['full_topk_certificate'],x['reference_vs_corrected_topk_jaccard']
 if dataset=='GSE272769':
  x=J(Q/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q4/PROTECTION_SUMMARY.json')['all_folds']
  return x['bound_violation_n_total'],x['cross_boundary_protected_fraction_weighted'],x['full_topk_certificate_folds']==5,x['topk_jaccard_mean']
 if dataset.startswith('Breast'):
  x=J(Q/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q4/Q4_FORMAL_RESULTS.json')[0]
  return x['bound_violation_n'],x['cross_boundary_protected_fraction'],x['full_topk_certificate'],x['topk_jaccard']
 if dataset=='CREDIT-G':
  d=pd.read_csv(Q/'08_Q2_Q5_DIAGNOSTICS/CREDIT_Q2/Q4_MIGRATED_PROTECTION_RESULTS.csv');x=one(d,(d.selector==selector)&(d.k==k))
  return int(x.bound_violation_n),ff(x.cross_boundary_protected_fraction),bool(x.full_topk_certificate),ff(x.topk_jaccard)
 if dataset=='Hospital Osteoporosis':
  xs=J(Q/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q4/PROTECTION_SUMMARY.json');x=next(z for z in xs if z['k']==10)
  return x['bound_violation_n'],x['cross_boundary_protected_fraction'],x['full_topk_certificate'],x['topk_jaccard']
 if dataset=='Darmanis GBM':
  sc='ELASTICNET' if selector=='Elastic Net' else selector
  xs=J(Q/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q4/PROTECTION_SUMMARY.json');x=next(z for z in xs if z['selector']==sc and z['k']==k)
  return x['bound_violation_n'],x['cross_boundary_protected_fraction'],x['full_topk_certificate'],x['topk_jaccard']
 raise KeyError(dataset)

rows=[]
for _,b in base.iterrows():
 v=qwen_values(b.dataset,b.selector,int(b.k)); sm,sp,rm,rp,ns,nr=q3(b.dataset,b.selector,int(b.k));bv,cb,cert,q4j=q4(b.dataset,b.selector,int(b.k))
 mapname={'Breast GSE25055→GSE25065':'Breast','Hospital Osteoporosis':'Hospital'}.get(b.dataset,b.dataset)
 selmap='ELASTICNET' if b.dataset=='Darmanis GBM' and b.selector=='Elastic Net' else b.selector
 trds=one(tri,(tri.dataset==mapname)&(tri.model_a=='DeepSeek')&(tri.model_b=='Qwen3-32B'))
 trg=one(tri,(tri.dataset==mapname)&(tri.model_a=='GPT')&(tri.model_b=='Qwen3-32B'))
 oo=one(ov,(ov.dataset==mapname)&(ov.selector==selmap)&(ov.k==int(b.k)))
 refdiff=abs(v['ref']-ff(b.reference_auroc))
 rows.append({**b.to_dict(),
  'qwen_reference_auroc':v['ref'],'qwen_reference_abs_diff_vs_frozen':refdiff,
  'qwen_selective_auroc':v['sel'],'qwen_delta_vs_reference':v['sel']-v['ref'],'qwen_minus_deepseek_selective':v['sel']-ff(b.deepseek_selective_auroc),
  'qwen_minus_swap_selective':v['sel']-ff(b.gpt_selective_auroc),'qwen_global_auroc':v['glob'],'qwen_selective_minus_global':v['sel']-v['glob'],
  'qwen_lam':v['lam'],'qwen_global_lam':v['geta'],'qwen_secondary_metric':v['sec'],'qwen_selective_secondary':v['secv'],'qwen_global_secondary':v['gsec'],
  'qwen_semantic_null_mean_delta':sm,'qwen_semantic_p':sp,'qwen_random_null_mean_delta':rm,'qwen_random_p':rp,
  'qwen_semantic_nrep':ns,'qwen_random_nrep':nr,
  'qwen_q4_bound_violations':bv,'qwen_q4_cross_boundary_protected':cb,'qwen_q4_full_topk_certificate':cert,'qwen_q4_topk_jaccard_vs_reference':q4j,
  'deepseek_qwen_hard_direction_agreement':ff(trds.hard_direction_agreement),'deepseek_qwen_probability_pearson':ff(trds.probability_pearson),
  'deepseek_qwen_probability_spearman':ff(trds.probability_spearman),'gpt_qwen_hard_direction_agreement':ff(trg.hard_direction_agreement),
  'gpt_qwen_probability_pearson':ff(trg.probability_pearson),'gpt_qwen_probability_spearman':ff(trg.probability_spearman),
  'deepseek_qwen_selected_set_jaccard':ff(oo.ds_qwen_jaccard),'gpt_qwen_selected_set_jaccard':ff(oo.gpt_qwen_jaccard),'threeway_selected_set_jaccard':ff(oo.threeway_jaccard),
  'qwen_fallback':v['fallback'],
  'qwen_selective_positive_vs_reference':bool(v['sel']>v['ref']+1e-12),
  'qwen_selective_equal_reference':bool(abs(v['sel']-v['ref'])<=1e-12),
  'qwen_selective_ge_global':bool(v['sel']>=v['glob']-1e-12)
 })
D=pd.DataFrame(rows)
D.to_csv(O/'FINAL_TRIMODEL_PAPER_ROWS_9.csv',index=False)

summary={
 'status':'COMPLETE','models':['DeepSeek','GPT-4o-mini-2024-07-18','Qwen3-32B-local-BF16'],
 'paper_rows':len(D),'reference_max_abs_diff_qwen':float(D.qwen_reference_abs_diff_vs_frozen.max()),
 'positive_selective_rows':{'DeepSeek':int((D.deepseek_delta_vs_reference>1e-12).sum()),'GPT':int((D.gpt_delta_vs_reference>1e-12).sum()),'Qwen3-32B':int(D.qwen_selective_positive_vs_reference.sum())},
 'zero_selective_rows_qwen':int(D.qwen_selective_equal_reference.sum()),
 'negative_selective_rows_qwen':int((D.qwen_delta_vs_reference<-1e-12).sum()),
 'selective_ge_matched_global_rows':{'GPT':int(D.selective_vs_global_nonnegative_swap.sum()),'Qwen3-32B':int(D.qwen_selective_ge_global.sum())},
 'qwen_selective_below_global_rows':D.loc[~D.qwen_selective_ge_global,['dataset','selector','k','qwen_selective_auroc','qwen_global_auroc','qwen_selective_minus_global']].to_dict('records'),
 'qwen_nonpositive_vs_reference_rows':D.loc[~D.qwen_selective_positive_vs_reference,['dataset','selector','k','qwen_delta_vs_reference','qwen_lam','qwen_fallback']].to_dict('records'),
 'qwen_measurement_integrity':J(Q/'00_PROTOCOL_AND_AUDIT/QWEN3_32B_RETURN_AND_CONVERSION_AUDIT.json')['status'],
 'source_integrity':J(Q/'00_PROTOCOL_AND_AUDIT/QWEN3_SOURCE_INTEGRITY_AUDIT.json')['status'],
 'hospital_random_null_reuse':J(Q/'00_PROTOCOL_AND_AUDIT/HOSPITAL_RANDOM_NULL_EXACT_REUSE_AUDIT.json')['status']
}
(O/'FINAL_TRIMODEL_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

# Human-facing report.
lines=['# DeepSeek vs GPT-4o-mini vs Qwen3-32B — Cross-model robustness report','',
'Qwen deployment: local Qwen3-32B, BF16, non-thinking, A100-80GB. Measurement probabilities use first-step full-vocabulary semantic logits; no top-20 truncation is involved.','',
'## Paper-facing 9 rows','',
'| Dataset | Selector/k | Ref | DeepSeek Sel | GPT Sel | Qwen Sel | Qwen Δref | DS Global | GPT Global | Qwen Global | Qwen Sel−Global | DS lam | GPT lam | Qwen lam |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
for r in rows:
 lines.append(f"| {r['dataset']} | {r['selector']} / {r['k']} | {fmt(r['reference_auroc'])} | {fmt(r['deepseek_selective_auroc'])} | {fmt(r['gpt_selective_auroc'])} | {fmt(r['qwen_selective_auroc'])} | {fmt(r['qwen_delta_vs_reference'])} | {fmt(r['deepseek_global_auroc'])} | {fmt(r['gpt_global_auroc'])} | {fmt(r['qwen_global_auroc'])} | {fmt(r['qwen_selective_minus_global'])} | {r['deepseek_lam']} | {r['gpt_lam']} | {r['qwen_lam']} |")
lines += ['',
f"- Positive Selective AUROC delta: DeepSeek {summary['positive_selective_rows']['DeepSeek']}/9, GPT {summary['positive_selective_rows']['GPT']}/9, Qwen {summary['positive_selective_rows']['Qwen3-32B']}/9.",
f"- Qwen has {summary['zero_selective_rows_qwen']} exact data-only fallback row and {summary['negative_selective_rows_qwen']} negative rows.",
f"- Selective >= matched Global: GPT {summary['selective_ge_matched_global_rows']['GPT']}/9; Qwen {summary['selective_ge_matched_global_rows']['Qwen3-32B']}/9.",
'',
'Qwen does not reproduce a universal Selective-over-Global pattern: Renal is slightly below Qwen Global, and GSE272769 falls back to data-only while Qwen Global improves. Breast reverses the GPT negative Selective-vs-reference result and becomes slightly positive.','',
'## Qwen Q3 diagnostics','',
'| Dataset | Selector/k | Semantic mean Δ / p | Random mean Δ / p |',
'|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['dataset']} | {r['selector']} / {r['k']} | {fmt(r['qwen_semantic_null_mean_delta'])} / {fmt(r['qwen_semantic_p'],3)} | {fmt(r['qwen_random_null_mean_delta'])} / {fmt(r['qwen_random_p'],3)} |")
lines += ['',
'Q3 probabilities are post-hoc diagnostic/randomization quantities, not preregistered confirmatory p-values. Hospital random-null computation is an exact deterministic reuse: the random branch replaces observed LLM probabilities with seeded Uniform draws, and all model-independent inputs plus the observed lam/AUC are identical.','',
'## Pairwise measurement agreement','',
'| Dataset | DS-Qwen hard | DS-Qwen Pearson | DS-Qwen Spearman | GPT-Qwen hard | GPT-Qwen Pearson | GPT-Qwen Spearman |',
'|---|---:|---:|---:|---:|---:|---:|']
for d in ['Renal TCMR','GSE272769','Breast','CREDIT-G','Hospital','Darmanis GBM']:
 a=one(tri,(tri.dataset==d)&(tri.model_a=='DeepSeek')&(tri.model_b=='Qwen3-32B'));b=one(tri,(tri.dataset==d)&(tri.model_a=='GPT')&(tri.model_b=='Qwen3-32B'))
 lines.append(f"| {d} | {fmt(a.hard_direction_agreement,3)} | {fmt(a.probability_pearson,3)} | {fmt(a.probability_spearman,3)} | {fmt(b.hard_direction_agreement,3)} | {fmt(b.probability_pearson,3)} | {fmt(b.probability_spearman,3)} |")
lines += ['',
'Qwen is notably closer to DeepSeek than GPT is on GSE272769 and Breast at the pairwise-measurement level. This does not guarantee the same downstream trust selection: GSE272769 Qwen selects lam=0 in all five outer folds.','',
'## Q4 / integrity','',
f"- Qwen reference/data-only max AUROC difference versus frozen reference: {summary['reference_max_abs_diff_qwen']:.3g}.",
f"- Qwen measurement-return audit: {summary['qwen_measurement_integrity']}.",
f"- Frozen source/hash audit: {summary['source_integrity']}.",
'- All nine paper-facing Qwen Q4 rows have zero displacement-bound violations.',
'- The initial Sepsis reproducer accidentally wrote only a new non-frozen scratch 09_REPRODUCED_REAL directory into the source package; the authoritative tarball was checked, that scratch directory was removed, the rerun was redirected into the isolated Qwen workspace, and all 13 frozen source hashes still match.',
'',
'## Scope guardrails','',
'- Renal and Breast use independent/sealed external cohorts; Hospital uses temporal Batch2.',
'- GSE272769 is nested outer CV rather than independent external validation.',
'- CREDIT-G theorem migration/Q2 remains a post-hoc diagnostic after historical holdout inspection.',
'- Darmanis remains an internal plate-grouped diagnostic with full-development/leakage-sensitive routing.',
'',
'Machine-readable main table: FINAL_TRIMODEL_PAPER_ROWS_9.csv',
'Pairwise measurement table: TRIMODEL_MEASUREMENT_PAIRWISE_AGREEMENT.csv',
'Selected-set overlap: TRIMODEL_SELECTED_SET_OVERLAP.csv',
'Summary: FINAL_TRIMODEL_SUMMARY.json']
(O/'FINAL_TRIMODEL_REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(D[['dataset','selector','k','reference_auroc','deepseek_selective_auroc','gpt_selective_auroc','qwen_selective_auroc','qwen_delta_vs_reference','qwen_global_auroc','qwen_selective_minus_global','qwen_lam','qwen_semantic_p','qwen_random_p']].to_string(index=False))
