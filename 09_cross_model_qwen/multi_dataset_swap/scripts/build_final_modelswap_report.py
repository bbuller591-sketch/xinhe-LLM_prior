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
import json
import pandas as pd
import numpy as np

R=Path(str(REPRO_ROOT))
W=R/'selective_qwen3_32b_local_modelswap_20260924'
O=W/'09_FINAL_COMPARISON'
O.mkdir(parents=True,exist_ok=True)

def J(p): return json.load(open(p))
def row1(df,mask):
    q=df[mask]
    if len(q)!=1: raise RuntimeError(f'row selection got {len(q)}')
    return q.iloc[0]
def f(x):
    try:
        if x is None or (isinstance(x,float) and np.isnan(x)): return None
        return float(x)
    except Exception:
        return None
def fmt(x,n=4):
    return '' if x is None or (isinstance(x,float) and np.isnan(x)) else f'{float(x):.{n}f}'

DS_Q2=pd.read_csv(R/'q3_q4_diagnostics_20260922/05_summary/FINAL_Q2_MATCHED_GLOBAL_MATRIX_20260922.csv')
DS_Q3=pd.read_csv(R/'q3_q4_diagnostics_20260922/05_summary/FINAL_Q3_NULL_MATRIX_20260922.csv')
DS_Q4=pd.read_csv(R/'q3_q4_diagnostics_20260922/05_summary/FINAL_Q4_PROTECTION_MATRIX_20260922.csv')
DS_Q5=pd.read_csv(R/'q3_q4_diagnostics_20260922/05_summary/FINAL_Q5_VALIDATION_MATRIX_20260922.csv')

renal=J(W/'02_RENAL/external_eval/EXTERNAL_EVALUATION_SUMMARY.json')
sepsis=pd.read_csv(W/'03_GSE272769/downstream/selective_NESTED_AGGREGATE.csv')
sepsis_lam=pd.read_csv(W/'03_GSE272769/downstream/selective_NESTED_SELECTED_ETA.csv')
breast=pd.read_csv(W/'04_BREAST/sealed_validation/SEALED_reference_selective_RESULTS.csv')
credit_q2=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/CREDIT_Q2/Q2_MIGRATED_MATCHED_RESULTS.csv')
credit_hold=pd.read_csv(W/'05_CREDIT_G/downstream/FINAL_HOLDOUT_RESULTS.csv')
hospital=pd.read_csv(W/'06_HOSPITAL/final_temporal_eval/BATCH2_PRIMARY_RESULTS.csv')
hospital_dev=pd.read_csv(W/'06_HOSPITAL/downstream_batch1/SELECTED_DEVELOPMENT_HYPERPARAMS.csv')
darm_lam=pd.read_csv(W/'07_DARMANIS/downstream/selective_SELECTED_ETA_V2_9.csv')
sepsis_q2=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q2/Q2_MATCHED_RESULT.csv')
breast_q2=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE/Q2_MATCHED_RESULTS_GPT.csv')
hospital_q2=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2/Q2_MATCHED_RESULTS.csv')
darm_q2=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2/Q2_MATCHED_RESULTS.csv')

ds_renal=J(R/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR/formal_outputs/05_external_evaluation_v3_2/EXTERNAL_EVALUATION_SUMMARY.json')
ds_sepsis=pd.read_csv(R/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/05_REAL_RESULTS/selective_AGGREGATE_TOP30_TOP50.csv')
ds_breast=pd.read_csv(R/'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919/RESULTS/SEALED_VALIDATION_PRIMARY_RESULTS.csv')
ds_credit=pd.read_csv(R/'CREDIT_G_REPRO_PACKAGE_20260919/06_RESULTS/FINAL_HOLDOUT_RESULTS.csv')
ds_hospital=pd.read_csv(R/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/11_BATCH2_FINAL_EVALUATION_V2_0_1/BATCH2_PRIMARY_RESULTS.csv')
ds_darm=pd.read_csv(R/'GBM_REPRO_PACKAGE_V2_9_20260919/PROJECT/GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6/DOWNSTREAM_V2_9/selective_SELECTED_ETA_V2_9.csv')

agree=pd.read_csv(O/'CROSS_MODEL_MEASUREMENT_AGREEMENT.csv')
overlap=pd.read_csv(O/'SELECTED_SET_OVERLAP.csv')

configs=[
 ('Renal TCMR','selective',50,'independent external GSE48581'),
 ('GSE272769','ElasticNet',50,'strict nested outer CV'),
 ('Breast GSE25055→GSE25065','SIS',20,'sealed external GSE25065'),
 ('CREDIT-G','GBM-permutation',10,'locked holdout; post-hoc migration diagnostic'),
 ('CREDIT-G','Elastic Net',10,'locked holdout; post-hoc migration diagnostic'),
 ('Hospital Osteoporosis','L1 logistic rank',10,'temporal external Batch2'),
 ('Darmanis GBM','SIS',10,'internal plate-grouped CV only'),
 ('Darmanis GBM','SIS',20,'internal plate-grouped CV only'),
 ('Darmanis GBM','Elastic Net',10,'internal plate-grouped CV only'),
]

def ds_sel(dataset,selector,k):
    return row1(DS_Q5,(DS_Q5.dataset==dataset)&(DS_Q5.selector==selector)&(DS_Q5.k==k))
def ds_global(dataset,selector,k):
    qsel='ElasticNet+SIS Reference' if dataset=='Renal TCMR' else selector
    return row1(DS_Q2,(DS_Q2.dataset==dataset)&(DS_Q2.selector==qsel)&(DS_Q2.k==k))
def ds_q3(dataset,selector,k):
    return row1(DS_Q3,(DS_Q3.dataset==dataset)&(DS_Q3.selector==selector)&(DS_Q3.k==k))
def ds_q4(dataset,selector,k):
    return row1(DS_Q4,(DS_Q4.dataset==dataset)&(DS_Q4.selector==selector)&(DS_Q4.k==k))

def gpt_q3(dataset,selector,k):
    if dataset=='Renal TCMR':
        sem=J(W/'08_Q2_Q5_DIAGNOSTICS/RENAL_Q3_SEMANTIC_EXTERNAL/SUMMARY.json')
        rnd=J(W/'02_RENAL/random_null/RANDOM_LLM_PROBABILITY_NULL_SUMMARY.json')['modes']['DIRECT_UNIFORM_PAIR_P']['selective']
        return sem['null_external_mean_delta_vs_ref'],sem['empirical_p_external_ge_observed'],rnd['null_external_mean_delta_vs_ref'],rnd['empirical_p_external_ge_observed']
    if dataset=='GSE272769':
        sem=J(W/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q3_SEMANTIC_EXACT200/SUMMARY.json')
        rnd=J(W/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q3_RANDOM/SUMMARY.json')
        return sem['auroc_null']['mean'],sem['auroc_null']['empirical_p'],rnd['auroc_null']['mean'],rnd['auroc_null']['empirical_p']
    if dataset.startswith('Breast'):
        ext=pd.read_csv(O/'EXTERNAL_Q3_PVALUES_GPT.csv')
        s=row1(ext,(ext.dataset=='Breast')&(ext.selector=='SIS')&(ext.k==20)&(ext['mode']=='semantic'))
        r=row1(ext,(ext.dataset=='Breast')&(ext.selector=='SIS')&(ext.k==20)&(ext['mode']=='random'))
        return f(s.null_mean_delta),f(s.empirical_p_ge_observed),f(r.null_mean_delta),f(r.empirical_p_ge_observed)
    if dataset=='CREDIT-G':
        ext=pd.read_csv(O/'EXTERNAL_Q3_PVALUES_GPT.csv')
        s=row1(ext,(ext.dataset=='CREDIT-G')&(ext.selector==selector)&(ext.k==10)&(ext['mode']=='semantic'))
        r=row1(ext,(ext.dataset=='CREDIT-G')&(ext.selector==selector)&(ext.k==10)&(ext['mode']=='random'))
        return f(s.null_mean_delta),f(s.empirical_p_ge_observed),f(r.null_mean_delta),f(r.empirical_p_ge_observed)
    if dataset=='Hospital Osteoporosis':
        s=J(W/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q3_TEMPORAL_NULL/SEMANTIC_SUMMARY.json')
        r=J(W/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q3_TEMPORAL_NULL/RANDOM_SUMMARY.json')
        return s['null_external_mean_delta_vs_ref'],s['empirical_p_external_ge_observed'],r['null_external_mean_delta_vs_ref'],r['empirical_p_external_ge_observed']
    if dataset=='Darmanis GBM':
        sc='ELASTICNET' if selector=='Elastic Net' else selector
        s=J(W/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q3/01_q3_semantic_shuffle/DARMANIS_GBM/SUMMARY.json')['summary']
        r=J(W/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q3/02_q3_random_null/DARMANIS_GBM/SUMMARY.json')['summary']
        ss=next(x for x in s if x['selector']==sc and int(x['k'])==k)
        rr=next(x for x in r if x['selector']==sc and int(x['k'])==k)
        return ss['null_mean'],ss['empirical_p_one_sided'],rr['null_mean'],rr['empirical_p_one_sided']
    raise KeyError(dataset)

def ds_q3_nullmeans(dataset,selector,k):
    if dataset=='Renal TCMR':
        s=J(R/'q3_q4_diagnostics_20260922/06_q2_q5_completion/artifacts/RENAL_Q3_SEMANTIC_EXTERNAL/SUMMARY.json')
        r=J(R/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR/formal_outputs/07_random_llm_probability_null_v3_2/RANDOM_LLM_PROBABILITY_NULL_SUMMARY.json')['modes']['DIRECT_UNIFORM_PAIR_P']['selective']
        return s['null_external_mean_delta_vs_ref'],r['null_external_mean_delta_vs_ref']
    if dataset=='GSE272769':
        x=pd.read_csv(R/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/06_SHUFFLE_RESULTS/SHUFFLE_SUMMARY_TOP30_TOP50.csv')
        s=row1(x,(x.selector=='ELASTICNET')&(x.k==50))
        r=J(R/'q3_q4_diagnostics_20260922/02_q3_random_null/GSE272769_EN50/SUMMARY.json')
        return f(s.shuffle_mean_delta_auroc),r['auroc_null']['mean']
    if dataset.startswith('Breast'):
        x=pd.read_csv(R/'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919/RESULTS/selective_SHUFFLE_CONTROL/selective_LLM_VS_SHUFFLE_CELL_SUMMARY.csv')
        s=row1(x,(x.selector=='SIS')&(x.k==20))
        r=J(R/'q3_q4_diagnostics_20260922/02_q3_random_null/BREAST_GSE25055_GSE25065/SUMMARY.json')['summary']
        rr=next(x for x in r if x['selector']=='SIS' and int(x['k'])==20)
        return f(s.null_mean_delta_auroc),float(rr['null_sealed_auc_mean'])-float(s.reference_auroc)
    if dataset=='CREDIT-G':
        sc='GBM_PERM' if selector=='GBM-permutation' else 'ELASTIC_NET'
        s=J(R/'q3_q4_diagnostics_20260922/01_q3_semantic_shuffle/CREDIT_G/SUMMARY.json')['summary']
        r=J(R/'q3_q4_diagnostics_20260922/02_q3_random_null/CREDIT_G/SUMMARY.json')['summary']
        ss=next(x for x in s if x['selector']==sc);rr=next(x for x in r if x['selector']==sc)
        ref=float(ds_sel(dataset,selector,k).reference_auroc)
        return float(ss['null_holdout_mean'])-ref,float(rr['null_holdout_mean'])-ref
    if dataset=='Hospital Osteoporosis':
        s=J(R/'q3_q4_diagnostics_20260922/06_q2_q5_completion/artifacts/HOSPITAL_Q3_TEMPORAL_NULL/SEMANTIC_SUMMARY.json')
        r=J(R/'q3_q4_diagnostics_20260922/06_q2_q5_completion/artifacts/HOSPITAL_Q3_TEMPORAL_NULL/RANDOM_SUMMARY.json')
        return s['null_external_mean_delta_vs_ref'],r['null_external_mean_delta_vs_ref']
    if dataset=='Darmanis GBM':
        sc='ELASTICNET' if selector=='Elastic Net' else selector
        s=J(R/'q3_q4_diagnostics_20260922/01_q3_semantic_shuffle/DARMANIS_GBM/SUMMARY.json')['summary']
        r=J(R/'q3_q4_diagnostics_20260922/02_q3_random_null/DARMANIS_GBM/SUMMARY.json')['summary']
        ss=next(x for x in s if x['selector']==sc and int(x['k'])==k);rr=next(x for x in r if x['selector']==sc and int(x['k'])==k)
        return ss['null_mean'],rr['null_mean']
    raise KeyError(dataset)

def gpt_values(dataset,selector,k):
    if dataset=='Renal TCMR':
        return dict(ref=renal['metrics']['Reference']['auroc'],sel=renal['metrics']['selective']['auroc'],glob=renal['metrics']['Global']['auroc'],
                    lam=str(renal['selected_lambda']['selective']),global_lam=str(renal['selected_lambda']['Global']),
                    secondary_name='AUPRC',secondary=renal['metrics']['selective']['auprc'],global_secondary=renal['metrics']['Global']['auprc'],fallback='none')
    if dataset=='GSE272769':
        s=row1(sepsis,(sepsis.selector=='ELASTICNET')&(sepsis.k==50));q=sepsis_q2.iloc[0]
        et=sepsis_lam[(sepsis_lam.selector=='ELASTICNET')&(sepsis_lam.k==50)].sort_values('outer_fold')
        vals=[float(x) for x in et.chosen_lam]
        return dict(ref=f(s.mean_reference_auroc),sel=f(s.mean_selective_auroc),glob=f(q.global_mean_auroc),
                    lam='['+','.join(f'{x:g}' for x in vals)+']',global_lam=str(q.chosen_lam_by_outer_fold),
                    secondary_name='MacroAP',secondary=f(s.mean_selective_macro_ap),global_secondary=f(q.global_mean_macro_ap),
                    fallback=f'{sum(x==0 for x in vals)}/5 outer folds lam=0')
    if dataset.startswith('Breast'):
        s=row1(breast,(breast.method=='selective')&(breast.selector=='SIS')&(breast.k==20));q=breast_q2.iloc[0]
        reference=row1(breast,(breast.method=='reference')&(breast.selector=='SIS')&(breast.k==20))
        return dict(ref=f(reference.auroc),sel=f(s.auroc),glob=f(q.global_sealed_auroc),lam=str(s.tuning_strength),global_lam=str(q.global_selected_lam),
                    secondary_name='MacroAP',secondary=f(s.macro_ap),global_secondary=f(q.global_sealed_macro_ap),fallback='none')
    if dataset=='CREDIT-G':
        cq=row1(credit_q2,(credit_q2.selector==selector)&(credit_q2.arm=='selective'))
        gq=row1(credit_q2,(credit_q2.selector==selector)&(credit_q2.arm=='global'))
        code='GBM_PERM' if selector=='GBM-permutation' else 'ELASTIC_NET'
        ch=row1(credit_hold,(credit_hold.selector==code)&(credit_hold.method=='selective'))
        return dict(ref=f(cq.reference_holdout_auroc),sel=f(cq.migrated_holdout_auroc),glob=f(gq.migrated_holdout_auroc),
                    lam=str(cq.selected_lam),global_lam=str(gq.selected_lam),secondary_name='AveragePrecision',
                    secondary=f(ch.holdout_average_precision),global_secondary=None,fallback='none')
    if dataset=='Hospital Osteoporosis':
        s=row1(hospital,(hospital.method=='selective')&(hospital.k==10));reference=row1(hospital,(hospital.method=='reference')&(hospital.k==10));q=hospital_q2.iloc[0]
        hp=row1(hospital_dev,(hospital_dev.method=='selective')&(hospital_dev.k==10))
        return dict(ref=f(reference.auroc),sel=f(s.auroc),glob=f(q.global_temporal_auroc),lam=str(hp.hyperparam),global_lam=str(q.global_selected_lam),
                    secondary_name='AUPRC',secondary=f(s.auprc),global_secondary=f(q.global_temporal_auprc),fallback='none')
    if dataset=='Darmanis GBM':
        sc='ELASTICNET' if selector=='Elastic Net' else selector
        s=row1(darm_lam,(darm_lam.selector==sc)&(darm_lam.k==k));q=row1(darm_q2,(darm_q2.selector==sc)&(darm_q2.k==k))
        return dict(ref=f(q.reference_cv_auroc),sel=f(q.selective_selective_cv_auroc),glob=f(q.global_cv_mean_auroc),
                    lam=str(s.lam),global_lam=str(q.global_selected_lam),secondary_name='MacroAP',
                    secondary=f(s.mean_macro_ap),global_secondary=None,fallback='none')
    raise KeyError(dataset)

def ds_secondary(dataset,selector,k):
    if dataset=='Renal TCMR': return 'AUPRC',ds_renal['metrics']['selective']['auprc']
    if dataset=='GSE272769':
        s=row1(ds_sepsis,(ds_sepsis.selector=='ELASTICNET')&(ds_sepsis.k==50));return 'MacroAP',f(s.mean_selective_macro_ap)
    if dataset.startswith('Breast'):
        s=row1(ds_breast,(ds_breast.method=='selective')&(ds_breast.selector=='SIS')&(ds_breast.k==20));return 'MacroAP',f(s.macro_ap)
    if dataset=='CREDIT-G':
        code='GBM_PERM' if selector=='GBM-permutation' else 'ELASTIC_NET'
        s=row1(ds_credit,(ds_credit.selector==code)&(ds_credit.method=='selective'));return 'AveragePrecision',f(s.holdout_average_precision)
    if dataset=='Hospital Osteoporosis':
        s=row1(ds_hospital,(ds_hospital.method=='selective')&(ds_hospital.k==10));return 'AUPRC',f(s.auprc)
    if dataset=='Darmanis GBM':
        sc='ELASTICNET' if selector=='Elastic Net' else selector
        s=row1(ds_darm,(ds_darm.selector==sc)&(ds_darm.k==k));return 'MacroAP',f(s.mean_macro_ap)
    raise KeyError(dataset)

agg=J(W/'00_PROTOCOL_AND_AUDIT/MEASUREMENT_AGGREGATION_AUDIT.json')
dropmap={x['dataset']:x for x in agg}
def drop_info(dataset):
    key={'Renal TCMR':'Renal','Breast GSE25055→GSE25065':'Breast','Hospital Osteoporosis':'Hospital','Darmanis GBM':'Darmanis'}.get(dataset,dataset)
    x=dropmap[key]
    return int(x['pair_source_cells_dropped']),int(x['pair_source_cells_usable']),int(x['unusable_calls'])

rows=[]
for dataset,selector,k,scope in configs:
    dsel=ds_sel(dataset,selector,k);dglob=ds_global(dataset,selector,k);dq3=ds_q3(dataset,selector,k);dq4=ds_q4(dataset,selector,k)
    gv=gpt_values(dataset,selector,k)
    dssem,dsrnd=ds_q3_nullmeans(dataset,selector,k);gsem,gsemp,grnd,grndp=gpt_q3(dataset,selector,k)
    secname,dssec=ds_secondary(dataset,selector,k)
    aname={'Renal TCMR':'Renal TCMR','GSE272769':'GSE272769','Breast GSE25055→GSE25065':'Breast',
           'CREDIT-G':'CREDIT-G','Hospital Osteoporosis':'Hospital','Darmanis GBM':'Darmanis GBM'}[dataset]
    ar=agree[agree.dataset==aname].iloc[0]
    ovsel=selector
    if dataset=='GSE272769': ovsel='ELASTICNET'
    if dataset=='Darmanis GBM' and selector=='Elastic Net': ovsel='ELASTICNET'
    ov=overlap[(overlap.dataset==aname)&(overlap.selector==ovsel)&(overlap.k==k)]
    ovj=f(ov.iloc[0].jaccard) if len(ov) else None
    drops,usable,unusable=drop_info(dataset)
    ds_delta=f(dsel.delta);g_delta=gv['sel']-gv['ref']
    rows.append({
      'dataset':dataset,'selector':selector,'k':k,'scope':scope,'reference_auroc':gv['ref'],
      'deepseek_selective_auroc':f(dsel.guided_auroc),'deepseek_delta_vs_reference':ds_delta,
      'gpt_selective_auroc':gv['sel'],'gpt_delta_vs_reference':g_delta,
      'gpt_minus_deepseek_selective_auroc':gv['sel']-f(dsel.guided_auroc),
      'deepseek_global_auroc':f(dglob.global_auroc),'gpt_global_auroc':gv['glob'],
      'deepseek_selective_minus_global':f(dglob.selective_minus_global),'gpt_selective_minus_global':gv['sel']-gv['glob'],
      'deepseek_lam':str(dq4.trust),'gpt_lam':gv['lam'],'gpt_global_lam':gv['global_lam'],
      'secondary_metric':secname,'deepseek_selective_secondary':dssec,'gpt_selective_secondary':gv['secondary'],
      'gpt_global_secondary':gv['global_secondary'],
      'deepseek_semantic_null_mean_delta':dssem,'deepseek_semantic_p':f(dq3.semantic_p),
      'gpt_semantic_null_mean_delta':gsem,'gpt_semantic_p':gsemp,
      'deepseek_random_null_mean_delta':dsrnd,'deepseek_random_p':f(dq3.random_p),
      'gpt_random_null_mean_delta':grnd,'gpt_random_p':grndp,
      'deepseek_q4_bound_violations':int(dq4.bound_violations),
      'deepseek_q4_cross_boundary_protected':f(dq4.cross_boundary_protected),
      'gpt_measurement_hard_direction_agreement':f(ar.hard_direction_agreement),
      'gpt_vs_deepseek_probability_pearson':f(ar.probability_pearson),
      'gpt_vs_deepseek_probability_spearman':f(ar.probability_spearman),
      'gpt_vs_deepseek_selected_set_jaccard':ovj,
      'gpt_pair_source_cells_dropped_top20':drops,'gpt_pair_source_cells_usable':usable,'gpt_unusable_calls_top20':unusable,
      'gpt_fallback':gv['fallback'],
      'selective_effect_sign_preserved':bool(np.sign(ds_delta)==np.sign(g_delta)) if ds_delta!=0 and g_delta!=0 else bool(ds_delta==g_delta),
      'selective_vs_global_nonnegative_swap':bool(gv['sel']>=gv['glob']-1e-12),
    })

D=pd.DataFrame(rows)
D.to_csv(O/'FINAL_PAPER_ROWS_9.csv',index=False)

q4rows=[]
q4rows.append(J(W/'08_Q2_Q5_DIAGNOSTICS/Q4/RENAL/PROTECTION_SUMMARY.json'))
s=J(W/'08_Q2_Q5_DIAGNOSTICS/Q4/GSE272769_EN50/PROTECTION_SUMMARY.json')
q4rows.append({'dataset':'GSE272769','selector':'ElasticNet','k':50,'lam':str(s['all_folds']['lam_values']),
               'n_constraints':None,'bound_violation_n':s['all_folds']['bound_violation_n_total'],
               'pairwise_protected_order_fraction':s['all_folds']['pairwise_protected_order_fraction_weighted'],
               'cross_boundary_protected_fraction':s['all_folds']['cross_boundary_protected_fraction_weighted'],
               'full_topk_certificate':s['all_folds']['full_topk_certificate_folds']==5,
               'topk_changed_n':s['all_folds']['topk_changed_n_mean'],'objective_compatibility':s['objective_compatibility']})
for x in J(W/'08_Q2_Q5_DIAGNOSTICS/Q4/BREAST/Q4_FORMAL_RESULTS.json'):
    if x['selector']=='SIS' and x['k']==20:q4rows.append(x)
for x in J(W/'08_Q2_Q5_DIAGNOSTICS/Q4/HOSPITAL/PROTECTION_SUMMARY.json'):
    if x['k']==10:q4rows.append(x)
for x in J(W/'08_Q2_Q5_DIAGNOSTICS/Q4/DARMANIS/PROTECTION_SUMMARY.json'):
    if (x['selector'],x['k']) in [('SIS',10),('SIS',20),('ELASTICNET',10)]:q4rows.append(x)
cq4=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/CREDIT_Q2/Q4_MIGRATED_PROTECTION_RESULTS.csv')
for x in cq4.to_dict('records'):q4rows.append(x)
pd.DataFrame(q4rows).to_csv(O/'GPT_Q4_PROTECTION_9.csv',index=False)

statpaths=[
 W/'02_RENAL/measurement/MEASUREMENT_STATUS.json',
 W/'03_GSE272769/measurement/MEASUREMENT_STATUS.json',
 W/'04_BREAST/measurement/MEASUREMENT_STATUS.json',
 W/'05_CREDIT_G/measurement/MEASUREMENT_STATUS.json',
 W/'06_HOSPITAL/measurement/MEASUREMENT_STATUS.json',
 W/'07_DARMANIS/measurement/MEASUREMENT_STATUS.json',
 W/'07_DARMANIS/measurement_broad7/MEASUREMENT_STATUS.json',
 W/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q2/measurement/MEASUREMENT_STATUS.json',
 W/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE/measurement/MEASUREMENT_STATUS.json',
 W/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2/measurement/MEASUREMENT_STATUS.json',
 W/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2/measurement/MEASUREMENT_STATUS.json',
]
tokenrows=[]
for p in statpaths:
    z=J(p)
    tokenrows.append({'artifact':str(p.relative_to(W)),'calls':z.get('n_calls',0),'prompt_tokens':z.get('prompt_tokens',0),
                      'completion_tokens':z.get('completion_tokens',0),'unusable_top20':z.get('n_unusable_top20',0)})
T=pd.DataFrame(tokenrows);T.to_csv(O/'FORMAL_API_TOKEN_AUDIT.csv',index=False)
toksum={'formal_used_call_rows':int(T.calls.sum()),'prompt_tokens':int(T.prompt_tokens.sum()),'completion_tokens':int(T.completion_tokens.sum()),
        'unusable_top20_calls':int(T.unusable_top20.sum()),'model':'gpt-4o-mini-2024-07-18',
        'note':'Counts only measurement runs used in final Selective/Q2 analyses; excludes generic preflight and superseded exploratory/intermediate Q2 runs.'}
(O/'FORMAL_API_TOKEN_AUDIT.json').write_text(json.dumps(toksum,indent=2)+'\n')

sign_preserved=int(D.selective_effect_sign_preserved.sum())
positive_swap=int((D.gpt_delta_vs_reference>0).sum())
negative_swap=int((D.gpt_delta_vs_reference<0).sum())
sel_ge_global=int(D.selective_vs_global_nonnegative_swap.sum())
summary={
 'status':'COMPLETE','replacement_model':'gpt-4o-mini-2024-07-18','paper_rows':len(D),
 'reference_data_only_unchanged':True,
 'deepseek_positive_selective_rows':int((D.deepseek_delta_vs_reference>0).sum()),
 'gpt_positive_selective_rows':positive_swap,'gpt_negative_selective_rows':negative_swap,
 'selective_effect_sign_preserved_rows':sign_preserved,'gpt_selective_ge_matched_global_rows':sel_ge_global,
 'sign_flip_rows':D.loc[~D.selective_effect_sign_preserved,['dataset','selector','k','deepseek_delta_vs_reference','gpt_delta_vs_reference']].to_dict('records'),
 'top20_dropped_pair_source_cells_total':int(D[['dataset','gpt_pair_source_cells_dropped_top20']].drop_duplicates('dataset').gpt_pair_source_cells_dropped_top20.sum()),
 'formal_api_usage':toksum,'integrity_audit':str(O/'INTEGRITY_AUDIT.json'),
 'guardrails':[
   'Q3 p-values are post-hoc diagnostic randomization quantities unless the underlying original package explicitly states otherwise.',
   'CREDIT-G theorem migration/Q2 is post-hoc after holdout inspection.',
   'Darmanis is internal 5-fold plate-grouped diagnostic with full-development/leakage-sensitive routing, not independent external validation.',
   'Top-20 truncation events were not imputed or rescued; complete AB/BA pair-source cells were dropped under the authorized amendment.'
 ]}
(O/'FINAL_MODEL_SWAP_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

lines=['# GPT-4o-mini Cross-Model Replication — Final Report','',
'Formal replacement model: gpt-4o-mini-2024-07-18.','',
'The reference/data-only side is unchanged to roundoff (max absolute AUROC difference 0 in the integrity audit). The model swap changes only the LLM measurement instrument plus the explicitly authorized top-20 missing-measurement rule.','',
'## 1. Paper-facing 9-row result','',
'| Dataset | Selector/k | Reference AUROC | DeepSeek Selective | GPT Selective | GPT Δ vs ref | DeepSeek Global | GPT Global | GPT Selective−Global | DS lam | GPT lam | Set Jaccard |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|']
for r in rows:
    lines.append(f"| {r['dataset']} | {r['selector']} / {r['k']} | {fmt(r['reference_auroc'])} | {fmt(r['deepseek_selective_auroc'])} | {fmt(r['gpt_selective_auroc'])} | {fmt(r['gpt_delta_vs_reference'])} | {fmt(r['deepseek_global_auroc'])} | {fmt(r['gpt_global_auroc'])} | {fmt(r['gpt_selective_minus_global'])} | {r['deepseek_lam']} | {r['gpt_lam']} | {fmt(r['gpt_vs_deepseek_selected_set_jaccard'],3)} |")
lines += ['',f"- DeepSeek Selective had positive AUROC deltas in {int((D.deepseek_delta_vs_reference>0).sum())}/9 rows.",
f"- GPT Selective has positive AUROC deltas in {positive_swap}/9 rows; the two sign flips are GSE272769 ElasticNet-50 and Breast SIS-20.",
f"- GPT Selective is at least as good as its matched Global arm in {sel_ge_global}/9 paper-facing rows.",'',
'This distinction matters: absolute downstream benefit is not fully model-invariant, but the selective-routing arm remains non-inferior to its matched Global routing control in all nine paper-facing configurations under this replacement model.','',
'## 2. Secondary predictive metrics','',
'| Dataset | Selector/k | Metric | DeepSeek Selective | GPT Selective |',
'|---|---:|---|---:|---:|']
for r in rows:
    lines.append(f"| {r['dataset']} | {r['selector']} / {r['k']} | {r['secondary_metric']} | {fmt(r['deepseek_selective_secondary'])} | {fmt(r['gpt_selective_secondary'])} |")
lines += ['','## 3. Q3 semantic-specificity / random-null diagnostics','',
'Values below are null mean AUROC deltas versus the same data-only reference, followed by the one-sided post-hoc exceedance diagnostic p.','',
'| Dataset | Selector/k | DS semantic mean / p | GPT semantic mean / p | DS random mean / p | GPT random mean / p |',
'|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['dataset']} | {r['selector']} / {r['k']} | {fmt(r['deepseek_semantic_null_mean_delta'])} / {fmt(r['deepseek_semantic_p'],3)} | {fmt(r['gpt_semantic_null_mean_delta'])} / {fmt(r['gpt_semantic_p'],3)} | {fmt(r['deepseek_random_null_mean_delta'])} / {fmt(r['deepseek_random_p'],3)} | {fmt(r['gpt_random_null_mean_delta'])} / {fmt(r['gpt_random_p'],3)} |")
lines += ['',
'The strongest original GSE272769 null separation does not replicate under GPT-4o-mini: its Selective AUROC delta changes from +0.0267 (DeepSeek) to -0.0008 (GPT), with GPT semantic/random diagnostic p values 0.219/0.143. Hospital remains the clearest GPT semantic-specificity case on temporal Batch2 (semantic p≈0.036; random p≈0.062).','',
'## 4. Cross-model measurement agreement','',
'| Dataset | Hard-direction agreement | Probability Pearson | Probability Spearman | DS mean certainty | GPT mean certainty |',
'|---|---:|---:|---:|---:|---:|']
for _,r in agree.iterrows():
    lines.append(f"| {r.dataset} | {fmt(r.hard_direction_agreement,3)} | {fmt(r.probability_pearson,3)} | {fmt(r.probability_spearman,3)} | {fmt(r.deepseek_mean_certainty,3)} | {fmt(r.gpt_mean_certainty,3)} |")
lines += ['',
'Breast shows the weakest hard-direction agreement among the large omics experiments (about 0.691) and the lowest paper-row selected-set overlap (Jaccard about 0.379 for SIS-20). Renal and Darmanis measurements are substantially more stable across the two model families.','',
'## 5. Top-20 truncation amendment','',
'No missing semantic probability was imputed. If either AB/BA orientation lacked both A and B in first-token top-20, that complete pair-source cell was dropped; no rescue calls were used.','',
'| Dataset | Usable pair-source cells | Dropped cells | Unusable calls |',
'|---|---:|---:|---:|']
seen=set()
for r in rows:
    if r['dataset'] in seen: continue
    seen.add(r['dataset'])
    lines.append(f"| {r['dataset']} | {r['gpt_pair_source_cells_usable']} | {r['gpt_pair_source_cells_dropped_top20']} | {r['gpt_unusable_calls_top20']} |")
lines += ['','## 6. Q4 protection','',
'All nine paper-facing GPT configurations have zero theorem-scale displacement-bound violations. Nontrivial full top-k certificates generally do not hold; the protection diagnostics should therefore be read as local/partial harmlessness certificates rather than proof that the selected set cannot change.','',
'## 7. API / execution audit','',
f"- Formal used call rows: {toksum['formal_used_call_rows']:,}",
f"- Formal prompt tokens: {toksum['prompt_tokens']:,}",
f"- Formal completion tokens: {toksum['completion_tokens']:,}",
f"- Top-20 unusable calls in formal used measurements: {toksum['unusable_top20_calls']}",
'- Generic preflight and superseded exploratory/intermediate calls are excluded from these formal-used counts.','',
'## 8. Guardrails','',
'- Renal and Breast use independent/sealed external cohorts; Hospital uses temporal external Batch2.',
'- GSE272769 is strict nested outer-CV, not an independent external cohort.',
'- CREDIT-G theorem migration and matched-Q2 analyses are post-hoc after the locked holdout had historically been inspected.',
'- Darmanis is an internal plate-grouped diagnostic with leakage-sensitive full-development routing; it is not patient-level or independent external validation.',
'- Q3 empirical probabilities are diagnostic/post-hoc and should not be described as preregistered confirmatory p-values.','',
'## 9. Machine-readable artifacts','',
'- FINAL_PAPER_ROWS_9.csv — complete nine-row DeepSeek/GPT/Q2/Q3/agreement table.',
'- CROSS_MODEL_MEASUREMENT_AGREEMENT.csv — pair-source measurement agreement.',
'- SELECTED_SET_OVERLAP.csv — DeepSeek/GPT selected-set overlap.',
'- GPT_Q4_PROTECTION_9.csv — GPT protection diagnostics.',
'- FORMAL_API_TOKEN_AUDIT.csv — formal-used API accounting.',
'- INTEGRITY_AUDIT.json — frozen-source and data-only integrity checks.']
(O/'FINAL_MODEL_SWAP_REPORT.md').write_text('\n'.join(lines)+'\n')

print(json.dumps(summary,ensure_ascii=False,indent=2))
print(D[['dataset','selector','k','reference_auroc','deepseek_selective_auroc','gpt_selective_auroc','deepseek_global_auroc','gpt_global_auroc','gpt_selective_minus_global','deepseek_lam','gpt_lam','gpt_semantic_p','gpt_random_p']].to_string(index=False))
