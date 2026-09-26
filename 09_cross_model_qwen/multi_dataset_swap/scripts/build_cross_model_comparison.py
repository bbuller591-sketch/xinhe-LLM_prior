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
import json,math
import numpy as np,pandas as pd
from scipy.stats import pearsonr,spearmanr

R=Path(str(REPRO_ROOT))
W=R/'selective_qwen3_32b_local_modelswap_20260924'
O=W/'09_FINAL_COMPARISON';O.mkdir(parents=True,exist_ok=True)

def corr(a,b,kind='pearson'):
    a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3 or np.std(a[m])==0 or np.std(b[m])==0:return None
    return float((pearsonr if kind=='pearson' else spearmanr)(a[m],b[m]).statistic)
def jacc(a,b):
    a=set(map(str,a));b=set(map(str,b))
    return float(len(a&b)/len(a|b)) if a|b else 1.0
def measurement_row(dataset,ds,gp,key,p_ds,p_gp,c_ds,c_gp,y_ds=None,y_gp=None,n_expected_ds=None,n_expected_gp=None):
    m=ds.merge(gp,on=key,suffixes=('_ds','_swap'),how='inner')
    if y_ds is None:
        yd=(m[p_ds+'_ds'].astype(float)>=.5).astype(int)
        yg=(m[p_gp+'_swap'].astype(float)>=.5).astype(int)
    else:
        yd=m[y_ds+'_ds'].astype(float).round().astype(int);yg=m[y_gp+'_swap'].astype(float).round().astype(int)
    out={
      'dataset':dataset,'deepseek_cells':int(len(ds)) if n_expected_ds is None else int(n_expected_ds),
      'gpt_usable_cells':int(len(gp)) if n_expected_gp is None else int(n_expected_gp),
      'common_cells':int(len(m)),'hard_direction_agreement':float((yd==yg).mean()) if len(m) else None,
      'deepseek_mean_certainty':float(m[c_ds+'_ds'].astype(float).mean()) if len(m) else None,
      'gpt_mean_certainty':float(m[c_gp+'_swap'].astype(float).mean()) if len(m) else None,
      'certainty_pearson':corr(m[c_ds+'_ds'],m[c_gp+'_swap']) if len(m) else None,
      'certainty_spearman':corr(m[c_ds+'_ds'],m[c_gp+'_swap'],'spearman') if len(m) else None,
    }
    if p_ds and p_gp:
      out.update({
        'probability_pearson':corr(m[p_ds+'_ds'],m[p_gp+'_swap']),
        'probability_spearman':corr(m[p_ds+'_ds'],m[p_gp+'_swap'],'spearman'),
        'mean_abs_probability_difference':float(np.mean(np.abs(m[p_ds+'_ds'].astype(float)-m[p_gp+'_swap'].astype(float)))) if len(m) else None,
      })
    else:
      out.update({'probability_pearson':None,'probability_spearman':None,'mean_abs_probability_difference':None})
    return out

mrs=[]
# Renal
ds=pd.read_csv(R/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR/formal_outputs/measurements/v3_2/SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv')
gp=pd.read_csv(W/'02_RENAL/measurements_compat/SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv')
mrs.append(measurement_row('Renal TCMR',ds,gp,['semantic_pair_id','arm'],'p_e_gene_i_gt_gene_j','p_e_gene_i_gt_gene_j','C_e','C_e'))
# Sepsis all unique cells
ds=pd.read_csv(R/'new_dataset_search_20260919/gse272769/measurement/postprocess/selective_PAIR_SOURCE_MEASUREMENTS.csv').drop_duplicates(['arm','unordered_pair_id'])
gp=pd.read_csv(W/'03_GSE272769/selective_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B_ALL.csv')
mrs.append(measurement_row('GSE272769',ds,gp,['arm','unordered_pair_id'],'p_i_over_j','p_i_over_j','certainty_1_minus_H','certainty_1_minus_H','hard_y_i_over_j','hard_y_i_over_j'))
# Breast
ds=pd.read_csv(R/'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919/MEASUREMENTS/selective/selective_PAIR_SOURCE_MEASUREMENTS.csv')
gp=pd.read_csv(W/'04_BREAST/selective_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B.csv')
mrs.append(measurement_row('Breast',ds,gp,['arm','unordered_pair_id'],'p_i_over_j','p_i_over_j','certainty_1_minus_H','certainty_1_minus_H','hard_y_i_over_j','hard_y_i_over_j'))
# Credit
ds=pd.read_csv(R/'CREDIT_G_REPRO_PACKAGE_20260919/06_RESULTS/selective_PAIR_MEASUREMENT.csv')
gp=pd.read_csv(W/'05_CREDIT_G/downstream/selective_PAIR_MEASUREMENT.csv')
mrs.append(measurement_row('CREDIT-G',ds,gp,['pair_id'],'p_canonical_a_order_neutral','p_canonical_a_order_neutral','c_edge_entropy','c_edge_entropy'))
# Hospital
ds=pd.read_csv(R/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/08_LLM_MEASUREMENT/BT_ANALYSIS_V1_8_1/selective_SELECTIVE30_SHARED_MEASUREMENTS.csv')
gp=pd.read_csv(W/'06_HOSPITAL/selective_SELECTIVE30_SHARED_MEASUREMENTS_QWEN3_32B.csv')
mrs.append(measurement_row('Hospital',ds,gp,['pair_id_selective'],'p_selective_A','p_selective_A','c_pair','c_pair'))
# Darmanis (downstream file stores hard y/c only for DeepSeek)
ds=pd.read_csv(R/'GBM_REPRO_PACKAGE_V2_9_20260919/PROJECT/GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6/DOWNSTREAM_V2_9/selective_COMPLETE_PAIR_SOURCE_MEASUREMENTS_V2_9.csv')
gp=pd.read_csv(W/'07_DARMANIS/selective_COMPLETE_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B.csv')
m=ds.merge(gp,on=['node_i','node_j','arm'],suffixes=('_ds','_swap'))
mrs.append({
 'dataset':'Darmanis GBM','deepseek_cells':len(ds),'gpt_usable_cells':len(gp),'common_cells':len(m),
 'hard_direction_agreement':float((m.y_ds.astype(int)==m.y_swap.astype(int)).mean()),
 'deepseek_mean_certainty':float(m.c_ds.mean()),'gpt_mean_certainty':float(m.c_swap.mean()),
 'certainty_pearson':corr(m.c_ds,m.c_swap),'certainty_spearman':corr(m.c_ds,m.c_swap,'spearman'),
 'probability_pearson':None,'probability_spearman':None,'mean_abs_probability_difference':None
})
mdf=pd.DataFrame(mrs);mdf.to_csv(O/'CROSS_MODEL_MEASUREMENT_AGREEMENT.csv',index=False)

# Selected-set overlap for paper-facing 9 rows.
ov=[]
# Renal
a=pd.read_csv(R/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR/formal_outputs/02_downstream_dev_v3_2/SELECTED_SUPPORTS_FROZEN.csv')
b=pd.read_csv(W/'02_RENAL/downstream_dev/SELECTED_SUPPORTS_FROZEN.csv')
sa=a[a.method=='selective'].gene.astype(str);sb=b[b.method=='selective'].gene.astype(str)
ov.append(dict(dataset='Renal TCMR',selector='selective',k=50,deepseek_n=len(sa),gpt_n=len(sb),intersection=len(set(sa)&set(sb)),jaccard=jacc(sa,sb),exact_set_match=set(sa)==set(sb)))
# Sepsis foldwise EN50
a=pd.read_csv(R/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/05_REAL_RESULTS/selective_OUTER_RESULTS_TOP30_TOP50.csv')
b=pd.read_csv(W/'03_GSE272769/downstream/selective_NESTED_OUTER_RESULTS.csv')
js=[];ins=[]
for f in range(1,6):
    ra=a[(a.outer_fold==f)&(a.selector=='ELASTICNET')&(a.k==50)].iloc[0]
    rb=b[(b.outer_fold==f)&(b.selector=='ELASTICNET')&(b.k==50)].iloc[0]
    aa=str(ra.selected_genes).split('|');bb=str(rb.selected_genes).split('|');js.append(jacc(aa,bb));ins.append(len(set(aa)&set(bb)))
ov.append(dict(dataset='GSE272769',selector='ELASTICNET',k=50,deepseek_n=50,gpt_n=50,intersection=float(np.mean(ins)),jaccard=float(np.mean(js)),exact_set_match=bool(all(x==1 for x in js)),note='mean across 5 outer-fold selected sets'))
# Breast SIS20
a=pd.read_csv(R/'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919/METHOD_ARTIFACTS/selective_FINAL/FINAL_selective_SELECTED_FEATURES.csv')
b=pd.read_csv(W/'04_BREAST/final_development_tuning/BREAST_GSE25055_GSE25065/FINAL_selective_SELECTED_FEATURES.csv')
aa=a[(a.selector=='SIS')&(a.k==20)].gene_symbol.astype(str);bb=b[(b.selector=='SIS')&(b.k==20)].gene_symbol.astype(str)
ov.append(dict(dataset='Breast',selector='SIS',k=20,deepseek_n=len(aa),gpt_n=len(bb),intersection=len(set(aa)&set(bb)),jaccard=jacc(aa,bb),exact_set_match=set(aa)==set(bb)))
# Credit
a=pd.read_csv(R/'CREDIT_G_REPRO_PACKAGE_20260919/06_RESULTS/FINAL_SELECTED_SETS_FREEZE.csv')
b=pd.read_csv(W/'05_CREDIT_G/downstream/FINAL_SELECTED_SETS_FREEZE.csv')
for sel,label in [('GBM_PERM','GBM-permutation'),('ELASTIC_NET','Elastic Net')]:
    ra=a[(a.selector==sel)&(a.method=='selective')].iloc[0]; rb=b[(b.selector==sel)&(b.method=='selective')].iloc[0]
    aa=str(ra.selected_features).split('|');bb=str(rb.selected_features).split('|')
    ov.append(dict(dataset='CREDIT-G',selector=label,k=10,deepseek_n=10,gpt_n=10,intersection=len(set(aa)&set(bb)),jaccard=jacc(aa,bb),exact_set_match=set(aa)==set(bb)))
# Hospital top10
a=pd.read_csv(R/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/09_BATCH1_DOWNSTREAM_V1_9/FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv')
b=pd.read_csv(W/'06_HOSPITAL/downstream_batch1/FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv')
ra=a[(a.method=='selective')&(a.k==10)].iloc[0];rb=b[(b.method=='selective')&(b.k==10)].iloc[0]
aa=str(ra.selected_set).split('|');bb=str(rb.selected_set).split('|')
ov.append(dict(dataset='Hospital',selector='L1 logistic rank',k=10,deepseek_n=10,gpt_n=10,intersection=len(set(aa)&set(bb)),jaccard=jacc(aa,bb),exact_set_match=set(aa)==set(bb)))
# Darmanis
a=pd.read_csv(R/'GBM_REPRO_PACKAGE_V2_9_20260919/PROJECT/GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6/DOWNSTREAM_V2_9/selective_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv')
b=pd.read_csv(W/'07_DARMANIS/downstream/selective_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv')
for sel,k in [('SIS',10),('SIS',20),('ELASTICNET',10)]:
    aa=a[(a.selector==sel)&(a.k==k)].gene_symbol.astype(str);bb=b[(b.selector==sel)&(b.k==k)].gene_symbol.astype(str)
    ov.append(dict(dataset='Darmanis GBM',selector=sel,k=k,deepseek_n=len(aa),gpt_n=len(bb),intersection=len(set(aa)&set(bb)),jaccard=jacc(aa,bb),exact_set_match=set(aa)==set(bb)))
odf=pd.DataFrame(ov);odf.to_csv(O/'SELECTED_SET_OVERLAP.csv',index=False)
summary={'measurement_agreement':mrs,'selected_set_overlap':ov}
(O/'CROSS_MODEL_COMPARISON.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x: bool(x) if isinstance(x,np.bool_) else x)+'\n')
print('\nMEASUREMENT AGREEMENT')
print(mdf.to_string(index=False))
print('\nSELECTED SET OVERLAP')
print(odf.to_string(index=False))
