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
G=R/'selective_gpt4omini_apiyi_modelswap_20260923'
Q=R/'selective_qwen3_32b_local_modelswap_20260924'
O=Q/'09_FINAL_COMPARISON';O.mkdir(parents=True,exist_ok=True)

def corr(a,b,rank=False):
    a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3 or np.std(a[m])==0 or np.std(b[m])==0:return np.nan
    return float((spearmanr if rank else pearsonr)(a[m],b[m]).statistic)
def jac(a,b):
    a=set(map(str,a));b=set(map(str,b));return len(a&b)/len(a|b) if a|b else 1.
def inv_entropy_prob(h,y):
    h=float(h); y=float(y)
    if h<=1e-15:q=1.0
    elif h>=1-1e-15:q=.5
    else:
        lo,hi=.5,1-1e-15
        for _ in range(80):
            m=(lo+hi)/2
            hm=-(m*math.log2(m)+(1-m)*math.log2(1-m))
            if hm>h:lo=m
            else:hi=m
        q=(lo+hi)/2
    return q if y>=.5 else 1-q

datasets={}
# DS/GPT/Qwen normalized tables (key,p,c,y).
# Renal
ds=pd.read_csv(R/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR/formal_outputs/measurements/v3_2/SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv')
gp=pd.read_csv(G/'02_RENAL/measurements_compat/SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv')
qw=pd.read_csv(Q/'02_RENAL/measurements_compat/SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv')
def renal(d):
    z=d[['semantic_pair_id','arm','p_e_gene_i_gt_gene_j','C_e']].copy();z['key']=z.semantic_pair_id.astype(str)+'|'+z.arm.astype(str)
    z['p']=z.p_e_gene_i_gt_gene_j.astype(float);z['c']=z.C_e.astype(float);z['y']=(z.p>=.5).astype(int);return z[['key','p','c','y']]
datasets['Renal TCMR']=(renal(ds),renal(gp),renal(qw))
# Sepsis
def std_pair(d):
    z=d.copy();z['key']=z.arm.astype(str)+'|'+z.unordered_pair_id.astype(str)
    z['p']=z.p_i_over_j.astype(float);z['c']=z.certainty_1_minus_H.astype(float);z['y']=z.hard_y_i_over_j.astype(float).round().astype(int)
    return z[['key','p','c','y']].drop_duplicates('key')
ds=pd.read_csv(R/'new_dataset_search_20260919/gse272769/measurement/postprocess/selective_PAIR_SOURCE_MEASUREMENTS.csv')
gp=pd.read_csv(G/'03_GSE272769/selective_PAIR_SOURCE_MEASUREMENTS_GPT4OMINI_ALL.csv')
qw=pd.read_csv(Q/'03_GSE272769/selective_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B_ALL.csv')
datasets['GSE272769']=(std_pair(ds),std_pair(gp),std_pair(qw))
# Breast
ds=pd.read_csv(R/'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919/MEASUREMENTS/selective/selective_PAIR_SOURCE_MEASUREMENTS.csv')
gp=pd.read_csv(G/'04_BREAST/selective_PAIR_SOURCE_MEASUREMENTS_GPT4OMINI.csv')
qw=pd.read_csv(Q/'04_BREAST/selective_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B.csv')
datasets['Breast']=(std_pair(ds),std_pair(gp),std_pair(qw))
# Credit
def credit(d):
    z=d.copy();z['key']=z.pair_id.astype(str);z['p']=z.p_canonical_a_order_neutral.astype(float);z['c']=z.c_edge_entropy.astype(float);z['y']=(z.p>=.5).astype(int)
    return z[['key','p','c','y']]
ds=pd.read_csv(R/'CREDIT_G_REPRO_PACKAGE_20260919/06_RESULTS/selective_PAIR_MEASUREMENT.csv')
gp=pd.read_csv(G/'05_CREDIT_G/downstream/selective_PAIR_MEASUREMENT.csv')
qw=pd.read_csv(Q/'05_CREDIT_G/downstream/selective_PAIR_MEASUREMENT.csv')
datasets['CREDIT-G']=(credit(ds),credit(gp),credit(qw))
# Hospital
def hosp(d):
    z=d.copy();z['key']=z.pair_id_selective.astype(str);z['p']=z.p_selective_A.astype(float);z['c']=z.c_pair.astype(float);z['y']=(z.p>=.5).astype(int)
    return z[['key','p','c','y']]
ds=pd.read_csv(R/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/08_LLM_MEASUREMENT/BT_ANALYSIS_V1_8_1/selective_SELECTIVE30_SHARED_MEASUREMENTS.csv')
gp=pd.read_csv(G/'06_HOSPITAL/selective_SELECTIVE30_SHARED_MEASUREMENTS_GPT4OMINI.csv')
qw=pd.read_csv(Q/'06_HOSPITAL/selective_SELECTIVE30_SHARED_MEASUREMENTS_QWEN3_32B.csv')
datasets['Hospital']=(hosp(ds),hosp(gp),hosp(qw))
# Darmanis
def darm(d,kind):
    z=d.copy();z['key']=z.node_i.astype(str)+'|'+z.node_j.astype(str)+'|'+z.arm.astype(str)
    if kind=='ds':
        z['p']=[inv_entropy_prob(h,y) for h,y in zip(z.H,z.y)]
    else:z['p']=z.p_i_over_j.astype(float)
    z['c']=z.c.astype(float);z['y']=z.y.astype(float).round().astype(int)
    return z[['key','p','c','y']]
ds=pd.read_csv(R/'GBM_REPRO_PACKAGE_V2_9_20260919/PROJECT/GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6/DOWNSTREAM_V2_9/selective_COMPLETE_PAIR_SOURCE_MEASUREMENTS_V2_9.csv')
gp=pd.read_csv(G/'07_DARMANIS/selective_COMPLETE_PAIR_SOURCE_MEASUREMENTS_GPT4OMINI.csv')
qw=pd.read_csv(Q/'07_DARMANIS/selective_COMPLETE_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B.csv')
datasets['Darmanis GBM']=(darm(ds,'ds'),darm(gp,'gp'),darm(qw,'qw'))

pairs=[('DeepSeek','GPT'),('DeepSeek','Qwen3-32B'),('GPT','Qwen3-32B')]
rows=[];three=[]
for name,(ds,gp,qw) in datasets.items():
    dd={'DeepSeek':ds,'GPT':gp,'Qwen3-32B':qw}
    selective=ds.rename(columns={x:x+'_ds' for x in ['p','c','y']}).merge(
        gp.rename(columns={x:x+'_gp' for x in ['p','c','y']}),on='key').merge(
        qw.rename(columns={x:x+'_qw' for x in ['p','c','y']}),on='key')
    three.append({'dataset':name,'common_threeway_cells':len(selective),
                  'threeway_hard_direction_agreement':float(((selective.y_ds==selective.y_gp)&(selective.y_ds==selective.y_qw)).mean()),
                  'deepseek_mean_certainty':float(selective.c_ds.mean()),'gpt_mean_certainty':float(selective.c_gp.mean()),'qwen_mean_certainty':float(selective.c_qw.mean())})
    for a,b in pairs:
        A=dd[a].rename(columns={x:x+'_a' for x in ['p','c','y']});B=dd[b].rename(columns={x:x+'_b' for x in ['p','c','y']})
        m=A.merge(B,on='key')
        rows.append({'dataset':name,'model_a':a,'model_b':b,'a_cells':len(dd[a]),'b_cells':len(dd[b]),'common_cells':len(m),
                     'hard_direction_agreement':float((m.y_a==m.y_b).mean()),
                     'probability_pearson':corr(m.p_a,m.p_b),'probability_spearman':corr(m.p_a,m.p_b,True),
                     'mean_abs_probability_difference':float(np.mean(np.abs(m.p_a-m.p_b))),
                     'certainty_pearson':corr(m.c_a,m.c_b),'certainty_spearman':corr(m.c_a,m.c_b,True),
                     'model_a_mean_certainty':float(m.c_a.mean()),'model_b_mean_certainty':float(m.c_b.mean())})
pd.DataFrame(rows).to_csv(O/'TRIMODEL_MEASUREMENT_PAIRWISE_AGREEMENT.csv',index=False)
pd.DataFrame(three).to_csv(O/'TRIMODEL_MEASUREMENT_THREEWAY_AGREEMENT.csv',index=False)

# Selected support overlap.
ov=[]
def addsets(dataset,selector,k,dsset,gpset,qwset,note=''):
    ss={'DeepSeek':set(map(str,dsset)),'GPT':set(map(str,gpset)),'Qwen3-32B':set(map(str,qwset))}
    common=len(ss['DeepSeek']&ss['GPT']&ss['Qwen3-32B'])
    union=len(ss['DeepSeek']|ss['GPT']|ss['Qwen3-32B'])
    rec={'dataset':dataset,'selector':selector,'k':k,'threeway_intersection':common,'threeway_jaccard':common/union if union else 1.,'note':note}
    for a,b,key in [('DeepSeek','GPT','ds_swap'),('DeepSeek','Qwen3-32B','ds_qwen'),('GPT','Qwen3-32B','gpt_qwen')]:
        rec[key+'_intersection']=len(ss[a]&ss[b]);rec[key+'_jaccard']=jac(ss[a],ss[b])
    ov.append(rec)
# Renal
a=pd.read_csv(R/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR/formal_outputs/02_downstream_dev_v3_2/SELECTED_SUPPORTS_FROZEN.csv')
b=pd.read_csv(G/'02_RENAL/downstream_dev/SELECTED_SUPPORTS_FROZEN.csv');c=pd.read_csv(Q/'02_RENAL/downstream_dev/SELECTED_SUPPORTS_FROZEN.csv')
addsets('Renal TCMR','selective',50,a[a.method=='selective'].gene,b[b.method=='selective'].gene,c[c.method=='selective'].gene)
# Sepsis foldwise aggregate average pairwise jaccards and intersections
a=pd.read_csv(R/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/05_REAL_RESULTS/selective_OUTER_RESULTS_TOP30_TOP50.csv')
b=pd.read_csv(G/'03_GSE272769/downstream/selective_NESTED_OUTER_RESULTS.csv');c=pd.read_csv(Q/'03_GSE272769/downstream/selective_NESTED_OUTER_RESULTS.csv')
fr=[]
for fold in range(1,6):
    sets=[]
    for d in [a,b,c]:
        x=d[(d.outer_fold==fold)&(d.selector=='ELASTICNET')&(d.k==50)].iloc[0]
        sets.append(set(str(x.selected_genes).split('|')))
    fr.append({'ds_swap_i':len(sets[0]&sets[1]),'ds_swap_j':jac(sets[0],sets[1]),'ds_qwen_i':len(sets[0]&sets[2]),'ds_qwen_j':jac(sets[0],sets[2]),
               'gpt_qwen_i':len(sets[1]&sets[2]),'gpt_qwen_j':jac(sets[1],sets[2]),'three_i':len(sets[0]&sets[1]&sets[2]),
               'three_j':len(sets[0]&sets[1]&sets[2])/len(sets[0]|sets[1]|sets[2])})
ov.append({'dataset':'GSE272769','selector':'ElasticNet','k':50,
           'threeway_intersection':float(np.mean([x['three_i'] for x in fr])),'threeway_jaccard':float(np.mean([x['three_j'] for x in fr])),
           'ds_swap_intersection':float(np.mean([x['ds_swap_i'] for x in fr])),'ds_swap_jaccard':float(np.mean([x['ds_swap_j'] for x in fr])),
           'ds_qwen_intersection':float(np.mean([x['ds_qwen_i'] for x in fr])),'ds_qwen_jaccard':float(np.mean([x['ds_qwen_j'] for x in fr])),
           'gpt_qwen_intersection':float(np.mean([x['gpt_qwen_i'] for x in fr])),'gpt_qwen_jaccard':float(np.mean([x['gpt_qwen_j'] for x in fr])),
           'note':'mean across five outer-fold supports'})
# Breast
a=pd.read_csv(R/'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919/METHOD_ARTIFACTS/selective_FINAL/FINAL_selective_SELECTED_FEATURES.csv')
b=pd.read_csv(G/'04_BREAST/final_development_tuning/BREAST_GSE25055_GSE25065/FINAL_selective_SELECTED_FEATURES.csv')
c=pd.read_csv(Q/'04_BREAST/final_development_tuning/BREAST_GSE25055_GSE25065/FINAL_selective_SELECTED_FEATURES.csv')
addsets('Breast','SIS',20,a[(a.selector=='SIS')&(a.k==20)].gene_symbol,b[(b.selector=='SIS')&(b.k==20)].gene_symbol,c[(c.selector=='SIS')&(c.k==20)].gene_symbol)
# Credit
a=pd.read_csv(R/'CREDIT_G_REPRO_PACKAGE_20260919/06_RESULTS/FINAL_SELECTED_SETS_FREEZE.csv');b=pd.read_csv(G/'05_CREDIT_G/downstream/FINAL_SELECTED_SETS_FREEZE.csv');c=pd.read_csv(Q/'05_CREDIT_G/downstream/FINAL_SELECTED_SETS_FREEZE.csv')
for sel,label in [('GBM_PERM','GBM-permutation'),('ELASTIC_NET','Elastic Net')]:
    aa=str(a[(a.selector==sel)&(a.method=='selective')].iloc[0].selected_features).split('|');bb=str(b[(b.selector==sel)&(b.method=='selective')].iloc[0].selected_features).split('|');cc=str(c[(c.selector==sel)&(c.method=='selective')].iloc[0].selected_features).split('|')
    addsets('CREDIT-G',label,10,aa,bb,cc)
# Hospital
a=pd.read_csv(R/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/09_BATCH1_DOWNSTREAM_V1_9/FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv')
b=pd.read_csv(G/'06_HOSPITAL/downstream_batch1/FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv');c=pd.read_csv(Q/'06_HOSPITAL/downstream_batch1/FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv')
addsets('Hospital','L1 logistic rank',10,str(a[(a.method=='selective')&(a.k==10)].iloc[0].selected_set).split('|'),str(b[(b.method=='selective')&(b.k==10)].iloc[0].selected_set).split('|'),str(c[(c.method=='selective')&(c.k==10)].iloc[0].selected_set).split('|'))
# Darmanis
a=pd.read_csv(R/'GBM_REPRO_PACKAGE_V2_9_20260919/PROJECT/GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6/DOWNSTREAM_V2_9/selective_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv')
b=pd.read_csv(G/'07_DARMANIS/downstream/selective_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv');c=pd.read_csv(Q/'07_DARMANIS/downstream/selective_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv')
for sel,k in [('SIS',10),('SIS',20),('ELASTICNET',10)]:
    addsets('Darmanis GBM',sel,k,a[(a.selector==sel)&(a.k==k)].gene_symbol,b[(b.selector==sel)&(b.k==k)].gene_symbol,c[(c.selector==sel)&(c.k==k)].gene_symbol)
pd.DataFrame(ov).to_csv(O/'TRIMODEL_SELECTED_SET_OVERLAP.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(ov).to_string(index=False))
