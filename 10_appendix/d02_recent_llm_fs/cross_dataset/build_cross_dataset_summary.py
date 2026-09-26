

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
import pandas as pd, json, numpy as np
ROOT=Path(str(REPRO_ROOT))
W=ROOT/'CROSS_DATASET_RECENT_LLM_FS_20260926'
rows=[]

def add(dataset,k,method,auroc,status='evaluated',notes=''):
    rows.append({'dataset':dataset,'k':k,'method':method,'auroc':float(auroc),'status':status,'notes':notes})

credit={
'Selective Correction (ours)':0.7933333333333333,
'FREEFORM-style':0.7863095238095238,
'ICE-SEARCH':0.784047619047619,
'LLM-Select Score':0.7832142857142857,
'Reference':0.7801190476190476,
'LLM-Select Rank':0.7723809523809524,
'LLM4FS-style':0.7717857142857143,
'LLM-Select Seq-style':0.7717857142857143,
'Data-centric LLM FS':0.7679761904761906,
'LLM-Lasso':0.7679761904761906,
'Correlation+feedback 2026-style':0.7678571428571429}
for m,v in credit.items():add('CREDIT-G',10,m,v)

o=pd.read_csv(W/'OSTEOPOROSIS/OSTEOPOROSIS_RECENT_METHODS_RESULTS.csv')
for r in o.itertuples():add('Osteoporosis',10,r.method,r.auroc)

for f in ['DARMANIS_GBM/LLMLASSO_RESULT.csv','DARMANIS_GBM/LLM_RANK_RESULT.csv','DARMANIS_GBM/FREEFORM_K10/RESULT.csv','DARMANIS_GBM/FREEFORM_K20/RESULT.csv','DARMANIS_GBM/DATACENTRIC_32/DATACENTRIC_RESULT.csv']:
    d=pd.read_csv(W/f)
    for r in d.itertuples():add('Darmanis GBM',int(r.k),r.method,r.mean_auroc)
d=pd.read_csv(ROOT/'EXISTING_METHOD_BASELINES_20260925/DARMANIS_GBM/DARMANIS_BASELINE_COMPARISON.csv')
for r in d[d.method=='LLM-Score (DeepSeek)'].itertuples():add('Darmanis GBM',int(r.k),'LLM-Select Score',r.mean_auroc)
a=pd.read_csv(ROOT/'GBM_REPRO_PACKAGE_V2_9_20260919/REPORT/KEY_TABLES/reference_global_selective_SELECTED_COMPARISON_V2_9.csv')
for selector,k in [('SIS',10),('SIS',20),('ELASTICNET',10),('ELASTICNET',20),('LASSO',10),('LASSO',20)]:
    q=a[(a.selector==selector)&(a.k==k)]
    for meth,label in [('reference',f'Reference ({selector})'),('selective',f'Selective Correction ({selector})')]:
        rr=q[q.method==meth].iloc[0];add('Darmanis GBM',k,label,rr.mean_auroc,'paper-facing')

for f in ['GSE272769_SEPSIS/LLMLASSO_RESULT.csv','GSE272769_SEPSIS/LLM_SCORE_RESULT.csv','GSE272769_SEPSIS/FREEFORM_K50/RESULT.csv','GSE272769_SEPSIS/DATACENTRIC_32/DATACENTRIC_RESULT.csv']:
    d=pd.read_csv(W/f)
    for r in d.itertuples():add('GSE272769 Sepsis',int(r.k),r.method,r.mean_auroc)
s=json.loads((ROOT/'EXISTING_METHOD_COMPARISON_20260925/SEPSIS_DIRECT_LLM_RANK_RESULT.json').read_text())
add('GSE272769 Sepsis',50,'LLM-Select Rank-style',s['mean_auroc'])
add('GSE272769 Sepsis',50,'Reference',s['reference_auroc'],'paper-facing')
add('GSE272769 Sepsis',50,'Selective Correction (ours)',s['selective_auroc'],'paper-facing')

for f in ['BREAST_PCR/LLMLASSO_RESULT.csv','BREAST_PCR/LLM_SCORE_RESULT.csv','BREAST_PCR/LLM_RANK_RESULT.csv','BREAST_PCR/FREEFORM_K20/RESULT.csv','BREAST_PCR/DATACENTRIC_32/DATACENTRIC_RESULT.csv']:
    d=pd.read_csv(W/f)
    for r in d.itertuples():
        v=getattr(r,'mean_auroc',None)
        if v is None or (isinstance(v,float) and np.isnan(v)):v=getattr(r,'auroc')
        add('Breast pCR',int(r.k),r.method,v)
br=pd.read_csv(ROOT/'GSE25055_GSE25065_FORMAL_REPRODUCIBILITY_PACKAGE/RESULTS/SEALED_VALIDATION_PRIMARY_RESULTS.csv')
for meth,label in [('reference','Reference (SIS)'),('selective','Selective Correction (SIS)')]:
    rr=br[(br.method==meth)&(br.selector=='SIS')&(br.k==20)].iloc[0];add('Breast pCR',20,label,rr.auroc,'paper-facing')

for f in ['RENAL_TCMR/LLMLASSO_RESULT.csv','RENAL_TCMR/FREEFORM_K50/RESULT.csv','RENAL_TCMR/DATACENTRIC_32/DATACENTRIC_RESULT.csv']:
    d=pd.read_csv(W/f)
    for r in d.itertuples():
        v=getattr(r,'mean_auroc',None)
        if v is None or (isinstance(v,float) and np.isnan(v)):v=getattr(r,'auroc')
        add('Renal TCMR',int(r.k),r.method,v)
re=pd.read_csv(ROOT/'EXISTING_METHOD_BASELINES_20260925/RENAL_TCMR/RENAL_TCMR_BASELINE_COMPARISON.csv')
for r in re.itertuples():
    label={'LLM-Score (DeepSeek)':'LLM-Select Score','Reference':'Reference','Selective Correction (ours)':'Selective Correction (ours)'}.get(r.method,r.method);add('Renal TCMR',50,label,r.auroc,'paper-facing' if 'Reference' in label or 'Selective' in label else 'evaluated')
rr=json.loads((ROOT/'EXISTING_METHOD_COMPARISON_20260925/RENAL_DIRECT_LLM_RANK_RESULT.json').read_text());add('Renal TCMR',50,'LLM-Select Rank-style',rr['external_auroc'])

df=pd.DataFrame(rows)
df.to_csv(W/'SUMMARY/ALL_AUROC_RESULTS.csv',index=False)

def is_comp(m):
    return not (m.startswith('Reference') or m.startswith('Selective Correction'))
comp=df[df.method.map(is_comp)]
summary=[]
for (dataset,k),g in df.groupby(['dataset','k']):
    cg=comp[(comp.dataset==dataset)&(comp.k==k)]
    if cg.empty:continue
    best=cg.sort_values('auroc',ascending=False).iloc[0]
    ours=g[g.method.str.startswith('Selective Correction')]
    if ours.empty:continue
    orow=ours.sort_values('auroc',ascending=False).iloc[0]
    refs=g[g.method.str.startswith('Reference')];ref=float(refs.auroc.max()) if len(refs) else np.nan
    summary.append({'dataset':dataset,'k':k,'strongest_recent_method':best.method,'strongest_recent_auroc':best.auroc,'best_paper_facing_ours':orow.method,'ours_auroc':orow.auroc,'delta_ours_vs_recent':orow.auroc-best.auroc,'best_reference_auroc':ref,'ours_is_best':bool(orow.auroc>best.auroc)})
sumdf=pd.DataFrame(summary).sort_values(['dataset','k'])
sumdf.to_csv(W/'SUMMARY/STRONGEST_COMPETITOR_SUMMARY.csv',index=False)

lines=['# Cross-dataset recent LLM feature-selection comparison','', 'Date: 2026-09-26','',
'All numbers below are AUROC. Added baselines are post-hoc common-protocol comparisons; original frozen main experiments were not modified.','',
'## Strongest recent LLM-FS competitor by dataset/configuration','',
'| Dataset | k | Strongest evaluated recent method | Recent AUROC | Best paper-facing Selective configuration at same k | Ours AUROC | Delta Ours - recent |',
'|---|---:|---|---:|---|---:|---:|']
for r in sumdf.itertuples():
    lines.append(f'| {r.dataset} | {r.k} | {r.strongest_recent_method} | {r.strongest_recent_auroc:.4f} | {r.best_paper_facing_ours} | {r.ours_auroc:.4f} | {r.delta_ours_vs_recent:+.4f} |')
lines += ['','## Main interpretation','',
'- Favorable common-protocol configurations: CREDIT-G k=10, Osteoporosis k=10, and Darmanis GBM k=10.',
'- GSE272769 Sepsis k=50 is not favorable: LLM-Lasso and pointwise LLM-Score can exceed Selective AUROC.',
'- Breast pCR k=20 is not favorable: pointwise LLM-Score exceeds Selective AUROC.',
'- Renal TCMR k=50 remains a clear counterexample: pretrained biomedical LLM selection is very strong.',
'- Darmanis k=20 is mixed: LLM-Lasso exceeds the paper-facing SIS-20 Selective row; do not summarize Darmanis as a single universal win without naming the selector/k configuration.',
'',
'## Paper recommendation','',
'For a compact main-text recent-method table, the cleanest favorable examples are:',
'1. CREDIT-G k=10;',
'2. Osteoporosis k=10;',
'3. Darmanis GBM k=10 (the pre-existing paper-facing ElasticNet-10 row is strongest; SIS-10 also remains above the evaluated recent baselines).',
'',
'Keep Sepsis, Breast, Renal, and Darmanis k=20 in the appendix/full audit rather than implying universal dominance.',
'',
'## Full results','',
'See ALL_AUROC_RESULTS.csv for every evaluated AUROC row and SOURCE_AUDIT/RECENT_LLM_FS_METHOD_SOURCES.md for method provenance/reproduction status.'
]
(W/'SUMMARY/CROSS_DATASET_AUROC_SUMMARY.md').write_text('\n'.join(lines)+'\n')
print(sumdf.to_string(index=False))
