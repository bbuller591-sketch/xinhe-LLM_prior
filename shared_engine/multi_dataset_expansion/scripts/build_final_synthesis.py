

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
import pandas as pd, numpy as np, json
from pathlib import Path
R=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
OUT=R/'22_FINAL_SYNTHESIS'; OUT.mkdir(parents=True,exist_ok=True)
ext=pd.read_csv(R/'21_SEALED_VALIDATION/SEALED_VALIDATION_ALL_RESULTS.csv')
rows=[]
for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
    p=ext[(ext.task==task)&(ext.analysis=='PRIMARY')].copy()
    for method in ['reference','global','global_certainty','selective']:
        g=p[p.method==method]
        rows.append({'task':task,'scope':'SEALED_PRIMARY','method':method,
                     'mean_auroc':g.auroc.mean(),'median_auroc':g.auroc.median(),
                     'mean_delta_vs_reference':g.delta_auroc_vs_reference.mean(),
                     'positive_cells':int((g.delta_auroc_vs_reference>0).sum()),
                     'zero_cells':int(np.isclose(g.delta_auroc_vs_reference,0,atol=1e-12).sum()),
                     'negative_cells':int((g.delta_auroc_vs_reference<0).sum())})
    # d10 sensitivity
    for method in ['global_D10','global_certainty_D10']:
        g=ext[(ext.task==task)&(ext.method==method)]
        rows.append({'task':task,'scope':'SEALED_D10_SENS','method':method,
                     'mean_auroc':g.auroc.mean(),'median_auroc':g.auroc.median(),
                     'mean_delta_vs_reference':g.delta_auroc_vs_reference.mean(),
                     'positive_cells':int((g.delta_auroc_vs_reference>0).sum()),
                     'zero_cells':int(np.isclose(g.delta_auroc_vs_reference,0,atol=1e-12).sum()),
                     'negative_cells':int((g.delta_auroc_vs_reference<0).sum())})
    # strict content if present
    for method in ['global_STRICT_CONTENT','global_certainty_STRICT_CONTENT']:
        g=ext[(ext.task==task)&(ext.method==method)]
        if len(g):
            rows.append({'task':task,'scope':'SEALED_STRICT_CONTENT','method':method,
                         'mean_auroc':g.auroc.mean(),'median_auroc':g.auroc.median(),
                         'mean_delta_vs_reference':g.delta_auroc_vs_reference.mean(),
                         'positive_cells':int((g.delta_auroc_vs_reference>0).sum()),
                         'zero_cells':int(np.isclose(g.delta_auroc_vs_reference,0,atol=1e-12).sum()),
                         'negative_cells':int((g.delta_auroc_vs_reference<0).sum())})
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'EXTERNAL_METHOD_AND_SENSITIVITY_SUMMARY.csv',index=False)

# Internal nested summaries.
internal=[]
for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
    selective=pd.read_csv(R/'12_selective_INTERNAL'/task/'selective_NESTED_AGGREGATE.csv')
    selective['method']='selective'
    internal.append(selective[['selector','k','method','mean_delta_auroc']].assign(task=task))
    mm=pd.read_csv(R/'17_global_INTERNAL/D20'/task/'global_NESTED_AGGREGATE.csv')
    internal.append(mm[['selector','k','method','mean_delta_auroc']].assign(task=task))
internal=pd.concat(internal,ignore_index=True)
external=ext[(ext.analysis=='PRIMARY')&(ext.method!='reference')][['task','selector','k','method','delta_auroc_vs_reference','auroc','reference_auroc']]
z=internal.merge(external,on=['task','selector','k','method'],validate='one_to_one')
z['same_sign_nonzero']=np.sign(z.mean_delta_auroc)==np.sign(z.delta_auroc_vs_reference)
z.to_csv(OUT/'INTERNAL_VS_EXTERNAL_DELTA.csv',index=False)

signsum=(z.groupby(['task','method'],as_index=False)
         .agg(internal_mean_delta=('mean_delta_auroc','mean'),
              external_mean_delta=('delta_auroc_vs_reference','mean'),
              sign_agreement_cells=('same_sign_nonzero','sum'),
              n_cells=('same_sign_nonzero','size')))
signsum.to_csv(OUT/'INTERNAL_VS_EXTERNAL_SUMMARY.csv',index=False)

# External top positive/negative cells descriptively.
primary=ext[(ext.analysis=='PRIMARY')&(ext.method!='reference')].copy()
toppos=primary.sort_values('delta_auroc_vs_reference',ascending=False).groupby('task').head(8)
topneg=primary.sort_values('delta_auroc_vs_reference',ascending=True).groupby('task').head(8)
toppos.to_csv(OUT/'TOP_EXTERNAL_POSITIVE_CELLS.csv',index=False)
topneg.to_csv(OUT/'TOP_EXTERNAL_NEGATIVE_CELLS.csv',index=False)

status={
 'status':'FINAL_SYNTHESIS_COMPLETE',
 'sealed_status':json.load(open(R/'21_SEALED_VALIDATION/SEALED_VALIDATION_STATUS.json'))['status'],
 'breast_primary_mean_delta':{m:float(summary[(summary.task=='BREAST_GSE25055_GSE25065')&(summary.scope=='SEALED_PRIMARY')&(summary.method==m)].mean_delta_vs_reference.iloc[0]) for m in ['global','global_certainty','selective']},
 'sepsis_primary_mean_delta':{m:float(summary[(summary.task=='SEPSIS_GSE65682')&(summary.scope=='SEALED_PRIMARY')&(summary.method==m)].mean_delta_vs_reference.iloc[0]) for m in ['global','global_certainty','selective']},
 'no_post_test_retuning':True
}
(OUT/'FINAL_SYNTHESIS_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
print('\nINTERNAL VS EXTERNAL\n',signsum.to_string(index=False))
print('\nTOP POSITIVE\n',toppos[['task','method','selector','k','auroc','delta_auroc_vs_reference']].to_string(index=False))
print('\nTOP NEGATIVE\n',topneg[['task','method','selector','k','auroc','delta_auroc_vs_reference']].to_string(index=False))
print('\n',json.dumps(status,indent=2))
