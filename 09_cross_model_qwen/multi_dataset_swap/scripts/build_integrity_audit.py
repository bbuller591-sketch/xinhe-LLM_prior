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
import csv,hashlib,json,math
import pandas as pd
R=Path(str(REPRO_ROOT));W=R/'selective_qwen3_32b_local_modelswap_20260924';O=W/'09_FINAL_COMPARISON';O.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
# Source manifest recheck
man=pd.read_csv(W/'00_PROTOCOL_AND_AUDIT/SOURCE_INPUT_MANIFEST.csv')
source_checks=[]
for r in man.itertuples():
 p=Path(r.absolute_path);cur=sha(p) if p.exists() else None
 source_checks.append({'path':str(p),'exists':p.exists(),'frozen_sha256':r.sha256,'current_sha256':cur,'match':cur==r.sha256})
# Selective measurement statuses
status_paths={
'Renal':W/'02_RENAL/measurement/MEASUREMENT_STATUS.json',
'GSE272769':W/'03_GSE272769/measurement/MEASUREMENT_STATUS.json',
'Breast':W/'04_BREAST/measurement/MEASUREMENT_STATUS.json',
'CREDIT-G':W/'05_CREDIT_G/measurement/MEASUREMENT_STATUS.json',
'Hospital':W/'06_HOSPITAL/measurement/MEASUREMENT_STATUS.json',
'Darmanis':W/'07_DARMANIS/measurement/MEASUREMENT_STATUS.json'}
statuses={k:json.load(open(p)) for k,p in status_paths.items()}
# Reference/data-only paper rows: original vs GPT result sources.
ds=pd.read_csv(R/'q3_q4_diagnostics_20260922/05_summary/FINAL_Q5_VALIDATION_MATRIX_20260922.csv')
# GPT reference numbers from final/q2 result files
rows=[]
# Renal
j=json.load(open(W/'02_RENAL/external_eval/EXTERNAL_EVALUATION_SUMMARY.json'))
rows.append(('Renal TCMR','selective',50,float(ds[(ds.dataset=='Renal TCMR')].iloc[0].reference_auroc),float(j['metrics']['Reference']['auroc'])))
# Sepsis
q=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q2/Q2_MATCHED_RESULT.csv').iloc[0]
rows.append(('GSE272769','ElasticNet',50,float(ds[(ds.dataset=='GSE272769')].iloc[0].reference_auroc),float(q.reference_mean_auroc)))
# Breast SIS20
q=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE/Q2_MATCHED_RESULTS_GPT.csv').iloc[0]
drow=ds[(ds.dataset.str.contains('Breast'))&(ds.selector=='SIS')&(ds.k==20)].iloc[0]
rows.append(('Breast','SIS',20,float(drow.reference_auroc),float(q.reference_sealed_auroc)))
# Credit
cq=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/CREDIT_Q2/Q2_MIGRATED_MATCHED_RESULTS.csv')
for sel,label in [('GBM-permutation','GBM-permutation'),('Elastic Net','Elastic Net')]:
 drow=ds[(ds.dataset=='CREDIT-G')&(ds.selector==sel)&(ds.k==10)].iloc[0]
 g=cq[(cq.selector==sel)&(cq.arm=='selective')].iloc[0]
 rows.append(('CREDIT-G',label,10,float(drow.reference_auroc),float(g.reference_holdout_auroc)))
# Hospital
q=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2/Q2_MATCHED_RESULTS.csv').iloc[0]
drow=ds[(ds.dataset=='Hospital Osteoporosis')&(ds.k==10)].iloc[0]
rows.append(('Hospital','L1 logistic rank',10,float(drow.reference_auroc),float(q.reference_temporal_auroc)))
# Darmanis
dq=pd.read_csv(W/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2/Q2_MATCHED_RESULTS.csv')
for sel,k in [('SIS',10),('SIS',20),('Elastic Net',10)]:
 key='ELASTICNET' if sel=='Elastic Net' else sel
 g=dq[(dq.selector==key)&(dq.k==k)].iloc[0]
 drow=ds[(ds.dataset=='Darmanis GBM')&(ds.selector==sel)&(ds.k==k)].iloc[0]
 rows.append(('Darmanis GBM',sel,k,float(drow.reference_auroc),float(g.reference_cv_auroc)))
ref=pd.DataFrame(rows,columns=['dataset','selector','k','deepseek_reference_auroc','gpt_reference_auroc'])
ref['abs_diff']=(ref.deepseek_reference_auroc-ref.gpt_reference_auroc).abs()
ref['match_roundoff']=ref.abs_diff<1e-12
ref.to_csv(O/'REFERENCE_DATA_ONLY_INTEGRITY.csv',index=False)
audit={
 'status':'PASS' if all(x['match'] for x in source_checks) and bool(ref.match_roundoff.all()) else 'FAIL',
 'source_manifest_all_match':all(x['match'] for x in source_checks),
 'source_manifest_checks':source_checks,
 'reference_data_only_all_match_roundoff':bool(ref.match_roundoff.all()),
 'reference_data_only_max_abs_diff':float(ref.abs_diff.max()),
 'selective_measurement_statuses':statuses,
 'top20_amendment':{
   'rule':'do not impute; drop complete pair-source cell if either AB/BA call lacks both semantic A/B in top20; no rescue calls',
   'audit_file':str(W/'00_PROTOCOL_AND_AUDIT/TOP20_TRUNCATION_AMENDMENT.md'),
   'aggregation_audit':json.load(open(W/'00_PROTOCOL_AND_AUDIT/MEASUREMENT_AGGREGATION_AUDIT.json'))
 },
 'chronology_checks':{
   'Renal':'lambda selected development CV before external GSE48581 evaluation',
   'GSE272769':'lam selected inner CV inside each outer fold; outer validation not used for lam tuning',
   'Breast':'final lam/support frozen from development before sealed GSE25065 evaluation',
   'CREDIT-G':'legacy holdout already inspected historically; Q2/theorem migration explicitly post-hoc diagnostic; no GPT retuning on holdout',
   'Hospital':'lam and predictor lambda selected on Batch1 only; temporal Batch2 final evaluation only',
   'Darmanis':'internal 5-fold plate-grouped diagnostic; no independent external cohort; full-development routing remains leakage-sensitive by design'
 }
}
(O/'INTEGRITY_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':audit['status'],'source_manifest_all_match':audit['source_manifest_all_match'],
                  'reference_max_abs_diff':audit['reference_data_only_max_abs_diff'],
                  'measurement_unusable':{k:v.get('n_unusable_top20',0) for k,v in statuses.items()}},indent=2))
