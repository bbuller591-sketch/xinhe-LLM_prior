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
import json,hashlib,pandas as pd
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
DOWN=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/02_RENAL/downstream_dev'))
SHUF=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/02_RENAL/semantic_shuffle'))
MEAS=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/02_RENAL/measurements_compat'))
FREEZE=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2/FINAL'
OUT=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/02_RENAL/pre_external_freeze'))
OUT.mkdir(parents=True,exist_ok=True)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((DOWN/'PRE_EXTERNAL_FREEZE_SUMMARY.json').read_text())
shuffle=json.loads((SHUF/'SEMANTIC_SHUFFLE_SUMMARY.json').read_text())
supports=pd.read_csv(DOWN/'SELECTED_SUPPORTS_FROZEN.csv')
assert set(supports.method)=={'Reference','Global','selective','selective_no_certainty-NoC'}
assert all(len(supports[supports.method==m])==50 for m in supports.method.unique())
assert summary['external_used'] is False and shuffle['external_used'] is False
assert summary['study_specific_llm_calls_reused']==400 and shuffle['new_llm_calls']==0

support_hash={}
support_genes={}
for m in ['Reference','Global','selective','selective_no_certainty-NoC']:
    z=supports[supports.method==m].sort_values('selected_rank')
    txt='\n'.join(z.gene.astype(str))+'\n'
    p=OUT/f'{m.replace("-","_")}_TOP50_FROZEN.txt';p.write_text(txt)
    support_hash[m]=sha(p);support_genes[m]=z.gene.astype(str).tolist()

files=[
 FREEZE/'STOP_GATE.json',FREEZE/'SHA256SUMS_FROZEN.csv',
 MEAS/'FORMAL_DEEPSEEK_V3_2_SUMMARY.json',MEAS/'FORMAL_PAIR_MEASUREMENT_QA_V3_2.json',
 MEAS/'SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv',MEAS/'GLOBAL_MEASUREMENTS_V3_2.csv',
 DOWN/'REFERENCE_SCORES_FROZEN.csv',DOWN/'FOLDLOCAL_LAMBDA_CV_ALL.csv',DOWN/'FOLDLOCAL_LAMBDA_CV_AGGREGATE.csv',
 DOWN/'SELECTED_SUPPORTS_FROZEN.csv',DOWN/'CORRECTED_SCORES_FROZEN.csv',DOWN/'SUPPORT_OVERLAP.csv',
 DOWN/'PRE_EXTERNAL_FREEZE_SUMMARY.json',DOWN/'PRE_EXTERNAL_SHA256.txt',
 SHUF/'SEMANTIC_SHUFFLE_1000.csv',SHUF/'SEMANTIC_SHUFFLE_SUMMARY.json',SHUF/'SHA256SUMS.txt'
]
manifest=[]
for p in files:
    assert p.exists(),p
    manifest.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
for m in support_hash:
    p=OUT/f'{m.replace("-","_")}_TOP50_FROZEN.txt'
    manifest.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
pd.DataFrame(manifest).to_csv(OUT/'PRE_EXTERNAL_FILE_HASHES.csv',index=False)

freeze={
 'version':'KIDNEY_TCMR_PRE_EXTERNAL_FINAL_FREEZE_V3_2',
 'freeze_date':'2026-09-21',
 'status':'PASS_EXTERNAL_UNSEAL_ALLOWED',
 'development_dataset':'GSE36059',
 'external_dataset':'GSE48581',
 'external_outcome_accessed_in_method_selection':False,
 'selected_lambda':summary['selected_lambda'],
 'k':50,
 'support_sha256':support_hash,
 'support_genes':support_genes,
 'measurement':{
   'study_specific_calls':400,
   'query_manifest_sha256':json.loads((MEAS/'FORMAL_DEEPSEEK_V3_2_SUMMARY.json').read_text())['query_manifest_sha256'],
   'selective_measurement_sha256':summary['selective_m4_shared_measurement_sha256'],
   'selective_m4_shared_exact_measurement':True,
   'new_calls_after_measurement':0
 },
 'semantic_shuffle':{
   'performed':True,'n':1000,'new_llm_calls':0,
   'selective_empirical_p_retuned':shuffle['selective']['empirical_p_retuned'],
   'selective_no_certainty_empirical_p_retuned':shuffle['selective_no_certainty-NoC']['empirical_p_retuned'],
   'shuffle_csv_sha256':sha(SHUF/'SEMANTIC_SHUFFLE_1000.csv'),
   'shuffle_summary_sha256':sha(SHUF/'SEMANTIC_SHUFFLE_SUMMARY.json')
 },
 'recipient_limitation':'Repeated recipients confirmed in both studies; public biopsy-to-recipient map unavailable. CV and final sample-level metrics must not be described as patient-independent/grouped.',
 'external_evaluation_rule':'After this freeze only: fit fixed StandardScaler + L2 logistic regression C=1,class_weight=balanced on full GSE36059 selected top-50 support separately for each arm; transform GSE48581 with training scaler; evaluate once. No tuning after external outcomes are read.',
 'pre_external_file_hash_manifest_sha256':sha(OUT/'PRE_EXTERNAL_FILE_HASHES.csv')
}
(OUT/'PRE_EXTERNAL_FINAL_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
(OUT/'PRE_EXTERNAL_FINAL_FREEZE.sha256').write_text(sha(OUT/'PRE_EXTERNAL_FINAL_FREEZE.json')+'  PRE_EXTERNAL_FINAL_FREEZE.json\n')
print(json.dumps(freeze,indent=2))
