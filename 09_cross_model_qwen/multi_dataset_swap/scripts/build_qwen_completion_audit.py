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
import json,pandas as pd,subprocess
R=Path(str(REPRO_ROOT))
Q=R/'selective_qwen3_32b_local_modelswap_20260924'
O=Q/'09_FINAL_COMPARISON'
def J(p):return json.load(open(p))
D=pd.read_csv(O/'FINAL_TRIMODEL_PAPER_ROWS_9.csv')
expected={'renal_semantic':1000,'renal_random':1000,'sepsis_semantic':200,'sepsis_random':1000,'breast_semantic':1000,'breast_random':1000,'credit_semantic':1000,'credit_random':1000,'hospital_semantic':1000,'hospital_random':1000,'darmanis_semantic':1000,'darmanis_random':1000}
actual={
'renal_semantic':J(Q/'08_Q2_Q5_DIAGNOSTICS/RENAL_Q3_SEMANTIC_EXTERNAL/SUMMARY.json')['n_replicates'],
'renal_random':J(Q/'02_RENAL/random_null/RANDOM_LLM_PROBABILITY_NULL_SUMMARY.json')['n_per_mode'],
'sepsis_semantic':J(Q/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q3_SEMANTIC_EXACT200/SUMMARY.json')['n_replicates'],
'sepsis_random':J(Q/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q3_RANDOM/SUMMARY.json')['n_replicates'],
'breast_semantic':J(Q/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q3_SEMANTIC/SUMMARY.json')['summary'][0]['n_replicates'],
'breast_random':J(Q/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q3_RANDOM/SUMMARY.json')['summary'][0]['n_replicates'],
'credit_semantic':J(Q/'08_Q2_Q5_DIAGNOSTICS/CREDIT_Q3/01_q3_semantic_shuffle/CREDIT_G/SUMMARY.json')['summary'][0]['n_replicates'],
'credit_random':J(Q/'08_Q2_Q5_DIAGNOSTICS/CREDIT_Q3/02_q3_random_null/CREDIT_G/SUMMARY.json')['summary'][0]['n_replicates'],
'hospital_semantic':J(Q/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q3_TEMPORAL_NULL/SEMANTIC_SUMMARY.json')['n_replicates'],
'hospital_random':J(Q/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q3_TEMPORAL_NULL/RANDOM_SUMMARY.json')['n_replicates'],
'darmanis_semantic':J(Q/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q3/01_q3_semantic_shuffle/DARMANIS_GBM/SUMMARY.json')['summary'][0]['n_replicates'],
'darmanis_random':J(Q/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q3/02_q3_random_null/DARMANIS_GBM/SUMMARY.json')['summary'][0]['n_replicates']}
ret=J(Q/'00_PROTOCOL_AND_AUDIT/QWEN3_32B_RETURN_AND_CONVERSION_AUDIT.json')
src=J(Q/'00_PROTOCOL_AND_AUDIT/QWEN3_SOURCE_INTEGRITY_AUDIT.json')
reuse=J(Q/'00_PROTOCOL_AND_AUDIT/HOSPITAL_RANDOM_NULL_EXACT_REUSE_AUDIT.json')
rt=J(Q/'00_PROTOCOL_AND_AUDIT/QWEN3_32B_MODEL_RUNTIME.json')
summ=J(O/'FINAL_TRIMODEL_SUMMARY.json')
status_paths=[
Q/'02_RENAL/measurement/MEASUREMENT_STATUS.json',
Q/'03_GSE272769/measurement/MEASUREMENT_STATUS.json',
Q/'04_BREAST/measurement/MEASUREMENT_STATUS.json',
Q/'05_CREDIT_G/measurement/MEASUREMENT_STATUS.json',
Q/'06_HOSPITAL/measurement/MEASUREMENT_STATUS.json',
Q/'07_DARMANIS/measurement/MEASUREMENT_STATUS.json',
Q/'07_DARMANIS/measurement_broad7/MEASUREMENT_STATUS.json',
Q/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q2/measurement/MEASUREMENT_STATUS.json',
Q/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE/measurement/MEASUREMENT_STATUS.json',
Q/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2/measurement/MEASUREMENT_STATUS.json',
Q/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2/measurement/MEASUREMENT_STATUS.json']
sts=[J(p) for p in status_paths]
main=[
J(Q/'02_RENAL/external_eval/EXTERNAL_EVALUATION_SUMMARY.json')['status'],
J(Q/'03_GSE272769/downstream/selective_NESTED_STATUS.json')['status'],
J(Q/'04_BREAST/sealed_validation/STATUS.json')['status'],
'COMPLETE' if (Q/'05_CREDIT_G/downstream/FINAL_HOLDOUT_RESULTS.csv').exists() else 'MISSING',
J(Q/'06_HOSPITAL/final_temporal_eval/STATUS.json')['status'],
J(Q/'07_DARMANIS/downstream/selective_STATUS_V2_9.json')['status']]
ps=subprocess.run(['bash','-lc',f"ps -ef | grep '{Q}/scripts' | grep -v grep | grep -v build_qwen_completion_audit.py || true"],capture_output=True,text=True).stdout.strip()
checks={
'returned_measurements_audit_pass':ret['status']=='PASS',
'returned_14542_rows':ret['measurement_rows']==14542 and ret['unique_ids']==14542,
'all_returned_calls_pass':ret['all_status_pass'],
'no_thinking':ret['no_thinking_contamination'] and rt['enable_thinking'] is False,
'no_context_truncation':ret['no_context_truncation'],
'hospital_format_all_pass':ret['hospital_parser_all_pass'],
'full_vocab_no_top20_loss':all((x.get('n_unusable_top20') or 0)==0 for x in sts),
'all_measurement_status_complete':all(x.get('status')=='COMPLETE' for x in sts),
'source_integrity_pass':src['status']=='PASS' and src['all_source_hashes_match'],
'sepsis_scratch_removed':not src['sepsis_execution_incident']['scratch_directory_present_after_remediation'],
'reference_exact_match_9_rows':len(D)==9 and float(D.qwen_reference_abs_diff_vs_frozen.max())==0.0,
'q3_replicate_counts_match':actual==expected,
'q4_9_rows_zero_bound_violations':len(D)==9 and bool((pd.to_numeric(D.qwen_q4_bound_violations)==0).all()),
'main_downstream_complete':all(('COMPLETE' in s) or ('PASS' in s) for s in main),
'trimodel_summary_complete':summ['status']=='COMPLETE',
'hospital_random_reuse_audited':reuse['status']=='PASS_EXACT_DETERMINISTIC_REUSE',
'no_live_qwen_workspace_jobs':ps=='',
'model_is_qwen3_32b_bf16_no_quantization':rt['model_id']=='Qwen3-32B' and rt['dtype']=='torch.bfloat16' and rt['quantization']=='none'}
audit={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'q3_expected':expected,'q3_actual':actual,'main_downstream_statuses':main,'live_process_output':ps,'returned_result_tar_sha256':'4933be7b955081381a00cfc368ad72ce84a77f50f50cdb04923dbdeecbe05d4a','input_bundle_sha256':'710566e19c1bd61986fd1ae043d7362cbf9817e3a3420dcf5afafd1c510e2db8','measurements_sha256':ret['measurements_sha256'],'final_summary':summ}
(O/'FINAL_QWEN_COMPLETION_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False,indent=2))
