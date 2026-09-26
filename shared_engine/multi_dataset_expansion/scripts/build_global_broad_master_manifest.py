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
import hashlib,json,csv
ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
B=ROOT/'14_global_BROAD_FREEZE'
REPORT=ROOT/'REPORT/PRE_global_BROAD_FREEZE_REPORT_20260919_1136_CST.md'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

files=[
 '14_global_BROAD_FREEZE/BROAD_GRAPH_SUMMARY.csv',
 '14_global_BROAD_FREEZE/BROAD_selective_OVERLAP_SUMMARY.csv',
 '14_global_BROAD_FREEZE/BROAD_RUNTIME_CONFIG_FROZEN.json',
 '14_global_BROAD_FREEZE/BUDGET/BROAD_global_TOKEN_COST_ESTIMATE.json',
 '14_global_BROAD_FREEZE/BUDGET/BROAD_D10_FALLBACK_BUDGET.json',
 '14_global_BROAD_FREEZE/BUDGET/BROAD_RUNTIME_ESTIMATE.json',
 '14_global_BROAD_FREEZE/QUERIES/BREAST_GSE25055_GSE25065_global_BROAD_D20_QUERIES.parquet',
 '14_global_BROAD_FREEZE/QUERIES/BREAST_GSE25055_GSE25065_global_BROAD_D20_QUERY_MANIFEST.csv',
 '14_global_BROAD_FREEZE/QUERIES/BREAST_GSE25055_GSE25065_global_BROAD_D20_QUERY_STATUS.json',
 '14_global_BROAD_FREEZE/QUERIES/SEPSIS_GSE65682_global_BROAD_D20_QUERIES.parquet',
 '14_global_BROAD_FREEZE/QUERIES/SEPSIS_GSE65682_global_BROAD_D20_QUERY_MANIFEST.csv',
 '14_global_BROAD_FREEZE/QUERIES/SEPSIS_GSE65682_global_BROAD_D20_QUERY_STATUS.json',
 '14_global_BROAD_FREEZE/QUERIES/ALL_BROAD_QUERY_STATUS.json',
 'scripts/build_global_broad_graphs.py',
 'scripts/build_global_broad_queries.py',
 'scripts/audit_broad_selective_overlap.py',
 'scripts/estimate_global_broad_budget.py',
 'scripts/estimate_global_d10_runtime.py',
 'REPORT/PRE_global_BROAD_FREEZE_REPORT_20260919_1136_CST.md',
 'REPORT/FINAL_DEVELOPMENT_selective_TUNING_FREEZE_20260919.md',
 'REPORT/NESTED_selective_SOLVER_V2_FREEZE_20260919.md',
 '13_FINAL_DEVELOPMENT_TUNING/BREAST_GSE25055_GSE25065/FINAL_DEVELOPMENT_TUNING_STATUS.json',
 '13_FINAL_DEVELOPMENT_TUNING/SEPSIS_GSE65682/FINAL_DEVELOPMENT_TUNING_STATUS.json',
 '11_selective_POSTPROCESS/selective_POSTPROCESS_STATUS.json'
]
rows=[]
for rel in files:
    p=ROOT/rel
    if not p.exists(): raise RuntimeError('MISSING '+rel)
    rows.append({'relative_path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
with open(B/'BROAD_MASTER_PRE_RUN_MANIFEST.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']); w.writeheader(); w.writerows(rows)
status={
 'status':'FROZEN_PRE_global_BROAD',
 'n_files':len(rows),
 'master_manifest_sha256':sha(B/'BROAD_MASTER_PRE_RUN_MANIFEST.csv'),
 'freeze_report_sha256':sha(REPORT),
 'runtime_config_sha256':sha(B/'BROAD_RUNTIME_CONFIG_FROZEN.json'),
 'logical_queries_d20':156400,'reusable_selective_queries':80,'new_api_queries':156320,
 'execution_authorized':False
}
(B/'BROAD_MASTER_PRE_RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
approval={
 'authorized':False,
 'reason':'Awaiting explicit user approval after broad freeze report review',
 'design':None,
 'authorized_new_api_queries':0,
 'authorized_model':None,
 'hard_cost_cap_usd':None,
 'master_manifest_sha256':status['master_manifest_sha256'],
 'runtime_config_sha256':status['runtime_config_sha256'],
 'authorized_by_user_after_freeze':False
}
(B/'BROAD_EXECUTION_APPROVAL.json').write_text(json.dumps(approval,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
print(json.dumps(approval,indent=2))
