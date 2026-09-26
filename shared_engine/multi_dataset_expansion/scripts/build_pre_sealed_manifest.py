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
OUT=ROOT/'20_PRE_SEALED_FREEZE'; OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

files=[
 'REPORT/FINAL_DEVELOPMENT_selective_TUNING_FREEZE_20260919.md',
 'REPORT/global_DEVELOPMENT_INTEGRATION_FREEZE_20260919.md',
 'REPORT/BROAD_CONTENT_CONTRACT_AMENDMENT_PRE_BT_20260919.md',
 '11_selective_POSTPROCESS/selective_POSTPROCESS_STATUS.json',
 '16_global_POSTPROCESS/global_POSTPROCESS_STATUS.json',
 '18_CONTENT_CONTRACT_SENSITIVITY/STRICT_CONTENT_SENSITIVITY_STATUS.json',
]
for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
    files += [
      f'03_FROZEN_DATA/{task}/FREEZE_MANIFEST.json',
      f'04_reference/{task}/reference_STATUS.json',
      f'12_selective_INTERNAL/{task}/selective_NESTED_STATUS.json',
      f'12_selective_INTERNAL/{task}/selective_NESTED_AGGREGATE.csv',
      f'13_FINAL_DEVELOPMENT_TUNING/{task}/FINAL_DEVELOPMENT_TUNING_STATUS.json',
      f'13_FINAL_DEVELOPMENT_TUNING/{task}/FINAL_SELECTOR_PARAMS.csv',
      f'13_FINAL_DEVELOPMENT_TUNING/{task}/FINAL_SELECTED_ETA.csv',
      f'13_FINAL_DEVELOPMENT_TUNING/{task}/FINAL_selective_SELECTED_FEATURES.csv',
      f'13_FINAL_DEVELOPMENT_TUNING/{task}/FINAL_ETA0_GATE.csv',
      f'17_global_INTERNAL/D20/{task}/global_INTERNAL_STATUS.json',
      f'17_global_INTERNAL/D20/{task}/global_NESTED_AGGREGATE.csv',
      f'17_global_INTERNAL/D20/{task}/global_FINAL_SELECTED_GAMMA.csv',
      f'17_global_INTERNAL/D20/{task}/global_FINAL_SELECTED_FEATURES.csv',
      f'17_global_INTERNAL/D20/{task}/global_FINAL_GAMMA0_GATE.csv',
      f'17_global_INTERNAL/D10/{task}/global_INTERNAL_STATUS.json',
      f'17_global_INTERNAL/D10/{task}/global_NESTED_AGGREGATE.csv',
      f'17_global_INTERNAL/D10/{task}/global_FINAL_SELECTED_GAMMA.csv',
      f'17_global_INTERNAL/D10/{task}/global_FINAL_SELECTED_FEATURES.csv',
    ]
files += [
 '19_global_STRICT_CONTENT/SEPSIS_GSE65682/global_INTERNAL_STATUS.json',
 '19_global_STRICT_CONTENT/SEPSIS_GSE65682/global_NESTED_AGGREGATE.csv',
 '19_global_STRICT_CONTENT/SEPSIS_GSE65682/global_FINAL_SELECTED_GAMMA.csv',
 '19_global_STRICT_CONTENT/SEPSIS_GSE65682/global_FINAL_SELECTED_FEATURES.csv',
]
rows=[]
for rel in files:
    p=ROOT/rel
    if not p.exists(): raise RuntimeError('MISSING '+rel)
    rows.append({'relative_path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
man=OUT/'PRE_SEALED_MANIFEST.csv'
with open(man,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']); w.writeheader(); w.writerows(rows)
status={
 'status':'DEVELOPMENT_FROZEN_READY_TO_OPEN_SEALED_VALIDATION',
 'n_frozen_artifacts':len(rows),
 'manifest_sha256':sha(man),
 'primary_methods':['reference','global_D20','global_certainty_D20','selective'],
 'sensitivities':['global_D10','global_certainty_D10','STRICT_CONTENT_D20'],
 'selectors':['LASSO','ELASTICNET','SIS'],'k_grid':[10,20,30],
 'sealed_validation_utility_used_before_this_freeze':False,
 'all_tuning_locked_before_sealed':True
}
(OUT/'PRE_SEALED_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
