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
import hashlib,json
from pathlib import Path
ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
D=ROOT/'10_PRE_BATCH2_FREEZE'
base=json.load(open(D/'PRE_BATCH2_FREEZE_MANIFEST.json'))
pred=json.load(open(D/'FINAL_PREDICTORS/FINAL_PREDICTOR_FREEZE_MANIFEST.json'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
base['final_predictor_freeze']={
 'manifest_path':'10_PRE_BATCH2_FREEZE/FINAL_PREDICTORS/FINAL_PREDICTOR_FREEZE_MANIFEST.json',
 'manifest_sha256':sha(D/'FINAL_PREDICTORS/FINAL_PREDICTOR_FREEZE_MANIFEST.json'),
 'internal_manifest_sha256':pred['manifest_sha256'],
 'n_unique_predictors':pred['n_unique_predictors'],
 'n_method_k_rows':pred['n_method_k_rows'],
 'batch2_accessed':False
}
base['status']='COMPLETE_METHOD_AND_PREDICTOR_FREEZE_BEFORE_FIRST_BATCH2_OPEN'
raw=json.dumps({k:v for k,v in base.items() if k!='freeze_manifest_sha256'},ensure_ascii=False,sort_keys=True,separators=(',',':'))
base['freeze_manifest_sha256_v2']=hashlib.sha256(raw.encode()).hexdigest()
(D/'PRE_BATCH2_FREEZE_MANIFEST_V2.json').write_text(json.dumps(base,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':base['status'],'freeze_manifest_sha256_v2':base['freeze_manifest_sha256_v2'],'predictor_manifest_sha256':base['final_predictor_freeze']['manifest_sha256']},ensure_ascii=False,indent=2))
