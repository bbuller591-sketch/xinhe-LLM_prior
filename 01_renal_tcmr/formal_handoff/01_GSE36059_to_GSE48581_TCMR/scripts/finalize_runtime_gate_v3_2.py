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
import pandas as pd,json,hashlib,shutil
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
F=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2/FINAL'
S=ROOT/'formal_outputs/measurements/runtime/DEEPSEEK_GENERIC_SMOKE_V3_2.json'
q=pd.read_csv(F/'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv',dtype=str,keep_default_na=False)
calc=q.prompt_text.map(lambda x:hashlib.sha256(x.encode()).hexdigest())
prompt_hash_ok=bool((calc==q.prompt_sha256).all())
bad=int((calc!=q.prompt_sha256).sum())
smoke=json.loads(S.read_text())
assert smoke.get('status')=='PASS'
assert smoke.get('provider_models')==['deepseek-flash']
assert len(smoke.get('fingerprints',[]))==1
assert prompt_hash_ok,(bad,'prompt hash mismatches')
shutil.copy2(S,F/'RUNTIME_SMOKE_FROZEN.json')
fp=smoke['fingerprints'][0]

# refresh checksum manifest including runtime smoke, excluding STOP_GATE and checksum file itself
rows=[]
for p in sorted(F.iterdir()):
    if p.is_file() and p.name not in ('SHA256SUMS_FROZEN.csv','STOP_GATE.json'):
        rows.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
pd.DataFrame(rows).to_csv(F/'SHA256SUMS_FROZEN.csv',index=False)
gate={
 'version':'STOP_GATE_V3_2_FINAL',
 'status':'PASS',
 'study_specific_llm_calls_allowed':True,
 'generic_smoke_allowed':False,
 'structural_gate':'PASS',
 'recipient_id_gate':'PASS_WITH_EXPLICIT_PUBLIC_ID_LIMITATION',
 'runtime_gate':'PASS',
 'requested_model':'deepseek-flash',
 'provider_model':'deepseek-flash',
 'runtime_model_fingerprint':fp,
 'runtime_smoke_sha256':hashlib.sha256((F/'RUNTIME_SMOKE_FROZEN.json').read_bytes()).hexdigest(),
 'query_prompt_hash_column_verified':prompt_hash_ok,
 'query_prompt_hash_mismatches':bad,
 'frozen_query_manifest_sha256':hashlib.sha256((F/'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv').read_bytes()).hexdigest(),
 'frozen_selective_pairset_sha256':hashlib.sha256((F/'SELECTIVE_PAIRSET_FROZEN_V3_2.csv').read_bytes()).hexdigest(),
 'frozen_global_pairset_sha256':hashlib.sha256((F/'GLOBAL_PAIRSET_FROZEN_V3_2.csv').read_bytes()).hexdigest(),
 'sha256_manifest_sha256':hashlib.sha256((F/'SHA256SUMS_FROZEN.csv').read_bytes()).hexdigest(),
 'study_specific_llm_calls_so_far':0,
 'formal_calls_expected':400,
 'aggregation_after_calls':'p_e=0.5*(p_AB(A)+p_BA(B)); C_e=1-H_binary(p_e)/ln2',
 'no_gse48581_outcome_used':True
}
(F/'STOP_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps(gate,indent=2))
