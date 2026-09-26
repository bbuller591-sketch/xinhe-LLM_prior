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
import hashlib,json,sys
ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
D=ROOT/'09_PRE_LLM_FREEZE'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

approval=json.load(open(D/'LLM_EXECUTION_APPROVAL.json'))
runtime=json.load(open(D/'LLM_RUNTIME_CONFIG_FROZEN.json'))
status=json.load(open(D/'MASTER_PRE_LLM_MANIFEST_STATUS.json'))
master=D/'MASTER_PRE_LLM_MANIFEST.csv'

if approval.get('authorized') is not True:
    raise SystemExit('NOT_AUTHORIZED: explicit post-freeze user approval has not been recorded')
if approval.get('authorized_by_user_after_freeze') is not True:
    raise SystemExit('NOT_AUTHORIZED_BY_USER_AFTER_FREEZE')
if int(approval.get('authorized_query_count',-1)) != 8862:
    raise SystemExit('AUTHORIZED_QUERY_COUNT_MISMATCH')
if approval.get('authorized_model') != 'deepseek-flash':
    raise SystemExit('AUTHORIZED_MODEL_MISMATCH')
if approval.get('master_manifest_sha256') != sha(master):
    raise SystemExit('MASTER_MANIFEST_HASH_MISMATCH')
if approval.get('runtime_config_sha256') != sha(D/'LLM_RUNTIME_CONFIG_FROZEN.json'):
    raise SystemExit('RUNTIME_CONFIG_HASH_MISMATCH')
if status.get('master_manifest_sha256') != sha(master):
    raise SystemExit('STATUS_MASTER_HASH_MISMATCH')
if status.get('runtime_config_sha256') != sha(D/'LLM_RUNTIME_CONFIG_FROZEN.json'):
    raise SystemExit('STATUS_RUNTIME_HASH_MISMATCH')
if runtime.get('requested_model') != 'deepseek-flash':
    raise SystemExit('RUNTIME_MODEL_MISMATCH')
if int(runtime.get('n_expected_queries',-1)) != 8862:
    raise SystemExit('RUNTIME_QUERY_COUNT_MISMATCH')

print(json.dumps({
 'status':'PRE_EXECUTION_APPROVAL_GATE_PASS',
 'authorized_queries':8862,
 'model':'deepseek-flash',
 'next_step':'generic no-experiment-data API smoke; freeze provider model and system_fingerprint before experiment calls'
},indent=2))
