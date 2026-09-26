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
import hashlib,json
ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
B=ROOT/'14_global_BROAD_FREEZE'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
ap=json.load(open(B/'BROAD_EXECUTION_APPROVAL.json'))
cfg=json.load(open(B/'BROAD_RUNTIME_CONFIG_FROZEN.json'))
st=json.load(open(B/'BROAD_MASTER_PRE_RUN_STATUS.json'))
man=B/'BROAD_MASTER_PRE_RUN_MANIFEST.csv'
checks=[
 ('authorized',ap.get('authorized') is True),
 ('authorized_by_user_after_freeze',ap.get('authorized_by_user_after_freeze') is True),
 ('design',ap.get('design')=='global_PRIMARY_BROAD_D20'),
 ('new_api_queries',int(ap.get('authorized_new_api_queries',-1))==156320),
 ('logical_queries',int(ap.get('logical_query_count_d20',-1))==156400),
 ('reuse',int(ap.get('reusable_existing_selective_queries',-1))==80),
 ('model',ap.get('authorized_model')=='deepseek-flash'),
 ('hard_cap',float(ap.get('hard_cost_cap_usd',-1))==35.0),
 ('master_hash_ap',ap.get('master_manifest_sha256')==sha(man)),
 ('runtime_hash_ap',ap.get('runtime_config_sha256')==sha(B/'BROAD_RUNTIME_CONFIG_FROZEN.json')),
 ('master_hash_status',st.get('master_manifest_sha256')==sha(man)),
 ('runtime_hash_status',st.get('runtime_config_sha256')==sha(B/'BROAD_RUNTIME_CONFIG_FROZEN.json')),
 ('cfg_model',cfg.get('requested_model')=='deepseek-flash'),
 ('cfg_new',int(cfg.get('expected_new_api_queries',-1))==156320),
 ('cfg_logical',int(cfg.get('logical_query_count_d20',-1))==156400),
 ('cfg_cap',float(cfg.get('hard_cost_cap_usd',-1))==35.0),
]
bad=[k for k,v in checks if not v]
if bad: raise SystemExit('BROAD_PREFLIGHT_FAIL '+','.join(bad))
print(json.dumps({'status':'BROAD_PRE_EXECUTION_APPROVAL_GATE_PASS','design':ap['design'],
                  'authorized_new_api_queries':156320,'logical_queries':156400,
                  'model':'deepseek-flash','hard_cost_cap_usd':35.0,
                  'master_manifest_sha256':sha(man),
                  'runtime_config_sha256':sha(B/'BROAD_RUNTIME_CONFIG_FROZEN.json'),
                  'next_step':'generic no-experiment-data smoke then fingerprint lock'},indent=2))
