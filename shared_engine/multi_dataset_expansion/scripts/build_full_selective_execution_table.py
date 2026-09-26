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
import pandas as pd,json,hashlib
ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
Q=ROOT/'08_PRE_LLM_QUERIES'
OUT=ROOT/'10_LLM_MEASUREMENT/02_FULL_selective'
OUT.mkdir(parents=True,exist_ok=True)
parts=[]
source={}
for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
    p=Q/f'{task}_selective_QUERIES_PREAUTH.csv'
    d=pd.read_csv(p,dtype=str,keep_default_na=False)
    parts.append(d)
    source[task]=hashlib.sha256(p.read_bytes()).hexdigest()
x=pd.concat(parts,ignore_index=True).sort_values('query_id',kind='mergesort').reset_index(drop=True)
assert len(x)==8862 and x.query_id.nunique()==8862
assert set(x.order)=={'AB','BA'}
for (arm,pair),z in x.groupby(['arm','unordered_pair_id']):
    assert len(z)==2 and set(z.order)=={'AB','BA'}
x.to_csv(OUT/'FULL_selective_QUERY_TABLE.csv',index=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={
 'status':'FROZEN_FULL_EXECUTION_TABLE',
 'n_queries':len(x),'n_pair_source_cells':len(x)//2,
 'n_unique_query_ids':x.query_id.nunique(),
 'arm_counts':x.groupby('arm').size().to_dict(),
 'source_query_table_sha256':source,
 'full_query_table_sha256':sha(OUT/'FULL_selective_QUERY_TABLE.csv'),
 'pilot_query_table_sha256':sha(ROOT/'10_LLM_MEASUREMENT/01_PILOT/PILOT_QUERY_TABLE.csv'),
 'pilot_calls_are_subset_and_must_be_reused_from_local_cache':True,
 'authorized_total_experiment_calls':8862
}
(OUT/'FULL_selective_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
