

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
import pandas as pd, hashlib, json, re
R=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
Q=R/'08_PRE_LLM_QUERIES'; B=R/'07_PRE_LLM_BUDGET'
expected={
 'BREAST_GSE25055_GSE25065':5308,
 'SEPSIS_GSE65682':3554,
}
for task,n in expected.items():
    p=Q/f'{task}_selective_QUERIES_PREAUTH.csv'
    d=pd.read_csv(p,dtype=str,keep_default_na=False)
    assert len(d)==n and d.query_id.nunique()==n
    assert (d.allowed_tokens=='A|B|U').all()
    assert set(d.order)=={'AB','BA'}
    assert (d.prompt_text.map(lambda x: hashlib.sha256(x.encode()).hexdigest())==d.prompt_sha256).all()
    for (pair,arm),z in d.groupby(['unordered_pair_id','arm']):
        assert len(z)==2 and set(z.order)=={'AB','BA'}
        zz=z.set_index('order')
        assert zz.loc['AB','gene_A']==zz.loc['BA','gene_B']
        assert zz.loc['AB','gene_B']==zz.loc['BA','gene_A']
        assert zz.evidence_packet_sha256.nunique()==1
    # hard leakage strings checked in prompt text, not query IDs.
    forbidden=['GSE25065','sealed validation','sealed-validation','X_sealed_validation','y_sealed_validation']
    hits={x:int(d.prompt_text.str.contains(x,case=False,regex=False).sum()) for x in forbidden}
    assert all(v==0 for v in hits.values()),hits
    print(task,'PASS rows',len(d),'pair-source',len(d)//2,'unique pairs',d.unordered_pair_id.nunique(),'forbidden_hits',hits)

rep=R/'REPORT/PRE_LLM_PRODUCTION_FREEZE_REPORT_20260919.md'
summary={
 'status':'READY_FOR_USER_APPROVAL_NOT_AUTHORIZED',
 'scope':'selective_SELECTIVE_ONLY',
 'breast_calls':5308,
 'sepsis_calls':3554,
 'total_calls':8862,
 'conservative_prompt_tokens':6440619,
 'completion_tokens':8862,
 'all_cache_miss_offpeak_usd':0.9714,
 'all_cache_miss_peak_usd':1.9428,
 'runtime_config_sha256':hashlib.sha256((B/'PROPOSED_RUNTIME_CONFIG_NOT_AUTHORIZED.json').read_bytes()).hexdigest(),
 'breast_query_csv_sha256':json.load(open(Q/'BREAST_GSE25055_GSE25065_QUERY_MANIFEST.json'))['query_csv_sha256'],
 'sepsis_query_csv_sha256':json.load(open(Q/'SEPSIS_GSE65682_QUERY_MANIFEST.json'))['query_csv_sha256'],
 'freeze_report_sha256':hashlib.sha256(rep.read_bytes()).hexdigest(),
 'production_llm_calls_issued':0
}
(B/'FINAL_FREEZE_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(B/'EXECUTION_APPROVAL.json').write_text(json.dumps({
 'authorized':False,
 'scope':'selective_SELECTIVE_ONLY',
 'freeze_report_sha256':summary['freeze_report_sha256'],
 'runtime_config_sha256':summary['runtime_config_sha256'],
 'breast_query_csv_sha256':summary['breast_query_csv_sha256'],
 'sepsis_query_csv_sha256':summary['sepsis_query_csv_sha256'],
 'note':'Set authorized=true only after explicit user approval for this exact frozen tranche.'
},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
