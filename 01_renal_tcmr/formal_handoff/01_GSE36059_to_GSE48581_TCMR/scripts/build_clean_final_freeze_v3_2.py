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
import shutil,json,hashlib,pandas as pd
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
P=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2'
F=P/'FINAL'
if F.exists():
    for p in F.iterdir():
        if p.is_file():p.unlink()
        elif p.is_dir():shutil.rmtree(p)
else:F.mkdir(parents=True)
sources={
 P/'SELECTIVE_PAIRSET_FROZEN_V3_2.csv':'SELECTIVE_PAIRSET_FROZEN_V3_2.csv',
 P/'GLOBAL_PAIRSET_FROZEN_V3_2.csv':'GLOBAL_PAIRSET_FROZEN_V3_2.csv',
 P/'GLOBAL_RANDOM_STREAM_PREFIX_FROZEN_V3_2.csv':'GLOBAL_RANDOM_STREAM_PREFIX_FROZEN_V3_2.csv',
 P/'EVIDENCE_SOURCE_MANIFEST_FROZEN_V3_2.csv':'EVIDENCE_SOURCE_MANIFEST_FROZEN_V3_2.csv',
 P/'GENE_EVIDENCE_LEDGER_FROZEN_V3_2.csv':'GENE_EVIDENCE_LEDGER_FROZEN_V3_2.csv',
 P/'LITERATURE_CONTEXT_SHORTLIST_FROZEN_V3_2.csv':'LITERATURE_CONTEXT_SHORTLIST_FROZEN_V3_2.csv',
 P/'PAIR_DOSSIERS_FROZEN_V3_2.jsonl':'PAIR_DOSSIERS_FROZEN_V3_2.jsonl',
 P/'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv':'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv',
 P/'RECIPIENT_ID_AUDIT_FROZEN.json':'RECIPIENT_ID_AUDIT_FROZEN.json',
 P/'RECIPIENT_ID_AUDIT_FROZEN.md':'RECIPIENT_ID_AUDIT_FROZEN.md',
 P/'METHOD_CONFIG_FROZEN_V3_2.json':'METHOD_CONFIG_FROZEN_V3_2.json',
 P/'ORIGINAL_FORMAL_CONFIG.json':'ORIGINAL_FORMAL_CONFIG.json',
 P/'STRUCTURAL_PRE_LLM_AUDIT.json':'STRUCTURAL_PRE_LLM_AUDIT.json',
 ROOT/'scripts/deepseek_generic_runtime_smoke_v3_2.py':'deepseek_generic_runtime_smoke_v3_2.py',
 ROOT/'scripts/run_formal_deepseek_v3_2.py':'run_formal_deepseek_v3_2.py',
 ROOT/'scripts/aggregate_formal_measurements_simple_abba_v3_2.py':'aggregate_formal_measurements_simple_abba_v3_2.py'
}
for s,n in sources.items():
    if not s.exists():raise FileNotFoundError(s)
    shutil.copy2(s,F/n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=pd.read_csv(F/'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv',dtype=str,keep_default_na=False)
assert len(q)==400 and q.semantic_pair_id.nunique()==200
assert not q.prompt_text.str.contains('GSE36059|GSE48581|GSE21374|GSE98320|GSE124203|GSE275126',case=False,regex=True).any()
# Snapshot checksum before gate; gate itself then points to this manifest hash.
rows=[]
for p in sorted(F.iterdir()):
    if p.is_file() and p.name not in ('SHA256SUMS_FROZEN.csv','STOP_GATE.json'):
        rows.append({'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
pd.DataFrame(rows).to_csv(F/'SHA256SUMS_FROZEN.csv',index=False)
gate={
 'version':'STOP_GATE_V3_2_FINAL',
 'status':'PASS_FOR_GENERIC_RUNTIME_SMOKE_ONLY',
 'study_specific_llm_calls_allowed':False,
 'generic_smoke_allowed':True,
 'recipient_id_gate':'PASS_WITH_EXPLICIT_PUBLIC_ID_LIMITATION',
 'structural_gate':'PASS',
 'runtime_gate':'PENDING_GENERIC_SMOKE',
 'frozen_query_manifest_sha256':sha(F/'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv'),
 'frozen_selective_pairset_sha256':sha(F/'SELECTIVE_PAIRSET_FROZEN_V3_2.csv'),
 'frozen_global_pairset_sha256':sha(F/'GLOBAL_PAIRSET_FROZEN_V3_2.csv'),
 'sha256_manifest_sha256':sha(F/'SHA256SUMS_FROZEN.csv'),
 'study_specific_llm_calls_so_far':0,
 'no_gse48581_outcome_used':True
}
(F/'STOP_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps(gate,indent=2))
print('\nFINAL package files:')
print(pd.read_csv(F/'SHA256SUMS_FROZEN.csv').to_string(index=False))
