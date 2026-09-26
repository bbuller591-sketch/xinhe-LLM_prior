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
import csv,json,hashlib,shutil
from collections import Counter,defaultdict

ROOT=Path(str(REPRO_ROOT))
SRC=ROOT/'selective_qwen3_32b_local_modelswap_20260924'
OUT=ROOT/'LOCAL_LLM_MEASUREMENT_BUNDLE_20260924'

if OUT.exists():
    shutil.rmtree(OUT)
for d in ['queries/csv','queries/jsonl','protocol','tools','return_outputs','audit','reference']:
    (OUT/d).mkdir(parents=True,exist_ok=True)

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_text(s): return sha_bytes(s.encode('utf-8'))
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

systems={
 'RENAL': (SRC/'00_PROTOCOL_AND_AUDIT/SYSTEM_RENAL.txt').read_text(),
 'ABU_GENERIC': (SRC/'00_PROTOCOL_AND_AUDIT/SYSTEM_ABU_GENERIC.txt').read_text(),
 'CREDIT': (SRC/'00_PROTOCOL_AND_AUDIT/SYSTEM_CREDIT.txt').read_text(),
 'HOSPITAL': (SRC/'00_PROTOCOL_AND_AUDIT/SYSTEM_HOSPITAL.txt').read_text(),
}
for k,v in systems.items():
    (OUT/'protocol'/f'SYSTEM_{k}.txt').write_text(v)

specs=[
 dict(bundle_id='renal_formal',dataset='Renal TCMR',phase='formal_combined_selective_and_global',
      path=SRC/'02_RENAL/RENAL_QWEN3_32B_FROZEN_QUERIES.csv',system='RENAL',parser='no_double',max_tokens=4,expected=400),
 dict(bundle_id='sepsis_selective',dataset='GSE272769',phase='selective',
      path=SRC/'03_GSE272769/GSE272769_QWEN3_32B_FROZEN_QUERIES.csv',system='ABU_GENERIC',parser='double_count_first',max_tokens=4,expected=3622),
 dict(bundle_id='breast_selective',dataset='Breast GSE25055->GSE25065',phase='selective',
      path=SRC/'04_BREAST/BREAST_QWEN3_32B_FROZEN_QUERIES.csv',system='ABU_GENERIC',parser='double_count_first',max_tokens=4,expected=5308),
 dict(bundle_id='credit_selective',dataset='CREDIT-G',phase='selective',
      path=SRC/'05_CREDIT_G/CREDIT_QWEN3_32B_FROZEN_QUERIES.csv',system='CREDIT',parser='credit',max_tokens=4,expected=18),
 dict(bundle_id='hospital_selective',dataset='Hospital Osteoporosis',phase='selective',
      path=SRC/'06_HOSPITAL/HOSPITAL_QWEN3_32B_FROZEN_QUERIES.csv',system='HOSPITAL',parser='hospital_structured',max_tokens=220,expected=60),
 dict(bundle_id='darmanis_selective',dataset='Darmanis GBM',phase='selective',
      path=SRC/'07_DARMANIS/DARMANIS_QWEN3_32B_FROZEN_QUERIES.csv',system='ABU_GENERIC',parser='double_count_first',max_tokens=4,expected=1500),
 dict(bundle_id='darmanis_broad7',dataset='Darmanis GBM',phase='required_broad7',
      path=SRC/'07_DARMANIS/DARMANIS_QWEN3_32B_REQUIRED_BROAD7_QUERIES.csv',system='ABU_GENERIC',parser='double_count_first',max_tokens=4,expected=14),
 dict(bundle_id='sepsis_q2_global',dataset='GSE272769',phase='q2_matched_global',
      path=SRC/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q2/Q2_GLOBAL_MATCHED_QUERIES_FROZEN.csv',system='ABU_GENERIC',parser='double_count_first',max_tokens=4,expected=804),
 dict(bundle_id='breast_q2_global',dataset='Breast GSE25055->GSE25065',phase='q2_matched_global_source_stratified',
      path=SRC/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE/BREAST_Q2_SOURCE_GPT_QUERIES.csv',system='ABU_GENERIC',parser='double_count_first',max_tokens=4,expected=2366),
 dict(bundle_id='hospital_q2_global',dataset='Hospital Osteoporosis',phase='q2_matched_global',
      path=SRC/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2/HOSPITAL_Q2_GPT_QUERIES.csv',system='HOSPITAL',parser='hospital_structured',max_tokens=220,expected=22),
 dict(bundle_id='darmanis_q2_global',dataset='Darmanis GBM',phase='q2_matched_global',
      path=SRC/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2/DARMANIS_Q2_REQUIRED_GPT_QUERIES.csv',system='ABU_GENERIC',parser='double_count_first',max_tokens=4,expected=428),
]

all_records=[]
manifest=[]
dataset_records=defaultdict(list)
for sp in specs:
    p=sp['path']
    if not p.exists(): raise FileNotFoundError(p)
    rows=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
    if len(rows)!=sp['expected']: raise RuntimeError((sp['bundle_id'],len(rows),sp['expected']))
    sys_text=systems[sp['system']]
    dest=OUT/'queries/csv'/f"{sp['bundle_id']}.csv"
    shutil.copy2(p,dest)
    for i,r in enumerate(rows):
        qid=r.get('query_id')
        if not qid: raise RuntimeError(f'missing query_id {sp["bundle_id"]} row {i}')
        portable_id=f"{sp['bundle_id']}::{qid}"
        prompt=r.get('prompt_text')
        ph=r.get('prompt_sha256')
        if prompt is None or ph is None: raise RuntimeError(f'missing prompt/hash {portable_id}')
        if sha_text(prompt)!=ph: raise RuntimeError(f'prompt hash mismatch {portable_id}')
        allowed=(r.get('allowed_tokens') or '').split('|')
        rec={
          'portable_query_id':portable_id,
          'original_query_id':qid,
          'bundle_id':sp['bundle_id'],
          'dataset':sp['dataset'],
          'phase':sp['phase'],
          'system_prompt':sys_text,
          'system_prompt_sha256':sha_text(sys_text),
          'user_prompt':prompt,
          'user_prompt_sha256':ph,
          'allowed_tokens':allowed,
          'parser_style_original':sp['parser'],
          'original_max_tokens':sp['max_tokens'],
          'temperature_original':1.0,
          'measurement_contract':{
             'need_first_semantic_token_logits':True,
             'preferred_local_extraction':'full_vocabulary_first_step_logits',
             'semantic_token_normalization':'decode(token_id).strip() grouped by A/B/U/T as applicable',
             'do_not_use_self_reported_confidence':True,
             'do_not_retrieve_or_browse':True
          },
          'metadata':{k:v for k,v in r.items() if k!='prompt_text'}
        }
        all_records.append(rec)
        dataset_records[sp['bundle_id']].append(rec)
    manifest.append({
      'bundle_id':sp['bundle_id'],'dataset':sp['dataset'],'phase':sp['phase'],
      'n_queries':len(rows),'system_key':sp['system'],'system_prompt_sha256':sha_text(sys_text),
      'parser_style_original':sp['parser'],'original_max_tokens':sp['max_tokens'],
      'source_csv':str(p),'source_csv_sha256':sha_file(p),
      'portable_csv':str(dest.relative_to(OUT)),'portable_csv_sha256':sha_file(dest)
    })

assert len(all_records)==14542, len(all_records)
assert len({r['portable_query_id'] for r in all_records})==len(all_records)

with (OUT/'queries/jsonl/all_queries.jsonl').open('w',encoding='utf-8') as f:
    for r in all_records: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
for bid,recs in dataset_records.items():
    with (OUT/'queries/jsonl'/f'{bid}.jsonl').open('w',encoding='utf-8') as f:
        for r in recs: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

keys=['bundle_id','dataset','phase','n_queries','system_key','system_prompt_sha256','parser_style_original','original_max_tokens','source_csv','source_csv_sha256','portable_csv','portable_csv_sha256']
with (OUT/'audit/QUERY_SCHEDULE_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(manifest)

pair_counts=Counter((r['system_prompt_sha256'],r['user_prompt_sha256']) for r in all_records)
dups=sum(v-1 for v in pair_counts.values() if v>1)
audit={
 'bundle_version':'LOCAL_LLM_MEASUREMENT_BUNDLE_20260924_V1',
 'created_from':str(SRC),
 'n_schedules':len(specs),
 'n_formal_call_rows':len(all_records),
 'n_unique_system_user_prompt_pairs':len(pair_counts),
 'n_repeated_prompt_rows':dups,
 'deduplication_policy':'NONE. Exact schedule rows are retained because some repeats are intentional measurement/repeat sentinels.',
 'contains_api_keys_or_secrets':False,
 'contains_old_model_answers':False,
 'contains_training_or_outcome_data':False,
 'q3_q4_q5_need_new_llm_calls':False,
 'note':'Complete LLM question schedules used by the final GPT-4o-mini Selective/Q2 analysis. Q3/Q4/Q5 are downstream/offline once measurements are returned.',
 'schedules':manifest
}
(OUT/'audit/BUNDLE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')

schema={
 'portable_query_id':'string; must match input',
 'original_query_id':'string',
 'model_id':'exact deployed model/checkpoint identifier',
 'model_revision':'commit/revision if available',
 'tokenizer_id':'tokenizer identifier/revision',
 'generated_text':'raw model generation',
 'first_generated_token':'decoded first generated token',
 'semantic_logit_A':'logsumexp over first-step vocabulary token logits whose decoded token .strip()==A',
 'semantic_logit_B':'same for B',
 'semantic_logit_U':'same for U when relevant; null if absent/not relevant',
 'semantic_logit_T':'optional diagnostic',
 'pA_vs_B':'exp(lA)/(exp(lA)+exp(lB))',
 'pB_vs_A':'1-pA_vs_B',
 'entropy_AB_bits':'binary entropy of pA_vs_B',
 'certainty':'1-entropy_AB_bits',
 'input_token_count':'integer',
 'generation_token_count':'integer',
 'runtime_seconds':'float',
 'status':'PASS/FAIL with reason',
 'raw_first_step_top_tokens':'optional compact diagnostic; full vocabulary logits need not be saved'
}
(OUT/'protocol/OUTPUT_SCHEMA.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')

validator="""#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
def sha_text(s): return hashlib.sha256(s.encode()).hexdigest()
audit=json.load(open(ROOT/'audit/BUNDLE_AUDIT.json',encoding='utf-8'))
rows=0; bad=[]; ids=set()
for line in open(ROOT/'queries/jsonl/all_queries.jsonl',encoding='utf-8'):
    if not line.strip(): continue
    r=json.loads(line); rows+=1
    if r['portable_query_id'] in ids: bad.append('duplicate portable id '+r['portable_query_id'])
    ids.add(r['portable_query_id'])
    if sha_text(r['system_prompt'])!=r['system_prompt_sha256']: bad.append('bad system hash '+r['portable_query_id'])
    if sha_text(r['user_prompt'])!=r['user_prompt_sha256']: bad.append('bad user hash '+r['portable_query_id'])
print('rows',rows,'expected',audit['n_formal_call_rows'])
print('unique portable ids',len(ids))
if rows!=audit['n_formal_call_rows']: bad.append('row count mismatch')
if bad:
    print('FAIL'); print('\\n'.join(bad[:20])); sys.exit(2)
print('PASS')
"""
(OUT/'tools/validate_bundle.py').write_text(validator)

inspect="""#!/usr/bin/env python3
import argparse,json
ap=argparse.ArgumentParser()
ap.add_argument('--model',required=True)
ap.add_argument('--trust-remote-code',action='store_true')
args=ap.parse_args()
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=args.trust_remote_code)
groups={x:[] for x in ['A','B','U','T']}
for i in range(len(tok)):
    try: s=tok.decode([i],skip_special_tokens=False)
    except Exception: continue
    z=s.strip()
    if z in groups: groups[z].append({'token_id':i,'decoded':s})
print(json.dumps({'model':args.model,'vocab_size':len(tok),'semantic_token_groups':groups},ensure_ascii=False,indent=2))
"""
(OUT/'tools/inspect_tokenizer_semantic_tokens.py').write_text(inspect)

return_readme="""# RETURN OUTPUTS

After the local-model run, place these files here before copying the bundle back:

- model_runtime.json
- measurements.jsonl
- run_status.json
- optional logs/

Do not edit the input query JSONLs.

measurements.jsonl should contain one record per portable_query_id, following ../protocol/OUTPUT_SCHEMA.json.

For a local Hugging Face/vLLM implementation, preserve raw generated text and extract first-step full-vocabulary logits if the runtime supports it. Do not replace logits with self-reported confidence.
"""
(OUT/'return_outputs/README.md').write_text(return_readme)

readme=f"""# LOCAL LLM MEASUREMENT BUNDLE - START HERE

This is a portable inference-only package for the six-dataset LLM-as-measurement experiment.

## Frozen contents

- 14,542 formal call rows
- exact system prompts
- exact user prompts and SHA256 hashes
- Selective schedules
- matched-Global/Q2 schedules
- AB/BA order
- evidence already embedded in prompts
- original allowed output tokens and parser metadata

No API key, secret, prior model answer, outcome vector, train/test data, or predictor matrix is included.

## Offline server task

Load a local language model and answer queries/jsonl/all_queries.jsonl.

Preferred local measurement: full-vocabulary logits at the first generated position. Decode every single vocabulary token and group token ids by decoded.strip() equal to A/B/U/T. Compute semantic log-mass with log-sum-exp, then normalize A versus B.

This avoids API top-20 truncation.

## Important

Do not automatically deduplicate rows. Some repeated calls are intentional.

Do not enable browsing, retrieval, tools, web search, or external knowledge injection. Evidence needed by evidence arms is already frozen in each prompt.

## Before run

1. python tools/validate_bundle.py
2. After model choice:
   python tools/inspect_tokenizer_semantic_tokens.py --model MODEL_PATH_OR_ID
3. Freeze model revision, tokenizer revision, dtype/quantization, chat template, inference engine version and runtime settings in return_outputs/model_runtime.json.
4. The model-specific runner will be finalized after the deployment choice/server details are supplied.

Q3/Q4/Q5 need no new LLM calls.

Expected formal call rows: 14,542.
"""
(OUT/'README_START_HERE.md').write_text(readme)

for src,name in [
 (SRC/'00_PROTOCOL_AND_AUDIT/MODEL_SWAP_FREEZE.md','QWEN3_32B_MODEL_SWAP_FREEZE.md'),
 (SRC/'00_PROTOCOL_AND_AUDIT/TOP20_TRUNCATION_AMENDMENT.md','QWEN3_32B_TOP20_AMENDMENT.md'),
 (SRC/'09_FINAL_COMPARISON/FINAL_MODEL_SWAP_REPORT.md','QWEN3_32B_FINAL_MODEL_SWAP_REPORT.md'),
 (SRC/'09_FINAL_COMPARISON/FINAL_PAPER_ROWS_9.csv','QWEN3_32B_FINAL_PAPER_ROWS_9.csv')]:
    shutil.copy2(src,OUT/'reference'/name)

files=[]
for p in sorted(x for x in OUT.rglob('*') if x.is_file()):
    files.append({'path':str(p.relative_to(OUT)),'sha256':sha_file(p),'bytes':p.stat().st_size})
with (OUT/'audit/SHA256SUMS.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['path','sha256','bytes']);w.writeheader();w.writerows(files)

print(json.dumps({
 'out':str(OUT),'formal_rows':len(all_records),'unique_prompt_pairs':len(pair_counts),
 'repeated_prompt_rows':dups,'files':len(files),
 'all_queries_bytes':(OUT/'queries/jsonl/all_queries.jsonl').stat().st_size
},indent=2))
