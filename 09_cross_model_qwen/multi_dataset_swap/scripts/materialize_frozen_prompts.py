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
import csv,gzip,json,hashlib,re

ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'selective_qwen3_32b_local_modelswap_20260924'
AUD=OUT/'00_PROTOCOL_AND_AUDIT'
AUD.mkdir(parents=True,exist_ok=True)

def sha_text(s): return hashlib.sha256(s.encode()).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

SYS_RENAL="Follow the user measurement instruction exactly. Do not call tools, browse the web, retrieve external information, or add explanation. Your entire response must be exactly one token: A or B."
SYS_ABU="Follow the user's measurement instruction exactly. Do not call tools, browse the web, or retrieve external information during this judgment. Output only the requested semantic token."
SYS_CREDIT="You are a measurement instrument for pairwise variable relevance in consumer-credit repayment risk. Follow the user instruction exactly. Do not call tools, browse the web, or retrieve external information. Do not try to identify or name a benchmark dataset. Your entire response must be exactly one token: A, B, or U."
(AUD/'SYSTEM_RENAL.txt').write_text(SYS_RENAL)
(AUD/'SYSTEM_ABU_GENERIC.txt').write_text(SYS_ABU)
(AUD/'SYSTEM_CREDIT.txt').write_text(SYS_CREDIT)

rows_audit=[]
def write_csv(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    return path

# Renal exact frozen prompt manifest
src=ROOT/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR/formal_outputs/01_pre_llm_freeze_v3_2/FINAL/DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv'
rr=list(csv.DictReader(src.open(encoding='utf-8-sig')))
assert len(rr)==400 and all(sha_text(r['prompt_text'])==r['prompt_sha256'] for r in rr)
for r in rr:r['allowed_tokens']='A|B'
dst=write_csv(rr,OUT/'02_RENAL/RENAL_QWEN3_32B_FROZEN_QUERIES.csv')
rows_audit.append(('Renal',str(src),sha_file(src),len(rr),str(dst),sha_file(dst),'original prompt_text exact copy'))

# Sepsis exact frozen prompts
src=ROOT/'new_dataset_search_20260919/gse272769/pre_llm/SEPSIS_GSE272769_selective_QUERIES_PREAUTH.csv'
rr=list(csv.DictReader(src.open(encoding='utf-8-sig')))
assert len(rr)==3622 and all(sha_text(r['prompt_text'])==r['prompt_sha256'] for r in rr)
for r in rr:r['allowed_tokens']='A|B|U'
dst=write_csv(rr,OUT/'03_GSE272769/GSE272769_QWEN3_32B_FROZEN_QUERIES.csv')
rows_audit.append(('GSE272769',str(src),sha_file(src),len(rr),str(dst),sha_file(dst),'original prompt_text exact copy'))

# Breast exact frozen prompts
src=ROOT/'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919/MEASUREMENTS/selective/selective_QUERIES_PREAUTH.csv'
rr=list(csv.DictReader(src.open(encoding='utf-8-sig')))
assert len(rr)==5308 and all(sha_text(r['prompt_text'])==r['prompt_sha256'] for r in rr)
for r in rr:r['allowed_tokens']='A|B|U'
dst=write_csv(rr,OUT/'04_BREAST/BREAST_QWEN3_32B_FROZEN_QUERIES.csv')
rows_audit.append(('Breast',str(src),sha_file(src),len(rr),str(dst),sha_file(dst),'original prompt_text exact copy'))

# CREDIT exact user_prompt copied into normalized prompt_text
src=ROOT/'CREDIT_G_REPRO_PACKAGE_20260919/04_LLM_RAW_AND_MANIFESTS/selective_QUERY_MANIFEST.csv'
orig=list(csv.DictReader(src.open(encoding='utf-8-sig')))
assert len(orig)==18
rr=[]
for r in orig:
    assert sha_text(r['user_prompt'])==r['user_prompt_sha256']
    x=dict(r);x['prompt_text']=x.pop('user_prompt');x['prompt_sha256']=x['user_prompt_sha256'];x['allowed_tokens']='A|B|U'
    rr.append(x)
dst=write_csv(rr,OUT/'05_CREDIT_G/CREDIT_QWEN3_32B_FROZEN_QUERIES.csv')
rows_audit.append(('CREDIT-G',str(src),sha_file(src),len(rr),str(dst),sha_file(dst),'user_prompt exact copy; normalized field name only'))

# Hospital: exact frozen actionable 30 + frozen packets + original renderer/spec
H=ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'
pair_src=H/'01_TASK_AND_IDENTITIES/FROZEN_ACTIONABLE_UNIQUE_PAIRS_30.csv'
pairs=list(csv.DictReader(pair_src.open(encoding='utf-8-sig')))
assert len(pairs)==30
spec_path=H/'02_PROTOCOL/PAIRWISE_PROMPT_SPEC_V1_8_1.json'
ps=json.loads(spec_path.read_text()); SYS_H=ps['system_template']; USER=ps['user_template']
(AUD/'SYSTEM_HOSPITAL.txt').write_text(SYS_H)
PACK=H/'07_BROAD37_EVIDENCE/V0_8_1_AUDITED_CORPUS_AND_PACKETS'
def packet_path(feature):
    safe=re.sub(r'[^A-Za-z0-9._-]+','_',feature)
    return PACK/f'{safe}_PACKET_V0_8_1.json'
def load_packet(feature):
    p=packet_path(feature)
    o=json.loads(p.read_text())
    assert o['packet_content_frozen'] is True and o['model_independent_packet'] is True
    return p,o
def render_cards(packet,prefix):
    lines=[]
    for c in packet['cards']:
        lines.append(re.sub(r'^E(\d\d):',prefix+r'.E\1:',c['card_text']))
    return '\n'.join(lines) if lines else 'NONE'
rr=[]; packet_rows=[]
for pair in pairs:
    fa,fb=pair['feature_A_canonical'],pair['feature_B_canonical']
    for ori,(da,db) in [('AB',(fa,fb)),('BA',(fb,fa))]:
        ppa,pa=load_packet(da);ppb,pb=load_packet(db)
        A_cards=render_cards(pa,'A');B_cards=render_cards(pb,'B')
        user=USER.format(FEATURE_A_NAME=pa['canonical_feature_name'],PACKET_A_CARDS_WITH_IDS_PREFIXED_A=A_cards,
                         FEATURE_B_NAME=pb['canonical_feature_name'],PACKET_B_CARDS_WITH_IDS_PREFIXED_B=B_cards)
        qid=f"HOSPITAL_selective_{pair['pair_id']}_{ori}"
        rr.append({
            'query_id':qid,'pair_id':pair['pair_id'],'order':ori,
            'semantic_feature_A':fa,'semantic_feature_B':fb,
            'gene_A':da,'gene_B':db,'display_feature_A':da,'display_feature_B':db,
            'k_membership':pair['k_membership'],'allowed_tokens':'A|B',
            'prompt_text':user,'prompt_sha256':sha_text(user),
            'packet_A_sha256':pa['packet_sha256'],'packet_B_sha256':pb['packet_sha256'],
            'packet_A_file_sha256':sha_file(ppa),'packet_B_file_sha256':sha_file(ppb),
            'prompt_spec_sha256':ps['prompt_spec_sha256']
        })
        packet_rows += [(str(ppa),sha_file(ppa),pa['packet_sha256'],da),(str(ppb),sha_file(ppb),pb['packet_sha256'],db)]
assert len(rr)==60
dst=write_csv(rr,OUT/'06_HOSPITAL/HOSPITAL_QWEN3_32B_FROZEN_QUERIES.csv')
rows_audit.append(('Hospital',str(pair_src),sha_file(pair_src),len(rr),str(dst),sha_file(dst),'deterministic original renderer from frozen 30 pairs + frozen packets/spec'))
# unique packet audit
seen={}
for x in packet_rows:seen[x[0]]=x
with (AUD/'HOSPITAL_PACKET_MANIFEST.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['absolute_path','file_sha256','embedded_packet_sha256','feature']);w.writerows(sorted(seen.values()))

# Darmanis: concatenate exact materialized JSONL schedules without changing prompts
base=ROOT/'GBM_REPRO_PACKAGE_V2_9_20260919/PROJECT/GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6/FINAL_QUERY_TABLES_V2_8'
files=[base/'selective_SELECTIVE_MISSING_ARM_IVY_V2_8.jsonl.gz',base/'selective_SELECTIVE_MISSING_ARM_G116_V2_8.jsonl.gz',base/'selective_SELECTIVE_MISSING_ARM_G132_V2_8.jsonl.gz']
rr=[]
for p in files:
    with gzip.open(p,'rt',encoding='utf-8') as f:
        zz=[json.loads(x) for x in f if x.strip()]
    assert all(sha_text(r['prompt_text'])==r['prompt_sha256'] for r in zz)
    rr.extend(zz)
assert len(rr)==1500 and len({r['query_id'] for r in rr})==1500
dst=write_csv(rr,OUT/'07_DARMANIS/DARMANIS_QWEN3_32B_FROZEN_QUERIES.csv')
rows_audit.append(('Darmanis', '|'.join(map(str,files)), '|'.join(sha_file(p) for p in files),len(rr),str(dst),sha_file(dst),'exact materialized JSONL rows concatenated'))

with (AUD/'PROMPT_MATERIALIZATION_AUDIT.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['dataset','source','source_sha256','n_queries','materialized_path','materialized_sha256','method']);w.writerows(rows_audit)
print(json.dumps({x[0]:x[3] for x in rows_audit},indent=2))
print('hospital_unique_packets',len(seen))
