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
import csv,json,hashlib,math,os,statistics
from collections import Counter,defaultdict
import numpy as np
import pandas as pd

ROOT=Path(str(REPRO_ROOT))
QW=ROOT/'selective_qwen3_32b_local_modelswap_20260924'
RET=ROOT/'qwen3_32b_local_return_20260924/extracted'
INP=ROOT/'LOCAL_LLM_MEASUREMENT_BUNDLE_20260924'
GPT=ROOT/'selective_qwen3_32b_local_modelswap_20260924'
AUD=QW/'00_PROTOCOL_AND_AUDIT'
AUD.mkdir(parents=True,exist_ok=True)

def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def sha_text(s): return hashlib.sha256(s.encode()).hexdigest()
def lse(vals):
    vals=[float(x) for x in vals if x is not None and np.isfinite(float(x))]
    if not vals: return None
    m=max(vals); return m+math.log(sum(math.exp(v-m) for v in vals))
def ent(p):
    p=min(max(float(p),1e-300),1-1e-16)
    hn=-(p*math.log(p)+(1-p)*math.log(1-p))
    hb=hn/math.log(2)
    return hn,hb,1-hb

run_status=json.load(open(RET/'return_outputs/run_status.json'))
runtime=json.load(open(RET/'return_outputs/model_runtime.json'))
preflight=json.load(open(RET/'audit/QWEN3_32B_PREFLIGHT.json'))
input_bundle_line=(RET/'INPUT_BUNDLE_SHA256.txt').read_text().strip().split()[0]
actual_input_sha=sha_file(ROOT/'LOCAL_LLM_MEASUREMENT_BUNDLE_20260924.tar.gz')
if input_bundle_line!=actual_input_sha:
    raise RuntimeError(f'INPUT BUNDLE SHA mismatch {input_bundle_line} != {actual_input_sha}')
measure_path=RET/'return_outputs/measurements.jsonl'
if sha_file(measure_path)!=run_status['measurements_sha256']:
    raise RuntimeError('measurements sha mismatch')

input_rows={}
for line in open(INP/'queries/jsonl/all_queries.jsonl',encoding='utf-8'):
    r=json.loads(line)
    input_rows[r['portable_query_id']]=r
meas={}
for line in open(measure_path,encoding='utf-8'):
    if not line.strip(): continue
    r=json.loads(line)
    pid=r['portable_query_id']
    if pid in meas: raise RuntimeError('duplicate measurement '+pid)
    meas[pid]=r
if set(meas)!=set(input_rows):
    raise RuntimeError(f'id set mismatch missing={len(set(input_rows)-set(meas))} extra={len(set(meas)-set(input_rows))}')
if len(meas)!=14542: raise RuntimeError(len(meas))

problems=[]
p_err=[]
hash_bad=[]
contract_bad=[]
thinking=[]
trunc=[]
hosp_bad=[]
raw_phase=Counter()
for pid,r in meas.items():
    inp=input_rows[pid]
    if r.get('status')!='PASS': problems.append((pid,r.get('status')))
    if not r.get('hash_match',False): hash_bad.append(pid)
    if r.get('thinking_contamination'): thinking.append(pid)
    if r.get('truncated'): trunc.append(pid)
    if str(r.get('system_prompt_sha256'))!=str(inp['system_prompt_sha256']) or str(r.get('user_prompt_sha256'))!=str(inp['user_prompt_sha256']):
        hash_bad.append(pid)
    la=float(r['semantic_logit_A']); lb=float(r['semantic_logit_B'])
    z=lse([la,lb]); pa=math.exp(la-z)
    if abs(pa-float(r['pA_vs_B']))>1e-10:
        p_err.append((pid,pa,r['pA_vs_B']))
    ft=str(r.get('first_generated_token','')).strip()
    if ft not in inp['allowed_tokens']: contract_bad.append((pid,ft,inp['allowed_tokens']))
    if inp['dataset']=='Hospital Osteoporosis':
        if (r.get('hospital_parser') or {}).get('hospital_parser_status')!='PASS': hosp_bad.append(pid)
    raw_phase[r.get('run_phase','?')]+=1
if problems or hash_bad or p_err or contract_bad or thinking or trunc or hosp_bad:
    raise RuntimeError({'problems':problems[:3],'hash_bad':hash_bad[:3],'p_err':p_err[:3],'contract_bad':contract_bad[:3],
                        'thinking':thinking[:3],'trunc':trunc[:3],'hosp_bad':hosp_bad[:3]})

# Bundle->source schedule path in the Qwen workspace.
schedule={
 'renal_formal':QW/'02_RENAL/RENAL_GPT4OMINI_FROZEN_QUERIES.csv',
 'sepsis_selective':QW/'03_GSE272769/GSE272769_GPT4OMINI_FROZEN_QUERIES.csv',
 'breast_selective':QW/'04_BREAST/BREAST_GPT4OMINI_FROZEN_QUERIES.csv',
 'credit_selective':QW/'05_CREDIT_G/CREDIT_GPT4OMINI_FROZEN_QUERIES.csv',
 'hospital_selective':QW/'06_HOSPITAL/HOSPITAL_GPT4OMINI_FROZEN_QUERIES.csv',
 'darmanis_selective':QW/'07_DARMANIS/DARMANIS_GPT4OMINI_FROZEN_QUERIES.csv',
 'darmanis_broad7':QW/'07_DARMANIS/DARMANIS_GPT4OMINI_REQUIRED_BROAD7_QUERIES.csv',
 'sepsis_q2_global':QW/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q2/Q2_GLOBAL_MATCHED_QUERIES_FROZEN.csv',
 'breast_q2_global':QW/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE/BREAST_Q2_SOURCE_GPT_QUERIES.csv',
 'hospital_q2_global':QW/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2/HOSPITAL_Q2_GPT_QUERIES.csv',
 'darmanis_q2_global':QW/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2/DARMANIS_Q2_REQUIRED_GPT_QUERIES.csv',
}
outdir={
 'renal_formal':QW/'02_RENAL/measurement',
 'sepsis_selective':QW/'03_GSE272769/measurement',
 'breast_selective':QW/'04_BREAST/measurement',
 'credit_selective':QW/'05_CREDIT_G/measurement',
 'hospital_selective':QW/'06_HOSPITAL/measurement',
 'darmanis_selective':QW/'07_DARMANIS/measurement',
 'darmanis_broad7':QW/'07_DARMANIS/measurement_broad7',
 'sepsis_q2_global':QW/'08_Q2_Q5_DIAGNOSTICS/GSE272769_Q2/measurement',
 'breast_q2_global':QW/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE/measurement',
 'hospital_q2_global':QW/'08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2/measurement',
 'darmanis_q2_global':QW/'08_Q2_Q5_DIAGNOSTICS/DARMANIS_Q2/measurement',
}

def find_argmax_logit(m):
    aid=int(m['argmax_token_id'])
    for x in m.get('raw_first_step_top5') or []:
        if int(x['token_id'])==aid:
            return float(x['logit'])
    raise RuntimeError('argmax logit absent '+m['portable_query_id'])

conversion_stats=[]
for bid,p in schedule.items():
    rows=list(csv.DictReader(open(p,encoding='utf-8-sig',newline='')))
    od=outdir[bid]; od.mkdir(parents=True,exist_ok=True)
    converted=[]
    direct_deltas=[]
    protocol_pas=[]
    for src in rows:
        pid=f"{bid}::{src['query_id']}"
        m=meas[pid]
        inp=input_rows[pid]
        style=inp['parser_style_original']
        ft=str(m['first_generated_token']).strip()
        direct={'A':float(m['semantic_logit_A']),'B':float(m['semantic_logit_B']),
                'U':None if m.get('semantic_logit_U') is None else float(m['semantic_logit_U']),
                'T':None if m.get('semantic_logit_T') is None else float(m['semantic_logit_T'])}
        L=direct.copy()
        arglog=find_argmax_logit(m)
        doubled=False
        if style=='double_count_first' and ft in L and L[ft] is not None:
            L[ft]=lse([L[ft],arglog])
            doubled=True
        z=lse([L['A'],L['B']]); pa=math.exp(L['A']-z); pb=1-pa
        hn,hb,C=ent(pa)
        direct_deltas.append(abs(pa-float(m['pA_vs_B'])))
        protocol_pas.append(pa)
        base={k:v for k,v in src.items() if k!='prompt_text'}
        common={
          'parse_status':'PASS',
          'content':m.get('generated_text',''),
          'first_token':ft,
          'reasoning_nonempty':False,
          'first_token_raw_logprob':np.nan,
          'first_token_raw_logit':arglog,
          'n_top_logprobs':int(m.get('vocab_dim',runtime['model_config']['vocab_size'])),
          'logp_A':L['A'],'logp_B':L['B'],'logp_U':L['U'],'logp_T':L['T'],
          'A_variants_top20':len(runtime['semantic_token_groups']['A'])+(1 if doubled and ft=='A' else 0),
          'B_variants_top20':len(runtime['semantic_token_groups']['B'])+(1 if doubled and ft=='B' else 0),
          'U_variants_top20':len(runtime['semantic_token_groups']['U'])+(1 if doubled and ft=='U' else 0),
          'T_variants_top20':len(runtime['semantic_token_groups']['T'])+(1 if doubled and ft=='T' else 0),
          'pA_vs_B':pa,'pB_vs_A':pb,'entropy_AB_nats':hn,'entropy_AB_bits':hb,'certainty':C,
          'finish_reason':'local_greedy',
          'timestamp_utc':run_status.get('generated_at'),
          'call_key':sha_text(pid+'|Qwen3-32B|full_vocab_first_step'),
          'requested_model':'Qwen3-32B','provider_model':'Qwen3-32B',
          'system_fingerprint':'LOCAL_QWEN3_32B_BF16_A100',
          'system_prompt_sha256':m['system_prompt_sha256'],
          'parser_style':style,
          'prompt_tokens':int(m['input_token_count']),
          'completion_tokens':int(m['generation_token_count']),
          'total_tokens':int(m['input_token_count'])+int(m['generation_token_count']),
          'latency_ms':float(m['runtime_seconds'])*1000,
          'attempt':1,'error':np.nan,
          'local_raw_pA_full_vocab':float(m['pA_vs_B']),
          'local_protocol_pA_after_original_parser':pa,
          'local_historical_first_token_double_count_applied':bool(doubled),
          'local_vocab_dim':int(m['vocab_dim'])
        }
        if bid.startswith('hospital_'):
            hp=m['hospital_parser']
            pdisp=pa
            order=str(src['order'])
            psem=pdisp if order=='AB' else 1-pdisp
            base.update({
              'parse_status':'PASS','content':m.get('generated_text',''),
              'p_display_A':pdisp,'pA_vs_B':pdisp,'pB_vs_A':1-pdisp,
              'entropy_AB_nats':hn,'entropy_AB_bits':hb,'certainty':C,
              'first_text_token':ft,'first_token':ft,'logp_A':L['A'],'logp_B':L['B'],
              'A_variants_top20':len(runtime['semantic_token_groups']['A']),
              'B_variants_top20':len(runtime['semantic_token_groups']['B']),
              'hard_winner_display':ft,
              'cites_A':'|'.join(hp.get('cites_a') or []),'cites_B':'|'.join(hp.get('cites_b') or []),
              'rationale':hp.get('rationale',''),'rationale_words':len(str(hp.get('rationale','')).split()),
              'finish_reason':'local_greedy','first_logprob_token':ft,'timestamp_utc':run_status.get('generated_at'),
              'call_key':common['call_key'],'requested_model':'Qwen3-32B','provider_model':'Qwen3-32B',
              'system_fingerprint':'LOCAL_QWEN3_32B_BF16_A100',
              'p_semantic_feature_A':psem,
              'hard_winner_semantic':src['semantic_feature_A'] if psem>=.5 else src['semantic_feature_B'],
              'prompt_tokens':common['prompt_tokens'],'completion_tokens':common['completion_tokens'],'total_tokens':common['total_tokens'],
              'latency_ms':common['latency_ms'],'attempt':1,'error':np.nan,
              'first_token_raw_logit':arglog,'local_raw_pA_full_vocab':float(m['pA_vs_B']),
              'local_vocab_dim':int(m['vocab_dim'])
            })
            converted.append(base)
        else:
            converted.append({**base,**common})
    df=pd.DataFrame(converted)
    df.to_csv(od/'CALLS_COMPLETE.csv',index=False)
    # Also keep exact local raw measurement subset for provenance.
    with open(od/'LOCAL_RAW_MEASUREMENTS.jsonl','w',encoding='utf-8') as f:
        for src in rows:
            pid=f"{bid}::{src['query_id']}"
            f.write(json.dumps(meas[pid],ensure_ascii=False,separators=(',',':'))+'\n')
    st={
      'dataset':rows[0].get('dataset',input_rows[f"{bid}::{rows[0]['query_id']}"]['dataset']),
      'bundle_id':bid,'status':'COMPLETE','n_calls':len(rows),'n_failures':0,'n_unusable_top20':0,
      'model':'Qwen3-32B','measurement_source':'LOCAL_FULL_VOCAB_FIRST_STEP_LOGITS',
      'parser_style_original':input_rows[f"{bid}::{rows[0]['query_id']}"]['parser_style_original'],
      'historical_double_count_reapplied':input_rows[f"{bid}::{rows[0]['query_id']}"]['parser_style_original']=='double_count_first',
      'prompt_tokens':int(df.prompt_tokens.sum()),'completion_tokens':int(df.completion_tokens.sum()),
      'first_token_counts':df.first_token.astype(str).value_counts().to_dict(),
      'system_fingerprints':{'LOCAL_QWEN3_32B_BF16_A100':len(rows)},
      'source_measurements_sha256':sha_file(measure_path),'manifest_sha256':sha_file(p),
      'max_abs_protocol_vs_raw_pA':float(max(direct_deltas)) if direct_deltas else 0.0
    }
    (od/'MEASUREMENT_STATUS.json').write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n')
    conversion_stats.append(st)

# Cross-check output row counts and IDs.
for bid,p in schedule.items():
    src=list(csv.DictReader(open(p,encoding='utf-8-sig')))
    got=pd.read_csv(outdir[bid]/'CALLS_COMPLETE.csv')
    if len(src)!=len(got) or set(r['query_id'] for r in src)!=set(got.query_id.astype(str)):
        raise RuntimeError('conversion id/count fail '+bid)

audit={
 'status':'PASS',
 'archive_sha256':'4933be7b955081381a00cfc368ad72ce84a77f50f50cdb04923dbdeecbe05d4a',
 'input_bundle_sha256':actual_input_sha,
 'returned_input_bundle_sha256':input_bundle_line,
 'measurements_sha256':sha_file(measure_path),
 'expected_rows':14542,'measurement_rows':len(meas),'unique_ids':len(meas),
 'all_status_pass':True,'hash_match_all':True,'no_thinking_contamination':True,'no_context_truncation':True,
 'hospital_parser_all_pass':True,'probability_recompute_max_abs_error':float(max(abs(float(meas[k]['pA_vs_B'])-(math.exp(float(meas[k]['semantic_logit_A'])-lse([float(meas[k]['semantic_logit_A']),float(meas[k]['semantic_logit_B'])])))) for k in meas)),
 'run_phase_counts':dict(raw_phase),
 'model_runtime':runtime,
 'run_status':run_status,
 'conversion_rule':{
    'base':'Use full-vocabulary first-step semantic log-sum-exp returned by the local Qwen runner.',
    'historical_parser_compatibility':'For Sepsis/Breast/Darmanis schedules marked double_count_first, re-apply the frozen historical parser quirk by adding the greedy first-token raw logit one additional time to that semantic pool before A/B normalization. Renal/CREDIT/Hospital do not double-count.',
    'why':'This preserves the original dataset-specific downstream probability extraction rule while replacing API top-20 truncation with exact local full-vocabulary semantic mass.',
    'raw_and_protocol_values_preserved':True
 },
 'conversion_schedules':conversion_stats
}
(AUD/'QWEN3_32B_RETURN_AND_CONVERSION_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
# Copy runtime/preflight/run status.
for src,name in [(RET/'return_outputs/model_runtime.json','QWEN3_32B_MODEL_RUNTIME.json'),
                 (RET/'return_outputs/run_status.json','QWEN3_32B_LOCAL_RUN_STATUS.json'),
                 (RET/'audit/QWEN3_32B_PREFLIGHT.json','QWEN3_32B_PREFLIGHT.json'),
                 (RET/'return_outputs/tokenizer_semantic_groups.json','QWEN3_32B_TOKENIZER_SEMANTIC_GROUPS.json')]:
    (AUD/name).write_bytes(src.read_bytes())

print(json.dumps({
 'status':'PASS','rows':len(meas),'run_phase_counts':dict(raw_phase),
 'hospital':sum(1 for x in meas.values() if x['dataset']=='Hospital Osteoporosis'),
 'conversion':[{k:x[k] for k in ['bundle_id','n_calls','parser_style_original','historical_double_count_reapplied','max_abs_protocol_vs_raw_pA']} for x in conversion_stats]
},ensure_ascii=False,indent=2))
