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
from datetime import datetime,timezone
import argparse,json,hashlib,math,time,urllib.request
import pandas as pd

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
CFG=ROOT/'07_PRE_LLM_BUDGET/PROPOSED_RUNTIME_CONFIG_NOT_AUTHORIZED.json'
APP=ROOT/'07_PRE_LLM_BUDGET/EXECUTION_APPROVAL.json'
FPLOCK=ROOT/'07_PRE_LLM_BUDGET/PROVIDER_FINGERPRINT_LOCK.json'
OUTROOT=ROOT/'09_LLM_MEASUREMENT'
SECRET=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'))

ALLOWED_INPUTS={
 'BREAST_GSE25055_GSE25065':ROOT/'08_PRE_LLM_QUERIES/BREAST_GSE25055_GSE25065_selective_QUERIES_PREAUTH.csv',
 'SEPSIS_GSE65682':ROOT/'08_PRE_LLM_QUERIES/SEPSIS_GSE65682_selective_QUERIES_PREAUTH.csv'
}

def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def sha_text(x): return hashlib.sha256(x.encode()).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def load_env():
    vals={}
    for line in SECRET.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1); vals[k.strip()]=v.strip()
    if not vals.get('DEEPSEEK_API_KEY'): raise RuntimeError('API_KEY_MISSING')
    return vals
def lse(xs):
    m=max(xs); return m+math.log(sum(math.exp(x-m) for x in xs))
def parse(resp,cfg):
    ch=resp['choices'][0]; content=(ch['message'].get('content') or '').strip()
    if content not in cfg['allowed_tokens']: return {'parse_status':'FAIL_CONTENT_CONTRACT','content':content}
    lp=(ch.get('logprobs') or {}).get('content') or []
    pos=next((x for x in lp if str(x.get('token','')).strip()),None)
    if pos is None: return {'parse_status':'FAIL_NO_LOGPROB_POSITION','content':content}
    pools={k:[] for k in cfg['allowed_tokens']}
    for x in pos.get('top_logprobs') or []:
        t=str(x.get('token','')).strip()
        if t in pools: pools[t].append(float(x['logprob']))
    ft=str(pos.get('token','')).strip()
    if ft in pools and pos.get('logprob') is not None: pools[ft].append(float(pos['logprob']))
    L={k:(lse(v) if v else None) for k,v in pools.items()}
    if cfg['require_A_and_B_in_top20'] and (L['A'] is None or L['B'] is None):
        return {'parse_status':'FAIL_AB_NOT_BOTH_TOP20','content':content,'logp':L}
    z=lse([L['A'],L['B']]); pA=math.exp(L['A']-z)
    H=0.0
    for p in (pA,1-pA):
        if p>0: H-=p*math.log(p,2)
    pU=None
    if L['U'] is not None:
        z3=lse([L['A'],L['B'],L['U']]); pU=math.exp(L['U']-z3)
    return {'parse_status':'PASS','content':content,'first_token':ft,
            'logp_A':L['A'],'logp_B':L['B'],'logp_U':L['U'],
            'pA_vs_B':pA,'entropy_AB_bits':H,'pU_threeway_topset':pU}
def api_call(vals,payload,cfg):
    err=None
    for a in range(cfg['retry_policy']['max_attempts']):
        try:
            req=urllib.request.Request(vals['DEEPSEEK_BASE_URL'].rstrip('/')+'/chat/completions',
                data=json.dumps(payload,ensure_ascii=False).encode(),
                headers={'Authorization':'Bearer '+vals['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=cfg['retry_policy']['request_timeout_seconds']) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            err=repr(e); time.sleep(min(30,2**a))
    raise RuntimeError('API_FAILED '+str(err))

ap=argparse.ArgumentParser()
ap.add_argument('--task',choices=list(ALLOWED_INPUTS),required=True)
ap.add_argument('--execute',action='store_true')
ap.add_argument('--limit',type=int,default=None)
args=ap.parse_args()

cfg=json.load(open(CFG)); inp=ALLOWED_INPUTS[args.task]
df=pd.read_csv(inp,dtype=str,keep_default_na=False)
manifest=json.load(open(ROOT/'08_PRE_LLM_QUERIES'/f'{args.task}_QUERY_MANIFEST.json'))
if sha_file(inp)!=manifest['query_csv_sha256']: raise RuntimeError('QUERY_MANIFEST_HASH_FAIL')
if df.query_id.nunique()!=len(df): raise RuntimeError('DUP_QUERY_ID')
if set(df.order)!={'AB','BA'}: raise RuntimeError('ORDER_SCHEMA_FAIL')
if not (df.allowed_tokens=='A|B|U').all(): raise RuntimeError('TOKEN_SCHEMA_FAIL')
if sha_text(cfg['system_prompt'])!=manifest['system_prompt_sha256']: raise RuntimeError('SYSTEM_PROMPT_HASH_FAIL')
for r in df.itertuples():
    if sha_text(r.prompt_text)!=r.prompt_sha256: raise RuntimeError('PROMPT_HASH_FAIL '+r.query_id)
for term in ['GSE25065','sealed validation','sealed-validation','X_sealed_validation','y_sealed_validation']:
    if df.prompt_text.str.contains(term,case=False,regex=False).any(): raise RuntimeError('HELDOUT_LEAK '+term)
pre={'status':'PASS','timestamp_utc':now(),'task':args.task,'n_queries':len(df),
     'input_sha256':sha_file(inp),'runtime_config_sha256':sha_file(CFG),
     'execute_requested':bool(args.execute),'llm_called':False}
print(json.dumps(pre,indent=2))
if not args.execute: raise SystemExit(0)

# execution gates
apv=json.load(open(APP))
if apv.get('authorized') is not True: raise RuntimeError('EXECUTION_NOT_AUTHORIZED')
if apv.get('scope')!='selective_SELECTIVE_ONLY': raise RuntimeError('APPROVAL_SCOPE_FAIL')
if apv.get('freeze_report_sha256')!=sha_file(ROOT/'REPORT/PRE_LLM_PRODUCTION_FREEZE_REPORT_20260919.md'):
    raise RuntimeError('APPROVAL_REPORT_HASH_MISMATCH')
if apv.get('runtime_config_sha256')!=sha_file(CFG): raise RuntimeError('APPROVAL_RUNTIME_HASH_MISMATCH')
keyname='breast_query_csv_sha256' if args.task.startswith('BREAST') else 'sepsis_query_csv_sha256'
if apv.get(keyname)!=sha_file(inp): raise RuntimeError('APPROVAL_QUERY_HASH_MISMATCH')
if not FPLOCK.exists(): raise RuntimeError('PROVIDER_FINGERPRINT_LOCK_MISSING')
fpl=json.load(open(FPLOCK))
if fpl.get('status')!='PASS_GENERIC_SMOKE_LOCKED': raise RuntimeError('FINGERPRINT_LOCK_NOT_PASS')
if fpl.get('requested_model')!=cfg['requested_model']: raise RuntimeError('FINGERPRINT_MODEL_LOCK_FAIL')

vals=load_env()
if vals.get('DEEPSEEK_MODEL')!=cfg['requested_model']: raise RuntimeError('MODEL_ENV_MISMATCH')
out=OUTROOT/args.task; cache=out/'cache'; summ=out/'summaries'
cache.mkdir(parents=True,exist_ok=True); summ.mkdir(parents=True,exist_ok=True)
run=df if args.limit is None else df.head(args.limit)
rows=[]
for _,r in run.iterrows():
    keyobj={'query_id':r.query_id,'prompt_sha256':r.prompt_sha256,
            'system_prompt_sha256':manifest['system_prompt_sha256'],
            'runtime_config_sha256':sha_file(CFG),'model':cfg['requested_model'],
            'thinking':cfg['thinking'],'temperature':cfg['temperature'],
            'max_tokens':cfg['max_tokens'],'logprobs':cfg['logprobs'],
            'top_logprobs':cfg['top_logprobs']}
    key=sha_text(json.dumps(keyobj,sort_keys=True,separators=(',',':')))
    cp=cache/f'{key}.json'
    if cp.exists():
        old=json.load(open(cp))
        if old['key_object']!=keyobj: raise RuntimeError('CACHE_KEY_OBJECT_MISMATCH')
        rows.append(old['summary']); continue
    payload={'model':cfg['requested_model'],
      'messages':[{'role':'system','content':cfg['system_prompt']},{'role':'user','content':r.prompt_text}],
      'thinking':cfg['thinking'],'temperature':cfg['temperature'],'max_tokens':cfg['max_tokens'],
      'logprobs':cfg['logprobs'],'top_logprobs':cfg['top_logprobs'],'stream':cfg['stream']}
    t0=time.time(); ts=now(); resp=api_call(vals,payload,cfg); latency=round((time.time()-t0)*1000,1)
    parsed=parse(resp,cfg); pm=resp.get('model'); fp=resp.get('system_fingerprint')
    if pm!=fpl['provider_model'] or fp!=fpl['system_fingerprint']:
        raise RuntimeError('PROVIDER_ROUTE_OR_FINGERPRINT_DRIFT')
    tok=parsed.get('first_token')
    sem=(r.gene_A if tok=='A' else (r.gene_B if tok=='B' else ('U' if tok=='U' else '')))
    s={'timestamp_utc':ts,'task':args.task,'query_id':r.query_id,'unordered_pair_id':r.unordered_pair_id,
       'arm':r.arm,'order':r.order,'gene_A':r.gene_A,'gene_B':r.gene_B,
       'prompt_sha256':r.prompt_sha256,'evidence_packet_sha256':r.evidence_packet_sha256,
       'call_key':key,'provider_model':pm,'system_fingerprint':fp,
       'parse_status':parsed.get('parse_status'),'response_token':tok,'semantic_choice_gene':sem,
       'logp_A':parsed.get('logp_A'),'logp_B':parsed.get('logp_B'),'logp_U':parsed.get('logp_U'),
       'pA_vs_B':parsed.get('pA_vs_B'),'entropy_AB_bits':parsed.get('entropy_AB_bits'),
       'pU_threeway_topset':parsed.get('pU_threeway_topset'),'latency_ms':latency,
       'prompt_tokens':resp.get('usage',{}).get('prompt_tokens'),
       'completion_tokens':resp.get('usage',{}).get('completion_tokens'),
       'total_tokens':resp.get('usage',{}).get('total_tokens')}
    cp.write_text(json.dumps({'key_object':keyobj,'summary':s,'parsed':parsed,'raw_response':resp},ensure_ascii=False,indent=2))
    rows.append(s)
    if s['parse_status']!='PASS':
        pd.DataFrame(rows).to_csv(summ/'CALLS_PARTIAL.csv',index=False)
        raise RuntimeError('HARD_MEASUREMENT_FAILURE '+r.query_id)
    if len(rows)%100==0:
        pd.DataFrame(rows).to_csv(summ/'CALLS_PARTIAL.csv',index=False)
        print('completed_or_cached',len(rows),flush=True)
pd.DataFrame(rows).to_csv(summ/'CALLS_COMPLETE.csv',index=False)
print(json.dumps({'status':'PASS','task':args.task,'n_rows':len(rows)},indent=2))
