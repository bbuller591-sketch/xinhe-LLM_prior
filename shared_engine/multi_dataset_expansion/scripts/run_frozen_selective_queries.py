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
import argparse,json,hashlib,math,time,urllib.request,urllib.error,threading,os
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
CFG=ROOT/'09_PRE_LLM_FREEZE/LLM_RUNTIME_CONFIG_FROZEN.json'
APV=ROOT/'09_PRE_LLM_FREEZE/LLM_EXECUTION_APPROVAL.json'
MASTER=ROOT/'09_PRE_LLM_FREEZE/MASTER_PRE_LLM_MANIFEST.csv'
LOCK=ROOT/'10_LLM_MEASUREMENT/PROVIDER_ROUTE_LOCK.json'
CACHE=ROOT/'10_LLM_MEASUREMENT/CACHE'
FAIL=ROOT/'10_LLM_MEASUREMENT/FAILURES'
CACHE.mkdir(parents=True,exist_ok=True); FAIL.mkdir(parents=True,exist_ok=True)

def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def sha_text(x): return hashlib.sha256(x.encode()).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def load_env(p):
    vals={}
    for line in Path(p).read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1); vals[k.strip()]=v.strip()
    if not vals.get('DEEPSEEK_API_KEY'): raise RuntimeError('API_KEY_MISSING')
    return vals
def lse(xs):
    if not xs:return None
    m=max(xs); return m+math.log(sum(math.exp(x-m) for x in xs))

def parse_response(resp,cfg):
    ch=resp['choices'][0]
    content=(ch['message'].get('content') or '').strip()
    if content not in cfg['allowed_tokens']:
        return {'parse_status':'FAIL_CONTENT_CONTRACT','content':content}
    lp=(ch.get('logprobs') or {}).get('content') or []
    pos=next((x for x in lp if str(x.get('token','')).strip()),None)
    if pos is None:
        return {'parse_status':'FAIL_NO_LOGPROB_POSITION','content':content}
    pools={k:[] for k in cfg['allowed_tokens']}
    seen_exact=set()
    for x in pos.get('top_logprobs') or []:
        raw=str(x.get('token','')); t=raw.strip()
        if t in pools and x.get('logprob') is not None:
            pools[t].append(float(x['logprob'])); seen_exact.add((raw,float(x['logprob'])))
    raw0=str(pos.get('token','')); t0=raw0.strip(); lp0=pos.get('logprob')
    if t0 in pools and lp0 is not None and (raw0,float(lp0)) not in seen_exact:
        pools[t0].append(float(lp0))
    L={k:lse(v) for k,v in pools.items()}
    if cfg['require_A_and_B_in_top20'] and (L['A'] is None or L['B'] is None):
        return {'parse_status':'FAIL_AB_NOT_BOTH_TOP20','content':content,'first_token':t0,'logp':L}
    z=lse([L['A'],L['B']]); pA=math.exp(L['A']-z)
    H=0.0
    for p in (pA,1-pA):
        if p>0: H-=p*math.log(p,2)
    pU=None
    if L['U'] is not None:
        z3=lse([L['A'],L['B'],L['U']]); pU=math.exp(L['U']-z3)
    return {'parse_status':'PASS','content':content,'first_token':t0,
            'logp_A':L['A'],'logp_B':L['B'],'logp_U':L['U'],
            'pA_vs_B':pA,'entropy_AB_bits':H,'pU_threeway_topset':pU}

def api_call(vals,payload,backoffs):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last=None
    for attempt in range(len(backoffs)):
        try:
            req=urllib.request.Request(vals['DEEPSEEK_BASE_URL'].rstrip('/')+'/chat/completions',
                data=json.dumps(payload,ensure_ascii=False).encode(),
                headers={'Authorization':'Bearer '+vals['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
            with opener.open(req,timeout=120) as r:
                return json.loads(r.read().decode()),attempt+1
        except urllib.error.HTTPError as e:
            last=repr(e)
            if not (e.code==429 or e.code>=500): raise
        except (urllib.error.URLError,TimeoutError,ConnectionError) as e:
            last=repr(e)
        if attempt+1<len(backoffs): time.sleep(backoffs[attempt])
    raise RuntimeError('API_TRANSPORT_EXHAUSTED '+str(last))

def peak_cost(summary):
    pt=int(summary.get('prompt_tokens') or 0); ct=int(summary.get('completion_tokens') or 0)
    hit=summary.get('prompt_cache_hit_tokens'); miss=summary.get('prompt_cache_miss_tokens')
    if hit is None or miss is None:
        miss=pt; hit=0
    return float(miss)*0.30/1e6 + float(hit)*0.006/1e6 + float(ct)*1.20/1e6

ap=argparse.ArgumentParser()
ap.add_argument('--input',required=True)
ap.add_argument('--stage',required=True)
ap.add_argument('--execute',action='store_true')
args=ap.parse_args()

cfg=json.load(open(CFG)); apv=json.load(open(APV)); lock=json.load(open(LOCK))
if apv.get('authorized') is not True or apv.get('authorized_by_user_after_freeze') is not True:
    raise RuntimeError('NOT_AUTHORIZED')
if apv.get('master_manifest_sha256')!=sha_file(MASTER): raise RuntimeError('MASTER_HASH_MISMATCH')
if apv.get('runtime_config_sha256')!=sha_file(CFG): raise RuntimeError('RUNTIME_HASH_MISMATCH')
if apv.get('authorized_model')!=cfg['requested_model']: raise RuntimeError('MODEL_APPROVAL_MISMATCH')
if lock['provider_model']!=cfg['requested_model'] or not lock.get('system_fingerprint'):
    raise RuntimeError('PROVIDER_LOCK_INVALID')

inp=Path(args.input)
df=pd.read_csv(inp,dtype=str,keep_default_na=False)
if df.query_id.nunique()!=len(df): raise RuntimeError('DUP_QUERY_ID')
if not (df.allowed_tokens=='A|B|U').all(): raise RuntimeError('TOKEN_SCHEMA_FAIL')
for r in df.itertuples():
    if sha_text(r.prompt_text)!=r.prompt_sha256: raise RuntimeError('PROMPT_HASH_FAIL '+r.query_id)
    if not r.evidence_packet_sha256: raise RuntimeError('EVIDENCE_HASH_MISSING '+r.query_id)

pre={'status':'PASS','stage':args.stage,'n_queries':len(df),'input_sha256':sha_file(inp),
     'runtime_config_sha256':sha_file(CFG),'provider_lock_sha256':sha_file(LOCK),
     'execute_requested':bool(args.execute)}
print(json.dumps(pre,indent=2))
if not args.execute: raise SystemExit(0)

vals=load_env(cfg['api_secret_env'])
if vals.get('DEEPSEEK_MODEL')!=cfg['requested_model']: raise RuntimeError('MODEL_ENV_MISMATCH')
system_hash=sha_text(cfg['system_prompt']); cfg_hash=sha_file(CFG); fp=lock['system_fingerprint']
backoffs=cfg['retry_policy']['backoff_seconds']
outdir=ROOT/'10_LLM_MEASUREMENT'/args.stage
summdir=outdir/'summaries'; summdir.mkdir(parents=True,exist_ok=True)

def one(row):
    keyobj={'query_id':row.query_id,'prompt_sha256':row.prompt_sha256,
      'evidence_packet_sha256':row.evidence_packet_sha256,
      'system_prompt_sha256':system_hash,'runtime_config_sha256':cfg_hash,
      'model':cfg['requested_model'],'provider_fingerprint':fp,
      'thinking':cfg['thinking'],'temperature':cfg['temperature'],
      'max_tokens':cfg['max_tokens'],'logprobs':cfg['logprobs'],'top_logprobs':cfg['top_logprobs']}
    key=sha_text(json.dumps(keyobj,sort_keys=True,separators=(',',':')))
    cp=CACHE/f'{key}.json'
    if cp.exists():
        obj=json.load(open(cp)); s=obj['summary']
        if s.get('parse_status')!='PASS' or s.get('system_fingerprint')!=fp or s.get('prompt_sha256')!=row.prompt_sha256:
            raise RuntimeError('INVALID_EXISTING_CACHE '+row.query_id)
        s=dict(s); s['cache_reused_local']=True
        return s
    payload={'model':cfg['requested_model'],
      'messages':[{'role':'system','content':cfg['system_prompt']},{'role':'user','content':row.prompt_text}],
      'thinking':cfg['thinking'],'temperature':cfg['temperature'],'max_tokens':cfg['max_tokens'],
      'logprobs':cfg['logprobs'],'top_logprobs':cfg['top_logprobs'],'stream':cfg['stream']}
    t0=time.time(); ts=now()
    resp,attempts=api_call(vals,payload,backoffs); lat=round((time.time()-t0)*1000,1)
    parsed=parse_response(resp,cfg)
    pm=resp.get('model'); rfp=resp.get('system_fingerprint')
    if pm!=lock['provider_model'] or rfp!=fp:
        parsed={'parse_status':'FAIL_PROVIDER_ROUTE_OR_FINGERPRINT',**parsed}
    tok=parsed.get('first_token')
    sem=row.gene_A if tok=='A' else (row.gene_B if tok=='B' else ('U' if tok=='U' else ''))
    usage=resp.get('usage') or {}; det=usage.get('prompt_tokens_details') or {}
    s={'timestamp_utc':ts,'query_id':row.query_id,'unordered_pair_id':row.unordered_pair_id,
       'arm':row.arm,'order':row.order,'query_role':row.query_role,
       'gene_A':row.gene_A,'gene_B':row.gene_B,'prompt_sha256':row.prompt_sha256,
       'evidence_packet_sha256':row.evidence_packet_sha256,'call_key':key,
       'requested_model':cfg['requested_model'],'provider_model':pm,'system_fingerprint':rfp,
       'parse_status':parsed.get('parse_status'),'response_token':tok,'semantic_choice_gene':sem,
       'logp_A':parsed.get('logp_A'),'logp_B':parsed.get('logp_B'),'logp_U':parsed.get('logp_U'),
       'pA_vs_B':parsed.get('pA_vs_B'),'entropy_AB_bits':parsed.get('entropy_AB_bits'),
       'pU_threeway_topset':parsed.get('pU_threeway_topset'),'latency_ms':lat,'attempts':attempts,
       'finish_reason':resp['choices'][0].get('finish_reason'),
       'prompt_tokens':usage.get('prompt_tokens'),'completion_tokens':usage.get('completion_tokens'),
       'total_tokens':usage.get('total_tokens'),
       'prompt_cache_hit_tokens':usage.get('prompt_cache_hit_tokens',det.get('cached_tokens')),
       'prompt_cache_miss_tokens':usage.get('prompt_cache_miss_tokens'),
       'cache_reused_local':False}
    obj={'key_object':keyobj,'summary':s,'parsed':parsed,'raw_response':resp}
    if s['parse_status']!='PASS':
        fpout=FAIL/f"{key}_{int(time.time())}.json"
        fpout.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
        return s
    tmp=cp.with_name(cp.name+f'.tmp.{os.getpid()}.{threading.get_ident()}')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(cp)
    return s

rows=[]; hard_cap=float(apv.get('hard_cost_cap_usd',3.0)); workers=int(cfg.get('parallelism',32))
records=list(df.itertuples(index=False))
for start in range(0,len(records),workers):
    batch=records[start:start+workers]
    bout=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(one,r):r for r in batch}
        for fut in as_completed(futs):
            bout.append(fut.result())
    rows.extend(bout)
    cur=pd.DataFrame(rows)
    cur.to_csv(summdir/'CALLS_PARTIAL.csv',index=False)
    fails=cur[cur.parse_status!='PASS']
    if len(fails):
        raise RuntimeError('HARD_MEASUREMENT_FAILURE '+fails[['query_id','parse_status']].to_json(orient='records'))
    cost=sum(peak_cost(s) for s in rows)
    print(json.dumps({'completed_or_cached':len(rows),'stage':args.stage,'peak_equivalent_cost_usd':cost}),flush=True)
    if cost>hard_cap: raise RuntimeError(f'HARD_COST_CAP_EXCEEDED {cost} > {hard_cap}')

res=pd.DataFrame(rows).sort_values('query_id',kind='mergesort').reset_index(drop=True)
res.to_csv(summdir/'CALLS_COMPLETE.csv',index=False)
status={'status':'PASS','stage':args.stage,'n_rows':len(res),'n_local_cache_reused':int(res.cache_reused_local.sum()),
        'n_new_api_calls':int((~res.cache_reused_local).sum()),
        'provider_model':lock['provider_model'],'system_fingerprint':fp,
        'all_parse_pass':bool((res.parse_status=='PASS').all()),
        'total_prompt_tokens':int(pd.to_numeric(res.prompt_tokens,errors='coerce').fillna(0).sum()),
        'total_completion_tokens':int(pd.to_numeric(res.completion_tokens,errors='coerce').fillna(0).sum()),
        'peak_equivalent_cost_usd':sum(peak_cost(s) for s in rows),
        'calls_complete_sha256':sha_file(summdir/'CALLS_COMPLETE.csv')}
(summdir/'RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
