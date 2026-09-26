#!/usr/bin/env python3
from __future__ import annotations

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
import argparse, concurrent.futures as cf, hashlib, json, math, os, random, socket, threading, time
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from pathlib import Path
from urllib import error, request
import pandas as pd

RETRY_HTTP={429,500,502,503,504}
TRANSPORT=(ConnectionResetError,TimeoutError,socket.timeout,error.URLError,RemoteDisconnected)
ALLOWED={'A','B','T','U'}

def now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def load_cfg(path): return json.loads(Path(path).read_text())
def load_env(cfg):
    vals={}
    p=Path(cfg['env_file'])
    for line in p.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1); vals[k.strip()]=v.strip().strip('"').strip("'")
    base=os.environ.get(cfg['base_url_env']) or vals.get(cfg['base_url_env'],'')
    key=os.environ.get(cfg['api_key_env']) or vals.get(cfg['api_key_env'],'')
    if not base or not key: raise RuntimeError('missing API config')
    return base.rstrip('/'),key
def lse(xs):
    if not xs:return None,0
    m=max(xs); return m+math.log(sum(math.exp(v-m) for v in xs)),len(xs)
def extract(obj):
    ch=obj['choices'][0]; content=(ch.get('message',{}).get('content') or '').strip()
    hard=content if content in ALLOWED else None
    toks=(ch.get('logprobs') or {}).get('content') or []
    first=next((x for x in toks if str(x.get('token','')).strip()),{})
    alts=first.get('top_logprobs') or []
    masses={}; counts={}
    for lab in 'ABTU':
        masses[lab],counts[lab]=lse([float(x['logprob']) for x in alts if str(x.get('token','')).strip()==lab])
    pA=None
    if masses['A'] is not None and masses['B'] is not None:
        m=max(masses['A'],masses['B']); ea=math.exp(masses['A']-m); eb=math.exp(masses['B']-m); pA=ea/(ea+eb)
    return {'hard_token':hard,'raw_content':content,'first_token':str(first.get('token','')).strip(),
            'logmass_A':masses['A'],'logmass_B':masses['B'],'logmass_T':masses['T'],'logmass_U':masses['U'],
            'variants_A':counts['A'],'variants_B':counts['B'],'variants_T':counts['T'],'variants_U':counts['U'],
            'pA_cond_AB':pA,'ab_prob_usable':pA is not None}
def make_payload(prompt,cfg):
    system,user=prompt.rstrip('\n').split('\n\n',1)
    p={'model':cfg['requested_model'],'messages':[{'role':'system','content':system},{'role':'user','content':user}],
       'temperature':cfg['temperature'],'max_tokens':cfg['max_tokens'],'stream':False,
       'logprobs':True,'top_logprobs':cfg['top_logprobs']}
    if cfg.get('thinking') is not None:p['thinking']=cfg['thinking']
    if cfg.get('top_p') is not None:p['top_p']=cfg['top_p']
    return p
def done_ids(path):
    if not path.exists():return set()
    out=set()
    for line in path.read_text().splitlines():
        if line.strip():
            o=json.loads(line)
            if o.get('final_status') not in {'RETRY'}: out.add(o['measurement_call_id'])
    return out
def one_call(row,cfg,base,key,root):
    prompt=(root/row['prompt_file']).read_text(encoding='utf-8')
    if sha(prompt.rstrip('\n'))!=row['prompt_sha256']:
        return {**row,'final_status':'LOCAL_HASH_FAIL','error':'prompt sha mismatch'}
    payload=make_payload(prompt,cfg)
    opener=request.build_opener(request.ProxyHandler({})) if cfg.get('disable_env_proxy') else request.build_opener()
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    for attempt in range(1,5):
        started=now()
        try:
            req=request.Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers=headers)
            with opener.open(req,timeout=120) as resp:
                status=resp.status; obj=json.loads(resp.read().decode())
            model=obj.get('model'); fp=obj.get('system_fingerprint')
            if cfg.get('expected_returned_model') and model!=cfg['expected_returned_model']:
                return {**row,'attempt':attempt,'timestamp_start':started,'timestamp_end':now(),'http_status':status,
                        'returned_model':model,'system_fingerprint':fp,'final_status':'RUNTIME_MISMATCH','error':'model mismatch'}
            if cfg.get('expected_system_fingerprint') and fp!=cfg['expected_system_fingerprint']:
                return {**row,'attempt':attempt,'timestamp_start':started,'timestamp_end':now(),'http_status':status,
                        'returned_model':model,'system_fingerprint':fp,'final_status':'RUNTIME_MISMATCH','error':'fingerprint mismatch'}
            ex=extract(obj)
            if ex['hard_token'] is None:
                st='BAD_RESPONSE'
            elif ex['hard_token'] in {'A','B'} and not ex['ab_prob_usable']:
                st='PROB_MISSING'
            else:
                st='SUCCESS'
            return {**row,'attempt':attempt,'timestamp_start':started,'timestamp_end':now(),'http_status':status,
                    'requested_model':cfg['requested_model'],'returned_model':model,'system_fingerprint':fp,
                    **ex,'final_status':st}
        except error.HTTPError as e:
            if e.code not in RETRY_HTTP:
                return {**row,'attempt':attempt,'timestamp_start':started,'timestamp_end':now(),'http_status':e.code,
                        'final_status':'NONRETRYABLE_HTTP','error':str(e)}
            if attempt==4:
                return {**row,'attempt':attempt,'timestamp_start':started,'timestamp_end':now(),'http_status':e.code,
                        'final_status':'FINAL_HTTP_FAIL','error':str(e)}
        except TRANSPORT as e:
            if attempt==4:
                return {**row,'attempt':attempt,'timestamp_start':started,'timestamp_end':now(),
                        'final_status':'FINAL_TRANSPORT_FAIL','error':type(e).__name__+': '+str(e)}
        time.sleep(min(30,2**attempt)*(0.5+random.random()))
    raise AssertionError
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--year',type=int,choices=[2024,2025],required=True)
    ap.add_argument('--schedule',type=Path,required=True); ap.add_argument('--out-log',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=Path('MODEL_CONFIG_DIRECT_DEEPSEEK_FLASH.json'))
    ap.add_argument('--workers',type=int,default=16); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--allow-2025',action='store_true'); ap.add_argument('--gate-2025',type=Path,default=Path('ALLOW_2025_AFTER_2024.json'))
    a=ap.parse_args()
    if a.year==2025:
        if not a.allow_2025 or not a.gate_2025.exists(): raise SystemExit('2025 locked: require --allow-2025 and gate file')
        gate=json.loads(a.gate_2025.read_text())
        if not gate.get('approved',False): raise SystemExit('2025 gate not approved')
    root=Path(str(REPRO_ROOT / '07_jkp153/phaseB'))
    cfg=load_cfg(a.config); base,key=load_env(cfg)
    df=pd.read_csv(a.schedule); rows=df.to_dict('records')
    done=done_ids(a.out_log); rows=[r for r in rows if r['measurement_call_id'] not in done]
    if a.limit: rows=rows[:a.limit]
    a.out_log.parent.mkdir(parents=True,exist_ok=True)
    processed=0; counts={}; stop=False
    # bounded submission window so runtime mismatch can stop quickly
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex, a.out_log.open('a',encoding='utf-8') as f:
        pending={}
        it=iter(rows)
        def fill():
            while len(pending)<a.workers*2 and not stop:
                try:r=next(it)
                except StopIteration:return
                pending[ex.submit(one_call,r,cfg,base,key,root)]=r['measurement_call_id']
        fill()
        while pending:
            fut=next(cf.as_completed(list(pending)))
            cid=pending.pop(fut)
            try:res=fut.result()
            except Exception as e:res={'measurement_call_id':cid,'final_status':'WORKER_EXCEPTION','error':type(e).__name__+': '+str(e)}
            f.write(json.dumps(res,ensure_ascii=False,sort_keys=True,default=str)+'\n'); f.flush()
            processed+=1; st=res.get('final_status','UNKNOWN'); counts[st]=counts.get(st,0)+1
            if st=='RUNTIME_MISMATCH':
                stop=True
                for q in pending:q.cancel()
            if processed%100==0:
                print(json.dumps({'processed_this_run':processed,'remaining_initial':len(rows)-processed,'status_counts':counts}),flush=True)
            if not stop: fill()
    print(json.dumps({'year':a.year,'processed_this_run':processed,'status_counts':counts,'stopped':stop,'out_log':str(a.out_log)},indent=2))
if __name__=='__main__': main()
