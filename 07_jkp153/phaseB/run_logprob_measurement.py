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
import argparse, hashlib, json, math, os, random, socket, time
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from pathlib import Path
from urllib import error, request
import pandas as pd

ALLOWED={'A','B','T','U'}
RETRY_HTTP={429,500,502,503,504}
TRANSPORT=(ConnectionResetError,TimeoutError,socket.timeout,error.URLError,RemoteDisconnected)

def now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def append(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
def load_json(path): return json.loads(Path(path).read_text())
def load_env(cfg):
    vals={}
    p=Path(cfg['env_file'])
    if p.exists():
        for line in p.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k,v=line.split('=',1); vals[k.strip()]=v.strip().strip('"').strip("'")
    base=os.environ.get(cfg['base_url_env']) or vals.get(cfg['base_url_env'],'')
    key=os.environ.get(cfg['api_key_env']) or vals.get(cfg['api_key_env'],'')
    if not base or not key: raise RuntimeError('configured API base/key missing')
    return base.rstrip('/'),key
def semantic_logmass(alts,label):
    xs=[float(x['logprob']) for x in alts if str(x.get('token','')).strip()==label]
    if not xs:return None,0
    m=max(xs); return m+math.log(sum(math.exp(v-m) for v in xs)),len(xs)
def extract(resp):
    ch=resp['choices'][0]
    content=(ch.get('message',{}).get('content') or '').strip()
    hard=content if content in ALLOWED else None
    toks=(ch.get('logprobs') or {}).get('content') or []
    first=next((x for x in toks if str(x.get('token','')).strip()),{})
    alts=first.get('top_logprobs') or []
    masses={}; counts={}
    for lab in 'ABTU':
        masses[lab],counts[lab]=semantic_logmass(alts,lab)
    pA=None
    if masses['A'] is not None and masses['B'] is not None:
        m=max(masses['A'],masses['B']); ea=math.exp(masses['A']-m); eb=math.exp(masses['B']-m); pA=ea/(ea+eb)
    return {
      'hard_token':hard,'raw_content':content,'first_token':str(first.get('token','')).strip(),
      'logmass_A':masses['A'],'logmass_B':masses['B'],'logmass_T':masses['T'],'logmass_U':masses['U'],
      'variants_A':counts['A'],'variants_B':counts['B'],'variants_T':counts['T'],'variants_U':counts['U'],
      'pA_cond_AB':pA,'ab_prob_usable':pA is not None,
    }
def payload(prompt,cfg):
    system,user=prompt.rstrip('\n').split('\n\n',1)
    out={'model':cfg['requested_model'],'messages':[{'role':'system','content':system},{'role':'user','content':user}],
         'temperature':cfg['temperature'],'max_tokens':cfg['max_tokens'],'stream':False,
         'logprobs':True,'top_logprobs':cfg['top_logprobs']}
    if cfg.get('top_p') is not None:
        out['top_p']=cfg['top_p']
    if cfg.get('thinking') is not None:
        out['thinking']=cfg['thinking']
    return out
def existing(path):
    if not path.exists():return set()
    out=set()
    for line in path.read_text().splitlines():
        if line.strip():
            o=json.loads(line)
            if o.get('final_status') in {'SUCCESS','FINAL_TRANSPORT_FAIL','FINAL_HTTP_FAIL','NONRETRYABLE_HTTP','BAD_RESPONSE','RUNTIME_MISMATCH'}:
                out.add(o['measurement_call_id'])
    return out
def mock_obj(token):
    # synthetic top-logprobs only for runner/parser unit testing
    probs={'A':(-0.2,-1.8),'B':(-1.7,-0.25),'T':(-1.0,-1.1),'U':(-1.2,-1.3)}
    la,lb=probs[token]
    return {'model':'mock','system_fingerprint':'mockfp','choices':[{'message':{'content':token},
      'logprobs':{'content':[{'token':token,'logprob':-0.1,'top_logprobs':[
        {'token':'A','logprob':la},{'token':' B','logprob':lb},{'token':'T','logprob':-3.0},{'token':' U','logprob':-4.0}]}]}}]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--year',type=int,choices=[2024,2025],required=True)
    ap.add_argument('--schedule',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=Path('MODEL_CONFIG_APIYI_DEEPSEEK_V3.json'))
    ap.add_argument('--out-log',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--mock-token',choices=['A','B','T','U'])
    ap.add_argument('--allow-2025',action='store_true')
    a=ap.parse_args()
    if a.year==2025 and not a.allow_2025:
        raise SystemExit('2025 Phase B calls locked; require --allow-2025 after 2024 freeze.')
    cfg=load_json(a.config)
    rows=pd.read_csv(a.schedule)
    if set(rows.origin_year.astype(int))!={a.year}: raise RuntimeError('schedule year mismatch')
    done=existing(a.out_log); base=key=None
    if not a.mock_token: base,key=load_env(cfg)
    n=0
    for r in rows.to_dict('records'):
        cid=r['measurement_call_id']
        if cid in done: continue
        prompt=Path(str(REPRO_ROOT / '07_jkp153/phaseB'))/r['prompt_file']
        txt=prompt.read_text(encoding='utf-8')
        if sha(txt.rstrip('\n'))!=r['prompt_sha256']: raise RuntimeError(f'prompt hash mismatch {cid}')
        pl=payload(txt,cfg)
        final=None
        for attempt in range(1,5):
            start=now()
            try:
                if a.mock_token:
                    obj=mock_obj(a.mock_token); status=200
                else:
                    req=request.Request(base+'/chat/completions',data=json.dumps(pl).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                    if cfg.get('disable_env_proxy'):
                        opener=request.build_opener(request.ProxyHandler({}))
                        resp_ctx=opener.open(req,timeout=120)
                    else:
                        resp_ctx=request.urlopen(req,timeout=120)
                    with resp_ctx as resp:
                        status=resp.status; obj=json.loads(resp.read().decode())
                    exp_model=cfg.get('expected_returned_model')
                    exp_fp=cfg.get('expected_system_fingerprint')
                    if exp_model and obj.get('model')!=exp_model:
                        final={**{k:r[k] for k in r},'attempt':attempt,'timestamp_start':start,'timestamp_end':now(),
                               'requested_model':cfg['requested_model'],'returned_model':obj.get('model'),
                               'system_fingerprint':obj.get('system_fingerprint'),'http_status':status,
                               'error':'returned model mismatch','final_status':'RUNTIME_MISMATCH'}
                        break
                    if exp_fp and obj.get('system_fingerprint')!=exp_fp:
                        final={**{k:r[k] for k in r},'attempt':attempt,'timestamp_start':start,'timestamp_end':now(),
                               'requested_model':cfg['requested_model'],'returned_model':obj.get('model'),
                               'system_fingerprint':obj.get('system_fingerprint'),'http_status':status,
                               'error':'system fingerprint mismatch','final_status':'RUNTIME_MISMATCH'}
                        break
                ex=extract(obj)
                final={**{k:r[k] for k in r},'attempt':attempt,'timestamp_start':start,'timestamp_end':now(),
                       'requested_model':cfg['requested_model'],'returned_model':obj.get('model'),
                       'system_fingerprint':obj.get('system_fingerprint'),'http_status':status,
                       **ex,'final_status':'SUCCESS' if ex['hard_token'] else 'BAD_RESPONSE'}
                break
            except error.HTTPError as e:
                final={**{k:r[k] for k in r},'attempt':attempt,'timestamp_start':start,'timestamp_end':now(),
                       'requested_model':cfg['requested_model'],'http_status':e.code,'error':str(e),
                       'final_status':'FINAL_HTTP_FAIL' if (e.code in RETRY_HTTP and attempt==4) else ('RETRY' if e.code in RETRY_HTTP else 'NONRETRYABLE_HTTP')}
                if e.code not in RETRY_HTTP or attempt==4: break
            except TRANSPORT as e:
                final={**{k:r[k] for k in r},'attempt':attempt,'timestamp_start':start,'timestamp_end':now(),
                       'requested_model':cfg['requested_model'],'error':type(e).__name__+': '+str(e),
                       'final_status':'FINAL_TRANSPORT_FAIL' if attempt==4 else 'RETRY'}
                if attempt==4: break
            if not a.mock_token: time.sleep(min(30,2**attempt)*(0.5+random.random()))
        append(a.out_log,final); n+=1
        if a.limit and n>=a.limit: break
    print(json.dumps({'year':a.year,'processed':n,'out_log':str(a.out_log),'mock':bool(a.mock_token)},indent=2))
if __name__=='__main__': main()
