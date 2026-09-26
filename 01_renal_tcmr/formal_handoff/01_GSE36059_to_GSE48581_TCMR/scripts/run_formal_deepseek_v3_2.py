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
import argparse,csv,json,math,time,urllib.request,urllib.error,hashlib,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd

ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
FREEZE=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2/FINAL'
MEAS=ROOT/'formal_outputs/measurements/v3_2'
CACHE=MEAS/'cache';CACHE.mkdir(parents=True,exist_ok=True)
MEAS.mkdir(parents=True,exist_ok=True)
SECRET=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'))
vals={}
for line in SECRET.read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1);vals[k.strip()]=v.strip()
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
BASE=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
API_KEY=vals['DEEPSEEK_API_KEY']
MODEL='deepseek-flash'
SYSTEM="Follow the user measurement instruction exactly. Do not call tools, browse the web, retrieve external information, or add explanation. Your entire response must be exactly one token: A or B."
lock=threading.Lock()

def semantic_logmass(alts,label):
    xs=[float(x['logprob']) for x in alts if str(x.get('token','')).strip()==label]
    if not xs:return None,0
    m=max(xs);return m+math.log(sum(math.exp(v-m) for v in xs)),len(xs)

def cache_key(row,fp):
    x={'query_id':row['query_id'],'prompt_sha256':row['prompt_sha256'],'model':MODEL,'fingerprint':fp,'system':SYSTEM,
       'temperature':1.0,'thinking':'disabled','top_logprobs':20}
    return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def call_one(row,expected_fp):
    key=cache_key(row,expected_fp);cp=CACHE/(key+'.json')
    if cp.exists():
        x=json.loads(cp.read_text())
        if not x.get('error'):return x
    payload={'model':MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':row['prompt_text']}],
             'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':4,'logprobs':True,'top_logprobs':20,'stream':False}
    headers={'Authorization':'Bearer '+API_KEY,'Content-Type':'application/json'}
    last=None
    for attempt in range(1,7):
        t0=time.time()
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=json.dumps(payload).encode(),headers=headers)
            with urllib.request.urlopen(req,timeout=120) as resp:obj=json.loads(resp.read().decode())
            ch=obj['choices'][0];content=(ch['message'].get('content') or '').strip()
            lp=(ch.get('logprobs') or {}).get('content') or []
            first=next((z for z in lp if str(z.get('token','')).strip()),{})
            tok=str(first.get('token','')).strip();alts=first.get('top_logprobs') or []
            la,na=semantic_logmass(alts,'A');lb,nb=semantic_logmass(alts,'B')
            provider=obj.get('model');fp=obj.get('system_fingerprint')
            if provider!=MODEL:raise RuntimeError(f'MODEL_MISMATCH expected={MODEL} got={provider}')
            if fp!=expected_fp:raise RuntimeError(f'FINGERPRINT_MISMATCH expected={expected_fp} got={fp}')
            if tok not in ('A','B'):raise RuntimeError(f'INVALID_FIRST_TOKEN:{tok!r}')
            if la is None or lb is None:raise RuntimeError('A_OR_B_MISSING_FROM_TOP20')
            mm=max(la,lb);ea,eb=math.exp(la-mm),math.exp(lb-mm);pA=ea/(ea+eb)
            x={**{k:row[k] for k in row if k!='prompt_text'},
               'provider_model':provider,'system_fingerprint':fp,'content':content,'first_token':tok,
               'semantic_A_variants_top20':na,'semantic_B_variants_top20':nb,'semantic_logmass_A':la,'semantic_logmass_B':lb,
               'p_presented_A_cond_AB':pA,'latency_sec':time.time()-t0,'attempt':attempt,'usage':obj.get('usage'),
               'raw_response':obj,'cache_key':key,'error':None}
            cp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
            return x
        except urllib.error.HTTPError as e:
            body=''
            try:body=e.read().decode(errors='replace')[:2000]
            except:pass
            last=f'HTTP {e.code}:{body}'
            if e.code not in (408,409,429,500,502,503,504):break
            time.sleep(min(2**attempt,30))
        except Exception as e:
            last=f'{type(e).__name__}:{e}'
            if any(x in last for x in ['MISMATCH','INVALID_FIRST_TOKEN','A_OR_B_MISSING']):break
            time.sleep(min(2**attempt,30))
    x={**{k:row[k] for k in row if k!='prompt_text'},'provider_model':None,'system_fingerprint':None,
       'content':None,'first_token':None,'p_presented_A_cond_AB':None,'attempt':6,'cache_key':key,'error':last,'raw_response':None}
    cp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    gate=json.loads((FREEZE/'STOP_GATE.json').read_text())
    if gate.get('status')!='PASS':raise RuntimeError('STOP_GATE is not PASS')
    smoke=json.loads((ROOT/'formal_outputs/measurements/runtime/DEEPSEEK_GENERIC_SMOKE_V3_2.json').read_text())
    if smoke.get('status')!='PASS' or len(smoke.get('fingerprints',[]))!=1:raise RuntimeError('generic smoke not frozen/pass')
    fp=smoke['fingerprints'][0]
    manifest=FREEZE/'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv'
    expected='4e41bea69f8015e530c9f3efed219182f79535999a12ba99310014bbc5c6c1a7'
    got=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if got!=expected:raise RuntimeError(f'MANIFEST_HASH_MISMATCH {got}')
    rows=list(csv.DictReader(manifest.open()))
    assert len(rows)==400
    outjson=MEAS/'FORMAL_DEEPSEEK_V3_2_RAW.jsonl';results={}
    if outjson.exists():
        for line in outjson.read_text().splitlines():
            if line.strip():
                x=json.loads(line);results[x['query_id']]=x
    todo=[r for r in rows if r['query_id'] not in results or results[r['query_id']].get('error')]
    print(f'total=400 cached/existing_ok={400-len(todo)} todo={len(todo)} workers={args.workers} fp={fp}',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut={ex.submit(call_one,r,fp):r['query_id'] for r in todo}
        n=0
        for f in as_completed(fut):
            x=f.result();results[x['query_id']]=x;n+=1
            with lock:
                with outjson.open('a') as h:h.write(json.dumps(x,ensure_ascii=False)+'\n')
            if x.get('error'):print('ERROR',x['query_id'],x['error'],flush=True)
            if n==1 or n%20==0 or n==len(todo):print(f'done {n}/{len(todo)}',flush=True)
    ordered=[results[r['query_id']] for r in rows]
    err=[x for x in ordered if x.get('error')]
    if err:
        (MEAS/'FORMAL_DEEPSEEK_V3_2_FAILURES.json').write_text(json.dumps(err[:30],indent=2)+'\n')
        raise SystemExit(2)
    slim=[{k:v for k,v in x.items() if k not in ('raw_response','usage')} for x in ordered]
    pd.DataFrame(slim).to_csv(MEAS/'FORMAL_DEEPSEEK_V3_2.csv',index=False)
    summary={'status':'COMPLETE','n_calls':400,'n_pairs':200,'provider_model':MODEL,'system_fingerprint':fp,
             'first_token_counts':pd.Series([x['first_token'] for x in ordered]).value_counts().to_dict(),
             'query_manifest_sha256':got,'raw_jsonl_sha256':hashlib.sha256(outjson.read_bytes()).hexdigest(),
             'study_specific_llm_calls':400}
    (MEAS/'FORMAL_DEEPSEEK_V3_2_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
