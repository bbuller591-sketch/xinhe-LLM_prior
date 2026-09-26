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
import argparse, csv, json, math, time, urllib.request, urllib.error, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
M=ROOT/'04_METHOD_FREEZE'
R=ROOT/'06_RESULTS'
R.mkdir(parents=True,exist_ok=True)
SECRET=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'))
vals={}
for line in SECRET.read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1); vals[k.strip()]=v.strip()
BASE=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
API_KEY=vals['DEEPSEEK_API_KEY']
MODEL='deepseek-flash'
EXPECTED_FP='aeb56401ca74e127821c4f9126dcb669'
SYSTEM="You are a measurement instrument for pairwise variable relevance in consumer-credit repayment risk. Follow the user instruction exactly. Do not call tools, browse the web, or retrieve external information. Do not try to identify or name a benchmark dataset. Your entire response must be exactly one token: A, B, or U."

lock=threading.Lock()

def semantic_logmass(alts,label):
    xs=[float(x['logprob']) for x in alts if str(x.get('token','')).strip()==label]
    if not xs: return None,0
    m=max(xs)
    return m+math.log(sum(math.exp(v-m) for v in xs)),len(xs)

def do_call(row):
    payload={
      'model':MODEL,
      'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':row['user_prompt']}],
      'thinking':{'type':'disabled'},
      'temperature':1.0,
      'max_tokens':4,
      'logprobs':True,
      'top_logprobs':20,
      'stream':False,
    }
    data=json.dumps(payload).encode()
    headers={'Authorization':'Bearer '+API_KEY,'Content-Type':'application/json'}
    last=None
    for attempt in range(1,7):
        t0=time.time()
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=data,headers=headers)
            with urllib.request.urlopen(req,timeout=120) as resp:
                obj=json.loads(resp.read().decode())
            ch=obj['choices'][0]
            content=(ch['message'].get('content') or '').strip()
            lp=(ch.get('logprobs') or {}).get('content') or []
            first=next((x for x in lp if str(x.get('token','')).strip()),{})
            first_tok=str(first.get('token','')).strip()
            alts=first.get('top_logprobs') or []
            la,na=semantic_logmass(alts,'A')
            lb,nb=semantic_logmass(alts,'B')
            lu,nu=semantic_logmass(alts,'U')
            provider_model=obj.get('model')
            fp=obj.get('system_fingerprint')
            if provider_model != MODEL:
                raise RuntimeError(f"MODEL_MISMATCH expected={MODEL} got={provider_model}")
            if fp != EXPECTED_FP:
                raise RuntimeError(f"FINGERPRINT_MISMATCH expected={EXPECTED_FP} got={fp}")
            if first_tok not in ('A','B','U'):
                raise RuntimeError(f"INVALID_FIRST_TOKEN {first_tok!r}")
            if first_tok != 'U' and (la is None or lb is None):
                raise RuntimeError("A_OR_B_MISSING_FROM_TOP20")
            pA=None
            if la is not None and lb is not None:
                mm=max(la,lb); ea,eb=math.exp(la-mm),math.exp(lb-mm)
                pA=ea/(ea+eb)
            return {
              **{k:row[k] for k in row if k!='user_prompt'},
              'provider_model':provider_model,'system_fingerprint':fp,
              'content':content,'first_token':first_tok,
              'semantic_A_variants_top20':na,'semantic_B_variants_top20':nb,'semantic_U_variants_top20':nu,
              'semantic_logmass_A':la,'semantic_logmass_B':lb,'semantic_logmass_U':lu,
              'p_presented_A_cond_AB':pA,
              'latency_sec':time.time()-t0,
              'attempt':attempt,
              'usage':obj.get('usage'),
              'raw_response':obj,
              'error':None,
            }
        except urllib.error.HTTPError as e:
            body=''
            try: body=e.read().decode(errors='replace')[:2000]
            except Exception: pass
            last=f"HTTP {e.code}: {body}"
            if e.code not in (408,409,429,500,502,503,504):
                break
            time.sleep(min(2**attempt,30))
        except Exception as e:
            last=f"{type(e).__name__}: {e}"
            if 'MISMATCH' in last or 'INVALID_FIRST_TOKEN' in last or 'A_OR_B_MISSING' in last:
                break
            time.sleep(min(2**attempt,30))
    return {
      **{k:row[k] for k in row if k!='user_prompt'},
      'error':last,'provider_model':None,'system_fingerprint':None,'content':None,'first_token':None,
      'p_presented_A_cond_AB':None,'attempt':6,'raw_response':None,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--out-stem',required=True)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    manifest=M/args.manifest
    rows=list(csv.DictReader(manifest.open()))
    for row in rows:
        for k in ['repeat_index']:
            row[k]=int(row[k])
        for k in ['primary_measurement','repeat_instability_sentinel']:
            row[k]=str(row[k]).lower()=='true'

    out_jsonl=R/(args.out_stem+'.jsonl')
    out_csv=R/(args.out_stem+'.csv')
    existing={}
    if out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            if line.strip():
                x=json.loads(line); existing[x['query_id']]=x
    todo=[r for r in rows if r['query_id'] not in existing or existing[r['query_id']].get('error')]
    print(f"manifest={len(rows)} existing_ok={len(rows)-len(todo)} todo={len(todo)} workers={args.workers}",flush=True)

    results=dict(existing)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(do_call,r):r['query_id'] for r in todo}
        done=0
        for fut in as_completed(futs):
            x=fut.result(); results[x['query_id']]=x; done+=1
            with lock:
                with out_jsonl.open('a') as f:
                    f.write(json.dumps(x,ensure_ascii=False)+'\n')
            if x.get('error'):
                print(f"ERROR {x['query_id']} {x['error']}",flush=True)
            if done==1 or done%20==0 or done==len(todo):
                print(f"done {done}/{len(todo)}",flush=True)

    ordered=[results[r['query_id']] for r in rows]
    errors=[x for x in ordered if x.get('error')]
    if errors:
        print(json.dumps({'status':'FAILED','n_errors':len(errors),'errors':[{'query_id':x['query_id'],'error':x['error']} for x in errors[:20]]},indent=2))
        raise SystemExit(2)

    slim=[]
    for x in ordered:
        slim.append({k:v for k,v in x.items() if k not in ('raw_response','usage')})
    import pandas as pd
    pd.DataFrame(slim).to_csv(out_csv,index=False)
    fps=sorted(set(x['system_fingerprint'] for x in ordered))
    summary={
      'status':'COMPLETE','manifest':args.manifest,'n_calls':len(ordered),
      'provider_models':sorted(set(x['provider_model'] for x in ordered)),
      'fingerprints':fps,
      'first_token_counts':pd.Series([x['first_token'] for x in ordered]).value_counts().to_dict(),
      'mean_latency_sec':sum(x['latency_sec'] for x in ordered)/len(ordered),
    }
    (R/(args.out_stem+'_SUMMARY.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
