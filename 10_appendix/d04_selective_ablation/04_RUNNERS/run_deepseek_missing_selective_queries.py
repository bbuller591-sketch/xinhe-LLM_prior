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
import argparse, concurrent.futures, datetime as dt, hashlib, json, math, os, time, urllib.request, urllib.error
import numpy as np
import pandas as pd

ROOT=Path(str(REPRO_ROOT / '10_appendix/d04_selective_ablation'))
SECRET=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'))
QPATH=ROOT/'02_QUERY_AUDIT/MISSING_selective_ABLATION_QUERIES_PREAUTH.csv'
OUTDIR=ROOT/'03_MEASUREMENTS'
OUTDIR.mkdir(parents=True,exist_ok=True)
OUT_PARQUET=OUTDIR/'MISSING_selective_CALLS_COMPLETE.parquet'
OUT_CSV=OUTDIR/'MISSING_selective_CALLS_COMPLETE.csv'
STATUS=OUTDIR/'MISSING_selective_CALL_STATUS.json'
SYSTEM_PROMPT="Follow the user's measurement instruction exactly. Do not call tools, browse the web, or retrieve external information during this judgment. Output only the requested semantic token."
MODEL='deepseek-flash'
EXPECTED_FP='aeb56401ca74e127821c4f9126dcb669'

def load_env():
    vals={}
    for line in SECRET.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1); vals[k.strip()]=v.strip().strip('"').strip("'")
    return vals

def logsumexp(vals):
    vals=[float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals: return float('-inf')
    m=max(vals)
    return float(m+math.log(sum(math.exp(v-m) for v in vals)))

def extract_logmasses(first):
    alts=[]
    tok=str(first.get('token',''))
    lp=first.get('logprob')
    if lp is not None: alts.append({'token':tok,'logprob':float(lp)})
    for x in first.get('top_logprobs') or []:
        if x.get('logprob') is not None:
            alts.append({'token':str(x.get('token','')),'logprob':float(x.get('logprob'))})
    buckets={'A':[],'B':[],'U':[]}
    for x in alts:
        t=x['token'].strip().upper()
        if t in buckets:
            buckets[t].append(x['logprob'])
    return {k:logsumexp(v) for k,v in buckets.items()}

def call_one(row, cfg, timeout=90, max_attempts=5):
    url=cfg['base'].rstrip()+'/chat/completions'
    headers={'Authorization':'Bearer '+cfg['key'],'Content-Type':'application/json'}
    payload={'model':MODEL,
             'messages':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':row['prompt_text']}],
             'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':4,
             'logprobs':True,'top_logprobs':20,'stream':False}
    data=json.dumps(payload,ensure_ascii=False).encode('utf-8')
    call_key=hashlib.sha256((row['query_id']+'|'+row['prompt_sha256']+'|'+MODEL).encode()).hexdigest()
    last=None; t0=time.time()
    for attempt in range(1,max_attempts+1):
        try:
            req=urllib.request.Request(url,data=data,headers=headers)
            with urllib.request.urlopen(req,timeout=timeout) as resp:
                obj=json.loads(resp.read().decode('utf-8'))
            choice=obj['choices'][0]
            msg=choice.get('message') or {}
            content=str(msg.get('content','')).strip()
            lpcontent=(choice.get('logprobs') or {}).get('content') or []
            first=lpcontent[0] if lpcontent else {}
            masses=extract_logmasses(first)
            la,lb,lu=masses['A'],masses['B'],masses['U']
            denom=logsumexp([la,lb])
            pAB=float(math.exp(la-denom)) if np.isfinite(denom) else float('nan')
            H=-(pAB*math.log(pAB,2)+(1-pAB)*math.log(1-pAB,2)) if 0<pAB<1 else 0.0
            denom3=logsumexp([la,lb,lu])
            pU=float(math.exp(lu-denom3)) if np.isfinite(denom3) else float('nan')
            token=str(first.get('token') or (content[:1] if content else '')).strip().upper()
            if token not in {'A','B','U'}:
                token=content[:1].strip().upper()
            parse_status='PASS' if token in {'A','B','U'} and np.isfinite(la) and np.isfinite(lb) else 'FAIL'
            sem={'A':row['gene_A'],'B':row['gene_B'],'U':'U'}.get(token,'')
            usage=obj.get('usage') or {}
            details=usage.get('prompt_tokens_details') or {}
            return {
                'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                'query_id':row['query_id'],'unordered_pair_id':row['unordered_pair_id'],'arm':row['arm'],'order':row['order'],
                'query_role':row.get('query_role','selective_ABLATION_CANDIDATE_UNIVERSE_COMPLETION'),
                'gene_A':row['gene_A'],'gene_B':row['gene_B'],'prompt_sha256':row['prompt_sha256'],'evidence_packet_sha256':row['evidence_packet_sha256'],
                'call_key':call_key,'requested_model':MODEL,'provider_model':obj.get('model'),'system_fingerprint':obj.get('system_fingerprint'),
                'parse_status':parse_status,'response_token':token,'semantic_choice_gene':sem,
                'logp_A':la,'logp_B':lb,'logp_U':lu,'pA_vs_B':pAB,'entropy_AB_bits':float(H),'pU_threeway_topset':pU,
                'latency_ms':round((time.time()-t0)*1000,1),'attempts':attempt,'finish_reason':choice.get('finish_reason'),
                'prompt_tokens':usage.get('prompt_tokens'),'completion_tokens':usage.get('completion_tokens'),'total_tokens':usage.get('total_tokens'),
                'prompt_cache_hit_tokens':details.get('cached_tokens',0),'prompt_cache_miss_tokens':None,'cache_reused_local':False,
                'error':''
            }
        except Exception as e:
            last=repr(e)
            time.sleep(min(2**attempt,20))
    return {k:row.get(k,'') for k in ['query_id','unordered_pair_id','arm','order','query_role','gene_A','gene_B','prompt_sha256','evidence_packet_sha256']} | {
        'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'call_key':call_key,'requested_model':MODEL,'provider_model':None,'system_fingerprint':None,
        'parse_status':'ERROR','response_token':None,'semantic_choice_gene':None,
        'logp_A':np.nan,'logp_B':np.nan,'logp_U':np.nan,'pA_vs_B':np.nan,'entropy_AB_bits':np.nan,'pU_threeway_topset':np.nan,
        'latency_ms':round((time.time()-t0)*1000,1),'attempts':max_attempts,'finish_reason':None,
        'prompt_tokens':None,'completion_tokens':None,'total_tokens':None,'prompt_cache_hit_tokens':None,'prompt_cache_miss_tokens':None,'cache_reused_local':False,
        'error':last}

def save(df):
    df=df.sort_values('query_id').reset_index(drop=True)
    df.to_parquet(OUT_PARQUET,index=False,compression='zstd')
    df.to_csv(OUT_CSV,index=False)
    status={'status':'RUNNING','rows':int(len(df)),'pass_rows':int((df.parse_status=='PASS').sum()) if 'parse_status' in df else 0,
            'errors':int((df.parse_status=='ERROR').sum()) if 'parse_status' in df else 0,
            'provider_models':sorted([str(x) for x in df.provider_model.dropna().unique()]) if 'provider_model' in df else [],
            'fingerprints':sorted([str(x) for x in df.system_fingerprint.dropna().unique()]) if 'system_fingerprint' in df else [],
            'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    STATUS.write_text(json.dumps(status,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    env=load_env(); cfg={'base':env.get('DEEPSEEK_BASE_URL','https://api.deepseek.com'),'key':env['DEEPSEEK_API_KEY']}
    q=pd.read_csv(QPATH,dtype=str,keep_default_na=False)
    if args.limit: q=q.head(args.limit).copy()
    done=pd.DataFrame()
    if OUT_PARQUET.exists():
        done=pd.read_parquet(OUT_PARQUET)
    done_ids=set(done.query_id.astype(str)) if len(done) else set()
    rows=[r._asdict() for r in q.itertuples(index=False) if str(r.query_id) not in done_ids]
    all_rows=[]
    if len(done): all_rows.append(done)
    print(json.dumps({'todo':len(rows),'already_done':len(done_ids),'workers':args.workers}),flush=True)
    batch=[]; completed=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(call_one,row,cfg) for row in rows]
        for fut in concurrent.futures.as_completed(futs):
            batch.append(fut.result()); completed+=1
            if len(batch)>=50 or completed==len(rows):
                cur=pd.DataFrame(batch); all_rows.append(cur); batch=[]
                merged=pd.concat(all_rows,ignore_index=True)
                merged=merged.drop_duplicates('query_id',keep='last')
                save(merged)
                print(json.dumps({'completed_new':completed,'total_saved':len(merged),'pass':int((merged.parse_status=='PASS').sum()),'errors':int((merged.parse_status=='ERROR').sum())}),flush=True)
    final=pd.read_parquet(OUT_PARQUET)
    ok=(len(final)==len(q) and (final.parse_status=='PASS').all() and set(final.provider_model)=={MODEL} and set(final.system_fingerprint)=={EXPECTED_FP})
    status=json.loads(STATUS.read_text())
    status['status']='PASS' if ok else 'FAIL'
    status['expected_rows']=int(len(q))
    status['all_parse_pass']=bool((final.parse_status=='PASS').all())
    status['expected_fingerprint']=EXPECTED_FP
    STATUS.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)
    if not ok: raise SystemExit(2)

if __name__=='__main__': main()
