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
import json,math,urllib.request
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
ENV=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'))
OUT=ROOT/'10_LLM_MEASUREMENT/00_GENERIC_SMOKE'
OUT.mkdir(parents=True,exist_ok=True)
vals={}
for line in ENV.read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1); vals[k.strip()]=v.strip()
key=vals.get('DEEPSEEK_API_KEY','')
if not key: raise RuntimeError('API_KEY_MISSING')
base=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
model='deepseek-flash'
payload={
 'model':model,
 'messages':[
  {'role':'system','content':'This is a generic API capability smoke test. Do not use tools or web. Follow the output instruction exactly.'},
  {'role':'user','content':'Valid response tokens are A, B, or U. Output exactly U as the first and only content token.'}],
 'thinking':{'type':'disabled'},
 'temperature':1.0,'max_tokens':4,'logprobs':True,'top_logprobs':20,'stream':False}
req=urllib.request.Request(base+'/chat/completions',data=json.dumps(payload).encode(),
 headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
# This host exports a local HTTP(S) proxy that may be unavailable. Bypass it
# explicitly; this changes transport only, not the frozen API payload.
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(req,timeout=90) as r:
    resp=json.loads(r.read().decode())
(OUT/'GENERIC_ABU_SMOKE_RESPONSE.json').write_text(json.dumps(resp,ensure_ascii=False,indent=2),encoding='utf-8')
ch=resp['choices'][0]; content=(ch['message'].get('content') or '').strip()
lp=(ch.get('logprobs') or {}).get('content') or []
first=next((x for x in lp if str(x.get('token','')).strip()),{})
alts=first.get('top_logprobs') or []
norm={}
for x in alts:
    t=str(x.get('token','')).strip()
    if t in {'A','B','U'}: norm.setdefault(t,[]).append(float(x['logprob']))
def lse(xs):
    if not xs:return None
    m=max(xs); return m+math.log(sum(math.exp(x-m) for x in xs))
LA,LB,LU=[lse(norm.get(k,[])) for k in ['A','B','U']]
pab=None
if LA is not None and LB is not None:
    m=max(LA,LB); pab=math.exp(LA-m)/(math.exp(LA-m)+math.exp(LB-m))
status={
 'status':'PASS' if content=='U' and LA is not None and LB is not None and LU is not None and resp.get('model')=='deepseek-flash' and resp.get('system_fingerprint') else 'FAIL',
 'timestamp_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
 'requested_model':model,'provider_model':resp.get('model'),'system_fingerprint':resp.get('system_fingerprint'),
 'content':content,'first_token':str(first.get('token','')).strip(),
 'A_in_top20':LA is not None,'B_in_top20':LB is not None,'U_in_top20':LU is not None,
 'logp_A':LA,'logp_B':LB,'logp_U':LU,'pA_vs_B':pab,
 'n_top_alternatives':len(alts),'usage':resp.get('usage',{}),
 'experiment_feature_data_sent':False}
(OUT/'GENERIC_ABU_SMOKE_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(status,ensure_ascii=False,indent=2))
if status['status']!='PASS': raise SystemExit(2)
