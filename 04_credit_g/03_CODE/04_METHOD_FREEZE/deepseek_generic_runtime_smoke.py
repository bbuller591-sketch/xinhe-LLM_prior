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
import json, math, urllib.request, hashlib
from pathlib import Path

ROOT=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
SECRET=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'))
vals={}
for line in SECRET.read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1); vals[k.strip()]=v.strip()

url=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')+'/chat/completions'
key=vals['DEEPSEEK_API_KEY']
model='deepseek-flash'
headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
system="Follow the user's measurement instruction exactly. Do not use tools or web access. Your entire response must be exactly one token: A, B, or U."
tests=[
 ("correctness_A","A: 2+2=4\nB: 2+2=5\nChoose the correct statement. Output exactly A, B, or U."),
 ("correctness_B","A: 3 is greater than 9\nB: 9 is greater than 3\nChoose the correct statement. Output exactly A, B, or U."),
 ("indeterminate_U","A and B are two unnamed objects with no properties or evidence. Choose which has a stronger unknown property, or U if impossible to determine. Output exactly A, B, or U."),
]
rows=[]
for name,user in tests:
    payload={
      'model':model,
      'messages':[{'role':'system','content':system},{'role':'user','content':user}],
      'thinking':{'type':'disabled'},
      'temperature':1.0,
      'max_tokens':4,
      'logprobs':True,
      'top_logprobs':20,
      'stream':False,
    }
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers)
    with urllib.request.urlopen(req,timeout=120) as resp:
        obj=json.loads(resp.read().decode())
    ch=obj['choices'][0]
    content=(ch['message'].get('content') or '').strip()
    lp=(ch.get('logprobs') or {}).get('content') or []
    first=next((x for x in lp if str(x.get('token','')).strip()),{})
    alts=first.get('top_logprobs') or []
    altmap={str(x.get('token','')).strip():float(x.get('logprob')) for x in alts}
    row={
      'test':name,
      'provider_model':obj.get('model'),
      'system_fingerprint':obj.get('system_fingerprint'),
      'content':content,
      'first_token':str(first.get('token','')).strip(),
      'has_A_top20':'A' in altmap,
      'has_B_top20':'B' in altmap,
      'has_U_top20':'U' in altmap,
      'logprob_A':altmap.get('A'),
      'logprob_B':altmap.get('B'),
      'logprob_U':altmap.get('U'),
    }
    if row['logprob_A'] is not None and row['logprob_B'] is not None:
        a,b=row['logprob_A'],row['logprob_B']; m=max(a,b)
        ea,eb=math.exp(a-m),math.exp(b-m)
        row['p_A_cond_AB']=ea/(ea+eb)
    else:
        row['p_A_cond_AB']=None
    rows.append(row)

out={
 'status':'GENERIC_RUNTIME_SMOKE_ONLY',
 'study_specific_calls':False,
 'requested_model':model,
 'tests':rows,
}
path=ROOT/'04_METHOD_FREEZE/DEEPSEEK_GENERIC_RUNTIME_SMOKE.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
