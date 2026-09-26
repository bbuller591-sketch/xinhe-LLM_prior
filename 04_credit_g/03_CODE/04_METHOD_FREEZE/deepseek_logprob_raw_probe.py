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
import json, urllib.request
from pathlib import Path
SECRET=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'))
vals={}
for line in SECRET.read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1); vals[k.strip()]=v.strip()
url=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')+'/chat/completions'
headers={'Authorization':'Bearer '+vals['DEEPSEEK_API_KEY'],'Content-Type':'application/json'}
payload={
 'model':'deepseek-flash',
 'messages':[
   {'role':'system','content':"Follow the user's instruction. Output exactly one token: A, B, or U."},
   {'role':'user','content':'A: 2+2=4\nB: 2+2=5\nChoose the correct statement. Output exactly A, B, or U.'}
 ],
 'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':4,'logprobs':True,'top_logprobs':20,'stream':False
}
req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers)
with urllib.request.urlopen(req,timeout=120) as resp:
    obj=json.loads(resp.read().decode())
ch=obj['choices'][0]
safe={
 'model':obj.get('model'),'system_fingerprint':obj.get('system_fingerprint'),
 'content':ch['message'].get('content'),
 'logprobs_content':(ch.get('logprobs') or {}).get('content')
}
print(json.dumps(safe,ensure_ascii=False,indent=2))
