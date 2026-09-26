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
import json, math, urllib.request, urllib.error, time, hashlib
from pathlib import Path

ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
F=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2/FINAL'
OUT=ROOT/'formal_outputs/measurements/runtime'
OUT.mkdir(parents=True,exist_ok=True)
gate=json.loads((F/'STOP_GATE.json').read_text())
if gate.get('status')!='PASS_FOR_GENERIC_RUNTIME_SMOKE_ONLY' or not gate.get('generic_smoke_allowed'):
    raise RuntimeError('STOP_GATE does not permit generic smoke')
SECRET=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'))
vals={}
for line in SECRET.read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1);vals[k.strip()]=v.strip()
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
url=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')+'/chat/completions'
headers={'Authorization':'Bearer '+vals['DEEPSEEK_API_KEY'],'Content-Type':'application/json'}
MODEL='deepseek-flash'
SYSTEM="Follow the user's measurement instruction exactly. Do not use tools or web access. Your entire response must be exactly one token: A or B."
tests=[
 ('A','A: 2+2=4\nB: 2+2=5\nChoose the correct statement. Output exactly A or B.'),
 ('B','A: 3 is greater than 9\nB: 9 is greater than 3\nChoose the correct statement. Output exactly A or B.'),
 ('A','A: The word cat contains three letters.\nB: The word cat contains five letters.\nChoose the correct statement. Output exactly A or B.')
]
def lm(alts,label):
    xs=[float(x['logprob']) for x in alts if str(x.get('token','')).strip()==label]
    if not xs:return None,0
    m=max(xs);return m+math.log(sum(math.exp(v-m) for v in xs)),len(xs)
rows=[]
for expected,user in tests:
    payload={'model':MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user}],
             'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':4,'logprobs':True,'top_logprobs':20,'stream':False}
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers)
    obj=None;err=None
    for attempt in range(1,7):
        try:
            with urllib.request.urlopen(req,timeout=120) as resp:obj=json.loads(resp.read().decode())
            break
        except Exception as e:
            err=f'{type(e).__name__}:{e}';time.sleep(min(2**attempt,30))
    if obj is None:raise RuntimeError(err)
    ch=obj['choices'][0];content=(ch['message'].get('content') or '').strip()
    lp=(ch.get('logprobs') or {}).get('content') or []
    first=next((x for x in lp if str(x.get('token','')).strip()),{})
    tok=str(first.get('token','')).strip();alts=first.get('top_logprobs') or []
    la,na=lm(alts,'A');lb,nb=lm(alts,'B')
    if tok not in ('A','B'):raise RuntimeError(f'INVALID_FIRST_TOKEN:{tok!r}')
    if la is None or lb is None:raise RuntimeError('A_OR_B_MISSING_FROM_TOP20')
    mm=max(la,lb);ea,eb=math.exp(la-mm),math.exp(lb-mm);pA=ea/(ea+eb)
    rows.append({'expected':expected,'provider_model':obj.get('model'),'system_fingerprint':obj.get('system_fingerprint'),
                 'content':content,'first_token':tok,'A_variants':na,'B_variants':nb,
                 'logmass_A':la,'logmass_B':lb,'pA_cond_AB':pA,'usage':obj.get('usage')})
    time.sleep(1)
models=sorted(set(x['provider_model'] for x in rows));fps=sorted(set(x['system_fingerprint'] for x in rows))
status={'status':'PASS' if all(x['first_token']==x['expected'] for x in rows) and len(models)==1 and len(fps)==1 else 'FAIL',
        'study_specific_calls':False,'contains_study_genes_or_evidence':False,'requested_model':MODEL,
        'provider_models':models,'fingerprints':fps,'tests':rows}
p=OUT/'DEEPSEEK_GENERIC_SMOKE_V3_2.json';p.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
if status['status']!='PASS':raise SystemExit(2)
