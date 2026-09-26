#!/usr/bin/env python3
import json, math
from pathlib import Path
from urllib import request
CFG=json.loads(Path('MODEL_CONFIG_DIRECT_DEEPSEEK_FLASH.json').read_text())
vals={}
for line in Path(CFG['env_file']).read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1); vals[k.strip()]=v.strip().strip('"').strip("'")
base=vals.get(CFG['base_url_env'],'https://api.deepseek.com').rstrip('/')
key=vals[CFG['api_key_env']]
url=base+'/chat/completions'
headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
system="Your entire response must be exactly one token: A, B, T, or U."
tests=[
 ('A',"Factor A statement is mathematically correct: two plus two equals four. Factor B says two plus two equals five. Choose the stronger statement."),
 ('B',"Factor A says nine is smaller than three. Factor B says nine is larger than three. Choose the stronger statement."),
 ('T',"Factor A says seven equals seven. Factor B also says seven equals seven. Use T for an exact tie."),
 ('U',"Factor A and Factor B are unnamed objects with no properties or evidence. Use U when there is insufficient evidence.")
]
def lse(xs):
    if not xs:return None
    m=max(xs); return m+math.log(sum(math.exp(v-m) for v in xs))
opener=request.build_opener(request.ProxyHandler({}))
rows=[]
for expected,user in tests:
    payload={'model':CFG['requested_model'],'messages':[{'role':'system','content':system},{'role':'user','content':user+' Return exactly A, B, T, or U.'}],
             'temperature':CFG['temperature'],'max_tokens':CFG['max_tokens'],'stream':False,
             'logprobs':True,'top_logprobs':CFG['top_logprobs'],'thinking':CFG['thinking']}
    req=request.Request(url,data=json.dumps(payload).encode(),headers=headers)
    with opener.open(req,timeout=120) as resp: obj=json.loads(resp.read().decode())
    ch=obj['choices'][0]; toks=(ch.get('logprobs') or {}).get('content') or []
    first=next((x for x in toks if str(x.get('token','')).strip()),{})
    alts=first.get('top_logprobs') or []
    masses={lab:lse([float(x['logprob']) for x in alts if str(x.get('token','')).strip()==lab]) for lab in 'ABTU'}
    rows.append({'expected':expected,'content':(ch.get('message',{}).get('content') or '').strip(),
                 'returned_model':obj.get('model'),'system_fingerprint':obj.get('system_fingerprint'),
                 'has_A':masses['A'] is not None,'has_B':masses['B'] is not None,
                 'has_T':masses['T'] is not None,'has_U':masses['U'] is not None,
                 'top_logprobs_count':len(alts)})
out={'study_specific':False,'runtime':CFG['requested_model'],'tests':rows}
Path('GENERIC_RUNTIME_SMOKE_PHASEB.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
