from pathlib import Path
from datetime import datetime,timezone
import pandas as pd,json,hashlib,math,time,urllib.request,os
B=Path('gse272769'); cfg=json.load(open(B/'pre_llm/LLM_RUNTIME_CONFIG_FROZEN.json')); app=json.load(open(B/'pre_llm/LLM_EXECUTION_APPROVAL.json')); lock=json.load(open(B/'pre_llm/PROVIDER_FINGERPRINT_LOCK.json')); inp=B/'tmp_k40_90_llm/K40_90_ADDITIONAL_QUERIES.csv'; out=B/'tmp_k40_90_llm/measurement';cache=out/'cache';summ=out/'summaries';cache.mkdir(parents=True,exist_ok=True);summ.mkdir(parents=True,exist_ok=True)
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lse(xs):m=max(xs);return m+math.log(sum(math.exp(x-m) for x in xs))
man=json.load(open(B/'tmp_k40_90_llm/FREEZE_MANIFEST.json')); assert man['user_authorized'] and man['query_sha256']==sh(inp); assert lock['status']=='PASS_GENERIC_SMOKE_LOCKED'
v={};secret=Path('../HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env')
for line in secret.read_text().splitlines():
 if '=' in line and not line.lstrip().startswith('#'):k,x=line.split('=',1);v[k.strip()]=x.strip()
import sys
df=pd.read_csv(inp,dtype=str,keep_default_na=False); wid=int(sys.argv[1]); nw=int(sys.argv[2]); df=df.iloc[wid::nw]; rows=[]
for i,r in df.iterrows():
 key=hashlib.sha256((r.query_id+'|'+r.prompt_sha256+'|'+sh(B/'pre_llm/LLM_RUNTIME_CONFIG_FROZEN.json')+'|'+lock['system_fingerprint']).encode()).hexdigest();cp=cache/f'{key}.json'
 if cp.exists(): rows.append(json.load(open(cp))['summary']);continue
 payload={'model':cfg['requested_model'],'messages':[{'role':'system','content':'Follow the user\'s measurement instruction exactly. Do not call tools, browse the web, or retrieve external information during this judgment. Output only the requested semantic token.'},{'role':'user','content':r.prompt_text}],'thinking':cfg['thinking'],'temperature':cfg['temperature'],'max_tokens':cfg['max_tokens'],'logprobs':True,'top_logprobs':20,'stream':False}
 err=None
 for a in range(6):
  try:
   req=urllib.request.Request(v['DEEPSEEK_BASE_URL'].rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+v['DEEPSEEK_API_KEY'],'Content-Type':'application/json'});t=time.time()
   with urllib.request.urlopen(req,timeout=120) as z:resp=json.loads(z.read().decode());break
  except Exception as e:err=e;time.sleep(min(30,2**a))
 else: raise RuntimeError(err)
 if resp.get('model')!=lock['provider_model'] or resp.get('system_fingerprint')!=lock['system_fingerprint']: raise RuntimeError('PROVIDER_DRIFT')
 ch=resp['choices'][0];content=(ch['message'].get('content') or '').strip();lp=(ch.get('logprobs') or {}).get('content') or [];pos=next((x for x in lp if str(x.get('token','')).strip()),None); pools={k:[] for k in 'ABU'}
 for x in (pos.get('top_logprobs') if pos else []) or []:
  tkn=str(x.get('token','')).strip()
  if tkn in pools:pools[tkn].append(float(x['logprob']))
 if pos:
  tkn=str(pos.get('token','')).strip()
  if tkn in pools:pools[tkn].append(float(pos['logprob']))
 if content not in pools or not pools['A'] or not pools['B']: raise RuntimeError('MEASUREMENT_CONTRACT '+r.query_id)
 L={k:(lse(x) if x else None) for k,x in pools.items()};z=lse([L['A'],L['B']]);pA=math.exp(L['A']-z);H=sum(-p*math.log(p,2) for p in [pA,1-pA] if p>0);s={'query_id':r.query_id,'unordered_pair_id':r.unordered_pair_id,'arm':r.arm,'order':r.order,'gene_A':r.gene_A,'gene_B':r.gene_B,'response_token':content,'semantic_choice_gene':r.gene_A if content=='A' else r.gene_B if content=='B' else 'U','logp_A':L['A'],'logp_B':L['B'],'logp_U':L['U'],'pA_vs_B':pA,'entropy_AB_bits':H,'provider_model':resp.get('model'),'system_fingerprint':resp.get('system_fingerprint'),'prompt_tokens':resp.get('usage',{}).get('prompt_tokens'),'completion_tokens':resp.get('usage',{}).get('completion_tokens'),'latency_ms':round((time.time()-t)*1000,1)};cp.write_text(json.dumps({'summary':s,'raw_response':resp},ensure_ascii=False));rows.append(s)
 if len(rows)%100==0:pd.DataFrame(rows).to_csv(summ/f'CALLS_PARTIAL_{wid}.csv',index=False);print('done',len(rows),flush=True)
pd.DataFrame(rows).to_csv(summ/f'CALLS_COMPLETE_{wid}.csv',index=False);print('PASS',len(rows))
