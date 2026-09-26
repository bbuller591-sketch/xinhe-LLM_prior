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
import argparse,json,re,time,urllib.request,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd

ROOT=Path(str(REPRO_ROOT))
ap=argparse.ArgumentParser()
ap.add_argument('--features',required=True)
ap.add_argument('--column',required=True)
ap.add_argument('--task',required=True)
ap.add_argument('--outdir',required=True)
ap.add_argument('--workers',type=int,default=32)
ap.add_argument('--temperature',type=float,default=1.0)
args=ap.parse_args()
OUT=Path(args.outdir);OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(args.features)
names=df[args.column].astype(str).tolist()
SECRET=ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'
env={}
for ln in SECRET.read_text().splitlines():
    if '=' in ln and not ln.lstrip().startswith('#'):
        k,v=ln.split('=',1);env[k.strip()]=v.strip()
BASE=env.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
MODEL=env.get('DEEPSEEK_MODEL','deepseek-flash')
proxy='http://127.0.0.1:7890'
OP=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
def call(p,retries=8):
    body=json.dumps({'model':MODEL,'messages':[{'role':'system','content':'Score the feature using pretrained task/domain knowledge only. Do not use tools, web access, retrieved evidence, or data.'},{'role':'user','content':p}],
                     'thinking':{'type':'disabled'},'temperature':args.temperature,'max_tokens':240,'stream':False}).encode()
    last=None
    for a in range(retries):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=body,headers={'Authorization':'Bearer '+env['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
            with OP.open(req,timeout=180) as f:x=json.loads(f.read().decode())
            return {'content':x['choices'][0]['message'].get('content',''),'model':x.get('model'),'usage':x.get('usage',{})}
        except Exception as e:
            last=repr(e);time.sleep(min(2**a,30))
    raise RuntimeError(last)
ck=OUT/'LLM_SCORE_RAW.jsonl';done={}
if ck.exists():
    for ln in ck.read_text().splitlines():
        try:
            z=json.loads(ln)
            if z.get('ok'):done[int(z['index'])]=z
        except:pass
lock=threading.Lock()
def one(i,name):
    p=f'''Task: {args.task}
Candidate feature: {name}
Give an importance score from 0 to 1 for predicting the target, where larger means more useful.
Judge only from the feature identity and pretrained domain knowledge.
Return exactly one JSON object: {{"score": <number>, "reasoning": "<one brief sentence>"}}.'''
    x=call(p);m=re.search(r'\{.*\}',x['content'],re.S)
    try:o=json.loads(m.group(0)) if m else {}
    except:o={}
    if 'score' not in o:
        mm=re.search(r'(?i)score[^0-9]*([01](?:\.\d+)?)',x['content'])
        if not mm:raise RuntimeError(f'parse {i} {name}: {x["content"]!r}')
        sc=float(mm.group(1));reason=x['content']
    else:sc=float(o['score']);reason=str(o.get('reasoning',''))
    z={'index':i,'feature':name,'score':sc,'reasoning':reason,'returned_model':x['model'],'usage':x['usage'],'ok':True}
    with lock:
        with ck.open('a') as f:f.write(json.dumps(z,ensure_ascii=False)+'\n')
    return z
todo=[(i,n) for i,n in enumerate(names) if i not in done]
print('existing',len(done),'todo',len(todo),flush=True)
errs=[]
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    fs={ex.submit(one,i,n):(i,n) for i,n in todo}
    for j,q in enumerate(as_completed(fs),1):
        i,n=fs[q]
        try:done[i]=q.result()
        except Exception as e:errs.append((i,n,e))
        if j%100==0:print(j,'/',len(todo),'errors',len(errs),flush=True)
for i,n,e in errs:
    try:done[i]=one(i,n)
    except Exception as ee:print('FINAL_FAIL',i,n,ee,flush=True)
miss=[i for i in range(len(names)) if i not in done]
if miss:raise RuntimeError(f'missing {len(miss)}')
out=pd.DataFrame([done[i] for i in range(len(names))])
out['tie_index']=out['index']
out=out.sort_values(['score','tie_index'],ascending=[False,True]).reset_index(drop=True)
out.to_csv(OUT/'LLM_SCORE.csv',index=False)
(OUT/'MANIFEST.json').write_text(json.dumps({'task':args.task,'features':args.features,'column':args.column,'n_features':len(names),'model_requested':MODEL,'temperature':args.temperature,'thinking':'disabled','data_seen_by_llm':False},indent=2)+'\n')
print(out[['index','feature','score']].head(30).to_string(index=False),flush=True)
