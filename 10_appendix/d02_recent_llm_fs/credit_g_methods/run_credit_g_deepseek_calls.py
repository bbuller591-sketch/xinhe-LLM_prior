

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
import os,json,re,urllib.request,time
from concurrent.futures import ThreadPoolExecutor,as_completed
OUT=Path(str(REPRO_ROOT / '10_appendix/d02_recent_llm_fs/credit_g_methods'))
BASE=os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
KEY=os.environ['DEEPSEEK_API_KEY']; MODEL=os.environ.get('DEEPSEEK_MODEL','deepseek-chat')
P='http://127.0.0.1:7890'
OP=urllib.request.build_opener(urllib.request.ProxyHandler({'http':P,'https':P}))
FEATURES=[
('checking_status','Status of existing checking account'),('duration','Duration, in months'),
('credit_history','Credit history (credits taken, paid back duly, delays, critical accounts)'),
('purpose','Purpose of the credit (e.g., car, television, education)'),('credit_amount','Credit amount'),
('savings_status','Status of savings accounts/bonds, in Deutsche Mark'),('employment','Number of years spent in current employment'),
('installment_commitment','Installment rate in percentage of disposable income'),('personal_status','Sex and marital status'),
('other_parties','Other debtors/guarantors (none/co-applicant/guarantor)'),('residence_since','Number of years spent in current residence'),
('property_magnitude','Property (e.g., real estate, life insurance)'),('age','Age'),
('other_payment_plans','Other installment plans (bank/stores/none)'),('housing','Housing (rent/own/for free)'),
('existing_credits','Number of existing credits at the bank'),('job','Job'),
('num_dependents','Number of people being liable to provide maintenance for'),
('own_telephone',"Telephone (none/registered under customer's name)"),('foreign_worker','Is a foreign worker (yes/no)')]
DESC=dict(FEATURES); KEYS=[k for k,_ in FEATURES]

def call(prompt,max_tokens):
    body={'model':MODEL,'messages':[{'role':'user','content':prompt}],'temperature':1.0,'max_tokens':max_tokens,'stream':False,'thinking':{'type':'disabled'}}
    raw=json.dumps(body).encode()
    err=None
    for a in range(6):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=raw,headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            with OP.open(req,timeout=180) as f: o=json.loads(f.read().decode())
            return {'ok':True,'content':o['choices'][0]['message'].get('content',''),'returned_model':o.get('model'),'usage':o.get('usage',{}),'attempt':a+1}
        except Exception as e:
            err=repr(e); time.sleep(min(16,2**a))
    return {'ok':False,'error':err}

def parse_keys(txt,cand):
    s=txt.lower(); q=[]
    for k in cand:
        ps=[x for x in [s.find('['+k.lower()+']'),s.find(k.lower())] if x>=0]
        if ps:q.append((min(ps),k))
    return [k for _,k in sorted(q)]

SCORE='''For each feature input by the user, your task is to provide a feature importance score (between 0 and 1; larger value indicates greater importance) for predicting whether an individual carries high credit risk and a reasoning behind how the importance score was assigned.

Return ONLY a JSON object with exactly two keys: "reasoning" (string) and "score" (number between 0 and 1).

Provide a score and reasoning for "{concept}" formatted according to the output schema above:'''

def one_score(x):
    k,d=x; r=call(SCORE.format(concept=d),220); sc=None
    if r.get('ok'):
        try:
            m=re.search(r'\{.*\}',r['content'],re.S); sc=float(json.loads(m.group(0) if m else r['content'])['score'])
        except Exception:
            nums=re.findall(r'(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)',r['content'])
            if nums:sc=float(nums[-1])
    return {'feature':k,'description':d,'parsed_score':sc,**r}

score=[]
with ThreadPoolExecutor(max_workers=10) as ex:
    for f in as_completed([ex.submit(one_score,x) for x in FEATURES]):score.append(f.result())
score.sort(key=lambda z:KEYS.index(z['feature']))
with open(OUT/'credit_g_llm_score_raw.jsonl','w') as f:
    for z in score:f.write(json.dumps(z,ensure_ascii=False)+'\n')
if any(z['parsed_score'] is None for z in score):raise RuntimeError('score parse failure')
score_order=[z['feature'] for z in sorted(score,key=lambda z:(-z['parsed_score'],KEYS.index(z['feature'])))]

def ltxt(ks):return '\n'.join(f'{i+1}. {DESC[k]} [{k}]' for i,k in enumerate(ks))
rank_prompt='''Given a list of features, rank them according to their importances in predicting whether an individual carries high credit risk. The ranking should be in descending order, starting with the most important feature.

Only output the ranking. Do not output dialogue or explanations for the ranking. Do not exclude any features in the ranking. Use each bracketed semantic key exactly once.

FEATURES:
'''+ltxt(KEYS)
rank=call(rank_prompt,1000); rank_order=parse_keys(rank.get('content',''),KEYS)
rank['parsed_order']=rank_order; rank['prompt']=rank_prompt
(OUT/'credit_g_llm_rank_raw.json').write_text(json.dumps(rank,indent=2,ensure_ascii=False)+'\n')
if len(rank_order)!=20:raise RuntimeError(f'rank parse {len(rank_order)}')

selected=[]; logs=[]
for step in range(10):
    cand=[k for k in KEYS if k not in selected]
    prompt='''Given a list of features already selected and a list of candidate features available, your task is to output the next feature that should be included to maximally improve the performance in predicting whether an individual carries high credit risk.

Output ONLY one bracketed semantic key from the candidate list and nothing else.

ALREADY SELECTED:
'''+(('\n'.join(f'- {DESC[k]} [{k}]' for k in selected)) if selected else '(none)')+'\n\nCANDIDATE FEATURES:\n'+ltxt(cand)
    r=call(prompt,80); parsed=parse_keys(r.get('content',''),cand)
    rec={'step':step+1,'selected_before':selected.copy(),'candidates':cand,'parsed':parsed,'prompt':prompt,**r}
    if not parsed:logs.append(rec);break
    rec['chosen']=parsed[0];selected.append(parsed[0]);logs.append(rec)
with open(OUT/'credit_g_llm_seq_raw.jsonl','w') as f:
    for z in logs:f.write(json.dumps(z,ensure_ascii=False)+'\n')
if len(selected)!=10:raise RuntimeError(f'seq parse {len(selected)}')
summary={'model_requested':MODEL,'score_order':score_order,'rank_order':rank_order,'seq_order':selected,
         'note':'Official LLM-Select CREDIT-G semantic concepts and default-style prompt wording; DeepSeek substituted for original GPT/Llama models.'}
(OUT/'CREDIT_G_LLM_ORDERS.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2))
