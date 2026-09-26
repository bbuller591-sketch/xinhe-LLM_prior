

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
import os, json, re, urllib.request, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'EXISTING_METHOD_COMPARISON_20260925'
PKG=ROOT/'CREDIT_G_REPRO_PACKAGE_20260919'
DATA=PKG/'01_DATA_AND_SPLITS'
BASE=os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
KEY=os.environ['DEEPSEEK_API_KEY']
MODEL=os.environ.get('DEEPSEEK_MODEL','deepseek-chat')
PROXY='http://127.0.0.1:7890'
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({'http':PROXY,'https':PROXY}))

FEATURES=[
('checking_status','Status of existing checking account'),
('duration','Duration, in months'),
('credit_history','Credit history (credits taken, paid back duly, delays, critical accounts)'),
('purpose','Purpose of the credit (e.g., car, television, education)'),
('credit_amount','Credit amount'),
('savings_status','Status of savings accounts/bonds, in Deutsche Mark'),
('employment','Number of years spent in current employment'),
('installment_commitment','Installment rate in percentage of disposable income'),
('personal_status','Sex and marital status'),
('other_parties','Other debtors/guarantors (none/co-applicant/guarantor)'),
('residence_since','Number of years spent in current residence'),
('property_magnitude','Property (e.g., real estate, life insurance)'),
('age','Age'),
('other_payment_plans','Other installment plans (bank/stores/none)'),
('housing','Housing (rent/own/for free)'),
('existing_credits','Number of existing credits at the bank'),
('job','Job'),
('num_dependents','Number of people being liable to provide maintenance for'),
('own_telephone',"Telephone (none/registered under customer's name)"),
('foreign_worker','Is a foreign worker (yes/no)'),
]
DESC=dict(FEATURES)
KEYS=[k for k,_ in FEATURES]

def call_llm(prompt,max_tokens=500,retries=6):
    body={'model':MODEL,'messages':[{'role':'user','content':prompt}],
          'temperature':1.0,'max_tokens':max_tokens,'stream':False,
          'thinking':{'type':'disabled'}}
    data=json.dumps(body).encode()
    err=None
    for a in range(retries):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=data,
                headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            with OPENER.open(req,timeout=180) as f:
                obj=json.loads(f.read().decode())
            return {'ok':True,'content':obj['choices'][0]['message'].get('content',''),
                    'returned_model':obj.get('model'),'attempt':a+1,
                    'usage':obj.get('usage',{})}
        except Exception as e:
            err=repr(e); time.sleep(min(16,2**a))
    return {'ok':False,'error':err}

SCORE_TEMPLATE='''For each feature input by the user, your task is to provide a feature importance score (between 0 and 1; larger value indicates greater importance) for predicting whether an individual carries high credit risk and a reasoning behind how the importance score was assigned.

Return ONLY a JSON object with exactly two keys: "reasoning" (string) and "score" (number between 0 and 1).

Provide a score and reasoning for "{concept}" formatted according to the output schema above:'''

def score_one(item):
    key,desc=item
    r=call_llm(SCORE_TEMPLATE.format(concept=desc),220)
    rec={'feature':key,'description':desc,**r}
    sc=None
    if r.get('ok'):
        txt=r['content']
        try:
            m=re.search(r'\{.*\}',txt,re.S); obj=json.loads(m.group(0) if m else txt)
            sc=float(obj['score'])
        except Exception:
            nums=re.findall(r'(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)',txt)
            if nums: sc=float(nums[-1])
    rec['parsed_score']=sc
    return rec

def run_score():
    recs=[]
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs=[ex.submit(score_one,x) for x in FEATURES]
        for f in as_completed(fs): recs.append(f.result())
    recs=sorted(recs,key=lambda z:KEYS.index(z['feature']))
    with open(OUT/'credit_g_llm_score_raw.jsonl','w') as f:
        for z in recs:f.write(json.dumps(z,ensure_ascii=False)+'\n')
    if any(z['parsed_score'] is None for z in recs):
        raise RuntimeError('LLM-Score parse incomplete')
    order=[z['feature'] for z in sorted(recs,key=lambda z:(-z['parsed_score'],KEYS.index(z['feature'])))]
    return order,recs

def list_text(keys):
    return '\n'.join(f'{i+1}. {DESC[k]} [{k}]' for i,k in enumerate(keys))

def parse_keys(txt,candidates):
    low=txt.lower(); found=[]
    for k in candidates:
        positions=[p for p in (low.find('['+k.lower()+']'),low.find(k.lower())) if p>=0]
        if positions: found.append((min(positions),k))
    out=[]
    for _,k in sorted(found):
        if k not in out:out.append(k)
    return out

def run_rank():
    prompt='''Given a list of features, rank them according to their importances in predicting whether an individual carries high credit risk. The ranking should be in descending order, starting with the most important feature.

Only output the ranking. Do not output dialogue or explanations for the ranking. Do not exclude any features in the ranking. Use each bracketed semantic key exactly once.

FEATURES:
'''+list_text(KEYS)
    r=call_llm(prompt,1000)
    order=parse_keys(r.get('content',''),KEYS)
    rec={'prompt':prompt,**r,'parsed_order':order}
    (OUT/'credit_g_llm_rank_raw.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False)+'\n')
    if len(order)!=20: raise RuntimeError(f'LLM-Rank parse got {len(order)}')
    return order,rec

def run_seq():
    selected=[]; logs=[]
    for step in range(10):
        cand=[k for k in KEYS if k not in selected]
        p='''Given a list of features already selected and a list of candidate features available, your task is to output the next feature that should be included to maximally improve the performance in predicting whether an individual carries high credit risk.

Output ONLY one bracketed semantic key from the candidate list and nothing else.

ALREADY SELECTED:
'''+(('\n'.join(f'- {DESC[k]} [{k}]' for k in selected)) if selected else '(none)')+'\n\nCANDIDATE FEATURES:\n'+list_text(cand)
        r=call_llm(p,80)
        parsed=parse_keys(r.get('content',''),cand)
        rec={'step':step+1,'selected_before':selected.copy(),'candidates':cand,'prompt':p,**r,'parsed':parsed}
        if not parsed:
            logs.append(rec)
            break
        chosen=parsed[0]; rec['chosen']=chosen; selected.append(chosen); logs.append(rec)
    with open(OUT/'credit_g_llm_seq_raw.jsonl','w') as f:
        for z in logs:f.write(json.dumps(z,ensure_ascii=False)+'\n')
    if len(selected)!=10: raise RuntimeError(f'LLM-Seq stopped at {len(selected)}')
    return selected,logs

CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[k for k in KEYS if k not in CAT]
def prep(cols):
    num=[c for c in NUM if c in cols]; cat=[c for c in CAT if c in cols]; tr=[]
    if num:tr.append(('num',StandardScaler(),num))
    if cat:tr.append(('cat',Pipeline([('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),('scale',StandardScaler())]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)

def evaluate(sel):
    X=pd.read_csv(DATA/'X.csv'); y=pd.read_csv(DATA/'y.csv')['label'].to_numpy()
    dev=np.load(DATA/'modern_dev_indices_seed20260918.npy'); hold=np.load(DATA/'modern_holdout_indices_seed20260918.npy')
    Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]
    Xh=X.iloc[hold].reset_index(drop=True); yh=y[hold]
    pipe=Pipeline([('prep',prep(sel)),('clf',LogisticRegression(penalty='l2',solver='lbfgs',C=0.01,fit_intercept=True,class_weight=None,max_iter=5000,tol=1e-5))])
    pipe.fit(Xd[sel],yd); ph=pipe.predict_proba(Xh[sel])[:,1]
    return {'holdout_auroc':float(roc_auc_score(yh,ph)),
            'holdout_average_precision':float(average_precision_score(yh,ph)),
            'holdout_log_loss':float(log_loss(yh,ph,labels=[0,1]))}

def main():
    score_order,_=run_score(); rank_order,_=run_rank(); seq_order,_=run_seq()
    rows=[]
    for m,o in [('Direct LLM-Score (LLM-Select style)',score_order),('Direct LLM-Rank (LLM-Select style)',rank_order),('LLM-Seq (LLM-Select style)',seq_order)]:
        sel=o[:10]; rows.append({'method':m,'k':10,'selected_features':'|'.join(sel),**evaluate(sel)})
    rows.extend([
      {'method':'Data-only Reference (GBM-permutation)','k':10,'selected_features':'frozen','holdout_auroc':0.780119,'holdout_average_precision':0.580819,'holdout_log_loss':0.512819},
      {'method':'Selective Correction (GBM-permutation, DeepSeek)','k':10,'selected_features':'frozen','holdout_auroc':0.793333,'holdout_average_precision':0.582880,'holdout_log_loss':0.508812},
      {'method':'All 20 semantic features','k':20,'selected_features':'|'.join(KEYS),'holdout_auroc':0.792024,'holdout_average_precision':0.622130,'holdout_log_loss':0.499551},
    ])
    df=pd.DataFrame(rows); df.to_csv(OUT/'CREDIT_G_LLMSELECT_COMPARISON.csv',index=False)
    meta={'status':'COMPLETE','model_requested':MODEL,'returned_models':sorted(set(z for z in [
        json.loads((OUT/'credit_g_llm_rank_raw.json').read_text()).get('returned_model')
    ] if z)),'score_order':score_order,'rank_order':rank_order,'seq_order':seq_order,
    'comparability':'LLM-Select prompt/concept style instantiated with DeepSeek on our frozen 800/200 split, k=10, and common L2 evaluator. Published LLM-Select path-area values are not directly compared numerically.'}
    (OUT/'CREDIT_G_LLMSELECT_COMPARISON_META.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    print(df[['method','k','holdout_auroc','holdout_average_precision','holdout_log_loss']].to_string(index=False))
    print('\nTOP10')
    for m,o in [('score',score_order),('rank',rank_order),('seq',seq_order)]:print(m,o[:10])
if __name__=='__main__':main()
