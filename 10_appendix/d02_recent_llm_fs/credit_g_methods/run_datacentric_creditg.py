

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
import os,json,re,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss

ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'EXISTING_METHOD_COMPARISON_20260925'/'RECENT_METHODS'
OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'CREDIT_G_REPRO_PACKAGE_20260919/01_DATA_AND_SPLITS'
DESC=json.loads((ROOT/'CREDIT_G_REPRO_PACKAGE_20260919/00_LEGACY_REFERENCE/credit_g/prompts/credit_g.json').read_text())['notes']['feature_descriptions']

# frozen data
X=pd.read_csv(DATA/'X.csv')
y=pd.read_csv(DATA/'y.csv')['label'].to_numpy(int)
dev=np.load(DATA/'modern_dev_indices_seed20260918.npy')
hold=np.load(DATA/'modern_holdout_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]
Xh=X.iloc[hold].reset_index(drop=True); yh=y[hold]
FEATURES=list(X.columns)
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[f for f in FEATURES if f not in CAT]

# DeepSeek runtime
secret=ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'
vals={}
for line in secret.read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1); vals[k.strip()]=v.strip()
BASE=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
KEY=vals['DEEPSEEK_API_KEY']; MODEL=vals.get('DEEPSEEK_MODEL','deepseek-flash')
proxy='http://127.0.0.1:7890'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))

def call(prompt,max_tokens=350,retries=6):
    payload={'model':MODEL,'messages':[{'role':'system','content':'Follow the feature-importance scoring instruction exactly. Do not use tools or web access.'},{'role':'user','content':prompt}],
             'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':max_tokens,'stream':False}
    raw=json.dumps(payload).encode(); err=None
    for a in range(retries):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=raw,headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            with opener.open(req,timeout=180) as f: obj=json.loads(f.read().decode())
            return {'ok':True,'content':obj['choices'][0]['message'].get('content',''),'provider_model':obj.get('model'),'usage':obj.get('usage',{}),'attempt':a+1}
        except Exception as e:
            err=repr(e);time.sleep(min(16,2**a))
    return {'ok':False,'error':err}

# Deterministic nested samples from development only.
rng=np.random.default_rng(20260926)
pos=np.where(yd==1)[0]; neg=np.where(yd==0)[0]
rng.shuffle(pos); rng.shuffle(neg)
shots=[16,32,64,128]
sample_idx={}
for n in shots:
    np_=max(1,round(n*yd.mean())); nn=n-np_
    sample_idx[n]=np.r_[pos[:np_],neg[:nn]]
    # shuffle only presentation order deterministically
    rr=np.random.default_rng(20260926+n); rr.shuffle(sample_idx[n])

def fmt(v):
    if isinstance(v,(float,np.floating)):
        return f'{float(v):.6g}'
    return str(v)

def make_prompt(feature,n):
    pairs=', '.join(f'({fmt(Xd.iloc[i][feature])}, {int(yd[i])})' for i in sample_idx[n])
    kind='categorical code' if feature in CAT else 'numeric value'
    return f'''For the given feature, your task is to provide a feature importance score between 0 and 1 (larger means more informative) for predicting whether a German credit applicant is a BAD/HIGH credit risk (target 1=bad/high risk, 0=good/low risk).

Feature: {feature}
Description: {DESC[feature]}
The stored feature value is a {kind} from the frozen benchmark.

Here are {n} development-only data points in the format (feature value, target value). Refer to them when determining how informative the feature is:
{pairs}

Return ONLY JSON with exactly two keys: "score" (number between 0 and 1) and "reasoning" (brief string).'''

def one(feature,n):
    p=make_prompt(feature,n); r=call(p)
    score=None
    if r.get('ok'):
        txt=r['content']
        try:
            m=re.search(r'\{.*\}',txt,re.S); o=json.loads(m.group(0) if m else txt); score=float(o['score'])
        except Exception:
            mm=re.findall(r'(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)',txt)
            if mm: score=float(mm[0])
    return {'method':'Data-centric LLM FS','shots':n,'feature':feature,'score':score,'prompt':p,**r}

records=[]
with ThreadPoolExecutor(max_workers=12) as ex:
    futs=[ex.submit(one,f,n) for n in shots for f in FEATURES]
    for fut in as_completed(futs): records.append(fut.result())
records=sorted(records,key=lambda z:(z['shots'],FEATURES.index(z['feature'])))
with open(OUT/'DATACENTRIC_DEEPSEEK_RAW.jsonl','w') as f:
    for z in records:f.write(json.dumps(z,ensure_ascii=False)+'\n')
if any(z['score'] is None for z in records): raise RuntimeError('score parse failure')

def prep(cols):
    num=[c for c in NUM if c in cols]; cat=[c for c in CAT if c in cols]; tr=[]
    if num: tr.append(('num',StandardScaler(),num))
    if cat: tr.append(('cat',Pipeline([('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),('scale',StandardScaler())]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)

def fit_predict(trainX,trainy,valX,cols):
    mdl=Pipeline([('prep',prep(cols)),('clf',LogisticRegression(C=0.01,solver='lbfgs',max_iter=5000,tol=1e-5))])
    mdl.fit(trainX[cols],trainy); return mdl.predict_proba(valX[cols])[:,1]

cv=StratifiedKFold(5,shuffle=True,random_state=20260926)
summary=[]
orders={}
for n in shots:
    rr=[z for z in records if z['shots']==n]
    order=[z['feature'] for z in sorted(rr,key=lambda z:(-z['score'],FEATURES.index(z['feature'])))]
    orders[n]=order; sel=order[:10]
    aucs=[]
    for tr,va in cv.split(Xd,yd):
        p=fit_predict(Xd.iloc[tr],yd[tr],Xd.iloc[va],sel)
        aucs.append(roc_auc_score(yd[va],p))
    summary.append({'shots':n,'selected_features':'|'.join(sel),'dev_cv_auroc':float(np.mean(aucs)),'dev_cv_sd':float(np.std(aucs,ddof=1))})
ss=pd.DataFrame(summary).sort_values(['dev_cv_auroc','shots'],ascending=[False,True])
chosen=int(ss.iloc[0].shots); sel=orders[chosen][:10]
ph=fit_predict(Xd,yd,Xh,sel)
result={'method':'Data-centric LLM FS (Li, Tan, Liu; DeepSeek adaptation)','chosen_shots_by_dev_cv':chosen,'k':10,'selected_features':sel,
        'dev_cv_auroc':float(ss.iloc[0].dev_cv_auroc),'holdout_auroc':float(roc_auc_score(yh,ph)),
        'holdout_ap':float(average_precision_score(yh,ph)),'holdout_log_loss':float(log_loss(yh,ph,labels=[0,1])),
        'selection_rule':'choose among 16/32/64/128-shot data-driven LLM score variants using development-only 5-fold AUROC; holdout evaluated once after choice',
        'comparability':'adapted from the paper data-driven prompt; fixed k=10 and common downstream evaluator'}
ss.to_csv(OUT/'DATACENTRIC_DEV_SELECTION.csv',index=False)
(OUT/'DATACENTRIC_RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(OUT/'DATACENTRIC_ORDERS.json').write_text(json.dumps({str(k):v for k,v in orders.items()},indent=2)+'\n')
print(ss.to_string(index=False))
print(json.dumps(result,indent=2,ensure_ascii=False))
