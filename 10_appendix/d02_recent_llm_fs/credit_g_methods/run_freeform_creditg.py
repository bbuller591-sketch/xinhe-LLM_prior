

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
import json,re,time,urllib.request,random
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss

ROOT=Path(str(REPRO_ROOT)); WORK=ROOT/'EXISTING_METHOD_COMPARISON_20260925';OUT=WORK/'RECENT_METHODS';OUT.mkdir(parents=True,exist_ok=True)
PKG=ROOT/'CREDIT_G_REPRO_PACKAGE_20260919';DATA=PKG/'01_DATA_AND_SPLITS'
X=pd.read_csv(DATA/'X.csv'); y=pd.read_csv(DATA/'y.csv')['label'].to_numpy(int)
dev=np.load(DATA/'modern_dev_indices_seed20260918.npy');hold=np.load(DATA/'modern_holdout_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev];Xh=X.iloc[hold].reset_index(drop=True);yh=y[hold]
FEATURES=list(X.columns)
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[f for f in FEATURES if f not in CAT]
DESC=json.loads((PKG/'00_LEGACY_REFERENCE/credit_g/prompts/credit_g.json').read_text())['notes']['feature_descriptions']

vals={}
for line in (ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env').read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1); vals[k.strip()]=v.strip()
BASE=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/');KEY=vals['DEEPSEEK_API_KEY'];MODEL=vals.get('DEEPSEEK_MODEL','deepseek-flash')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}))
def call(prompt,retries=6):
    payload={'model':MODEL,'messages':[{'role':'system','content':'Use your pretrained knowledge to perform feature selection. Do not use tools, web access, retrieved evidence, or data. Return only the requested feature key(s).'},{'role':'user','content':prompt}],
             'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':250,'stream':False}
    raw=json.dumps(payload).encode();err=None
    for a in range(retries):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=raw,headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            with opener.open(req,timeout=180) as f:o=json.loads(f.read().decode())
            return {'ok':True,'content':o['choices'][0]['message'].get('content',''),'provider_model':o.get('model')}
        except Exception as e:err=repr(e);time.sleep(min(16,2**a))
    return {'ok':False,'error':err}

def parse_one(txt,cand):
    hits=[]
    for f in cand:
        p=txt.find(f)
        if p>=0:hits.append((p,f))
    return sorted(hits)[0][1] if hits else None

def list_block(cand):
    return '\n'.join(f'- {f}: {DESC[f]}' for f in cand)

# FREEFORM-style iterative self-consistency.
selected=[];raw=[]
for step in range(10):
    cand=[f for f in FEATURES if f not in selected]
    runs=1 if step<2 else 3   # matches published code behavior before 11 selected
    def one(rep):
        order=cand.copy(); random.Random(20260926+step*100+rep).shuffle(order)
        p=f'''Select the SINGLE most important remaining feature for predicting whether a German credit applicant is a BAD/HIGH credit risk.

Features already selected: {selected if selected else "(none)"}
Remaining candidates:
{list_block(order)}

Consider the task carefully, then return ONLY the exact feature key of your single choice.'''
        r=call(p);return {'step':step+1,'rep':rep,'prompt':p,**r,'parsed':parse_one(r.get('content',''),cand)}
    with ThreadPoolExecutor(max_workers=runs) as ex:
        rr=[f.result() for f in as_completed([ex.submit(one,j) for j in range(runs)])]
    raw.extend(rr);votes=[z['parsed'] for z in rr if z['parsed']]
    if not votes:raise RuntimeError(f'FREEFORM iterative parse failure step {step+1}')
    cnt=Counter(votes); chosen=sorted(cnt.items(),key=lambda kv:(-kv[1],FEATURES.index(kv[0])))[0][0]
    selected.append(chosen)
iter_sel=selected

# FREEFORM-style single-bucket pyramid consensus: 10 independent top-10 proposals then frequency aggregation.
def proposal(rep):
    order=FEATURES.copy();random.Random(20261926+rep).shuffle(order)
    p=f'''Select the 10 most important features for predicting whether a German credit applicant is a BAD/HIGH credit risk.

Candidate features:
{list_block(order)}

Use pretrained domain knowledge only. Return ONLY a JSON array of exactly 10 distinct exact feature keys, ordered from most to least important.'''
    r=call(p);sel=[]
    if r.get('ok'):
        try:
            m=re.search(r'\[.*\]',r['content'],re.S);a=json.loads(m.group(0) if m else r['content']);sel=[str(x) for x in a if str(x) in FEATURES]
        except Exception:
            hits=[]
            for f in FEATURES:
                q=r['content'].find(f)
                if q>=0:hits.append((q,f))
            sel=[f for _,f in sorted(hits)]
    return {'variant':'pyramid','rep':rep,'prompt':p,**r,'parsed':sel,'valid':len(sel)==10 and len(set(sel))==10}
with ThreadPoolExecutor(max_workers=10) as ex:
    pyr=[f.result() for f in as_completed([ex.submit(proposal,j) for j in range(10)])]
raw.extend(pyr)
valid=[z['parsed'] for z in pyr if z['valid']]
if not valid:raise RuntimeError('FREEFORM pyramid no valid proposals')
counts=Counter(f for s in valid for f in s)
# frequency, then average within-list rank, then frozen order
avg_rank={f:np.mean([s.index(f) if f in s else 10 for s in valid]) for f in FEATURES}
pyr_sel=sorted(FEATURES,key=lambda f:(-counts[f],avg_rank[f],FEATURES.index(f)))[:10]

with open(OUT/'FREEFORM_RAW.jsonl','w') as f:
    for z in raw:f.write(json.dumps(z,ensure_ascii=False)+'\n')

def prep(cols):
    tr=[];num=[c for c in NUM if c in cols];cat=[c for c in CAT if c in cols]
    if num:tr.append(('num',StandardScaler(),num))
    if cat:tr.append(('cat',Pipeline([('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),('scale',StandardScaler())]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)
def pred(a,ya,b,sel):
    m=Pipeline([('prep',prep(sel)),('clf',LogisticRegression(C=0.01,solver='lbfgs',max_iter=5000,tol=1e-5))]);m.fit(a[sel],ya);return m.predict_proba(b[sel])[:,1]
cv=StratifiedKFold(5,shuffle=True,random_state=20260926)
variants={'iterative_self_consistency':iter_sel,'pyramid_consensus':pyr_sel}
vrows=[]
for name,sel in variants.items():
    auc=[]
    for tr,va in cv.split(Xd,yd):auc.append(roc_auc_score(yd[va],pred(Xd.iloc[tr],yd[tr],Xd.iloc[va],sel)))
    vrows.append({'variant':name,'selected_features':'|'.join(sel),'dev_cv_auroc':float(np.mean(auc)),'dev_cv_sd':float(np.std(auc,ddof=1))})
vdf=pd.DataFrame(vrows).sort_values(['dev_cv_auroc','variant'],ascending=[False,True])
chosen=vdf.iloc[0].variant;sel=variants[chosen]
ph=pred(Xd,yd,Xh,sel)
res={'method':'FREEFORM-style DeepSeek adaptation','chosen_variant_by_dev_cv':chosen,'k':10,'selected_features':sel,
     'dev_cv_auroc':float(vdf.iloc[0].dev_cv_auroc),'holdout_auroc':float(roc_auc_score(yh,ph)),
     'holdout_ap':float(average_precision_score(yh,ph)),'holdout_log_loss':float(log_loss(yh,ph,labels=[0,1])),
     'method_note':'Adapted FREEFORM iterative self-consistency / hierarchical consensus mechanisms from genotype feature selection to CREDIT-G; selection uses task semantics only.',
     'reproduction_status':'method-style domain adaptation, not exact genotype-task reproduction'}
vdf.to_csv(OUT/'FREEFORM_DEV_VARIANTS.csv',index=False)
(OUT/'FREEFORM_RESULT.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n')
print(vdf.to_string(index=False));print(json.dumps(res,indent=2,ensure_ascii=False))
