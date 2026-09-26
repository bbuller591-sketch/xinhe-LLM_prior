

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
import json,re,time,urllib.request
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss

ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'EXISTING_METHOD_COMPARISON_20260925/RECENT_METHODS';OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'CREDIT_G_REPRO_PACKAGE_20260919/01_DATA_AND_SPLITS'
prompt_info=json.loads((ROOT/'CREDIT_G_REPRO_PACKAGE_20260919/00_LEGACY_REFERENCE/credit_g/prompts/credit_g.json').read_text())
DESC=prompt_info['notes']['feature_descriptions']
X=pd.read_csv(DATA/'X.csv'); y=pd.read_csv(DATA/'y.csv')['label'].to_numpy(int)
dev=np.load(DATA/'modern_dev_indices_seed20260918.npy');hold=np.load(DATA/'modern_holdout_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev];Xh=X.iloc[hold].reset_index(drop=True);yh=y[hold]
FEATURES=list(X.columns)
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[f for f in FEATURES if f not in CAT]

# DeepSeek
vals={}
for line in (ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env').read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1);vals[k.strip()]=v.strip()
BASE=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/');KEY=vals['DEEPSEEK_API_KEY'];MODEL=vals.get('DEEPSEEK_MODEL','deepseek-flash')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}))
def call(prompt):
    payload={'model':MODEL,'messages':[{'role':'system','content':'Follow the feature penalty-factor instruction exactly. Do not use tools or web access.'},{'role':'user','content':prompt}],
             'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':1800,'stream':False}
    raw=json.dumps(payload).encode();err=None
    for a in range(6):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=raw,headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            with opener.open(req,timeout=180) as f:o=json.loads(f.read().decode())
            return {'ok':True,'content':o['choices'][0]['message'].get('content',''),'provider_model':o.get('model'),'usage':o.get('usage',{})}
        except Exception as e:err=repr(e);time.sleep(min(16,2**a))
    return {'ok':False,'error':err}

flines='\n'.join(f'- {f}: {DESC[f]}' for f in FEATURES)
prompt=f'''Context: We want to predict whether a German credit applicant is a BAD/HIGH credit risk (1=bad/high risk, 0=good/low risk) from 20 applicant attributes. Before fitting a sparse logistic model, assign LLM-Lasso-style penalty factors to the semantic features.

Task: Provide an integer penalty factor for every feature from 2 to 5 inclusive:
- 2 = strongly associated with the target; penalize least.
- 3 = moderately strong relevance.
- 4 = weaker relevance.
- 5 = minimal relevance; penalize most.

Base the factors only on general domain knowledge of credit risk. Do not use data, tools, web access, or retrieved evidence.

Return ONLY a JSON object mapping each exact feature key to its integer penalty factor. Include all 20 exactly once.

Features:
{flines}'''
raw=call(prompt)
(OUT/'LLMLASSO_DEEPSEEK_RAW.json').write_text(json.dumps({'prompt':prompt,**raw},ensure_ascii=False,indent=2)+'\n')
if not raw.get('ok'):raise RuntimeError(raw)
m=re.search(r'\{.*\}',raw['content'],re.S);pf=json.loads(m.group(0) if m else raw['content'])
pf={k:float(pf[k]) for k in FEATURES}
assert all(v in [2,3,4,5] for v in pf.values())

# Design builder: semantic groups -> standardized columns, preserving group identity.
def design_fit(df):
    arrays=[]; groups=[]; state={}
    for f in FEATURES:
        if f in NUM:
            sc=StandardScaler().fit(df[[f]])
            z=sc.transform(df[[f]])
            arrays.append(z);groups.append(f);state[f]=('num',sc)
        else:
            enc=OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False).fit(df[[f]])
            z0=enc.transform(df[[f]])
            sc=StandardScaler().fit(z0)
            z=sc.transform(z0)
            arrays.append(z);groups.extend([f]*z.shape[1]);state[f]=('cat',enc,sc)
    return np.hstack(arrays),groups,state
def design_transform(df,state):
    arrays=[]
    for f in FEATURES:
        st=state[f]
        if st[0]=='num':arrays.append(st[1].transform(df[[f]]))
        else:arrays.append(st[2].transform(st[1].transform(df[[f]])))
    return np.hstack(arrays)

def topk_from_weighted_l1(trainX,trainy,exp,C,k=10):
    Z,groups,state=design_fit(trainX)
    wsem={f:(1.0/(1.0/pf[f])**0 if False else None) for f in FEATURES} # placeholder for audit readability
    # Official LLM-Lasso with penalty-factor scores uses importance=1/pf and then penalty=1/importance^i = pf^i.
    weights=np.array([pf[g]**exp for g in groups],float)
    weights=weights/weights.mean()
    Zw=Z/weights[None,:]
    mdl=LogisticRegression(C=float(C),penalty='l1',solver='liblinear',max_iter=5000,class_weight=None)
    mdl.fit(Zw,trainy)
    theta=mdl.coef_.ravel()
    beta=theta/weights
    gscore={f:0.0 for f in FEATURES}
    for b,g in zip(beta,groups):gscore[g]=max(gscore[g],abs(float(b)))
    active=[f for f in FEATURES if gscore[f] > 1e-10]
    order=sorted(active,key=lambda f:(-gscore[f],FEATURES.index(f)))
    return (order[:k] if len(active)>=k else None),gscore,len(active)

def common_eval(trainX,trainy,valX,valy,sel):
    # exact common evaluator
    arrays_tr=[];arrays_va=[]
    for f in sel:
        if f in NUM:
            sc=StandardScaler().fit(trainX[[f]]);arrays_tr.append(sc.transform(trainX[[f]]));arrays_va.append(sc.transform(valX[[f]]))
        else:
            enc=OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False).fit(trainX[[f]])
            a=enc.transform(trainX[[f]]);b=enc.transform(valX[[f]])
            sc=StandardScaler().fit(a);arrays_tr.append(sc.transform(a));arrays_va.append(sc.transform(b))
    A=np.hstack(arrays_tr);B=np.hstack(arrays_va)
    mdl=LogisticRegression(C=0.01,solver='lbfgs',max_iter=5000,tol=1e-5).fit(A,trainy)
    return mdl.predict_proba(B)[:,1]

exps=list(range(0,6))
Cs=np.logspace(-3,1.5,14)
cv=StratifiedKFold(5,shuffle=True,random_state=20260926)
rows=[]
for exp in exps:
    for C in Cs:
        aucs=[];supports=[]
        for tr,va in cv.split(Xd,yd):
            sel,_,nactive=topk_from_weighted_l1(Xd.iloc[tr],yd[tr],exp,C,10)
            if sel is None:
                aucs=[]; supports=[]; break
            p=common_eval(Xd.iloc[tr],yd[tr],Xd.iloc[va],yd[va],sel)
            aucs.append(roc_auc_score(yd[va],p));supports.append('|'.join(sel))
        if len(aucs)==5:
            rows.append({'exponent':exp,'C_weighted_l1':float(C),'mean_dev_auc':float(np.mean(aucs)),'sd_dev_auc':float(np.std(aucs,ddof=1)),'supports':'||'.join(supports)})
cvdf=pd.DataFrame(rows)
# ties: lower exponent (less LLM trust), then smaller C
best=cvdf.sort_values(['mean_dev_auc','exponent','C_weighted_l1'],ascending=[False,True,True]).iloc[0]
exp=int(best.exponent);C=float(best.C_weighted_l1)
sel,gscore,nactive=topk_from_weighted_l1(Xd,yd,exp,C,10)
if sel is None: raise RuntimeError('selected dev config does not produce 10 active semantic groups on full development')
ph=common_eval(Xd,yd,Xh,yh,sel)
result={'method':'LLM-Lasso (DeepSeek, common-k adaptation)','k':10,'penalty_factors':pf,'chosen_exponent':exp,'chosen_weighted_l1_C':C,
        'selected_features':sel,'weighted_l1_active_groups_full_dev':nactive,'dev_cv_auroc':float(best.mean_dev_auc),'holdout_auroc':float(roc_auc_score(yh,ph)),
        'holdout_ap':float(average_precision_score(yh,ph)),'holdout_log_loss':float(log_loss(yh,ph,labels=[0,1])),
        'method_note':'LLM-Lasso official penalty-factor transform: score PF -> importance=1/PF -> penalty proportional to PF^i; i and sparse path strength selected on development CV. Semantic top-k extracted from weighted-L1 group coefficient magnitude, then evaluated with common L2 downstream evaluator.',
        'reproduction_status':'faithful adaptation to fixed semantic k=10; not byte-for-byte Adelie reproduction'}
cvdf.to_csv(OUT/'LLMLASSO_DEV_GRID.csv',index=False)
(OUT/'LLMLASSO_RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
print('\nTOP DEV CONFIGS')
print(cvdf.sort_values('mean_dev_auc',ascending=False).head(12).to_string(index=False))
