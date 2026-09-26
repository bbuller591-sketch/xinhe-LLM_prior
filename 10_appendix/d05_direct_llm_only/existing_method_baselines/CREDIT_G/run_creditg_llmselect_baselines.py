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
import json, re, time, math, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl, sys

# Prevent duplicate execution if an orchestration call is replayed.
_lock_fh=open(str(REPRO_ROOT / '10_appendix/d05_direct_llm_only/existing_method_baselines/CREDIT_G/RUN.lock'),'w')
try:
    fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('Another baseline run is already active; exiting duplicate.')
    sys.exit(0)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(str(REPRO_ROOT))
SRC=ROOT/'CREDIT_G_REPRO_PACKAGE_20260919'
OUT=ROOT/'EXISTING_METHOD_BASELINES_20260925'/'CREDIT_G'
OUT.mkdir(parents=True, exist_ok=True)
SECRET=ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'

FEATURES=[
"checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment",
"installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age",
"other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
DESC=[
"Status of existing checking account",
"Duration, in months",
"Credit history (credits taken, paid back duly, delays, critical accounts)",
"Purpose of the credit (e.g., car, television, education)",
"Credit amount",
"Status of savings accounts/bonds, in Deutsche Mark",
"Number of years spent in current employment",
"Installment rate in percentage of disposable income",
"Sex and marital status",
"Other debtors/guarantors (none/co-applicant/guarantor)",
"Number of years spent in current residence",
"Property (e.g., real estate, life insurance)",
"Age",
"Other installment plans (bank/stores/none)",
"Housing (rent/own/for free)",
"Number of existing credits at the bank",
"Job",
"Number of people being liable to provide maintenance for",
"Telephone (none/registered under customer's name)",
"Is a foreign worker (yes/no)"]
ID2DESC=dict(zip(FEATURES,DESC))
DESC2ID=dict(zip(DESC,FEATURES))

CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment",
"personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing",
"existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[x for x in FEATURES if x not in CAT]

CONTEXT=("Context: Using data collected at a German bank, we wish to build a machine learning model that can "
"accurately predict whether a client carries high or low credit risk (target variable). The dataset contains "
"a total of 20 features (e.g., credit history, savings account status). Prior to training the model, we first "
"want to identify a subset of the 20 features that are most important for reliable prediction of the target variable.")

def load_env():
    d={}
    for line in SECRET.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1); d[k.strip()]=v.strip()
    assert d.get('DEEPSEEK_API_KEY')
    return d

ENV=load_env()
BASE=ENV.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
MODEL=ENV.get('DEEPSEEK_MODEL','deepseek-flash')
NO_PROXY_OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def call_ds(messages,max_tokens=800,retries=6):
    body=json.dumps({
        'model':MODEL,
        'messages':messages,
        'thinking':{'type':'disabled'},
        'temperature':0,
        'max_tokens':max_tokens,
        'stream':False,
    }).encode()
    req=urllib.request.Request(BASE+'/chat/completions',data=body,
        headers={'Authorization':'Bearer '+ENV['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
    last=None
    for a in range(retries):
        try:
            with NO_PROXY_OPENER.open(req,timeout=120) as f:
                x=json.loads(f.read().decode())
            return {'content':x['choices'][0]['message']['content'],'usage':x.get('usage',{}),
                    'returned_model':x.get('model'),'raw_id':x.get('id')}
        except Exception as e:
            last=e; time.sleep(min(2**a,20))
    raise RuntimeError(last)

def extract_json_obj(s):
    m=re.search(r'\{.*\}',s,re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None

def prep(cols):
    cols=list(cols); num=[c for c in NUM if c in cols]; cat=[c for c in CAT if c in cols]
    tr=[]
    if num: tr.append(('num',StandardScaler(),num))
    if cat:
        tr.append(('cat',Pipeline([
            ('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),
            ('scale',StandardScaler()),
        ]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)

DATA=SRC/'01_DATA_AND_SPLITS'
X=pd.read_csv(DATA/'X.csv')
y=pd.read_csv(DATA/'y.csv')['label'].to_numpy()
dev=np.load(DATA/'modern_dev_indices_seed20260918.npy')
hold=np.load(DATA/'modern_holdout_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]
Xh=X.iloc[hold].reset_index(drop=True); yh=y[hold]
assert len(dev)==800 and len(hold)==200 and len(set(dev).intersection(set(hold)))==0

def fit_eval(sel):
    pipe=Pipeline([
        ('prep',prep(sel)),
        ('clf',LogisticRegression(penalty='l2',solver='lbfgs',C=0.01,fit_intercept=True,
                                  class_weight=None,max_iter=5000,tol=1e-5))
    ])
    pipe.fit(Xd[sel],yd)
    p=pipe.predict_proba(Xh[sel])[:,1]
    return {
        'auroc':float(roc_auc_score(yh,p)),
        'ap':float(average_precision_score(yh,p)),
        'log_loss':float(log_loss(yh,p,labels=[0,1])),
    }

def dev_cv(sel):
    pipe=Pipeline([
        ('prep',prep(sel)),
        ('clf',LogisticRegression(penalty='l2',solver='lbfgs',C=0.01,fit_intercept=True,
                                  class_weight=None,max_iter=5000,tol=1e-5))
    ])
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260925)
    return float(cross_val_score(pipe,Xd[sel],yd,cv=cv,scoring='roc_auc',n_jobs=5).mean())

# ---- LLM-Score, adapted from official LLM-Select prompt template ----
def score_one(fid):
    desc=ID2DESC[fid]
    prompt=f"""{CONTEXT}
For each feature input by the user, your task is to provide a feature importance score (between 0 and 1; larger value indicates greater importance) for predicting whether an individual carries high credit risk and a reasoning behind how the importance score was assigned.

Return exactly one JSON object with keys \"score\" (number from 0 to 1) and \"reasoning\" (short string).

Provide a score and reasoning for \"{desc}\" formatted according to the output schema above:"""
    x=call_ds([{'role':'user','content':prompt}],max_tokens=300)
    obj=extract_json_obj(x['content'])
    if not obj or 'score' not in obj:
        m=re.search(r'(?i)score[^0-9]*([01](?:\.\d+)?)',x['content'])
        if not m: raise RuntimeError(f'Cannot parse score for {fid}: {x["content"]}')
        score=float(m.group(1)); reasoning=x['content']
    else:
        score=float(obj['score']); reasoning=str(obj.get('reasoning',''))
    return fid,score,reasoning,x

score_rows=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    fut={ex.submit(score_one,f):f for f in FEATURES}
    for q in as_completed(fut):
        fid,score,reasoning,raw=q.result()
        score_rows.append({'feature':fid,'description':ID2DESC[fid],'score':score,'reasoning':reasoning,
                           'returned_model':raw.get('returned_model'),'usage':json.dumps(raw.get('usage',{}))})
score_df=pd.DataFrame(score_rows).sort_values(['score','feature'],ascending=[False,True]).reset_index(drop=True)
score_df.to_csv(OUT/'LLM_SCORE_DEEPSEEK.csv',index=False)
score_sel=score_df.feature.iloc[:10].tolist()

# ---- LLM-Rank, official prompt semantics, one full ranking ----
concept_lines='\n'.join([f'{i+1}. {DESC[i]} [{FEATURES[i]}]' for i in range(20)])
rank_prompt=f"""Given a list of features, rank them according to their importances in predicting whether an individual carries high credit risk. The ranking should be in descending order, starting with the most important feature.

Only output the ranking. Do not output dialogue or explanations for the ranking. Do not exclude any features in the ranking.

Rank all 20 features in the following list:
{concept_lines}

Output exactly 20 lines. On each line output only the bracketed feature ID, without brackets."""
rank_raw=call_ds([{'role':'user','content':rank_prompt}],max_tokens=700)
tokens=re.findall(r'\b(?:'+ '|'.join(map(re.escape,FEATURES)) + r')\b',rank_raw['content'])
rank_order=[]
for t in tokens:
    if t not in rank_order: rank_order.append(t)
if len(rank_order)<20:
    # fallback fuzzy/substring by descriptions
    for f in FEATURES:
        if f not in rank_order: rank_order.append(f)
rank_order=rank_order[:20]
(OUT/'LLM_RANK_RAW.txt').write_text(rank_raw['content']+'\n')
(OUT/'LLM_RANK_ORDER.json').write_text(json.dumps(rank_order,indent=2)+'\n')
rank_sel=rank_order[:10]

# ---- LLM-Seq-style: LLM-Score initialization + development-only CV feedback ----
seq=[score_df.feature.iloc[0]]
seq_log=[{'step':1,'selected':seq.copy(),'dev_cv_auroc':dev_cv(seq),'source':'LLM-Score initialization'}]
history=[]
while len(seq)<10:
    cv_now=dev_cv(seq)
    candidates=[f for f in FEATURES if f not in seq]
    cand_lines='\n'.join([f'- {f}: {ID2DESC[f]}' for f in candidates])
    sel_lines='\n'.join([f'- {f}: {ID2DESC[f]}' for f in seq])
    system=CONTEXT+"\nGiven a list of features already selected and a list of candidate features available, your task is to output the next feature that should be included to maximally improve the performance in predicting whether an individual carries high credit risk."
    user=f"""I used the features:
{sel_lines}
and the trained model achieved a development 5-fold CV AUROC of {cv_now:.6f}.
What feature should I add next from:
{cand_lines}
Give me just the feature ID to add (no other text)."""
    messages=[{'role':'system','content':system}]
    messages.extend(history)
    messages.append({'role':'user','content':user})
    raw=call_ds(messages,max_tokens=80)
    out=raw['content'].strip()
    hit=None
    for f in candidates:
        if re.search(r'\b'+re.escape(f)+r'\b',out):
            hit=f;break
    if hit is None:
        # robust fallback: choose candidate whose description occurs, else top score among candidates
        for f in candidates:
            if ID2DESC[f].lower() in out.lower(): hit=f;break
    if hit is None:
        hit=score_df[score_df.feature.isin(candidates)].iloc[0].feature
    seq.append(hit)
    history.extend([{'role':'user','content':user},{'role':'assistant','content':hit}])
    seq_log.append({'step':len(seq),'selected':seq.copy(),'added':hit,'prompt_output':out,
                    'dev_cv_auroc':dev_cv(seq)})
(OUT/'LLM_SEQ_ORDER.json').write_text(json.dumps(seq,indent=2)+'\n')
(OUT/'LLM_SEQ_LOG.json').write_text(json.dumps(seq_log,indent=2)+'\n')

# Reference and ours from authoritative final results
auth=pd.read_csv(SRC/'06_RESULTS/FINAL_HOLDOUT_RESULTS.csv')
ref=auth[(auth.selector=='GBM_PERM')&(auth.method=='reference')].iloc[0]
ours=auth[(auth.selector=='GBM_PERM')&(auth.method=='selective')].iloc[0]

rows=[]
for method,sel in [('LLM-Score (DeepSeek)',score_sel),('LLM-Rank (DeepSeek)',rank_sel),('LLM-Seq-style (DeepSeek)',seq)]:
    met=fit_eval(sel)
    rows.append({'method':method,'k':len(sel),'selected_features':'|'.join(sel),**met})
rows += [
    {'method':'Reference (GBM-permutation)','k':int(ref.k),'selected_features':ref.selected_features,
     'auroc':float(ref.holdout_auroc),'ap':float(ref.holdout_average_precision),'log_loss':float(ref.holdout_log_loss)},
    {'method':'Selective Correction (ours)','k':int(ours.k),'selected_features':ours.selected_features,
     'auroc':float(ours.holdout_auroc),'ap':float(ours.holdout_average_precision),'log_loss':float(ours.holdout_log_loss)},
]
res=pd.DataFrame(rows)
res['delta_vs_reference_auroc']=res.auroc-float(ref.holdout_auroc)
res['delta_vs_ours_auroc']=res.auroc-float(ours.holdout_auroc)
res.to_csv(OUT/'CREDIT_G_BASELINE_COMPARISON.csv',index=False)

manifest={
 'created':'2026-09-25',
 'source_dataset_package':str(SRC),
 'llm_select_repo':'taekb/llm-select',
 'llm_select_commit_observed':'a2ea8b50d24540b26af2cfdf34a2b14277b34dee',
 'deepseek_requested_model':MODEL,
 'deepseek_base_url':BASE,
 'temperature':0,
 'k':10,
 'split':'existing frozen 800/200 development/holdout',
 'evaluator':'same frozen L2 logistic evaluator, C=0.01',
 'holdout_used_for_selection':False,
 'llm_seq_note':'LLM-Select-style dialogue with LLM-Score initialization; feedback uses development-only 5-fold CV AUROC, never final holdout.',
}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('\n=== CREDIT-G BASELINE COMPARISON ===')
print(res[['method','k','auroc','delta_vs_reference_auroc','ap','log_loss','selected_features']].to_string(index=False))
print('\nSaved to',OUT)
