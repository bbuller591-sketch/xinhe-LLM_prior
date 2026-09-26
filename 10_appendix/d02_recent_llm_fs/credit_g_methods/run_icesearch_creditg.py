

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
import json,re,time,urllib.request,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

ROOT=Path(str(REPRO_ROOT))
WORK=ROOT/'EXISTING_METHOD_COMPARISON_20260925'
OUT=WORK/'RECENT_METHODS';OUT.mkdir(parents=True,exist_ok=True)
PKG=ROOT/'CREDIT_G_REPRO_PACKAGE_20260919'
DATA=PKG/'01_DATA_AND_SPLITS'
X=pd.read_csv(DATA/'X.csv');y=pd.read_csv(DATA/'y.csv')['label'].to_numpy(int)
dev=np.load(DATA/'modern_dev_indices_seed20260918.npy');hold=np.load(DATA/'modern_holdout_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev];Xh=X.iloc[hold].reset_index(drop=True);yh=y[hold]
FEATURES=list(X.columns)
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[f for f in FEATURES if f not in CAT]
DESC=json.loads((PKG/'00_LEGACY_REFERENCE/credit_g/prompts/credit_g.json').read_text())['notes']['feature_descriptions']

# DeepSeek
vals={}
for line in (ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env').read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1);vals[k.strip()]=v.strip()
BASE=vals.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/');KEY=vals['DEEPSEEK_API_KEY'];MODEL=vals.get('DEEPSEEK_MODEL','deepseek-flash')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}))
def call(prompt,retries=6):
    payload={'model':MODEL,'messages':[{'role':'system','content':'You are participating in an evolutionary feature-selection search. Follow the requested output format exactly. Do not use tools or web access.'},{'role':'user','content':prompt}],
             'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':400,'stream':False}
    raw=json.dumps(payload).encode();err=None
    for a in range(retries):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=raw,headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            with opener.open(req,timeout=180) as f:o=json.loads(f.read().decode())
            return {'ok':True,'content':o['choices'][0]['message'].get('content',''),'provider_model':o.get('model'),'usage':o.get('usage',{})}
        except Exception as e:err=repr(e);time.sleep(min(16,2**a))
    return {'ok':False,'error':err}

def prep(cols):
    tr=[];num=[c for c in NUM if c in cols];cat=[c for c in CAT if c in cols]
    if num:tr.append(('num',StandardScaler(),num))
    if cat:tr.append(('cat',Pipeline([('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),('scale',StandardScaler())]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)

def common_predict(trainX,trainy,valX,sel):
    mdl=Pipeline([('prep',prep(sel)),('clf',LogisticRegression(C=0.01,solver='lbfgs',max_iter=5000,tol=1e-5))])
    mdl.fit(trainX[sel],trainy);return mdl.predict_proba(valX[sel])[:,1]

cv10=StratifiedKFold(10,shuffle=True,random_state=20260926)
score_cache={}
def key(sel):return '|'.join(sorted(sel,key=FEATURES.index))
def dev_score(sel):
    k=key(sel)
    if k in score_cache:return score_cache[k]
    vals=[]
    for tr,va in cv10.split(Xd,yd):
        p=common_predict(Xd.iloc[tr],yd[tr],Xd.iloc[va],sel)
        vals.append(float(roc_auc_score(yd[va],p)))
    score_cache[k]={'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1)),'folds':vals}
    return score_cache[k]

# Initial populations: multiple data-driven selectors plus direct LLM selectors.
frozen=pd.read_csv(PKG/'06_RESULTS/FINAL_SELECTED_SETS_FREEZE.csv')
def frozen_set(selector):
    r=frozen[(frozen.selector==selector)&(frozen.method=='reference')].iloc[0]
    return str(r.selected_features).split('|')
initial=[]
for name in ['GBM_PERM','L1','ELASTIC_NET']:
    initial.append((f'data_{name}',frozen_set(name)))

# Mutual information (semantic variables; categorical flags respected)
mi=mutual_info_classif(Xd.to_numpy(),yd,discrete_features=np.array([f in CAT for f in FEATURES]),random_state=20260926)
initial.append(('data_MI',[FEATURES[i] for i in np.argsort(-mi)[:10]]))
rf=RandomForestClassifier(n_estimators=700,max_features='sqrt',min_samples_leaf=3,class_weight='balanced',random_state=20260926,n_jobs=64).fit(Xd,yd)
pi=permutation_importance(rf,Xd,yd,n_repeats=10,random_state=20260926,scoring='roc_auc',n_jobs=64)
initial.append(('data_RFperm',[FEATURES[i] for i in np.argsort(-pi.importances_mean)[:10]]))

ord=json.loads((WORK/'CREDIT_G_SCORE_RANK_ORDERS.json').read_text())
initial.append(('llm_score',ord['score_order'][:10]));initial.append(('llm_rank',ord['rank_order'][:10]))
seq=json.loads((WORK/'CREDIT_G_SEQ_ORDER.json').read_text())['seq_order']
initial.append(('llm_seq_style',seq[:10]))

pool={}
history=[]
for name,sel in initial:
    if len(sel)==10 and len(set(sel))==10:
        sc=dev_score(sel);pool[key(sel)]={'features':sel,'score':sc['mean'],'origin':name,'epoch':0}
        history.append({'epoch':0,'role':name,'features':'|'.join(sel),'dev_cv_auroc':sc['mean'],'status':'initial'})

roles=[
'credit-risk analyst','bank loan underwriter','consumer-credit economist','financial risk manager','statistician',
'machine-learning feature-selection researcher','bank regulator','behavioral economist','credit-scoring practitioner',
'fraud and risk data scientist','causal inference researcher','robust statistics researcher','portfolio risk analyst',
'consumer finance researcher','model validation specialist','responsible AI credit-risk auditor','skeptical ensemble designer']

feature_block='\n'.join(f'- {f}: {DESC[f]}' for f in FEATURES)
def retained(pool):
    vals=sorted(pool.values(),key=lambda z:z['score'],reverse=True)
    top=vals[:5];bottom=vals[-3:] if len(vals)>=8 else vals[5:]
    out=[];seen=set()
    for z in top+bottom:
        kk=key(z['features'])
        if kk not in seen:out.append(z);seen.add(kk)
    return out

def prompt_for(role,epoch,ret):
    lines=[]
    for i,z in enumerate(ret,1):
        quality='HIGH' if z in sorted(ret,key=lambda q:q['score'],reverse=True)[:min(5,len(ret))] else 'LOW'
        lines.append(f'{i}. {quality} CV AUROC={z["score"]:.4f}; subset={z["features"]}')
    return f'''Act as a {role}. We are predicting BAD/HIGH credit risk from the German Credit benchmark and must select exactly 10 of the 20 semantic features.

This is an ICE-SEARCH-style evolutionary step. Below are previously evaluated feature subsets with DEVELOPMENT-ONLY 10-fold CV AUROC. Use them as in-context evolutionary evidence: preserve useful combinations, avoid repeatedly poor combinations, and make crossover/mutation changes using your domain knowledge. Do not use any held-out test information.

CURRENT POPULATION:
{chr(10).join(lines)}

AVAILABLE FEATURES:
{feature_block}

Propose ONE new subset of exactly 10 DISTINCT features. It should be a meaningful mutation/crossover rather than simply copying one population member.

Return ONLY a JSON array of the 10 exact feature keys, with no explanation.'''

def propose(role,epoch,ret):
    p=prompt_for(role,epoch,ret);r=call(p)
    sel=[]
    if r.get('ok'):
        try:
            m=re.search(r'\[.*\]',r['content'],re.S);a=json.loads(m.group(0) if m else r['content'])
            sel=[str(x) for x in a if str(x) in FEATURES]
        except Exception:
            # fallback: order by first appearance of feature key
            low=r['content'];hits=[]
            for f in FEATURES:
                j=low.find(f)
                if j>=0:hits.append((j,f))
            sel=[f for _,f in sorted(hits)]
    ok=len(sel)==10 and len(set(sel))==10
    return {'role':role,'epoch':epoch,'prompt':p,'response':r,'features':sel,'valid':ok}

raw=[]
for epoch in range(1,9):
    ret=retained(pool)
    proposals=[]
    with ThreadPoolExecutor(max_workers=17) as ex:
        fs=[ex.submit(propose,role,epoch,ret) for role in roles]
        for f in as_completed(fs):proposals.append(f.result())
    for z in proposals:
        raw.append(z)
        if not z['valid']:
            history.append({'epoch':epoch,'role':z['role'],'features':'','dev_cv_auroc':np.nan,'status':'invalid'})
            continue
        sc=dev_score(z['features'])
        kk=key(z['features'])
        cand={'features':z['features'],'score':sc['mean'],'origin':z['role'],'epoch':epoch}
        if kk not in pool or sc['mean']>pool[kk]['score']:pool[kk]=cand
        history.append({'epoch':epoch,'role':z['role'],'features':'|'.join(z['features']),'dev_cv_auroc':sc['mean'],'status':'valid'})
    # Evolutionary survival: top 5 and bottom 3 across current union, matching ICE-SEARCH exploration principle.
    survivors=retained(pool)
    pool={key(z['features']):z for z in survivors}
    best=max(pool.values(),key=lambda z:z['score'])
    print(f'epoch {epoch}: pool={len(pool)} best_dev={best["score"]:.6f} {best["features"]}')

# final dev winner among final retained population
best=max(pool.values(),key=lambda z:z['score']);sel=best['features']
ph=common_predict(Xd,yd,Xh,sel)
result={'method':'ICE-SEARCH (DeepSeek adaptation)','k':10,'epochs':8,'roles_per_epoch':17,
        'selected_features':sel,'dev_cv_auroc':best['score'],'origin':best['origin'],'winning_epoch':best['epoch'],
        'holdout_auroc':float(roc_auc_score(yh,ph)),'holdout_ap':float(average_precision_score(yh,ph)),
        'holdout_log_loss':float(log_loss(yh,ph,labels=[0,1])),
        'method_note':'ICE-SEARCH-style in-context evolutionary search: mixed classical/LLM initialization, CV-scored populations, role-conditioned crossover/mutation, top-5 + bottom-3 survival, 8 epochs; all search feedback development-only.',
        'reproduction_status':'faithful adaptation to CREDIT-G fixed k=10 and common L2 evaluator; original paper is medical-domain and uses its own downstream models.'}
pd.DataFrame(history).to_csv(OUT/'ICESEARCH_HISTORY.csv',index=False)
with open(OUT/'ICESEARCH_RAW.jsonl','w') as f:
    for z in raw:f.write(json.dumps(z,ensure_ascii=False)+'\n')
(OUT/'ICESEARCH_RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(OUT/'ICESEARCH_SCORE_CACHE.json').write_text(json.dumps(score_cache,indent=2)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
