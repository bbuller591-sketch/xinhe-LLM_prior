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
import json,re,time,urllib.request,fcntl,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd

ROOT=Path(str(REPRO_ROOT))
SRC=ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'
REPRO=SRC/'16_OVERALL_REPORT_AND_REPRO_PACKAGE/HOSPITAL_OSTEOPOROSIS_LOW_TSCORE_REPRO'
OUT=ROOT/'EXISTING_METHOD_BASELINES_20260925/OSTEOPOROSIS'
OUT.mkdir(parents=True,exist_ok=True)
lock=open(OUT/'RUN.lock','w')
try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    print('duplicate active; exit'); sys.exit(0)

SECRET=SRC/'LOCAL_SECRETS/deepseek_api.env'
env={}
for ln in SECRET.read_text().splitlines():
    if '=' in ln and not ln.lstrip().startswith('#'):
        k,v=ln.split('=',1);env[k.strip()]=v.strip()
BASE=env.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
MODEL=env.get('DEEPSEEK_MODEL','deepseek-flash')
OP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def call(prompt,max_tokens=700,retries=6):
    body=json.dumps({'model':MODEL,'messages':[{'role':'user','content':prompt}],
                     'thinking':{'type':'disabled'},'temperature':0,'max_tokens':max_tokens,'stream':False}).encode()
    req=urllib.request.Request(BASE+'/chat/completions',data=body,
        headers={'Authorization':'Bearer '+env['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
    last=None
    for a in range(retries):
        try:
            with OP.open(req,timeout=120) as f: x=json.loads(f.read().decode())
            return {'content':x['choices'][0]['message'].get('content',''),'model':x.get('model'),'usage':x.get('usage',{})}
        except Exception as e:
            last=e;time.sleep(min(2**a,20))
    raise RuntimeError(last)

feat=pd.read_csv(REPRO/'frozen_inputs/FROZEN_BROAD_SELECTOR_FEATURES_37.csv')
assert len(feat)==37
IDS=feat.feature_name.astype(str).tolist()
NAMES=feat.canonical_english_name.astype(str).tolist()
NAME=dict(zip(IDS,NAMES))

TASK=("We want to predict site-level osteoporosis-range low bone mineral density in a hospital DXA cohort. "
      "The binary target is whether the current lumbar-spine or hip skeletal site has T-score <= -2.5. "
      "Candidate predictors are demographic variables and routine blood laboratory measurements. "
      "Do not use any patient data, fitted model statistics, retrieved literature, or external tools; judge only from the task description and feature meaning.")

def score_one(fid):
    p=f"""{TASK}
Feature: {NAME[fid]} (ID: {fid})
Give an importance score from 0 to 1 for predicting the target. Higher means more useful.
Return exactly one JSON object: {{"score": <number>, "reasoning": "<brief reason>"}}."""
    x=call(p,300)
    m=re.search(r'\{.*\}',x['content'],re.S)
    obj=json.loads(m.group(0)) if m else {}
    if 'score' not in obj:
        mm=re.search(r'(?i)score[^0-9]*([01](?:\.\d+)?)',x['content'])
        if not mm: raise RuntimeError(f'parse score {fid}: {x["content"]!r}')
        sc=float(mm.group(1));reason=x['content']
    else: sc=float(obj['score']);reason=str(obj.get('reasoning',''))
    return fid,sc,reason,x

rows=[]
with ThreadPoolExecutor(max_workers=12) as ex:
    fs={ex.submit(score_one,f):f for f in IDS}
    for q in as_completed(fs):
        fid,sc,reason,x=q.result()
        rows.append({'feature':fid,'canonical_name':NAME[fid],'score':sc,'reasoning':reason,
                     'returned_model':x['model'],'usage':json.dumps(x['usage'])})
sdf=pd.DataFrame(rows).sort_values(['score','feature'],ascending=[False,True]).reset_index(drop=True)
sdf.to_csv(OUT/'LLM_SCORE_DEEPSEEK.csv',index=False)
score_sel=sdf.feature.iloc[:10].tolist()

listing='\n'.join(f'{i+1}. {f}: {NAME[f]}' for i,f in enumerate(IDS))
rp=f"""{TASK}
Rank all 37 candidate features from most to least important for predicting the target.
{listing}
Output exactly 37 lines, each containing only the feature ID. Do not omit or duplicate a feature."""
rx=call(rp,1200)
(OUT/'LLM_RANK_RAW.txt').write_text(rx['content']+'\n')
order=[]
# IDs are simple fNN; robust exact extraction
for t in re.findall(r'\bf\d{2}\b',rx['content']):
    if t in IDS and t not in order: order.append(t)
for f in IDS:
    if f not in order: order.append(f)
order=order[:37]
(OUT/'LLM_RANK_ORDER.json').write_text(json.dumps(order,ensure_ascii=False,indent=2)+'\n')
rank_sel=order[:10]

# Authoritative fitting/evaluation machinery
DATA=ROOT/'hospital_osteoporosis_dataonly_pilot_20260917'
CANON=ROOT/'hospital_osteoporosis_canonical_20260917'
sys.path.insert(0,str(DATA/'scripts'))
import v2_core as V
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json'))
all_features=list(man['selectable_features'])
full=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(
    pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),
    on=['patient_uid','site']).reset_index(drop=True)
b1=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
assert len(b1)==892 and b1.patient_uid.nunique()==664
site1=b1.site.to_numpy();y1=b1.y.to_numpy(int);groups=b1.patient_uid.to_numpy()

def choose_fit(sel):
    # canonicalize feature order to authoritative data order
    sel=[f for f in all_features if f in set(sel)]
    assert len(sel)==10,(len(sel),sel)
    X=b1[sel].to_numpy(float)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=91019)
    splits=list(cv.split(np.zeros(len(y1)),y1,groups))
    curve=[];best=(-1,None)
    for lam in V.LAM_GRID:
        aucs=[]
        for tr,va in splits:
            w,info,sc,nctx=V.fit_full(X[tr],site1[tr],y1[tr],lam,kind='l2')
            pp=V.predict_full(w,sc,X[va],site1[va])
            aucs.append(float(roc_auc_score(y1[va],pp)))
        mean=float(np.mean(aucs));curve.append({'lambda':float(lam),'mean_auroc':mean,'folds':aucs})
        if mean>best[0]:best=(mean,float(lam))
    lam=best[1]
    w,info,sc,nctx=V.fit_full(X,site1,y1,lam,kind='l2')
    return sel,lam,best[0],w,sc,nctx,curve

Xall=pd.read_csv(CANON/'canonical/site_level_X.csv')
Y=pd.read_csv(CANON/'canonical/site_level_y.csv')
Xall=Xall.copy();Xall['y']=pd.to_numeric(Y.iloc[:,0]).astype(int).to_numpy()
b2=Xall[Xall.cohort.astype(str).eq('batch2')].copy()
b2=b2[b2.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
assert len(b2)==727 and b2.patient_uid.nunique()==538
y2=b2.y.to_numpy(int);site2=b2.site.to_numpy()

def evaluate(method,raw_sel):
    sel,lam,cv_auc,w,sc,nctx,curve=choose_fit(raw_sel)
    z=b2[sel].copy()
    for c in sel:
        if c=='DXA_性别': z[c]=z[c].astype(str).str.strip().map({'女':1.0,'男':0.0})
        else: z[c]=pd.to_numeric(z[c],errors='coerce')
    X=z.to_numpy(float)
    p=V.predict_full(w,sc,X,site2)
    return {'method':method,'k':10,'selected_features':'|'.join(sel),'batch1_cv_auroc':cv_auc,
            'predictor_lambda':lam,'auroc':float(roc_auc_score(y2,p)),
            'auprc':float(average_precision_score(y2,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y2,(p>=0.5).astype(int))),
            'cv_curve':json.dumps(curve)}

new=[evaluate('LLM-Score (DeepSeek)',score_sel),evaluate('LLM-Rank (DeepSeek)',rank_sel)]
auth=pd.read_csv(SRC/'11_BATCH2_FINAL_EVALUATION_V2_0_1/BATCH2_PRIMARY_RESULTS.csv')
for meth,label in [('reference','Reference (L1)'),('selective','Selective Correction (ours)')]:
    r=auth[(auth.method==meth)&(auth.k==10)].iloc[0]
    new.append({'method':label,'k':10,'selected_features':r.selected_set,'batch1_cv_auroc':np.nan,
                'predictor_lambda':r.predictor_l2_lambda if 'predictor_l2_lambda' in r.index else np.nan,
                'auroc':float(r.auroc),'auprc':float(r.auprc),'balanced_accuracy':float(r.balanced_accuracy),'cv_curve':''})
res=pd.DataFrame(new)
ref=float(res.loc[res.method=='Reference (L1)','auroc'].iloc[0])
ours=float(res.loc[res.method=='Selective Correction (ours)','auroc'].iloc[0])
res['delta_vs_reference_auroc']=res.auroc-ref
res['delta_vs_ours_auroc']=res.auroc-ours
res.to_csv(OUT/'OSTEOPOROSIS_BASELINE_COMPARISON.csv',index=False)

manifest={'status':'POST_HOC_SUPPLEMENTAL_BASELINE','reason':'baseline added after original Batch2 evaluation; feature selection itself used no Batch2 values',
          'requested_model':MODEL,'temperature':0,'thinking':'disabled','k':10,
          'feature_input':'canonical English name only + exact task description; no retrieved evidence or data statistics',
          'training':'Batch1 primary 892 rows/664 patients; same 5-fold patient-grouped lambda selection seed 91019',
          'evaluation':'authoritative Batch2 v2.0.1 preprocessing, 727 rows/538 patients',
          'batch2_used_for_feature_selection_or_tuning':False}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(res[['method','auroc','delta_vs_reference_auroc','auprc','balanced_accuracy','selected_features']].to_string(index=False))
