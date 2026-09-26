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
import sys,json,re,time,urllib.request,random,math
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,log_loss
from sklearn.feature_selection import mutual_info_classif

ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'CROSS_DATASET_RECENT_LLM_FS_20260926/OSTEOPOROSIS';OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'
REPRO=SRC/'16_OVERALL_REPORT_AND_REPRO_PACKAGE/HOSPITAL_OSTEOPOROSIS_LOW_TSCORE_REPRO'
BASEPURE=ROOT/'EXISTING_METHOD_BASELINES_20260925/OSTEOPOROSIS'
feat=pd.read_csv(REPRO/'frozen_inputs/FROZEN_BROAD_SELECTOR_FEATURES_37.csv')
IDS=feat.feature_name.astype(str).tolist();NAMES=dict(zip(feat.feature_name.astype(str),feat.canonical_english_name.astype(str)))
assert len(IDS)==37

# authoritative dev/external machinery
DATA=ROOT/'hospital_osteoporosis_dataonly_pilot_20260917'; CANON=ROOT/'hospital_osteoporosis_canonical_20260917'
sys.path.insert(0,str(DATA/'scripts'));import v2_core as V
man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json'));all_features=list(man['selectable_features'])
full=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),on=['patient_uid','site']).reset_index(drop=True)
b1=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True);y1=b1.y.to_numpy(int);site1=b1.site.to_numpy();groups=b1.patient_uid.to_numpy()
assert len(b1)==892
Xall=pd.read_csv(CANON/'canonical/site_level_X.csv');Y=pd.read_csv(CANON/'canonical/site_level_y.csv');Xall=Xall.copy();Xall['y']=pd.to_numeric(Y.iloc[:,0]).astype(int).to_numpy();b2=Xall[Xall.cohort.astype(str).eq('batch2') & Xall.site.isin(V.SITE_PRIMARY)].reset_index(drop=True);y2=b2.y.to_numpy(int);site2=b2.site.to_numpy()
assert len(b2)==727

# normalized numeric matrix for selectors that require data
def numeric_frame(df):
    z=pd.DataFrame(index=df.index)
    for f in all_features:
        if f=='DXA_性别': z[f]=df[f].astype(str).str.strip().map({'女':1.0,'男':0.0,'female':1.0,'male':0.0,'1.0':1.0,'0.0':0.0,'1':1.0,'0':0.0})
        else:z[f]=pd.to_numeric(df[f],errors='coerce')
    return z
Z1=numeric_frame(b1)
med=Z1.median(axis=0);Z1=Z1.fillna(med)
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=91019);splits=list(cv.split(np.zeros(len(y1)),y1,groups))

def canonical(sel):
    s=set(sel);ret=[f for f in all_features if f in s];assert len(ret)==len(s)
    return ret

fitness_cache={}
def dev_score(raw_sel):
    sel=canonical(raw_sel); kk='|'.join(sel)
    if kk in fitness_cache:return fitness_cache[kk]
    X=b1[sel].to_numpy(float)
    best=(-1,None,[])
    for lam in V.LAM_GRID:
        vals=[]
        for tr,va in splits:
            w,info,sc,nctx=V.fit_full(X[tr],site1[tr],y1[tr],lam,kind='l2');p=V.predict_full(w,sc,X[va],site1[va]);vals.append(float(roc_auc_score(y1[va],p)))
        m=float(np.mean(vals))
        if m>best[0]:best=(m,float(lam),vals)
    fitness_cache[kk]={'mean':best[0],'lambda':best[1],'folds':best[2]}
    return fitness_cache[kk]

def evaluate(raw_sel):
    sel=canonical(raw_sel);fit=dev_score(sel);X=b1[sel].to_numpy(float);w,info,sc,nctx=V.fit_full(X,site1,y1,fit['lambda'],kind='l2')
    zz=numeric_frame(b2)[sel].to_numpy(float);p=V.predict_full(w,sc,zz,site2)
    return {'selected_features':'|'.join(sel),'batch1_cv_auroc':fit['mean'],'predictor_lambda':fit['lambda'],'auroc':float(roc_auc_score(y2,p)),'auprc':float(average_precision_score(y2,p)),'balanced_accuracy':float(balanced_accuracy_score(y2,p>=.5))}

# DeepSeek
env={}
for ln in (SRC/'LOCAL_SECRETS/deepseek_api.env').read_text().splitlines():
    if '=' in ln and not ln.lstrip().startswith('#'):
        k,v=ln.split('=',1);env[k.strip()]=v.strip()
BASE=env.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/');MODEL=env.get('DEEPSEEK_MODEL','deepseek-flash')
proxy='http://127.0.0.1:7890';OP=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
def call(prompt,max_tokens=1000,retries=7):
    body=json.dumps({'model':MODEL,'messages':[{'role':'system','content':'Follow the feature-selection instruction exactly. Do not use web access or tools.'},{'role':'user','content':prompt}],'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':max_tokens,'stream':False}).encode()
    last=None
    for a in range(retries):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=body,headers={'Authorization':'Bearer '+env['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
            with OP.open(req,timeout=240) as f:o=json.loads(f.read().decode())
            return {'ok':True,'content':o['choices'][0]['message'].get('content',''),'model':o.get('model'),'usage':o.get('usage',{})}
        except Exception as e:last=repr(e);time.sleep(min(2**a,20))
    return {'ok':False,'error':last}
TASK='Predict whether the current lumbar-spine or hip skeletal site has osteoporosis-range low bone mineral density (T-score <= -2.5) using demographic and routine blood laboratory variables. The skeletal site is handled separately by the downstream model.'
listing='\n'.join(f'- {f}: {NAMES[f]}' for f in IDS)
rawlog=[]

# Existing pure semantic Score/Rank
score_df=pd.read_csv(BASEPURE/'LLM_SCORE_DEEPSEEK.csv');score_order=score_df.sort_values(['score','feature'],ascending=[False,True]).feature.tolist();rank_order=json.loads((BASEPURE/'LLM_RANK_ORDER.json').read_text())
methods={'LLM-Select Score':score_order[:10],'LLM-Select Rank':rank_order[:10]}

# Data-centric pointwise, several shot sizes chosen dev-only
rng=np.random.default_rng(20260926);indices=np.arange(len(b1));rng.shuffle(indices)
shot_options=[32,64,128]
for n in shot_options:
    sample=indices[:n]
    rec=[]
    def dc_one(fid):
        vals=[]
        for i in sample:
            v=b1.iloc[i][fid]
            if fid=='DXA_性别':v={'女':'female','男':'male'}.get(str(v).strip(),str(v))
            vals.append(f'({v}, {int(y1[i])})')
        p=f'''{TASK}
Feature: {NAMES[fid]} (ID {fid})
Here are {n} development-only observations as (feature value, target): {", ".join(vals)}
Give an importance score from 0 to 1 based on the provided feature-target data and feature meaning.
Return ONLY JSON: {{"score": <number>, "reasoning": "<brief>"}}.'''
        z=call(p,300);txt=z.get('content','');m=re.search(r'\{.*\}',txt,re.S)
        try:o=json.loads(m.group(0) if m else txt);sc=float(o['score'])
        except:
            mm=re.search(r'(?i)score[^0-9]*([01](?:\.\d+)?)',txt);sc=float(mm.group(1)) if mm else np.nan
        return {'feature':fid,'score':sc,'prompt':p,**z}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for q in as_completed([ex.submit(dc_one,f) for f in IDS]):rec.append(q.result())
    if any(not np.isfinite(z['score']) for z in rec):raise RuntimeError('data-centric parse failure')
    order=[z['feature'] for z in sorted(rec,key=lambda z:(-z['score'],IDS.index(z['feature'])))]
    methods[f'Data-centric-{n}']=order[:10];rawlog += [{'kind':f'datacentric-{n}',**z} for z in rec]
best_dc=max(shot_options,key=lambda n:dev_score(methods[f'Data-centric-{n}'])['mean'])
methods['Data-centric LLM FS']=methods[f'Data-centric-{best_dc}']

# LLM-Lasso: pure LLM scores -> feature-specific penalty; dev-only grouped CV tunes exponent/C.
smap=dict(zip(score_df.feature.astype(str),score_df.score.astype(float)))
llrows=[]
for exp in range(0,6):
    for C in np.logspace(-3,1.3,12):
        fold_auc=[];fold_sel=[];ok=True
        for tr,va in splits:
            A=Z1.iloc[tr][all_features].to_numpy(float);B=Z1.iloc[va][all_features].to_numpy(float)
            sc=StandardScaler().fit(A);Az=sc.transform(A);Bz=sc.transform(B)
            w=np.array([(1/max(smap[f],.05))**exp for f in all_features]);w/=w.mean()
            mod=LogisticRegression(C=float(C),penalty='l1',solver='liblinear',class_weight='balanced',max_iter=5000).fit(Az/w,y1[tr])
            beta=mod.coef_.ravel()/w;active=np.flatnonzero(np.abs(beta)>1e-10)
            if len(active)<10:ok=False;break
            top=active[np.argsort(-np.abs(beta[active]))[:10]];sel=[all_features[j] for j in top]
            # common downstream model on selected support within this fold, fixed best lambda over 5-lambda grid on same fold is avoided; use middle lambda 33.33 for fitness.
            Xt=b1.iloc[tr][sel].to_numpy(float);Xv=b1.iloc[va][sel].to_numpy(float);ww,info,scc,nctx=V.fit_full(Xt,site1[tr],y1[tr],33.333333333333336,kind='l2');pp=V.predict_full(ww,scc,Xv,site1[va])
            fold_auc.append(float(roc_auc_score(y1[va],pp)));fold_sel.append('|'.join(sel))
        if ok:llrows.append({'exp':exp,'C':float(C),'mean_auc':float(np.mean(fold_auc)),'supports':'||'.join(fold_sel)})
lldf=pd.DataFrame(llrows);best=lldf.sort_values(['mean_auc','exp','C'],ascending=[False,True,True]).iloc[0]
A=Z1[all_features].to_numpy(float);sc=StandardScaler().fit(A);Az=sc.transform(A);w=np.array([(1/max(smap[f],.05))**int(best.exp) for f in all_features]);w/=w.mean();mod=LogisticRegression(C=float(best.C),penalty='l1',solver='liblinear',class_weight='balanced',max_iter=5000).fit(Az/w,y1);beta=mod.coef_.ravel()/w;active=np.flatnonzero(np.abs(beta)>1e-10);top=active[np.argsort(-np.abs(beta[active]))[:10]];methods['LLM-Lasso']=[all_features[j] for j in top]
lldf.to_csv(OUT/'LLMLASSO_DEV_GRID.csv',index=False)

# FREEFORM style: iterative self-consistency + pyramid consensus, choose variant on dev.
selected=[]
for step in range(10):
    cand=[f for f in IDS if f not in selected];runs=1 if step<2 else 3
    def fr_one(rep):
        oo=cand.copy();random.Random(20260926+step*101+rep).shuffle(oo)
        p=f'''{TASK}
Already selected: {selected if selected else "(none)"}
Remaining candidates:
{chr(10).join("- "+f+": "+NAMES[f] for f in oo)}
Select the SINGLE most important remaining feature using pretrained medical knowledge only. Return ONLY the exact feature ID.'''
        z=call(p,120);hits=[(z.get('content','').find(f),f) for f in cand if z.get('content','').find(f)>=0];ch=sorted(hits)[0][1] if hits else None
        return {'kind':'freeform_iter','step':step,'rep':rep,'parsed':ch,'prompt':p,**z}
    with ThreadPoolExecutor(max_workers=runs) as ex: rr=[q.result() for q in as_completed([ex.submit(fr_one,r) for r in range(runs)])]
    rawlog += rr;votes=[z['parsed'] for z in rr if z['parsed']];ch=Counter(votes).most_common(1)[0][0];selected.append(ch)
iter_sel=selected
def pyramid_one(rep):
    oo=IDS.copy();random.Random(20261926+rep).shuffle(oo)
    p=f'''{TASK}
Candidate features:
{chr(10).join("- "+f+": "+NAMES[f] for f in oo)}
Select exactly 10 distinct features using pretrained medical knowledge only. Return ONLY a JSON array of 10 exact feature IDs.'''
    z=call(p,350);txt=z.get('content','');sel=[]
    try:m=re.search(r'\[.*\]',txt,re.S);a=json.loads(m.group(0) if m else txt);sel=[str(x) for x in a if str(x) in IDS]
    except:pass
    return {'kind':'freeform_pyramid','rep':rep,'parsed':sel,'valid':len(sel)==10 and len(set(sel))==10,'prompt':p,**z}
with ThreadPoolExecutor(max_workers=10) as ex:pyr=[q.result() for q in as_completed([ex.submit(pyramid_one,r) for r in range(10)])]
rawlog += pyr;valid=[z['parsed'] for z in pyr if z['valid']];cnt=Counter(f for s in valid for f in s);avgr={f:np.mean([s.index(f) if f in s else 10 for s in valid]) for f in IDS};pyr_sel=sorted(IDS,key=lambda f:(-cnt[f],avgr[f],IDS.index(f)))[:10]
methods['FREEFORM-style']=max([iter_sel,pyr_sel],key=lambda s:dev_score(s)['mean'])

# ICE-SEARCH
ref_auth=pd.read_csv(SRC/'11_BATCH2_FINAL_EVALUATION_V2_0_1/BATCH2_PRIMARY_RESULTS.csv');refset=str(ref_auth[(ref_auth.method=='reference')&(ref_auth.k==10)].iloc[0].selected_set).split('|')
# MI initialization on imputed numeric matrix
mi=mutual_info_classif(Z1[all_features].to_numpy(float),y1,random_state=20260926)
mi_sel=[all_features[i] for i in np.argsort(-mi)[:10]]
pool={}
for origin,sel in [('ref',refset),('llm-score',methods['LLM-Select Score']),('llm-rank',methods['LLM-Select Rank']),('mi',mi_sel),('freeform',methods['FREEFORM-style'])]:
    d=dev_score(sel);pool['|'.join(canonical(sel))]={'features':sel,'score':d['mean'],'origin':origin,'epoch':0}
roles=['bone-metabolism clinician','endocrinologist','clinical pathologist','geriatrician','statistician','feature-selection researcher','nephrologist','hematologist','robust-model auditor','skeptical ensemble designer']
def retain(pp):
    v=sorted(pp.values(),key=lambda z:z['score'],reverse=True);x=v[:5]+(v[-2:] if len(v)>6 else []);o=[];seen=set()
    for z in x:
        k='|'.join(canonical(z['features']))
        if k not in seen:o.append(z);seen.add(k)
    return o
for ep in range(1,7):
    ret=retain(pool);lines='\n'.join(f'{i+1}. CV AUROC={z["score"]:.4f}; subset={z["features"]}' for i,z in enumerate(ret))
    def ice_one(role):
        p=f'''Act as a {role}. {TASK}
This is an ICE-SEARCH-style evolutionary feature-selection step. Development-only evaluated subsets:
{lines}
Candidates:
{listing}
Propose ONE new subset of exactly 10 distinct feature IDs by crossover/mutation, using both the CV feedback and domain knowledge. Return ONLY a JSON array.'''
        z=call(p,350);sel=[]
        try:m=re.search(r'\[.*\]',z.get('content',''),re.S);a=json.loads(m.group(0) if m else z['content']);sel=[str(x) for x in a if str(x) in IDS]
        except:pass
        return {'kind':'ice','epoch':ep,'role':role,'parsed':sel,'valid':len(sel)==10 and len(set(sel))==10,'prompt':p,**z}
    with ThreadPoolExecutor(max_workers=10) as ex:rr=[q.result() for q in as_completed([ex.submit(ice_one,r) for r in roles])]
    rawlog += rr
    for z in rr:
        if z['valid']:
            d=dev_score(z['parsed']);pool['|'.join(canonical(z['parsed']))]={'features':z['parsed'],'score':d['mean'],'origin':z['role'],'epoch':ep}
    pool={'|'.join(canonical(z['features'])):z for z in retain(pool)}
methods['ICE-SEARCH']=max(pool.values(),key=lambda z:z['score'])['features']

# LLM4FS-style: 200 development rows in one prompt.
rng=np.random.default_rng(20260927);idx=np.arange(len(b1));rng.shuffle(idx);idx=idx[:200]
sample_df=numeric_frame(b1).iloc[idx][all_features].copy();sample_df.columns=IDS;sample_df['Class']=y1[idx]
p=f'''{TASK}
Apply Random-Forest-style reasoning to the following 200 development-only samples. Score ALL 37 candidate features from 0 to 1, all scores different. Return ONLY a JSON array of objects {{"feature":"<ID>","score":<number>}} including each feature ID exactly once.
Feature names:
{listing}
Samples:
{sample_df.to_csv(index=False)}'''
z=call(p,5000);rawlog.append({'kind':'llm4fs','prompt':p,**z});txt=z.get('content','');arr=[]
try:m=re.search(r'\[.*\]',txt,re.S);arr=json.loads(m.group(0) if m else txt)
except:pass
scores={}
for o in arr:
    f=str(o.get('feature',''))
    if f in IDS:
        try:scores[f]=float(o['score'])
        except:pass
if len(scores)==37:methods['LLM4FS-style']=sorted(IDS,key=lambda f:(-scores[f],IDS.index(f)))[:10]

# 2026 correlation+feedback iterative selection
corr={f:float(np.corrcoef(Z1[f].to_numpy(float),y1)[0,1]) for f in all_features};sel=[]
for step in range(10):
    cand=[f for f in IDS if f not in sel]
    fb='no selected features yet' if not sel else f'current development CV AUROC={dev_score(sel)["mean"]:.4f}'
    lines='\n'.join(f'- {f}: corr={corr[f]:+.4f}; {NAMES[f]}' for f in cand)
    p=f'''{TASK}
Already selected: {sel if sel else "(none)"}; {fb}.
Remaining features with development-only Pearson correlation:
{lines}
Choose the SINGLE next feature considering semantics, correlation, complementarity, and current validation feedback. Return ONLY the exact feature ID.'''
    z=call(p,120);rawlog.append({'kind':'corr_feedback','step':step,'prompt':p,**z});hits=[(z.get('content','').find(f),f) for f in cand if z.get('content','').find(f)>=0]
    if not hits:raise RuntimeError('corr feedback parse')
    sel.append(sorted(hits)[0][1])
methods['Correlation+feedback 2026-style']=sel

# Remove internal data-centric variants from main methods
for n in shot_options:methods.pop(f'Data-centric-{n}',None)

# Evaluate every final method once on Batch2
rows=[]
for name,sel in methods.items():
    try:rows.append({'method':name,**evaluate(sel)})
    except Exception as e:rows.append({'method':name,'error':repr(e)})
# authoritative anchors
for meth,label in [('reference','Reference'),('selective','Selective Correction (ours)')]:
    rr=ref_auth[(ref_auth.method==meth)&(ref_auth.k==10)].iloc[0];rows.append({'method':label,'selected_features':rr.selected_set,'batch1_cv_auroc':np.nan,'predictor_lambda':rr.predictor_l2_lambda,'auroc':float(rr.auroc),'auprc':float(rr.auprc),'balanced_accuracy':float(rr.balanced_accuracy)})
res=pd.DataFrame(rows);res.to_csv(OUT/'OSTEOPOROSIS_RECENT_METHODS_RESULTS.csv',index=False)
with open(OUT/'RAW_LLM_CALLS.jsonl','w') as f:
    for z in rawlog:f.write(json.dumps(z,ensure_ascii=False,default=str)+'\n')
(OUT/'METHOD_SELECTIONS.json').write_text(json.dumps({k:v for k,v in methods.items()},ensure_ascii=False,indent=2)+'\n')
(OUT/'MANIFEST.json').write_text(json.dumps({'status':'POST_HOC_COMMON_PROTOCOL_COMPARISON','k':10,'development':'Batch1 primary 892 rows/664 patients','evaluation':'Batch2 primary 727 rows/538 patients','holdout_used_for_selection':False,'model':MODEL,'temperature':1.0},indent=2)+'\n')
print(res[['method','auroc','batch1_cv_auroc','selected_features']].sort_values('auroc',ascending=False).to_string(index=False))
