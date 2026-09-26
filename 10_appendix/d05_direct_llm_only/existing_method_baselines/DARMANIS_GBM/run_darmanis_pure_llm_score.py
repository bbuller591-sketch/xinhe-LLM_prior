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
import json,re,time,urllib.request,fcntl,sys,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd

ROOT=Path(str(REPRO_ROOT))
GBMROOT=ROOT/'gbm_canonical_recovery_20260917'
PKG=ROOT/'GBM_REPRO_PACKAGE_V2_9_20260919'
OUT=ROOT/'EXISTING_METHOD_BASELINES_20260925/DARMANIS_GBM'
OUT.mkdir(parents=True,exist_ok=True)
lock=open(OUT/'RUN.lock','w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    print('duplicate active');sys.exit(0)

SECRET=ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'
env={}
for ln in SECRET.read_text().splitlines():
    if '=' in ln and not ln.lstrip().startswith('#'):
        k,v=ln.split('=',1);env[k.strip()]=v.strip()
BASE=env.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
MODEL=env.get('DEEPSEEK_MODEL','deepseek-flash')
OP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

cf=pd.read_csv(GBMROOT/'canonical/candidate_features_canonical.csv')
assert len(cf)==2000
items=[]
for i,r in cf.iterrows():
    sym='' if pd.isna(r.get('gene_symbol')) else str(r.get('gene_symbol')).strip()
    ens=str(r['ensembl_gene_id_stable'])
    items.append({'node':int(i),'symbol':sym,'ensembl':ens,'name':sym if sym else ens})

TASK=("We want to select gene-expression features for classifying glioblastoma neoplastic tumor-core cells "
      "versus tumor-periphery cells in a human single-cell transcriptomic study. Judge each gene only from its "
      "gene identity and your pretrained biomedical knowledge. Do not use expression data, fitted statistics, "
      "retrieved literature, web search, tools, or any dataset-specific memorized ranking.")

def call(p,max_tokens=220,retries=8):
    body=json.dumps({'model':MODEL,'messages':[{'role':'user','content':p}],
                     'thinking':{'type':'disabled'},'temperature':0,'max_tokens':max_tokens,'stream':False}).encode()
    req=urllib.request.Request(BASE+'/chat/completions',data=body,
        headers={'Authorization':'Bearer '+env['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
    last=None
    for a in range(retries):
        try:
            with OP.open(req,timeout=120) as f:x=json.loads(f.read().decode())
            return {'content':x['choices'][0]['message'].get('content',''),'model':x.get('model'),'usage':x.get('usage',{})}
        except Exception as e:
            last=e;time.sleep(min(2**a,30))
    raise RuntimeError(last)

checkpoint=OUT/'LLM_SCORE_RAW.jsonl';done={}
if checkpoint.exists():
    for ln in checkpoint.read_text().splitlines():
        try:
            z=json.loads(ln)
            if z.get('ok'):done[int(z['node'])]=z
        except:pass
wlock=threading.Lock()

def one(it):
    p=f"""{TASK}
Gene: {it['name']}
Give a feature-importance score from 0 to 1 for predicting the target, where larger means more useful.
Return exactly one JSON object with keys \"score\" (number from 0 to 1) and \"reasoning\" (one brief sentence)."""
    x=call(p);m=re.search(r'\{.*\}',x['content'],re.S)
    try:obj=json.loads(m.group(0)) if m else {}
    except:obj={}
    if 'score' not in obj:
        mm=re.search(r'(?i)score[^0-9]*([01](?:\.\d+)?)',x['content'])
        if not mm:raise RuntimeError(f'parse node={it["node"]} {it["name"]}: {x["content"]!r}')
        sc=float(mm.group(1));reason=x['content']
    else:sc=float(obj['score']);reason=str(obj.get('reasoning',''))
    rec={**it,'score':sc,'reasoning':reason,'returned_model':x['model'],'usage':x['usage'],'ok':True}
    with wlock:
        with checkpoint.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    return rec

todo=[it for it in items if it['node'] not in done]
print(f'existing={len(done)} todo={len(todo)}',flush=True)
errs=[]
with ThreadPoolExecutor(max_workers=32) as ex:
    fs={ex.submit(one,it):it for it in todo}
    for i,q in enumerate(as_completed(fs),1):
        it=fs[q]
        try:done[it['node']]=q.result()
        except Exception as e:errs.append((it,e))
        if i%100==0:print(f'{i}/{len(todo)} errors={len(errs)}',flush=True)
for it,e in errs:
    try:done[it['node']]=one(it)
    except Exception as ee:print('FINAL_FAIL',it['node'],ee,flush=True)
missing=[it['node'] for it in items if it['node'] not in done]
if missing:raise RuntimeError(f'missing {len(missing)}')

df=pd.DataFrame([done[i] for i in range(2000)])
rng=np.random.default_rng(20260925);df['tie_jitter']=rng.uniform(-1e-10,1e-10,len(df))
df=df.sort_values(['score','tie_jitter'],ascending=[False,False]).reset_index(drop=True)
df.to_csv(OUT/'LLM_SCORE_DEEPSEEK_2000.csv',index=False)
order=df.node.astype(int).tolist()
for k in [10,20,30]:
    (OUT/f'LLM_SCORE_TOP{k}_FROZEN.txt').write_text('\n'.join(map(str,order[:k]))+'\n')

# Same authoritative internal plate-grouped evaluator.
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
X=np.load(GBMROOT/'canonical/X_candidate_aligned_log1p.npy').astype(float)
y=pd.read_csv(GBMROOT/'canonical/y_canonical.csv')['y'].to_numpy(int)
mp=pd.read_csv(GBMROOT/'canonical/cell_sample_mapping_canonical.csv')
ann=pd.read_csv(GBMROOT/'metadata/darmanis_cell_annotation.csv')[['cell_key','plate']]
m=mp.merge(ann,left_on='cell_id',right_on='cell_key',how='left',validate='one_to_one')
groups=m['plate'].astype(str).to_numpy()
SEED=2026091817
splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))

def ev(cols,tr,va):
    sc=StandardScaler().fit(X[tr][:,cols])
    Xt=sc.transform(X[tr][:,cols]);Xv=sc.transform(X[va][:,cols])
    mod=LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',class_weight='balanced',
                           max_iter=5000,random_state=SEED).fit(Xt,y[tr])
    p=mod.predict_proba(Xv)[:,1]
    ap1=average_precision_score(y[va],p);ap0=average_precision_score(1-y[va],1-p)
    return {'auroc':float(roc_auc_score(y[va],p)),'macro_ap':float((ap1+ap0)/2),
            'balanced_accuracy':float(balanced_accuracy_score(y[va],p>=0.5))}

rows=[]
for k in [10,20,30]:
    cols=np.asarray(order[:k],int)
    for fi,(tr,va) in enumerate(splits,1):
        rows.append({'method':'LLM-Score (DeepSeek)','k':k,'fold':fi,**ev(cols,tr,va)})
fold=pd.DataFrame(rows);fold.to_csv(OUT/'LLM_SCORE_FOLD_RESULTS.csv',index=False)
agg=fold.groupby(['method','k'],as_index=False).agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),
    mean_macro_ap=('macro_ap','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'))

auth=pd.read_csv(PKG/'REPORT/KEY_TABLES/reference_global_selective_SELECTED_COMPARISON_V2_9.csv')
# Paper-facing SIS k10/k20 and EN k10 are enough to compare with pure fixed support.
cmp=[]
for k in [10,20]:
    rr=auth[(auth.selector=='SIS')&(auth.k==k)].iloc[0]
    ours=float(rr.selective_mean_auroc if 'selective_mean_auroc' in rr.index else rr.selective_AUROC if 'selective_AUROC' in rr.index else np.nan)
    ref=float(rr.reference_mean_auroc if 'reference_mean_auroc' in rr.index else rr.reference_AUROC if 'reference_AUROC' in rr.index else np.nan)
    ll=float(agg[agg.k==k].mean_auroc.iloc[0])
    cmp += [{'configuration':f'SIS-{k}','method':'LLM-Score (DeepSeek)','k':k,'mean_auroc':ll},
            {'configuration':f'SIS-{k}','method':'Reference','k':k,'mean_auroc':ref},
            {'configuration':f'SIS-{k}','method':'Selective Correction (ours)','k':k,'mean_auroc':ours}]
# If column detection failed, recover from known row column names by dumping auth columns.
comp=pd.DataFrame(cmp)
comp.to_csv(OUT/'DARMANIS_BASELINE_COMPARISON.csv',index=False)
(OUT/'AUTH_COLUMNS.json').write_text(json.dumps(auth.columns.tolist(),indent=2)+'\n')
manifest={'status':'POST_HOC_SUPPLEMENTAL_BASELINE','requested_model':MODEL,'temperature':0,'thinking':'disabled',
          'method':'pointwise pure LLM-Score across frozen 2000 genes; fixed seeded tie-break; top-k fixed across folds',
          'task':'tumor core vs tumor periphery','evaluation':'same 5-fold StratifiedGroupKFold by plate, seed 2026091817, same L2 logistic evaluator',
          'data_used_by_llm_selection':False,'caveat':'supplemental baseline designed after existing internal-CV results'}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PURE LLM\n',agg.to_string(index=False),flush=True)
print('COMPARE\n',comp.to_string(index=False),flush=True)
