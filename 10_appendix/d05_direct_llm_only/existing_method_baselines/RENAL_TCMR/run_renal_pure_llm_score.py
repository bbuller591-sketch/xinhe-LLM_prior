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
import json,re,time,urllib.request,fcntl,sys,threading,io
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd

ROOT=Path(str(REPRO_ROOT))
SRC=ROOT/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR'
BASE=ROOT/'dataset_screening_20260921'
OUT=ROOT/'EXISTING_METHOD_BASELINES_20260925/RENAL_TCMR'
OUT.mkdir(parents=True,exist_ok=True)
lock=open(OUT/'RUN.lock','w')
try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    print('duplicate active; exit');sys.exit(0)

SECRET=ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env'
env={}
for ln in SECRET.read_text().splitlines():
    if '=' in ln and not ln.lstrip().startswith('#'):
        k,v=ln.split('=',1);env[k.strip()]=v.strip()
BASEURL=env.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
MODEL=env.get('DEEPSEEK_MODEL','deepseek-flash')
OP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

cand=pd.read_csv(SRC/'CANDIDATE_UNIVERSE_FROZEN.csv')
genes=cand.gene.astype(str).tolist()
assert len(genes)==2000 and len(set(genes))==2000

TASK=("We want to select genes from pretreatment/diagnostic renal-allograft biopsy gene-expression measurements "
      "for classifying T-cell-mediated rejection (TCMR) versus non-TCMR transplant biopsy states. "
      "Judge a gene only from its gene identity and your pretrained biomedical knowledge. "
      "Do not use patient data, fitted statistics, retrieved literature, web search, tools, or any dataset-specific memorized ranking.")

def call(prompt,max_tokens=220,retries=8):
    body=json.dumps({'model':MODEL,'messages':[{'role':'user','content':prompt}],
                     'thinking':{'type':'disabled'},'temperature':0,'max_tokens':max_tokens,'stream':False}).encode()
    req=urllib.request.Request(BASEURL+'/chat/completions',data=body,
        headers={'Authorization':'Bearer '+env['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
    last=None
    for a in range(retries):
        try:
            with OP.open(req,timeout=120) as f:x=json.loads(f.read().decode())
            return {'content':x['choices'][0]['message'].get('content',''),'model':x.get('model'),'usage':x.get('usage',{})}
        except Exception as e:
            last=e;time.sleep(min(2**a,30))
    raise RuntimeError(last)

checkpoint=OUT/'LLM_SCORE_RAW.jsonl'
done={}
if checkpoint.exists():
    for ln in checkpoint.read_text().splitlines():
        try:
            z=json.loads(ln)
            if z.get('ok') and z.get('gene'):done[z['gene']]=z
        except:pass
wlock=threading.Lock()

def one(g):
    p=f"""{TASK}
Gene: {g}
Give a feature-importance score from 0 to 1 for predicting the target, where larger means more likely to be useful.
Return exactly one JSON object with keys \"score\" (number from 0 to 1) and \"reasoning\" (one brief sentence)."""
    x=call(p)
    m=re.search(r'\{.*\}',x['content'],re.S)
    try:obj=json.loads(m.group(0)) if m else {}
    except:obj={}
    if 'score' not in obj:
        mm=re.search(r'(?i)score[^0-9]*([01](?:\.\d+)?)',x['content'])
        if not mm: raise RuntimeError(f'parse {g}: {x["content"]!r}')
        sc=float(mm.group(1));reason=x['content']
    else:sc=float(obj['score']);reason=str(obj.get('reasoning',''))
    rec={'gene':g,'score':sc,'reasoning':reason,'returned_model':x['model'],'usage':x['usage'],'ok':True}
    with wlock:
        with checkpoint.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    return rec

todo=[g for g in genes if g not in done]
print(f'resume existing={len(done)} todo={len(todo)}',flush=True)
errs=[]
with ThreadPoolExecutor(max_workers=32) as ex:
    fs={ex.submit(one,g):g for g in todo}
    for i,q in enumerate(as_completed(fs),1):
        g=fs[q]
        try:done[g]=q.result()
        except Exception as e:
            errs.append((g,repr(e)))
            with wlock:
                with (OUT/'ERRORS.jsonl').open('a') as f:f.write(json.dumps({'gene':g,'error':repr(e)})+'\n')
        if i%100==0:print(f'completed {i}/{len(todo)} errors={len(errs)}',flush=True)

# Retry any failures serial-ish.
for g,e in list(errs):
    try:done[g]=one(g)
    except Exception as ee:print('FINAL_FAIL',g,ee,flush=True)
missing=[g for g in genes if g not in done]
if missing: raise RuntimeError(f'{len(missing)} genes still missing: {missing[:10]}')

rows=[done[g] for g in genes]
df=pd.DataFrame(rows)
# Official LLM-Score-style fixed seeded random tie-break.
rng=np.random.default_rng(20260925)
df['tie_jitter']=rng.uniform(-1e-10,1e-10,len(df))
df=df.sort_values(['score','tie_jitter'],ascending=[False,False]).reset_index(drop=True)
df.to_csv(OUT/'LLM_SCORE_DEEPSEEK_2000.csv',index=False)
top50=df.gene.iloc[:50].tolist()
(OUT/'LLM_SCORE_TOP50_FROZEN.txt').write_text('\n'.join(top50)+'\n')

# Evaluation follows the original frozen external evaluator exactly, except support.
sys.path.insert(0,str(BASE/'src'))
from geo_utils import read_geo_metadata,read_geo_expression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

txt=(BASE/'data/GPL570_full.txt').read_text(errors='replace').replace('\r','')
block=txt.split('!platform_table_begin\n',1)[1].split('!platform_table_end',1)[0]
ann=pd.read_csv(io.StringIO(block),sep='\t',dtype=str)[['ID','Gene Symbol']].dropna()
ann=ann[~ann['Gene Symbol'].isin(['---',''])]
ann['Gene Symbol']=ann['Gene Symbol'].str.split(' /// ').str[0].str.strip()
probe2gene=dict(zip(ann.ID.astype(str),ann['Gene Symbol'].astype(str)))

def load(acc):
    m=read_geo_metadata(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz'))
    e=read_geo_expression(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz')).set_index('feature_id')
    if acc=='GSE36059':
        lab=m['diagnosis_tcmr_abmr_mixed_non_rejecting'].astype(str)
        m=m[lab.isin(['TCMR','ABMR','MIXED','non-rejecting'])].copy()
        m['y']=(m['diagnosis_tcmr_abmr_mixed_non_rejecting']=='TCMR').astype(int)
    else:
        lab=m['diagnosis_tcmr_non_tcmr_nephrectomies'].astype(str)
        m=m[lab.isin(['TCMR','non-TCMR'])].copy()
        m['y']=(m['diagnosis_tcmr_non_tcmr_nephrectomies']=='TCMR').astype(int)
    sam=[s for s in m.geo_accession if s in e.columns]
    m=m.set_index('geo_accession').loc[sam]
    sub=e[sam].copy();sub['gene']=[probe2gene.get(str(i),'') for i in sub.index];sub=sub[sub.gene!='']
    ge=sub.groupby('gene',sort=False).median(numeric_only=True)
    return m,ge,sam

md,gd,sd=load('GSE36059')
me,ge,se=load('GSE48581')
assert len(md)==403 and md.y.sum()==35 and len(me)==300 and me.y.sum()==32
Xd=gd.loc[genes,sd].T.to_numpy(float);yd=md.y.to_numpy(int)
Xe=ge.loc[genes,se].T.to_numpy(float);ye=me.y.to_numpy(int)
med=np.nanmedian(Xd,axis=0)
for A in [Xd,Xe]:
    rr,cc=np.where(~np.isfinite(A));A[rr,cc]=med[cc]
gidx={g:i for i,g in enumerate(genes)}

def eval_support(gs):
    ids=np.asarray([gidx[g] for g in gs],int)
    sc=StandardScaler().fit(Xd[:,ids])
    mdl=LogisticRegression(C=1.,penalty='l2',solver='liblinear',class_weight='balanced',
                           max_iter=3000,random_state=20261044).fit(sc.transform(Xd[:,ids]),yd)
    p=mdl.predict_proba(sc.transform(Xe[:,ids]))[:,1]
    return {'auroc':float(roc_auc_score(ye,p)),'auprc':float(average_precision_score(ye,p)),
            'balanced_accuracy':float(balanced_accuracy_score(ye,(p>=.5).astype(int)))}

ourmet=pd.read_csv(SRC/'formal_outputs/05_external_evaluation_v3_2/EXTERNAL_METRICS_POINT.csv')
out=[{'method':'LLM-Score (DeepSeek)','k':50,'selected_features':'|'.join(top50),**eval_support(top50)}]
for srcmeth,label in [('Reference','Reference'),('selective','Selective Correction (ours)')]:
    r=ourmet[ourmet.method==srcmeth].iloc[0]
    support=(SRC/f'formal_outputs/04_pre_external_final_freeze_v3_2/{srcmeth}_TOP50_FROZEN.txt').read_text().split()
    out.append({'method':label,'k':50,'selected_features':'|'.join(support),
                'auroc':float(r.auroc),'auprc':float(r.auprc),'balanced_accuracy':float(r.balanced_accuracy)})
res=pd.DataFrame(out)
ref=float(res.loc[res.method=='Reference','auroc'].iloc[0]);ours=float(res.loc[res.method=='Selective Correction (ours)','auroc'].iloc[0])
res['delta_vs_reference_auroc']=res.auroc-ref;res['delta_vs_ours_auroc']=res.auroc-ours
res.to_csv(OUT/'RENAL_TCMR_BASELINE_COMPARISON.csv',index=False)
manifest={'status':'POST_HOC_SUPPLEMENTAL_BASELINE','requested_model':MODEL,'temperature':0,'thinking':'disabled',
          'method':'pointwise LLM-Score on all 2000 frozen candidate genes; fixed seeded tie-break; top50',
          'llm_input':'task semantics + gene symbol only; no patient data, fitted statistics, retrieved evidence, web, or tools',
          'evaluation':'same original StandardScaler + L2 logistic C=1 balanced evaluator; development GSE36059 -> GSE48581 external',
          'external_outcome_used_for_feature_selection_or_tuning':False,
          'caveat':'baseline designed after original external evaluation, therefore supplemental/post-hoc comparison'}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(res[['method','auroc','delta_vs_reference_auroc','auprc','balanced_accuracy']].to_string(index=False),flush=True)
