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
import argparse,json,re,time,urllib.request,random,math,io,sys,warnings
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
warnings.filterwarnings('ignore')
ROOT=Path(str(REPRO_ROOT));BASEOUT=ROOT/'CROSS_DATASET_RECENT_LLM_FS_20260926'
ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['darmanis','sepsis','breast','renal'],required=True);ap.add_argument('--k',type=int,required=True);ap.add_argument('--workers',type=int,default=24);args=ap.parse_args();ds=args.dataset;k=args.k
OUT={'darmanis':BASEOUT/'DARMANIS_GBM','sepsis':BASEOUT/'GSE272769_SEPSIS','breast':BASEOUT/'BREAST_PCR','renal':BASEOUT/'RENAL_TCMR'}[ds]/f'FREEFORM_K{k}';OUT.mkdir(parents=True,exist_ok=True)

def load():
    if ds=='darmanis':
        B=ROOT/'gbm_canonical_recovery_20260917';X=np.load(B/'canonical/X_candidate_aligned_log1p.npy').astype(float);y=pd.read_csv(B/'canonical/y_canonical.csv')['y'].to_numpy(int);f=pd.read_csv(B/'canonical/candidate_features_canonical.csv');names=(f.gene_symbol.fillna('').astype(str).where(f.gene_symbol.fillna('').astype(str).str.len()>0,f.ensembl_gene_id_stable.astype(str))).tolist();mp=pd.read_csv(B/'canonical/cell_sample_mapping_canonical.csv');ann=pd.read_csv(B/'metadata/darmanis_cell_annotation.csv')[['cell_key','plate']];groups=mp.merge(ann,left_on='cell_id',right_on='cell_key',validate='one_to_one')['plate'].astype(str).to_numpy();outer=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026091817).split(X,y,groups));return X,y,None,None,names,outer,'lbfgs',2026091817
    if ds=='sepsis':
        B=ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/01_FROZEN_DATA';X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);names=pd.read_csv(B/'features_p1500.csv').gene_symbol.astype(str).tolist();sp=pd.read_csv(B/'OUTER_SPLITS.csv');outer=[]
        for f in sorted(sp.fold.unique()):outer.append((sp[(sp.fold==f)&(sp.role=='outer_train')].sample_index.astype(int).to_numpy(),sp[(sp.fold==f)&(sp.role=='outer_validation')].sample_index.astype(int).to_numpy()))
        return X,y,None,None,names,outer,'lbfgs',20260919
    if ds=='breast':
        B=ROOT/'GSE25055_GSE25065_FORMAL_REPRODUCIBILITY_PACKAGE/DATA/FROZEN/BREAST_GSE25055_GSE25065';X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);Xe=np.load(B/'X_sealed_validation.npy').astype(float);ye=np.load(B/'y_sealed_validation.npy').astype(int);names=pd.read_csv(B/'features_p2000.csv').gene_symbol.astype(str).tolist();return X,y,Xe,ye,names,[(np.arange(len(y)),None)],'lbfgs',2026091901
    SRC=ROOT/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR';names=pd.read_csv(SRC/'CANDIDATE_UNIVERSE_FROZEN.csv').gene.astype(str).tolist();cache=BASEOUT/'RENAL_TCMR/RENAL_FULL_MATRICES.npz';z=np.load(cache);return z['X'],z['y'],z['Xe'],z['ye'],names,[(np.arange(len(z['y'])),None)],'liblinear',20261044
X,y,Xe,ye,names,outer,solver,seed=load();assert k<=len(names)
TASK={'darmanis':'classify glioblastoma neoplastic tumor-core cells versus tumor-periphery cells from gene expression',
      'sepsis':'predict 30-day mortality among adult ICU sepsis patients from whole-blood gene expression',
      'breast':'predict pathological complete response versus residual disease after neoadjuvant breast-cancer therapy from pretreatment tumor gene expression',
      'renal':'classify T-cell-mediated rejection (TCMR) versus non-TCMR renal-allograft biopsy states from gene expression'}[ds]
keys=[f'g{i:04d}' for i in range(len(names))];name_of=dict(zip(keys,names));key_of={n:q for q,n in name_of.items()}
env={}
for ln in (ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env').read_text().splitlines():
    if '=' in ln and not ln.lstrip().startswith('#'):a,b=ln.split('=',1);env[a.strip()]=b.strip()
BASE=env.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/');MODEL=env.get('DEEPSEEK_MODEL','deepseek-flash');OP=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}))
def call(prompt,retries=7):
    body=json.dumps({'model':MODEL,'messages':[{'role':'system','content':'Perform hierarchical knowledge-driven feature selection. Do not use tools, web access, retrieved evidence, or dataset values. Return only requested keys.'},{'role':'user','content':prompt}],'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':1800,'stream':False}).encode();last=None
    for a in range(retries):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=body,headers={'Authorization':'Bearer '+env['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
            with OP.open(req,timeout=240) as f:o=json.loads(f.read().decode())
            return {'content':o['choices'][0]['message'].get('content',''),'model':o.get('model')}
        except Exception as e:last=repr(e);time.sleep(min(2**a,30))
    raise RuntimeError(last)

def split_list(xs,bucket_size=75):
    if len(xs)<=bucket_size:return [xs]
    nb=max(round(len(xs)/bucket_size),1);base=len(xs)//nb;r=len(xs)%nb;out=[];st=0
    for i in range(nb):
        en=st+base+(1 if i<r else 0);out.append(xs[st:en]);st=en
    return out

rawpath=OUT/'RAW_CALLS.jsonl'
current=keys.copy();roundno=0
while len(current)>k:
    roundno+=1;rr=random.Random(seed+k*1000+roundno);rr.shuffle(current);buckets=split_list(current,75);final=(len(buckets)==1);retries=12 if final else 3
    print(ds,'k',k,'round',roundno,'n',len(current),'buckets',len(buckets),'retries',retries,flush=True)
    counts=[Counter() for _ in buckets];rank_sum=[Counter() for _ in buckets]
    jobs=[]
    def one(bi,rep,bucket):
        need=min(k,len(bucket));ordered=bucket.copy();random.Random(seed+roundno*100000+bi*100+rep).shuffle(ordered);listing='\n'.join(f'[{q}] {name_of[q]}' for q in ordered)
        p=f'''Task: {TASK}.
From this bucket of candidate genes, select the {need} most relevant features using pretrained biomedical knowledge. This is one stage of a FREEFORM-style hierarchical/pyramid selection with self-consistency.
Return ONLY the {need} bracketed keys, one per line, ranked most to least relevant. Do not add explanations.
{listing}'''
        z=call(p);seen=[]
        for q in re.findall(r'g\d{4}',z['content']):
            if q in bucket and q not in seen:seen.append(q)
        return bi,rep,p,z,seen[:need]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fs=[ex.submit(one,bi,rep,b) for bi,b in enumerate(buckets) for rep in range(retries)]
        for q in as_completed(fs):
            bi,rep,p,z,sel=q.result()
            with rawpath.open('a') as f:f.write(json.dumps({'round':roundno,'bucket':bi,'rep':rep,'prompt':p,'response':z['content'],'parsed':sel},ensure_ascii=False)+'\n')
            for rnk,g in enumerate(sel):counts[bi][g]+=1;rank_sum[bi][g]+=rnk
    nxt=[]
    for bi,b in enumerate(buckets):
        need=min(k,len(b));ordered=sorted(b,key=lambda q:(-counts[bi][q],(rank_sum[bi][q]/counts[bi][q] if counts[bi][q] else 1e9),b.index(q)));nxt.extend(ordered[:need])
    # de-duplicate while preserving hierarchy order
    current=list(dict.fromkeys(nxt))
    if len(current)<k:raise RuntimeError('hierarchical reduction fell below k')
selected=current[:k];selnames=[name_of[q] for q in selected]
(OUT/'SELECTED.json').write_text(json.dumps({'keys':selected,'features':selnames},ensure_ascii=False,indent=2)+'\n')
idx=[int(q[1:]) for q in selected]
rows=[]
for fi,(tr,va) in enumerate(outer,1):
    sc=StandardScaler().fit(X[tr][:,idx]);m=LogisticRegression(C=1.,penalty='l2',solver=solver,class_weight='balanced',max_iter=5000,random_state=seed+fi).fit(sc.transform(X[tr][:,idx]),y[tr])
    if va is None:B,yb=Xe,ye
    else:B,yb=X[va],y[va]
    p=m.predict_proba(sc.transform(B[:,idx]))[:,1];rows.append({'dataset':ds,'method':'FREEFORM-pyramid','k':k,'fold':fi,'auroc':float(roc_auc_score(yb,p)),'ap':float(average_precision_score(yb,p)),'balanced_accuracy':float(balanced_accuracy_score(yb,p>=.5)),'selected_features':'|'.join(selnames)})
df=pd.DataFrame(rows);df.to_csv(OUT/'FOLD_RESULTS.csv',index=False);agg=df.groupby(['dataset','method','k'],as_index=False).agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),mean_ap=('ap','mean'));agg.to_csv(OUT/'RESULT.csv',index=False);(OUT/'MANIFEST.json').write_text(json.dumps({'method':'FREEFORM hierarchical/pyramid adaptation','bucket_size':75,'initial_retries':3,'final_retries':12,'semantic_only':True,'model':MODEL},indent=2)+'\n');print(agg.to_string(index=False))
