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
import argparse,json,re,time,urllib.request,threading,io,sys,warnings
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
warnings.filterwarnings('ignore')
ROOT=Path(str(REPRO_ROOT));BASEOUT=ROOT/'CROSS_DATASET_RECENT_LLM_FS_20260926'
ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['darmanis','sepsis','breast','renal'],required=True);ap.add_argument('--shots',type=int,default=32);ap.add_argument('--workers',type=int,default=48);args=ap.parse_args();ds=args.dataset
OUT={'darmanis':BASEOUT/'DARMANIS_GBM','sepsis':BASEOUT/'GSE272769_SEPSIS','breast':BASEOUT/'BREAST_PCR','renal':BASEOUT/'RENAL_TCMR'}[ds]/f'DATACENTRIC_{args.shots}';OUT.mkdir(parents=True,exist_ok=True)

def load():
    if ds=='darmanis':
        B=ROOT/'gbm_canonical_recovery_20260917';X=np.load(B/'canonical/X_candidate_aligned_log1p.npy').astype(float);y=pd.read_csv(B/'canonical/y_canonical.csv')['y'].to_numpy(int);f=pd.read_csv(B/'canonical/candidate_features_canonical.csv');names=(f.gene_symbol.fillna('').astype(str).where(f.gene_symbol.fillna('').astype(str).str.len()>0,f.ensembl_gene_id_stable.astype(str))).tolist();mp=pd.read_csv(B/'canonical/cell_sample_mapping_canonical.csv');ann=pd.read_csv(B/'metadata/darmanis_cell_annotation.csv')[['cell_key','plate']];groups=mp.merge(ann,left_on='cell_id',right_on='cell_key',validate='one_to_one')['plate'].astype(str).to_numpy();outer=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026091817).split(X,y,groups));return X,y,None,None,names,outer,[10,20],'lbfgs',2026091817
    if ds=='sepsis':
        B=ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/01_FROZEN_DATA';X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);names=pd.read_csv(B/'features_p1500.csv').gene_symbol.astype(str).tolist();sp=pd.read_csv(B/'OUTER_SPLITS.csv');outer=[]
        for f in sorted(sp.fold.unique()):outer.append((sp[(sp.fold==f)&(sp.role=='outer_train')].sample_index.astype(int).to_numpy(),sp[(sp.fold==f)&(sp.role=='outer_validation')].sample_index.astype(int).to_numpy()))
        return X,y,None,None,names,outer,[50],'lbfgs',20260919
    if ds=='breast':
        B=ROOT/'GSE25055_GSE25065_FORMAL_REPRODUCIBILITY_PACKAGE/DATA/FROZEN/BREAST_GSE25055_GSE25065';X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);Xe=np.load(B/'X_sealed_validation.npy').astype(float);ye=np.load(B/'y_sealed_validation.npy').astype(int);names=pd.read_csv(B/'features_p2000.csv').gene_symbol.astype(str).tolist();return X,y,Xe,ye,names,[(np.arange(len(y)),None)],[20],'lbfgs',2026091901
    SRC=ROOT/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR';names=pd.read_csv(SRC/'CANDIDATE_UNIVERSE_FROZEN.csv').gene.astype(str).tolist();cache=BASEOUT/'RENAL_TCMR/RENAL_FULL_MATRICES.npz'
    if cache.exists():z=np.load(cache);X=z['X'];y=z['y'];Xe=z['Xe'];ye=z['ye']
    else:
        BASE=ROOT/'dataset_screening_20260921';sys.path.insert(0,str(BASE/'src'));from geo_utils import read_geo_metadata,read_geo_expression
        txt=(BASE/'data/GPL570_full.txt').read_text(errors='replace').replace('\r','');block=txt.split('!platform_table_begin\n',1)[1].split('!platform_table_end',1)[0];ann=pd.read_csv(io.StringIO(block),sep='\t',dtype=str)[['ID','Gene Symbol']].dropna();ann=ann[~ann['Gene Symbol'].isin(['---',''])];ann['Gene Symbol']=ann['Gene Symbol'].str.split(' /// ').str[0].str.strip();p2g=dict(zip(ann.ID.astype(str),ann['Gene Symbol'].astype(str)))
        def ld(acc):
            m=read_geo_metadata(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz'));e=read_geo_expression(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz')).set_index('feature_id')
            if acc=='GSE36059':
                lab=m['diagnosis_tcmr_abmr_mixed_non_rejecting'].astype(str);m=m[lab.isin(['TCMR','ABMR','MIXED','non-rejecting'])].copy();m['y']=(m['diagnosis_tcmr_abmr_mixed_non_rejecting']=='TCMR').astype(int)
            else:
                lab=m['diagnosis_tcmr_non_tcmr_nephrectomies'].astype(str);m=m[lab.isin(['TCMR','non-TCMR'])].copy();m['y']=(m['diagnosis_tcmr_non_tcmr_nephrectomies']=='TCMR').astype(int)
            sam=[q for q in m.geo_accession if q in e.columns];m=m.set_index('geo_accession').loc[sam];sub=e[sam].copy();sub['gene']=[p2g.get(str(i),'') for i in sub.index];sub=sub[sub.gene!=''];ge=sub.groupby('gene',sort=False).median(numeric_only=True);return m,ge,sam
        md,gd,sd=ld('GSE36059');me,ge,se=ld('GSE48581');X=gd.loc[names,sd].T.to_numpy(float);y=md.y.to_numpy(int);Xe=ge.loc[names,se].T.to_numpy(float);ye=me.y.to_numpy(int);med=np.nanmedian(X,axis=0)
        for A in [X,Xe]:
            rr,cc=np.where(~np.isfinite(A));A[rr,cc]=med[cc]
        np.savez_compressed(cache,X=X,y=y,Xe=Xe,ye=ye)
    return X,y,Xe,ye,names,[(np.arange(len(y)),None)],[50],'liblinear',20261044
X,y,Xe,ye,names,outer,ks,solver,seed=load();assert X.shape[1]==len(names)

TASK={'darmanis':'classify glioblastoma neoplastic tumor-core cells versus tumor-periphery cells from gene expression',
      'sepsis':'predict 30-day mortality among adult ICU sepsis patients from whole-blood gene expression',
      'breast':'predict pathological complete response versus residual disease after neoadjuvant breast-cancer therapy from pretreatment tumor gene expression',
      'renal':'classify T-cell-mediated rejection (TCMR) versus non-TCMR renal-allograft biopsy states from gene expression'}[ds]

env={}
for ln in (ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env').read_text().splitlines():
    if '=' in ln and not ln.lstrip().startswith('#'):k,v=ln.split('=',1);env[k.strip()]=v.strip()
BASE=env.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/');MODEL=env.get('DEEPSEEK_MODEL','deepseek-flash');proxy='http://127.0.0.1:7890';OP=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
def call(prompt,retries=8):
    body=json.dumps({'model':MODEL,'messages':[{'role':'system','content':'Estimate feature importance from the supplied development-only feature/target observations and feature identity. Do not use tools or web access.'},{'role':'user','content':prompt}],'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':260,'stream':False}).encode();last=None
    for a in range(retries):
        try:
            req=urllib.request.Request(BASE+'/chat/completions',data=body,headers={'Authorization':'Bearer '+env['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
            with OP.open(req,timeout=180) as f:o=json.loads(f.read().decode())
            return {'content':o['choices'][0]['message'].get('content',''),'model':o.get('model'),'usage':o.get('usage',{})}
        except Exception as e:last=repr(e);time.sleep(min(2**a,30))
    raise RuntimeError(last)

def common_eval(Xtr,ytr,Xv,yv,cols,rs):
    sc=StandardScaler().fit(Xtr[:,cols]);m=LogisticRegression(C=1.,penalty='l2',solver=solver,class_weight='balanced',max_iter=5000,random_state=rs).fit(sc.transform(Xtr[:,cols]),ytr);p=m.predict_proba(sc.transform(Xv[:,cols]))[:,1];return float(roc_auc_score(yv,p)),float(average_precision_score(yv,p)),float(balanced_accuracy_score(yv,p>=.5))

rows=[];all_selected={}
for fi,(tr,va) in enumerate(outer,1):
    tr=np.asarray(tr,int);rng=np.random.default_rng(seed+fi*101+args.shots);pos=tr[y[tr]==1].copy();neg=tr[y[tr]==0].copy();rng.shuffle(pos);rng.shuffle(neg);npo=max(1,round(args.shots*len(pos)/len(tr)));sample=np.r_[pos[:npo],neg[:args.shots-npo]];rng.shuffle(sample)
    ck=OUT/f'fold{fi}_RAW.jsonl';done={}
    if ck.exists():
        for ln in ck.read_text().splitlines():
            try:z=json.loads(ln);done[int(z['index'])]=z
            except:pass
    lock=threading.Lock()
    def one(j):
        pairs=', '.join(f'({X[i,j]:.6g}, {int(y[i])})' for i in sample)
        p=f'''Task: {TASK}.
Feature/gene: {names[j]}
Here are {len(sample)} DEVELOPMENT/TRAINING-ONLY observations as (feature value, target): {pairs}
Give an importance score from 0 to 1 for predicting the target using both the observed feature-target pattern and feature identity. Return ONLY JSON: {{"score": <number>, "reasoning": "<brief>"}}.'''
        z=call(p);txt=z['content'];m=re.search(r'\{.*\}',txt,re.S)
        try:o=json.loads(m.group(0) if m else txt);sc=float(o['score'])
        except:
            mm=re.search(r'(?i)score[^0-9]*([01](?:\.\d+)?)',txt)
            if not mm:raise RuntimeError(f'parse {j} {names[j]}')
            sc=float(mm.group(1))
        rec={'index':j,'feature':names[j],'score':sc,'fold':fi,'prompt':p,'response':txt,'model':z['model']}
        with lock:
            with ck.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        return rec
    todo=[j for j in range(len(names)) if j not in done];print(ds,'fold',fi,'existing',len(done),'todo',len(todo),flush=True);errs=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(one,j):j for j in todo}
        for ii,q in enumerate(as_completed(fs),1):
            j=fs[q]
            try:done[j]=q.result()
            except Exception as e:errs.append((j,e))
            if ii%200==0:print(' fold',fi,ii,'/',len(todo),'err',len(errs),flush=True)
    for j,e in errs:
        try:done[j]=one(j)
        except Exception as ee:print('FINAL_FAIL',fi,j,ee,flush=True)
    if len(done)!=len(names):raise RuntimeError(f'fold {fi} incomplete {len(done)}/{len(names)}')
    scores=np.array([done[j]['score'] for j in range(len(names))]);order=np.lexsort((np.arange(len(scores)),-scores))
    for k in ks:
        cols=order[:k];all_selected[f'fold{fi}_k{k}']=cols.tolist()
        if va is None:auc,apv,ba=common_eval(X[tr],y[tr],Xe,ye,cols,seed+fi)
        else:auc,apv,ba=common_eval(X[tr],y[tr],X[va],y[va],cols,seed+fi)
        rows.append({'dataset':ds,'method':f'Data-centric LLM FS ({args.shots}-shot)','fold':fi,'k':k,'auroc':auc,'ap':apv,'balanced_accuracy':ba,'selected_features':'|'.join(names[j] for j in cols)})
res=pd.DataFrame(rows);res.to_csv(OUT/'DATACENTRIC_FOLD_RESULTS.csv',index=False)
agg=res.groupby(['dataset','method','k'],as_index=False).agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),mean_ap=('ap','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'));agg.to_csv(OUT/'DATACENTRIC_RESULT.csv',index=False)
(OUT/'SELECTED_SUPPORTS.json').write_text(json.dumps(all_selected,indent=2)+'\n')
(OUT/'MANIFEST.json').write_text(json.dumps({'dataset':ds,'shots':args.shots,'fold_specific_for_internal_cv':ds in ['darmanis','sepsis'],'external_outcomes_seen_by_llm':False,'model':MODEL},indent=2)+'\n')
print(agg.to_string(index=False),flush=True)
