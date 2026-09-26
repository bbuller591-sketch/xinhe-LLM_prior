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
import argparse,json,re,time,urllib.request,sys,io,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
warnings.filterwarnings('ignore')
ROOT=Path(str(REPRO_ROOT));BASEOUT=ROOT/'CROSS_DATASET_RECENT_LLM_FS_20260926'
ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['darmanis','breast'],required=True);args=ap.parse_args();ds=args.dataset
OUT={'darmanis':BASEOUT/'DARMANIS_GBM','breast':BASEOUT/'BREAST_PCR'}[ds];OUT.mkdir(parents=True,exist_ok=True)
if ds=='darmanis':
 B=ROOT/'gbm_canonical_recovery_20260917';X=np.load(B/'canonical/X_candidate_aligned_log1p.npy').astype(float);y=pd.read_csv(B/'canonical/y_canonical.csv')['y'].to_numpy(int);f=pd.read_csv(B/'canonical/candidate_features_canonical.csv');names=(f.gene_symbol.fillna('').astype(str).where(f.gene_symbol.fillna('').astype(str).str.len()>0,f.ensembl_gene_id_stable.astype(str))).tolist();mp=pd.read_csv(B/'canonical/cell_sample_mapping_canonical.csv');ann=pd.read_csv(B/'metadata/darmanis_cell_annotation.csv')[['cell_key','plate']];groups=mp.merge(ann,left_on='cell_id',right_on='cell_key',validate='one_to_one')['plate'].astype(str).to_numpy();outer=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026091817).split(X,y,groups));ks=[10,20];task='classify glioblastoma neoplastic tumor-core cells versus tumor-periphery cells from gene expression';solver='lbfgs';seed=2026091817;Xe=ye=None
else:
 B=ROOT/'GSE25055_GSE25065_FORMAL_REPRODUCIBILITY_PACKAGE/DATA/FROZEN/BREAST_GSE25055_GSE25065';X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);Xe=np.load(B/'X_sealed_validation.npy').astype(float);ye=np.load(B/'y_sealed_validation.npy').astype(int);names=pd.read_csv(B/'features_p2000.csv').gene_symbol.astype(str).tolist();outer=[(np.arange(len(y)),None)];ks=[20];task='predict pathological complete response versus residual disease after neoadjuvant breast-cancer therapy from pretreatment tumor gene expression';solver='lbfgs';seed=2026091901
keys=[f'g{i:04d}' for i in range(len(names))];mapping=dict(zip(keys,names));listing='\n'.join(f'[{k}] {n}' for k,n in zip(keys,names))
env={}
for ln in (ROOT/'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918/LOCAL_SECRETS/deepseek_api.env').read_text().splitlines():
 if '=' in ln and not ln.lstrip().startswith('#'):k,v=ln.split('=',1);env[k.strip()]=v.strip()
BASE=env.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/');MODEL=env.get('DEEPSEEK_MODEL','deepseek-flash');OP=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}))
maxk=max(ks);prompt=f'''Task: {task}.
Below is the full frozen candidate gene universe. Using pretrained biomedical knowledge only, select the {maxk} genes most likely to be useful for the task. Do not use tools, web access, retrieved evidence, or dataset values. Return ONLY the {maxk} bracketed keys in ranked order, one per line, without explanations.
{listing}'''
body=json.dumps({'model':MODEL,'messages':[{'role':'system','content':'Rank genes by pretrained biomedical knowledge only. Return only requested keys.'},{'role':'user','content':prompt}],'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':2000,'stream':False}).encode();last=None
for a in range(8):
 try:
  req=urllib.request.Request(BASE+'/chat/completions',data=body,headers={'Authorization':'Bearer '+env['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
  with OP.open(req,timeout=240) as q:o=json.loads(q.read().decode())
  txt=o['choices'][0]['message'].get('content','');break
 except Exception as e:last=repr(e);time.sleep(min(2**a,30))
else:raise RuntimeError(last)
selkeys=[]
for k in re.findall(r'g\d{4}',txt):
 if k in mapping and k not in selkeys:selkeys.append(k)
if len(selkeys)<maxk:raise RuntimeError(f'parsed only {len(selkeys)} keys')
order=[names[int(k[1:])] for k in selkeys[:maxk]]
(OUT/'LLM_RANK_RAW.json').write_text(json.dumps({'prompt':prompt,'response':txt,'selected_keys':selkeys[:maxk],'selected_genes':order,'model':o.get('model')},ensure_ascii=False,indent=2)+'\n')
idx={n:i for i,n in enumerate(names)}
def ev(A,ya,B,yb,gs,rs):
 cols=np.array([idx[g] for g in gs]);sc=StandardScaler().fit(A[:,cols]);m=LogisticRegression(C=1.,penalty='l2',solver=solver,class_weight='balanced',max_iter=5000,random_state=rs).fit(sc.transform(A[:,cols]),ya);p=m.predict_proba(sc.transform(B[:,cols]))[:,1];return float(roc_auc_score(yb,p)),float(average_precision_score(yb,p)),float(balanced_accuracy_score(yb,p>=.5))
rows=[]
for k in ks:
 gs=order[:k]
 for fi,(tr,va) in enumerate(outer,1):
  if va is None:auc,apv,ba=ev(X[tr],y[tr],Xe,ye,gs,seed)
  else:auc,apv,ba=ev(X[tr],y[tr],X[va],y[va],gs,seed+fi)
  rows.append({'dataset':ds,'method':'LLM-Select Rank-style','k':k,'fold':fi,'auroc':auc,'ap':apv,'balanced_accuracy':ba,'selected_features':'|'.join(gs)})
res=pd.DataFrame(rows);res.to_csv(OUT/'LLM_RANK_FOLD_RESULTS.csv',index=False);agg=res.groupby(['dataset','method','k'],as_index=False).agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),mean_ap=('ap','mean'));agg.to_csv(OUT/'LLM_RANK_RESULT.csv',index=False);print(agg.to_string(index=False))
