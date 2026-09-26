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
import argparse,json,io,sys,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
warnings.filterwarnings('ignore')
ROOT=Path(str(REPRO_ROOT))
ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['darmanis','sepsis','breast','renal'],required=True);args=ap.parse_args();ds=args.dataset
BASEOUT=ROOT/'CROSS_DATASET_RECENT_LLM_FS_20260926'
OUT={'darmanis':BASEOUT/'DARMANIS_GBM','sepsis':BASEOUT/'GSE272769_SEPSIS','breast':BASEOUT/'BREAST_PCR','renal':BASEOUT/'RENAL_TCMR'}[ds];OUT.mkdir(parents=True,exist_ok=True)

def load_dataset():
    if ds=='darmanis':
        B=ROOT/'gbm_canonical_recovery_20260917'
        X=np.load(B/'canonical/X_candidate_aligned_log1p.npy').astype(float);y=pd.read_csv(B/'canonical/y_canonical.csv')['y'].to_numpy(int)
        feat=pd.read_csv(B/'canonical/candidate_features_canonical.csv');names=(feat.gene_symbol.fillna('').astype(str).where(feat.gene_symbol.fillna('').astype(str).str.len()>0,feat.ensembl_gene_id_stable.astype(str))).tolist()
        mp=pd.read_csv(B/'canonical/cell_sample_mapping_canonical.csv');ann=pd.read_csv(B/'metadata/darmanis_cell_annotation.csv')[['cell_key','plate']];groups=mp.merge(ann,left_on='cell_id',right_on='cell_key',validate='one_to_one')['plate'].astype(str).to_numpy()
        outer=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026091817).split(X,y,groups))
        score=pd.read_csv(ROOT/'EXISTING_METHOD_BASELINES_20260925/DARMANIS_GBM/LLM_SCORE_DEEPSEEK_2000.csv').sort_values('node');s=score.score.to_numpy(float)
        return {'kind':'internal','X':X,'y':y,'names':names,'groups':groups,'outer':outer,'score':s,'ks':[10,20],'solver':'lbfgs','seed':2026091817}
    if ds=='sepsis':
        B=ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/01_FROZEN_DATA'
        X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);names=pd.read_csv(B/'features_p1500.csv').gene_symbol.astype(str).tolist();sp=pd.read_csv(B/'OUTER_SPLITS.csv');outer=[]
        for f in sorted(sp.fold.unique()):
            tr=sp[(sp.fold==f)&(sp.role=='outer_train')].sample_index.astype(int).to_numpy();va=sp[(sp.fold==f)&(sp.role=='outer_validation')].sample_index.astype(int).to_numpy();outer.append((tr,va))
        score=pd.read_csv(OUT/'LLM_SCORE/LLM_SCORE.csv').sort_values('index');s=score.score.to_numpy(float)
        return {'kind':'internal','X':X,'y':y,'names':names,'groups':None,'outer':outer,'score':s,'ks':[50],'solver':'lbfgs','seed':20260919}
    if ds=='breast':
        B=ROOT/'GSE25055_GSE25065_FORMAL_REPRODUCIBILITY_PACKAGE/DATA/FROZEN/BREAST_GSE25055_GSE25065'
        X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);Xe=np.load(B/'X_sealed_validation.npy').astype(float);ye=np.load(B/'y_sealed_validation.npy').astype(int);names=pd.read_csv(B/'features_p2000.csv').gene_symbol.astype(str).tolist()
        score=pd.read_csv(OUT/'LLM_SCORE/LLM_SCORE.csv').sort_values('index');s=score.score.to_numpy(float)
        return {'kind':'external','X':X,'y':y,'Xe':Xe,'ye':ye,'names':names,'groups':None,'score':s,'ks':[20],'solver':'lbfgs','seed':2026091901}
    # renal
    SRC=ROOT/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR';BASE=ROOT/'dataset_screening_20260921';cand=pd.read_csv(SRC/'CANDIDATE_UNIVERSE_FROZEN.csv');names=cand.gene.astype(str).tolist()
    cache=OUT/'RENAL_FULL_MATRICES.npz'
    if cache.exists():
        z=np.load(cache);X=z['X'];y=z['y'];Xe=z['Xe'];ye=z['ye']
    else:
        sys.path.insert(0,str(BASE/'src'));from geo_utils import read_geo_metadata,read_geo_expression
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
    score=pd.read_csv(ROOT/'EXISTING_METHOD_BASELINES_20260925/RENAL_TCMR/LLM_SCORE_DEEPSEEK_2000.csv');score=score.set_index('gene').loc[names].reset_index();s=score.score.to_numpy(float)
    return {'kind':'external','X':X,'y':y,'Xe':Xe,'ye':ye,'names':names,'groups':None,'score':s,'ks':[50],'solver':'liblinear','seed':20261044}
D=load_dataset();X=D['X'];y=D['y'];names=D['names'];score=np.clip(np.asarray(D['score'],float),.05,1.0);p=X.shape[1];assert p==len(score)==len(names)

def common_eval(Xtr,ytr,Xv,yv,cols,seed):
    sc=StandardScaler().fit(Xtr[:,cols]);A=sc.transform(Xtr[:,cols]);B=sc.transform(Xv[:,cols])
    m=LogisticRegression(C=1.0,penalty='l2',solver=D['solver'],class_weight='balanced',max_iter=5000,random_state=seed).fit(A,ytr);pr=m.predict_proba(B)[:,1]
    return float(roc_auc_score(yv,pr)),float(average_precision_score(yv,pr)),float(balanced_accuracy_score(yv,pr>=.5))

def weighted_support(Xtr,ytr,k,exp,C):
    sc=StandardScaler().fit(Xtr);Z=sc.transform(Xtr);w=(1.0/score)**exp;w/=w.mean()
    m=LogisticRegression(C=float(C),penalty='l1',solver='liblinear',class_weight='balanced',max_iter=5000).fit(Z/w,ytr);beta=m.coef_.ravel()/w;active=np.flatnonzero(np.abs(beta)>1e-10)
    if len(active)<k:return None,len(active)
    top=active[np.argsort(-np.abs(beta[active]))[:k]]
    return top,len(active)

EXPS=[0,1,2,3];CS=[0.01,0.03,0.1,0.3,1.0,3.0]
def tune(train_idx,k,groups=None):
    Xt=X[train_idx];yt=y[train_idx];gg=None if groups is None else groups[train_idx]
    if gg is None:inner=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=D['seed']+37+k).split(Xt,yt))
    else:inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=D['seed']+37+k).split(Xt,yt,gg))
    rows=[]
    for exp in EXPS:
        for C in CS:
            vals=[];ok=True
            for ii,(a,b) in enumerate(inner):
                cols,na=weighted_support(Xt[a],yt[a],k,exp,C)
                if cols is None:ok=False;break
                auc,ap,ba=common_eval(Xt[a],yt[a],Xt[b],yt[b],cols,D['seed']+ii);vals.append(auc)
            if ok:rows.append({'exp':exp,'C':C,'mean_auc':float(np.mean(vals))})
    if not rows:raise RuntimeError('no viable weighted-l1 config')
    df=pd.DataFrame(rows);best=df.sort_values(['mean_auc','exp','C'],ascending=[False,True,True]).iloc[0]
    return int(best.exp),float(best.C),float(best.mean_auc),df

allrows=[];grids=[]
if D['kind']=='internal':
    for k in D['ks']:
        for fi,(tr,va) in enumerate(D['outer'],1):
            exp,C,cvauc,g=tune(np.asarray(tr),k,D['groups']);cols,na=weighted_support(X[tr],y[tr],k,exp,C);auc,apv,ba=common_eval(X[tr],y[tr],X[va],y[va],cols,D['seed']+fi)
            allrows.append({'dataset':ds,'method':'LLM-Lasso','k':k,'fold':fi,'exp':exp,'C':C,'inner_cv_auroc':cvauc,'active_groups':na,'auroc':auc,'ap':apv,'balanced_accuracy':ba,'selected_indices':'|'.join(map(str,cols.tolist())),'selected_features':'|'.join(names[j] for j in cols)})
            g['k']=k;g['outer_fold']=fi;grids.append(g)
    fold=pd.DataFrame(allrows);agg=fold.groupby(['dataset','method','k'],as_index=False).agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),mean_ap=('ap','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'))
    fold.to_csv(OUT/'LLMLASSO_FOLD_RESULTS.csv',index=False);pd.concat(grids,ignore_index=True).to_csv(OUT/'LLMLASSO_INNER_GRIDS.csv',index=False);agg.to_csv(OUT/'LLMLASSO_RESULT.csv',index=False);print(agg.to_string(index=False))
else:
    allidx=np.arange(len(y))
    for k in D['ks']:
        exp,C,cvauc,g=tune(allidx,k,None);cols,na=weighted_support(X,y,k,exp,C);auc,apv,ba=common_eval(X,y,D['Xe'],D['ye'],cols,D['seed'])
        allrows.append({'dataset':ds,'method':'LLM-Lasso','k':k,'exp':exp,'C':C,'dev_cv_auroc':cvauc,'active_groups':na,'auroc':auc,'ap':apv,'balanced_accuracy':ba,'selected_indices':'|'.join(map(str,cols.tolist())),'selected_features':'|'.join(names[j] for j in cols)})
        g['k']=k;grids.append(g)
    res=pd.DataFrame(allrows);res.to_csv(OUT/'LLMLASSO_RESULT.csv',index=False);pd.concat(grids,ignore_index=True).to_csv(OUT/'LLMLASSO_DEV_GRID.csv',index=False);print(res.to_string(index=False))
(OUT/'LLMLASSO_MANIFEST.json').write_text(json.dumps({'dataset':ds,'method':'LLM-Lasso fixed-k adaptation','score_source':'pointwise LLM-Score','exponents':EXPS,'C_grid':CS,'inner_selection':'development/outer-train only','external_or_outer_validation_used_for_tuning':False},indent=2)+'\n')
