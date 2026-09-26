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
from datetime import datetime,timezone
import hashlib,json,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
PRE=ROOT/'20_PRE_SEALED_FREEZE'
OUT=ROOT/'21_SEALED_VALIDATION'
OUT.mkdir(parents=True,exist_ok=True)
EXPECTED='cb17a3702dec69bcfed8fa2c48bff0d4da91d044b391cce2d103f1bc2c444c45'
SEED=2026091901

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

if sha(PRE/'PRE_SEALED_MANIFEST.csv')!=EXPECTED:
    raise RuntimeError('PRE_SEALED_MANIFEST_HASH_MISMATCH')
pst=json.load(open(PRE/'PRE_SEALED_STATUS.json'))
if not pst.get('all_tuning_locked_before_sealed') or pst.get('sealed_validation_utility_used_before_this_freeze'):
    raise RuntimeError('PRE_SEALED_STATUS_INVALID')

event={'status':'SEALED_VALIDATION_OPENED_FOR_ONE_TIME_EVALUATION',
       'timestamp_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
       'pre_sealed_manifest_sha256':EXPECTED,
       'rule':'No tuning or support changes after this event; test used only for final evaluation.'}
(OUT/'SEALED_OPEN_EVENT.json').write_text(json.dumps(event,indent=2),encoding='utf-8')

def sis_score(X,y):
    yc=y-y.mean(); xc=X-X.mean(axis=0)
    den=np.sqrt((xc*xc).sum(axis=0)*(yc*yc).sum())
    return np.nan_to_num(np.abs((xc*yc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0)

def sparse_score(X,y,C,ratio):
    sc=StandardScaler().fit(X); Z=sc.transform(X)
    pen='l1' if float(ratio)==1.0 else 'elasticnet'
    m=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=pen,
        l1_ratio=None if pen=='l1' else float(ratio),max_iter=5000,tol=1e-4,
        random_state=SEED,fit_intercept=True,n_jobs=1)
    m.fit(Z,y)
    nit=int(m.n_iter_[0]); conv=bool(nit<m.max_iter)
    if not conv: raise RuntimeError('FINAL_reference_SPARSE_NONCONVERGENCE')
    return np.abs(m.coef_[0]),nit

def stable_topk(D,k):
    return np.lexsort((np.arange(len(D)),-D))[:k]

def downstream(Xtr,ytr,Xte,yte,cols):
    sc=StandardScaler().fit(Xtr[:,cols])
    A=sc.transform(Xtr[:,cols]); B=sc.transform(Xte[:,cols])
    m=LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',class_weight='balanced',
                         max_iter=5000,random_state=SEED)
    m.fit(A,ytr)
    p=m.predict_proba(B)[:,1]
    app=average_precision_score(yte,p); apn=average_precision_score(1-yte,1-p)
    return {
      'auroc':float(roc_auc_score(yte,p)),
      'ap_positive':float(app),'ap_negative':float(apn),'macro_ap':float((app+apn)/2),
      'balanced_accuracy':float(balanced_accuracy_score(yte,p>=0.5)),
      'test_prevalence':float(yte.mean()),
      'downstream_iter':int(m.n_iter_[0]),
      'probabilities':p
    }

all_rows=[]; pred_rows=[]; support_rows=[]
for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
    src=ROOT/'03_FROZEN_DATA'/task
    # This is the first utility use of these sealed files after PRE_SEALED freeze.
    Xtr=np.load(src/'X_development.npy').astype(float)
    ytr=np.load(src/'y_development.npy').astype(int)
    Xte=np.load(src/'X_sealed_validation.npy').astype(float)
    yte=np.load(src/'y_sealed_validation.npy').astype(int)
    feat=pd.read_csv(src/'features_p2000.csv')
    if Xtr.shape[1]!=2000 or Xte.shape[1]!=2000: raise RuntimeError('P_MISMATCH')
    if len(feat)!=2000: raise RuntimeError('FEATURE_COUNT_MISMATCH')
    pars=pd.read_csv(ROOT/'13_FINAL_DEVELOPMENT_TUNING'/task/'FINAL_SELECTOR_PARAMS.csv')
    m3sel=pd.read_csv(ROOT/'13_FINAL_DEVELOPMENT_TUNING'/task/'FINAL_selective_SELECTED_FEATURES.csv')
    m12d20=pd.read_csv(ROOT/'17_global_INTERNAL/D20'/task/'global_FINAL_SELECTED_FEATURES.csv')
    m12d10=pd.read_csv(ROOT/'17_global_INTERNAL/D10'/task/'global_FINAL_SELECTED_FEATURES.csv')
    strictp=ROOT/'19_global_STRICT_CONTENT'/task/'global_FINAL_SELECTED_FEATURES.csv'
    strict=pd.read_csv(strictp) if strictp.exists() else None

    # Frozen reference full-development support.
    m0support={}
    for sel in ['LASSO','ELASTICNET','SIS']:
        rr=pars[pars.selector==sel].iloc[0]
        if sel=='SIS': D=sis_score(Xtr,ytr); sit=0
        else: D,sit=sparse_score(Xtr,ytr,float(rr.C),float(rr.l1_ratio))
        for k in [10,20,30]:
            cols=stable_topk(D,k)
            m0support[(sel,k)]=cols
            for rank,j in enumerate(cols,1):
                support_rows.append({'task':task,'method':'reference','selector':sel,'k':k,'rank':rank,
                                     'feature_index':int(j),'gene_symbol':str(feat.iloc[j].gene_symbol),
                                     'tuning_strength':0.0})

    supports={}
    # Primary reference.
    for key,v in m0support.items(): supports[('reference',*key)]=v
    # Primary selective.
    for (sel,k),g in m3sel.groupby(['selector','k']):
        g=g.sort_values('rank')
        cols=g.feature_index.to_numpy(int)
        if len(cols)!=int(k): raise RuntimeError('selective_SUPPORT_SIZE')
        supports[('selective',str(sel),int(k))]=cols
        lam=float(g.chosen_lam.iloc[0])
        for rank,j in enumerate(cols,1):
            support_rows.append({'task':task,'method':'selective','selector':sel,'k':int(k),'rank':rank,
                                 'feature_index':int(j),'gene_symbol':str(feat.iloc[j].gene_symbol),
                                 'tuning_strength':lam})
    # Primary d20 global/global_certainty.
    for r in m12d20.itertuples():
        cols=np.array([int(x) for x in str(r.selected_indices).split('|')],int)
        supports[(str(r.method),str(r.selector),int(r.k))]=cols
        for rank,j in enumerate(cols,1):
            support_rows.append({'task':task,'method':str(r.method),'selector':str(r.selector),'k':int(r.k),'rank':rank,
                                 'feature_index':int(j),'gene_symbol':str(feat.iloc[j].gene_symbol),
                                 'tuning_strength':float(r.chosen_lam)})
    # Sensitivity supports.
    for r in m12d10.itertuples():
        cols=np.array([int(x) for x in str(r.selected_indices).split('|')],int)
        supports[(str(r.method)+'_D10',str(r.selector),int(r.k))]=cols
    if strict is not None:
        for r in strict.itertuples():
            cols=np.array([int(x) for x in str(r.selected_indices).split('|')],int)
            supports[(str(r.method)+'_STRICT_CONTENT',str(r.selector),int(r.k))]=cols

    # Evaluate each frozen support independently, with no test-based selection.
    methods_primary=['reference','global','global_certainty','selective']
    methods_sens=['global_D10','global_certainty_D10'] + (['global_STRICT_CONTENT','global_certainty_STRICT_CONTENT'] if strict is not None else [])
    for method in methods_primary+methods_sens:
        for sel in ['LASSO','ELASTICNET','SIS']:
            for k in [10,20,30]:
                key=(method,sel,k)
                if key not in supports: continue
                cols=supports[key]
                if len(cols)!=k: raise RuntimeError(f'SUPPORT_SIZE {task} {key} {len(cols)}')
                met=downstream(Xtr,ytr,Xte,yte,cols)
                row={'task':task,'analysis':'PRIMARY' if method in methods_primary else 'SENSITIVITY',
                     'method':method,'selector':sel,'k':k,
                     'n_development':len(ytr),'n_test':len(yte),'test_positive':int(yte.sum()),
                     **{x:v for x,v in met.items() if x!='probabilities'},
                     'selected_indices':'|'.join(map(str,cols.tolist())),
                     'selected_genes':'|'.join(feat.iloc[cols].gene_symbol.astype(str).tolist())}
                all_rows.append(row)
                for i,(yy,pp) in enumerate(zip(yte,met['probabilities'])):
                    pred_rows.append({'task':task,'analysis':row['analysis'],'method':method,'selector':sel,'k':k,
                                      'test_index':i,'y_true':int(yy),'p_positive':float(pp)})

res=pd.DataFrame(all_rows)
# Matched primary deltas to reference, selector x k.
base=res[(res.analysis=='PRIMARY')&(res.method=='reference')][['task','selector','k','auroc','macro_ap','balanced_accuracy']].rename(
    columns={'auroc':'reference_auroc','macro_ap':'reference_macro_ap','balanced_accuracy':'reference_balanced_accuracy'})
res=res.merge(base,on=['task','selector','k'],how='left',validate='many_to_one')
res['delta_auroc_vs_reference']=res.auroc-res.reference_auroc
res['delta_macro_ap_vs_reference']=res.macro_ap-res.reference_macro_ap
res['delta_balanced_accuracy_vs_reference']=res.balanced_accuracy-res.reference_balanced_accuracy
res.to_csv(OUT/'SEALED_VALIDATION_ALL_RESULTS.csv',index=False)
pd.DataFrame(pred_rows).to_parquet(OUT/'SEALED_VALIDATION_PREDICTIONS.parquet',index=False,compression='zstd')
pd.DataFrame(support_rows).to_csv(OUT/'FROZEN_PRIMARY_SUPPORTS_USED.csv',index=False)

primary=res[res.analysis=='PRIMARY'].copy()
primary.to_csv(OUT/'SEALED_VALIDATION_PRIMARY_RESULTS.csv',index=False)
summary=(primary.groupby(['task','method'],as_index=False)
         .agg(mean_auroc=('auroc','mean'),median_auroc=('auroc','median'),
              mean_delta_auroc_vs_reference=('delta_auroc_vs_reference','mean'),
              median_delta_auroc_vs_reference=('delta_auroc_vs_reference','median'),
              positive_delta_cells=('delta_auroc_vs_reference',lambda x:int((x>0).sum())),
              zero_delta_cells=('delta_auroc_vs_reference',lambda x:int(np.isclose(x,0,atol=1e-12).sum())),
              negative_delta_cells=('delta_auroc_vs_reference',lambda x:int((x<0).sum())),
              mean_macro_ap=('macro_ap','mean'),
              mean_delta_macro_ap_vs_reference=('delta_macro_ap_vs_reference','mean')))
summary.to_csv(OUT/'SEALED_VALIDATION_PRIMARY_METHOD_SUMMARY.csv',index=False)

status={'status':'SEALED_VALIDATION_COMPLETE_NO_RETUNING',
        'pre_sealed_manifest_sha256':EXPECTED,
        'n_primary_rows':int(len(primary)),'n_all_rows':int(len(res)),
        'tasks':sorted(res.task.unique().tolist()),
        'test_sizes':{t:int(res[res.task==t].n_test.iloc[0]) for t in res.task.unique()},
        'test_positives':{t:int(res[res.task==t].test_positive.iloc[0]) for t in res.task.unique()},
        'no_test_based_tuning':True,'no_post_test_support_changes':True,
        'primary_results_sha256':sha(OUT/'SEALED_VALIDATION_PRIMARY_RESULTS.csv'),
        'all_results_sha256':sha(OUT/'SEALED_VALIDATION_ALL_RESULTS.csv'),
        'predictions_sha256':sha(OUT/'SEALED_VALIDATION_PREDICTIONS.parquet')}
(OUT/'SEALED_VALIDATION_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
print('\nPRIMARY METHOD SUMMARY\n',summary.to_string(index=False))
print('\nPRIMARY CELL RESULTS\n',primary[['task','method','selector','k','auroc','delta_auroc_vs_reference','macro_ap','delta_macro_ap_vs_reference']].to_string(index=False))
