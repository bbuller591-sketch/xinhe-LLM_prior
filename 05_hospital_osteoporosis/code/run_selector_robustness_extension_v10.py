#!/usr/bin/env python3
from __future__ import annotations

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
import json, math, sys, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
DATA=Path(str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917'))
CANON=Path(str(REPRO_ROOT / 'hospital_osteoporosis_canonical_20260917'))
OUT=ROOT/'13_SELECTOR_ROBUSTNESS_EXPLORATORY_V1_0'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(DATA/'scripts'))
import v2_core as V

ALPHAS=[0.25,0.50,0.75]
ENET_LAMS=[float(x) for x in V.LAM_GRID_EXTENDED]
RIDGE_LAMS=[float(x) for x in V.LAM_GRID]
KGRID=[5,10]
SEED_CV=2026091801
SEED_RESAMPLE=2026091802
N_RESAMPLES=200
RESAMPLE_FRAC=0.80

def folds(y,g,seed,n=5):
    return list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))

def metrics(y,p):
    return {'auroc':float(roc_auc_score(y,p)),
            'auprc':float(average_precision_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,(p>=0.5).astype(int)))}

def enet_fit_full(Xraw,site,y,lam,alpha,include_site=True,max_iter=30000,tol=1e-9):
    sc=V.fit_scaler_v2(Xraw)
    Xs=V.apply_scaler_v2(Xraw,sc)
    Xd,mask=V.design(Xs,site,include_site=include_site)
    Xd=np.ascontiguousarray(Xd,float); y=np.asarray(y,float); mask=np.asarray(mask,float)
    L=0.25*V.lambda_max_power(Xd)+2.0*float(lam)*(1.0-float(alpha))
    step=1.0/L if L>0 else 1.0
    w=np.zeros(Xd.shape[1]); z=w.copy(); t=1.0
    l1=float(lam)*float(alpha)*mask
    l2=float(lam)*(1.0-float(alpha))*mask
    for it in range(max_iter):
        g=V.grad_logloss(Xd,y,z)+2.0*l2*z
        v=z-step*g
        wn=np.sign(v)*np.maximum(np.abs(v)-step*l1,0.0)
        # unpenalized coordinates should not be soft-thresholded
        wn[mask==0]=v[mask==0]
        tn=(1+math.sqrt(1+4*t*t))/2
        zn=wn+((t-1)/tn)*(wn-w)
        if np.max(np.abs(wn-w))<tol:
            w=wn; break
        w,z,t=wn,zn,tn
    nctx=V.forced_context_cols(site,include_site)
    return w,sc,nctx

def enet_predict(w,sc,Xraw,site):
    return V.predict_full(w,sc,Xraw,site)

def choose_enet(X,site,y,g):
    rows=[]; best=None
    spl=folds(y,g,SEED_CV,5)
    for alpha in ALPHAS:
        for lam in ENET_LAMS:
            aucs=[]
            for tri,vai in spl:
                w,sc,nctx=enet_fit_full(X[tri],site[tri],y[tri],lam,alpha)
                aucs.append(float(roc_auc_score(y[vai],enet_predict(w,sc,X[vai],site[vai]))))
            m=float(np.mean(aucs)); sd=float(np.std(aucs,ddof=1))
            rows.append({'alpha':alpha,'lambda':lam,'mean_auroc':m,'sd_auroc':sd})
            key=(m,alpha,lam)
            if best is None or key>best[0]:
                best=(key,alpha,lam)
    tab=pd.DataFrame(rows).sort_values(['mean_auroc','alpha','lambda'],ascending=[False,False,False])
    return best[1],best[2],tab

def choose_ridge(X,site,y,g):
    rows=[]; best=None
    spl=folds(y,g,SEED_CV,5)
    for lam in RIDGE_LAMS:
        aucs=[]
        for tri,vai in spl:
            w,info,sc,nctx=V.fit_full(X[tri],site[tri],y[tri],lam,kind='l2')
            aucs.append(float(roc_auc_score(y[vai],V.predict_full(w,sc,X[vai],site[vai]))))
        m=float(np.mean(aucs)); sd=float(np.std(aucs,ddof=1))
        rows.append({'lambda':lam,'mean_auroc':m,'sd_auroc':sd})
        key=(m,lam)
        if best is None or key>best[0]: best=(key,lam)
    return best[1],pd.DataFrame(rows).sort_values(['mean_auroc','lambda'],ascending=[False,False])

def choose_predictor_l2(X,site,y,g):
    rows=[]; best=None
    spl=folds(y,g,91019,5)
    for lam in V.LAM_GRID:
        aucs=[]
        for tri,vai in spl:
            w,info,sc,nctx=V.fit_full(X[tri],site[tri],y[tri],lam,kind='l2')
            aucs.append(float(roc_auc_score(y[vai],V.predict_full(w,sc,X[vai],site[vai]))))
        m=float(np.mean(aucs))
        rows.append({'lambda':float(lam),'mean_auroc':m,'sd_auroc':float(np.std(aucs,ddof=1))})
        if best is None or (m,float(lam))>best[0]: best=((m,float(lam)),float(lam))
    return best[1],pd.DataFrame(rows)

def stable_topk(score,secondary,features,k):
    d=pd.DataFrame({'feature':features,'score':np.asarray(score,float),'secondary':np.asarray(secondary,float),'idx':np.arange(len(features))})
    d=d.sort_values(['score','secondary','idx'],ascending=[False,False,True],kind='mergesort')
    return list(d.feature.head(k))

# ---------------- Batch1 data ----------------
man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json'))
features=list(man['selectable_features'])
devX=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv')
devY=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y'])
dev=devX.merge(devY,on=['patient_uid','site'])
dev=dev[dev.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
assert len(dev)==892 and dev.patient_uid.nunique()==664
X=dev[features].to_numpy(float); site=dev.site.to_numpy(); y=dev.y.to_numpy(int); groups=dev.patient_uid.to_numpy()

# ---------------- selectors ----------------
selector_rows=[]
selected=[]

# S0 existing L1-reference reference
m0sets=pd.read_csv(ROOT/'09_BATCH1_DOWNSTREAM_V1_9/FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv')
for k in KGRID:
    rr=m0sets[(m0sets.method=='reference')&(m0sets.k==k)].iloc[0]
    ss=str(rr.selected_set).split('|')
    selected.append({'selector':'S0_L1_LASSO','k':k,'selected_set':'|'.join(ss),
                     'selector_hp':'lambda='+str(rr.selector_l1_lam_full_batch1),
                     'selection_source':'existing_frozen_main_experiment'})

# S1 ElasticNet
a_en,l_en,cv_en=choose_enet(X,site,y,groups)
cv_en.to_csv(OUT/'ELASTICNET_SELECTOR_CV.csv',index=False)
w_en,sc_en,nctx_en=enet_fit_full(X,site,y,l_en,a_en)
D_en=V.clinical_scores(w_en,nctx_en)
for f,d in zip(features,D_en): selector_rows.append({'selector':'S1_ELASTICNET','feature':f,'score':float(d)})
for k in KGRID:
    ss=stable_topk(D_en,D_en,features,k)
    selected.append({'selector':'S1_ELASTICNET','k':k,'selected_set':'|'.join(ss),
                     'selector_hp':f'alpha={a_en};lambda={l_en}','selection_source':'Batch1_only'})

# S2 RidgeRank
l_r,cv_r=choose_ridge(X,site,y,groups)
cv_r.to_csv(OUT/'RIDGERANK_SELECTOR_CV.csv',index=False)
w_r,info_r,sc_r,nctx_r=V.fit_full(X,site,y,l_r,kind='l2')
D_r=V.clinical_scores(w_r,nctx_r)
for f,d in zip(features,D_r): selector_rows.append({'selector':'S2_RIDGERANK','feature':f,'score':float(d)})
for k in KGRID:
    ss=stable_topk(D_r,D_r,features,k)
    selected.append({'selector':'S2_RIDGERANK','k':k,'selected_set':'|'.join(ss),
                     'selector_hp':f'lambda={l_r}','selection_source':'Batch1_only'})

# S3 SubsampleTopK using the already-frozen main L1 lambda 3.3333...
l1_frozen=float(m0sets.iloc[0].selector_l1_lam_full_batch1)
uids=np.unique(groups)
scores=np.zeros((N_RESAMPLES,len(features))); ranks=np.zeros_like(scores)
for b in range(N_RESAMPLES):
    rng=np.random.default_rng(SEED_RESAMPLE+b)
    take=rng.choice(len(uids),size=int(round(RESAMPLE_FRAC*len(uids))),replace=False)
    mask=np.isin(groups,uids[take]); idx=np.where(mask)[0]
    w,info,sc,nctx=V.fit_full(X[idx],site[idx],y[idx],l1_frozen,kind='l1')
    d=V.clinical_scores(w,nctx); scores[b]=d; ranks[b]=rankdata(-d,method='average')
    if (b+1)%50==0: print('resample',b+1,flush=True)
np.savez_compressed(OUT/'SUBSAMPLETOPK_RESAMPLES.npz',scores=scores,ranks=ranks,features=np.array(features),lambda_l1=l1_frozen)
mean_rank=ranks.mean(axis=0)
for k in KGRID:
    pi=(ranks<=k).mean(axis=0)
    for f,pv,mr in zip(features,pi,mean_rank):
        selector_rows.append({'selector':f'S3_SUBSAMPLETOPK_K{k}','feature':f,'score':float(pv),'mean_rank':float(mr)})
    # tie: higher pi, lower mean rank, frozen order
    secondary=-mean_rank
    ss=stable_topk(pi,secondary,features,k)
    selected.append({'selector':'S3_SUBSAMPLETOPK','k':k,'selected_set':'|'.join(ss),
                     'selector_hp':f'200x80pct;L1lambda={l1_frozen}','selection_source':'Batch1_only'})

pd.DataFrame(selector_rows).to_csv(OUT/'SELECTOR_FEATURE_SCORES.csv',index=False)
sel=pd.DataFrame(selected)
sel.to_csv(OUT/'BATCH1_FROZEN_SELECTED_SETS.csv',index=False)

# ---------------- fit downstream predictors on full Batch1 only ----------------
pred_index=[]; pred_cache={}
for r in sel.itertuples():
    feat=str(r.selected_set).split('|')
    # canonicalize to original feature order
    feat=tuple(f for f in features if f in set(feat))
    key=feat
    if key not in pred_cache:
        inds=[features.index(f) for f in feat]
        Xs=X[:,inds]
        lam,tab=choose_predictor_l2(Xs,site,y,groups)
        mid='P%02d'%(len(pred_cache)+1)
        tab.to_csv(OUT/f'{mid}_PREDICTOR_L2_CV.csv',index=False)
        w,info,sc,nctx=V.fit_full(Xs,site,y,lam,kind='l2')
        np.savez_compressed(OUT/f'{mid}_MODEL.npz',w=w,med=sc[0],mu=sc[1],sd=sc[2],nctx=np.array([nctx]))
        pred_cache[key]=(mid,lam,w,sc,nctx)
    mid,lam,w,sc,nctx=pred_cache[key]
    pred_index.append({'selector':r.selector,'k':int(r.k),'selector_hp':r.selector_hp,'selected_set':'|'.join(feat),
                       'predictor_id':mid,'predictor_l2_lambda':lam})
idx=pd.DataFrame(pred_index)
idx.to_csv(OUT/'FROZEN_PREDICTOR_INDEX.csv',index=False)

# ---------------- Batch2 exploratory evaluation ----------------
canonX=pd.read_csv(CANON/'canonical/site_level_X.csv')
canonY=pd.read_csv(CANON/'canonical/site_level_y.csv')
canonX=canonX.copy(); canonX['y']=pd.to_numeric(canonY.iloc[:,0]).astype(int).to_numpy()
b2=canonX[(canonX.cohort.astype(str)=='batch2') & canonX.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
assert len(b2)==727 and b2.patient_uid.nunique()==538
yb=b2.y.to_numpy(int); sb=b2.site.to_numpy(); uid=b2.patient_uid.astype(str).to_numpy()

preds={}
rows=[]
for mid,g in idx.groupby('predictor_id'):
    r=g.iloc[0]; feat=str(r.selected_set).split('|')
    z=b2[feat].copy()
    for c in feat:
        if c=='DXA_性别': z[c]=z[c].astype(str).str.strip().map({'女':1.0,'男':0.0})
        else: z[c]=pd.to_numeric(z[c],errors='coerce')
    if 'DXA_性别' in feat and z['DXA_性别'].isna().any(): raise RuntimeError('SEX_MAPPING_FAILURE')
    Xb=z.to_numpy(float)
    dat=np.load(OUT/f'{mid}_MODEL.npz'); w=dat['w']; sc=(dat['med'],dat['mu'],dat['sd'])
    pb=V.predict_full(w,sc,Xb,sb); preds[mid]=pb

for r in idx.itertuples():
    m=metrics(yb,preds[r.predictor_id])
    rows.append({'selector':r.selector,'k':int(r.k),'selector_hp':r.selector_hp,'selected_set':r.selected_set,
                 'predictor_id':r.predictor_id,'predictor_l2_lambda':float(r.predictor_l2_lambda),**m})
res=pd.DataFrame(rows)

# compare with authoritative original reference within each k
main=pd.read_csv(ROOT/'11_BATCH2_FINAL_EVALUATION_V2_0_1/BATCH2_PRIMARY_RESULTS.csv')
base={k:main[(main.method=='reference')&(main.k==k)].iloc[0] for k in KGRID}
for met in ['auroc','auprc','balanced_accuracy']:
    res['delta_'+met+'_vs_original_reference']=res.apply(lambda r:float(r[met])-float(base[int(r.k)][met]),axis=1)

# overlap with original reference
for k in KGRID:
    bset=set(m0sets[(m0sets.method=='reference')&(m0sets.k==k)].iloc[0].selected_set.split('|'))
    mask=res.k==k
    res.loc[mask,'jaccard_vs_original_reference']=res.loc[mask,'selected_set'].map(lambda x:len(set(x.split('|'))&bset)/len(set(x.split('|'))|bset))

res.to_csv(OUT/'BATCH2_EXPLORATORY_DATAONLY_RESULTS.csv',index=False)

# patient-cluster paired bootstrap against original reference predictions reconstructed from original frozen predictors
orig_idx=pd.read_csv(ROOT/'10_PRE_BATCH2_FREEZE/FINAL_PREDICTORS/FINAL_PREDICTOR_INDEX.csv')
orig_preds={}
for k in KGRID:
    rr=orig_idx[(orig_idx.method=='reference')&(orig_idx.k==k)].iloc[0]
    feat=str(rr.feature_order).split('|'); z=b2[feat].copy()
    for c in feat:
        if c=='DXA_性别': z[c]=z[c].astype(str).str.strip().map({'女':1.0,'男':0.0})
        else: z[c]=pd.to_numeric(z[c],errors='coerce')
    dat=np.load(ROOT/f'10_PRE_BATCH2_FREEZE/FINAL_PREDICTORS/{rr.model_id}_MODEL.npz')
    orig_preds[k]=V.predict_full(dat['w'],(dat['med'],dat['mu'],dat['sd']),z.to_numpy(float),sb)

rng=np.random.default_rng(2026091803)
patients=np.unique(uid); B=2000
bootrows=[]
for b in range(B):
    samp=rng.choice(patients,size=len(patients),replace=True)
    inds=np.concatenate([np.where(uid==u)[0] for u in samp])
    yy=yb[inds]
    if len(np.unique(yy))<2: continue
    for r in idx.itertuples():
        p=preds[r.predictor_id][inds]; p0=orig_preds[int(r.k)][inds]
        bootrows.append({'b':b,'selector':r.selector,'k':int(r.k),
                         'd_auroc':float(roc_auc_score(yy,p)-roc_auc_score(yy,p0)),
                         'd_auprc':float(average_precision_score(yy,p)-average_precision_score(yy,p0))})
bd=pd.DataFrame(bootrows)
cis=[]
for (s,k),g in bd.groupby(['selector','k']):
    cis.append({'selector':s,'k':int(k),
                'd_auroc_lo':float(g.d_auroc.quantile(.025)),'d_auroc_hi':float(g.d_auroc.quantile(.975)),
                'p_d_auroc_gt0':float((g.d_auroc>0).mean()),
                'd_auprc_lo':float(g.d_auprc.quantile(.025)),'d_auprc_hi':float(g.d_auprc.quantile(.975)),
                'p_d_auprc_gt0':float((g.d_auprc>0).mean())})
ci=pd.DataFrame(cis)
final=res.merge(ci,on=['selector','k'],how='left')
final.to_csv(OUT/'BATCH2_EXPLORATORY_DATAONLY_RESULTS_WITH_BOOTSTRAP.csv',index=False)

status={'status':'PASS_POSTHOC_EXPLORATORY','n_batch1_rows':len(dev),'n_batch1_patients':int(dev.patient_uid.nunique()),
        'n_batch2_rows':len(b2),'n_batch2_patients':int(b2.patient_uid.nunique()),
        'selectors':['S0_L1_LASSO','S1_ELASTICNET','S2_RIDGERANK','S3_SUBSAMPLETOPK'],
        'elasticnet_alpha':a_en,'elasticnet_lambda':l_en,'ridge_lambda':l_r,'subsample_l1_lambda':l1_frozen,
        'batch2_already_seen_before_extension':True,
        'interpretation':'descriptive robustness only; cannot upgrade primary confirmatory claims'}
(OUT/'SELECTOR_ROBUSTNESS_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(status,ensure_ascii=False,indent=2))
print('\nSELECTED SETS\n',sel.to_string(index=False))
print('\nBATCH2 EXPLORATORY\n',final.to_string(index=False))
