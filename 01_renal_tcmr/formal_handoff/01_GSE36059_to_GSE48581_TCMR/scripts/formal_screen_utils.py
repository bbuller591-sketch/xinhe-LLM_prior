import json, math, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

RNG_SEED=20260921

def variance_filter(X, p_candidate):
    v=np.nanvar(X,axis=0)
    good=np.isfinite(v) & (v>1e-12)
    idx=np.where(good)[0]
    idx=idx[np.argsort(v[idx])[::-1]]
    return idx[:min(p_candidate,len(idx))]

def sis_scores(X,y):
    y=np.asarray(y,float)
    yc=y-y.mean()
    Xm=X-np.nanmean(X,axis=0)
    num=np.nansum(Xm*yc[:,None],axis=0)
    den=np.sqrt(np.nansum(Xm*Xm,axis=0)*np.sum(yc*yc))
    s=np.divide(num,den,out=np.zeros_like(num,float),where=den>0)
    return np.nan_to_num(np.abs(s))

def _auc(y,p):
    try: return float(roc_auc_score(y,p))
    except: return float('nan')

def _eval_probs(y,p,thr=.5):
    return {
      'auc':_auc(y,p),
      'auprc':float(average_precision_score(y,p)),
      'balanced_accuracy':float(balanced_accuracy_score(y,(p>=thr).astype(int)))
    }

def choose_sis_k(X,y,k_grid,folds=5):
    cv=StratifiedKFold(n_splits=min(folds,int(np.bincount(y).min())),shuffle=True,random_state=RNG_SEED)
    rows=[]
    for k in k_grid:
      vals=[]
      for tr,va in cv.split(X,y):
        sc=sis_scores(X[tr],y[tr]); ids=np.argsort(sc)[::-1][:k]
        mdl=make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',C=1.0,solver='liblinear',class_weight='balanced',max_iter=2000,random_state=RNG_SEED))
        mdl.fit(X[tr][:,ids],y[tr])
        vals.append(_auc(y[va],mdl.predict_proba(X[va][:,ids])[:,1]))
      rows.append({'k':int(k),'cv_auc':float(np.nanmean(vals))})
    best=max(rows,key=lambda r:(r['cv_auc'],-r['k']))
    return best['k'],rows

def choose_regularized(X,y,kind='l1',folds=5):
    Cs=[0.01,0.03,0.1,0.3,1.0]
    ratios=[None] if kind=='l1' else [0.2,0.5,0.8]
    cv=StratifiedKFold(n_splits=min(folds,int(np.bincount(y).min())),shuffle=True,random_state=RNG_SEED)
    rows=[]
    for C in Cs:
      for ratio in ratios:
        vals=[]
        for tr,va in cv.split(X,y):
          if kind=='l1':
            lr=LogisticRegression(penalty='l1',C=C,solver='liblinear',class_weight='balanced',max_iter=3000,random_state=RNG_SEED)
          else:
            lr=LogisticRegression(penalty='elasticnet',l1_ratio=ratio,C=C,solver='saga',class_weight='balanced',max_iter=3000,tol=1e-3,random_state=RNG_SEED)
          mdl=make_pipeline(StandardScaler(),lr)
          mdl.fit(X[tr],y[tr])
          vals.append(_auc(y[va],mdl.predict_proba(X[va])[:,1]))
        rows.append({'C':C,'l1_ratio':ratio,'cv_auc':float(np.nanmean(vals))})
    best=max(rows,key=lambda r:r['cv_auc'])
    return best,rows

def fit_regularized(X,y,best,kind='l1'):
    if kind=='l1':
      lr=LogisticRegression(penalty='l1',C=best['C'],solver='liblinear',class_weight='balanced',max_iter=3000,random_state=RNG_SEED)
    else:
      lr=LogisticRegression(penalty='elasticnet',l1_ratio=best['l1_ratio'],C=best['C'],solver='saga',class_weight='balanced',max_iter=3000,tol=1e-3,random_state=RNG_SEED)
    mdl=make_pipeline(StandardScaler(),lr); mdl.fit(X,y); return mdl

def tune_reference_k(X,y,en_best,k_grid,folds=5):
    cv=StratifiedKFold(n_splits=min(folds,int(np.bincount(y).min())),shuffle=True,random_state=RNG_SEED+17)
    rows=[]
    for k in k_grid:
      vals=[]
      for tr,va in cv.split(X,y):
        rankmdl=fit_regularized(X[tr],y[tr],en_best,'en')
        coef=np.abs(rankmdl[-1].coef_[0])
        ids=np.argsort(coef)[::-1][:k]
        pred=make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',C=1,solver='liblinear',class_weight='balanced',max_iter=2000,random_state=RNG_SEED))
        pred.fit(X[tr][:,ids],y[tr])
        vals.append(_auc(y[va],pred.predict_proba(X[va][:,ids])[:,1]))
      rows.append({'k':int(k),'cv_auc':float(np.nanmean(vals))})
    best=max(rows,key=lambda r:(r['cv_auc'],-r['k']))
    return best['k'],rows

def resampling_boundary(X,y,feature_names,en_best,k,R=200,frac=.8):
    rng=np.random.default_rng(20261044)
    n,p=X.shape
    scores=np.zeros((R,p),float)
    top=np.zeros((R,p),bool)
    for r in range(R):
      # stratified subsample without replacement
      ids=[]
      for cls in [0,1]:
        ix=np.where(y==cls)[0]
        m=max(2,int(np.floor(frac*len(ix))))
        ids.extend(rng.choice(ix,size=m,replace=False).tolist())
      ids=np.array(sorted(ids))
      mdl=fit_regularized(X[ids],y[ids],en_best,'en')
      sc=np.abs(mdl[-1].coef_[0])
      # continuous tie-breaker with tiny SIS component, to define pair order when EN coefficients tie at zero
      sc=sc + 1e-8*sis_scores(X[ids],y[ids])
      scores[r]=sc
      sel=np.argsort(sc)[::-1][:k]; top[r,sel]=True
    incl=top.mean(axis=0)
    # Jaccard across all resample pairs
    js=[]
    for a in range(R):
      for b in range(a+1,R):
        inter=np.logical_and(top[a],top[b]).sum(); uni=np.logical_or(top[a],top[b]).sum()
        js.append(inter/uni if uni else 1.0)
    # candidate pair pool: features ever near/inside boundary, top 3k by mean score or inclusion in (0,1)
    mean_scores=scores.mean(axis=0)
    pool=set(np.argsort(mean_scores)[::-1][:min(3*k,p)].tolist())
    pool.update(np.where((incl>0)&(incl<1))[0].tolist())
    pool=sorted(pool)
    pairs=[]
    for ai,i in enumerate(pool):
      for j in pool[ai+1:]:
        q=float(np.mean(np.logical_xor(top[:,i],top[:,j])))
        prob=float(np.mean(scores[:,i]>scores[:,j]) + 0.5*np.mean(scores[:,i]==scores[:,j]))
        b=float(1-2*abs(prob-.5))
        u=q*b
        if q>0:
          pairs.append((u,q,b,i,j,prob))
    pairs.sort(reverse=True)
    top_pairs=[{
      'U':float(u),'Q':float(q),'B':float(b),
      'feature_i':str(feature_names[i]),'feature_j':str(feature_names[j]),
      'p_i_gt_j':float(prob),'incl_i':float(incl[i]),'incl_j':float(incl[j])
    } for u,q,b,i,j,prob in pairs[:100]]
    summary={
      'R':R,'subsample_fraction':frac,'k':int(k),
      'mean_pairwise_jaccard':float(np.mean(js)) if js else 1.0,
      'median_pairwise_jaccard':float(np.median(js)) if js else 1.0,
      'features_inclusion_10_90':int(np.sum((incl>=.1)&(incl<=.9))),
      'features_inclusion_25_75':int(np.sum((incl>=.25)&(incl<=.75))),
      'pairs_U_ge_0_10':int(sum(x[0]>=.10 for x in pairs)),
      'pairs_U_ge_0_25':int(sum(x[0]>=.25 for x in pairs)),
      'pairs_U_ge_0_40':int(sum(x[0]>=.40 for x in pairs)),
      'max_U':float(pairs[0][0]) if pairs else 0.0,
      'top_pairs':top_pairs
    }
    feat=pd.DataFrame({'feature':feature_names,'inclusion_prob':incl,'mean_abs_reference_score':mean_scores})
    feat=feat.sort_values(['inclusion_prob','mean_abs_reference_score'],ascending=False)
    return summary,feat,scores,top

def run_screen(X,y,feature_names,outdir,p_candidate=2000,X_external=None,y_external=None,external_feature_names=None,internal_holdout=.25):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    X=np.asarray(X,float); y=np.asarray(y,int); feature_names=np.asarray(feature_names).astype(str)
    p_raw=int(X.shape[1])
    # If external present, align must already be done. Candidate filtering is dev-only.
    ids=variance_filter(X,p_candidate)
    X=X[:,ids]; feature_names=feature_names[ids]
    if X_external is not None:
      Xtr,ytr=X,y
      Xte,yte=np.asarray(X_external,float)[:,ids],np.asarray(y_external,int)
      split='external'
    else:
      tr,te=train_test_split(np.arange(len(y)),test_size=internal_holdout,stratify=y,random_state=RNG_SEED)
      Xtr,ytr=X[tr],y[tr]; Xte,yte=X[te],y[te]; split='internal_holdout'
    # median impute from training
    med=np.nanmedian(Xtr,axis=0)
    for A in [Xtr,Xte]:
      rr,cc=np.where(~np.isfinite(A)); A[rr,cc]=med[cc]

    k_grid=[k for k in [10,20,30,50,75,100] if k < Xtr.shape[1] and k < max(10,len(ytr))]
    if not k_grid: k_grid=[min(10,Xtr.shape[1])]
    sis_k,sis_cv=choose_sis_k(Xtr,ytr,k_grid)
    sis_sc=sis_scores(Xtr,ytr); sis_ids=np.argsort(sis_sc)[::-1][:sis_k]
    sis_m=make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',C=1,solver='liblinear',class_weight='balanced',max_iter=2000,random_state=RNG_SEED))
    sis_m.fit(Xtr[:,sis_ids],ytr); sis_perf=_eval_probs(yte,sis_m.predict_proba(Xte[:,sis_ids])[:,1])

    l1_best,l1_cv=choose_regularized(Xtr,ytr,'l1')
    l1_m=fit_regularized(Xtr,ytr,l1_best,'l1'); l1_perf=_eval_probs(yte,l1_m.predict_proba(Xte)[:,1])

    en_best,en_cv=choose_regularized(Xtr,ytr,'en')
    en_m=fit_regularized(Xtr,ytr,en_best,'en'); en_perf=_eval_probs(yte,en_m.predict_proba(Xte)[:,1])

    # ExtraTrees nonlinear baseline and ranking (fixed lightweight screen)
    et=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,max_features='sqrt',class_weight='balanced',random_state=RNG_SEED,n_jobs=-1)
    et.fit(Xtr,ytr); et_perf=_eval_probs(yte,et.predict_proba(Xte)[:,1])

    ref_k,ref_k_cv=tune_reference_k(Xtr,ytr,en_best,k_grid)
    boundary,feat_tbl,scores,top=resampling_boundary(Xtr,ytr,feature_names,en_best,ref_k,R=200,frac=.8)

    pd.DataFrame(sis_cv).to_csv(outdir/'sis_cv.csv',index=False)
    pd.DataFrame(l1_cv).to_csv(outdir/'l1_cv.csv',index=False)
    pd.DataFrame(en_cv).to_csv(outdir/'elasticnet_cv.csv',index=False)
    pd.DataFrame(ref_k_cv).to_csv(outdir/'reference_k_cv.csv',index=False)
    feat_tbl.to_csv(outdir/'resampling_feature_stability.csv',index=False)
    pd.DataFrame(boundary['top_pairs']).to_csv(outdir/'top_boundary_pairs.csv',index=False)
    pd.DataFrame({'feature':feature_names}).to_csv(outdir/'candidate_features.csv',index=False)

    summary={
      'n_total':int(len(y)),'n_dev':int(len(ytr)),'n_eval':int(len(yte)),
      'positive_total':int(y.sum()),'positive_dev':int(ytr.sum()),'positive_eval':int(yte.sum()),
      'p_raw':p_raw,
      'p_candidate':int(X.shape[1]),'eval_design':split,
      'sis':{'best_k':int(sis_k),'cv_best_auc':max(r['cv_auc'] for r in sis_cv),**sis_perf},
      'l1':{'best':l1_best,**l1_perf},
      'elastic_net':{'best':en_best,**en_perf},
      'extra_trees':et_perf,
      'reference_k':int(ref_k),
      'boundary':boundary
    }
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
