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
import json,math,time,warnings,hashlib,sys
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
sys.path.insert(0,str(ROOT/'scripts'))
import run_strict_nested_selective_routing as route
MEAS=pd.read_csv(str(REPRO_ROOT / '02_gse272769_sepsis/gse272769/measurement/postprocess/selective_PAIR_SOURCE_MEASUREMENTS.csv'))
MEAS['task']='SEPSIS_GSE272769'
SEED=2026091901
LAMS=[0,0.03,0.1,0.3,1,3,10]
KGRID=[10,20,30]
SELECTORS=['LASSO','ELASTICNET','SIS']

def sis_score(X,y):
    yc=y-y.mean(); xc=X-X.mean(axis=0)
    den=np.sqrt((xc*xc).sum(axis=0)*(yc*yc).sum())
    return np.nan_to_num(np.abs((xc*yc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0)

def sparse_score_outer_reference(X,y,C,l1_ratio,seed):
    """Original reference SAGA score. Outer evaluation must use this exactly."""
    sc=StandardScaler().fit(X); Z=sc.transform(X)
    penalty='l1' if float(l1_ratio)==1.0 else 'elasticnet'
    mod=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=penalty,
        l1_ratio=None if penalty=='l1' else float(l1_ratio),max_iter=5000,tol=1e-4,
        random_state=int(seed),fit_intercept=True,n_jobs=1)
    mod.fit(Z,y)
    nit=int(mod.n_iter_[0]); conv=bool(nit<mod.max_iter)
    return np.abs(mod.coef_[0]),nit,conv,'reference_SAGA'

def sparse_score_inner_exact(X,y,C,l1_ratio,seed):
    """Same convex objective as reference, solved to tight KKT tolerance for inner lam tuning."""
    D,ok,nnz=route.fit_sparse(X,y,float(C),float(l1_ratio),int(seed))
    return D,-1,bool(ok),'EXACT_WEIGHTED_PROXNEWTON'

def selector_score(X,y,selector,C=None,l1_ratio=None,seed=0,stage='inner'):
    if selector=='SIS': return sis_score(X,y),0,True,'SIS_PEARSON'
    if stage=='inner':
        return sparse_score_inner_exact(X,y,C,l1_ratio,seed)
    if stage=='outer':
        return sparse_score_outer_reference(X,y,C,l1_ratio,seed)
    raise ValueError(stage)

def data_anchor(D):
    r=rankdata(-D,method='average')
    u=1-(r-0.5)/len(D)
    return norm.ppf(np.clip(u,1e-6,1-1e-6))

def stable_topk(z,D,k):
    return np.lexsort((np.arange(len(z)),-D,-z))[:k]

def optimize_selective(sD,constraints,lam):
    if lam==0 or not constraints: return sD.copy(),True,0,0.0
    ii=np.array([x['i'] for x in constraints],int)
    jj=np.array([x['j'] for x in constraints],int)
    yy=np.array([x['y'] for x in constraints],float)
    w=np.array([x['w'] for x in constraints],float)
    W=float(w.sum())
    if W<=0: return sD.copy(),True,0,0.0
    p=len(sD)
    def fg(z):
        dz=z[ii]-z[jj]
        ce=np.logaddexp(0,dz)-yy*dz
        f=0.5*np.mean((z-sD)**2)+lam*np.dot(w,ce)/W
        grad=(z-sD)/p
        rr=lam*(w/W)*(expit(dz)-yy)
        np.add.at(grad,ii,rr); np.add.at(grad,jj,-rr)
        return float(f),grad
    res=minimize(lambda z:fg(z),sD.copy(),jac=True,method='L-BFGS-B',
                 options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    return res.x,bool(res.success),int(res.nit),W

def eval_selected(X,y,tr,va,cols):
    sc=StandardScaler().fit(X[tr][:,cols])
    Xt=sc.transform(X[tr][:,cols]); Xv=sc.transform(X[va][:,cols])
    mod=LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',class_weight='balanced',
                           max_iter=5000,random_state=SEED)
    mod.fit(Xt,y[tr])
    p=mod.predict_proba(Xv)[:,1]
    app=average_precision_score(y[va],p); apn=average_precision_score(1-y[va],1-p)
    return {'auroc':float(roc_auc_score(y[va],p)),'ap_positive':float(app),
            'ap_negative':float(apn),'macro_ap':float(0.5*(app+apn)),
            'balanced_accuracy':float(balanced_accuracy_score(y[va],p>=0.5)),
            'validation_prevalence':float(y[va].mean())}

def build_constraints(task,fold,selector,k,feat):
    fp=ROOT/'06_selective_ROUTING'/task/f'fold{fold}_{selector}'/f'K{k}_PAIR_CONFUSION.csv'
    cf=pd.read_csv(fp)
    if 'pair_type' not in cf.columns or len(cf)==0: return [],{'n_actionable_pairs':0,'n_source_constraints':0,'total_weight':0.0,'n_U_constraints':0}
    cf=cf[cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION'].copy()
    if len(cf)==0: return [],{'n_actionable_pairs':0,'n_source_constraints':0,'total_weight':0.0,'n_U_constraints':0}
    name_to_idx={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}
    mt=MEAS[MEAS.task==task]
    mmap={}
    for r in mt.itertuples():
        mmap[(str(r.unordered_pair_id),str(r.arm))]=r
    rows=[]; nU=0
    for r in cf.itertuples():
        a=str(r.feature_A); b=str(r.feature_B)
        pair='||'.join(sorted([a,b]))
        avail=[q for (pp,arm),q in mmap.items() if pp==pair]
        m=len(avail)
        if m==0: continue
        if a not in name_to_idx or b not in name_to_idx: raise RuntimeError('GENE_INDEX_MISSING')
        i=name_to_idx[a]; j=name_to_idx[b]
        for q in avail:
            if str(q.gene_i)==a and str(q.gene_j)==b: yy=float(q.hard_y_i_over_j)
            elif str(q.gene_i)==b and str(q.gene_j)==a: yy=1.0-float(q.hard_y_i_over_j)
            else: raise RuntimeError(f'ORIENTATION_FAIL {task} {pair} {q.arm}')
            c=float(q.certainty_1_minus_H)
            w=float(r.actionable_boundary_score)*c/m
            rows.append({'i':i,'j':j,'y':yy,'w':w,'arm':str(q.arm),'pair':pair,
                         'certainty':c,'actionable':float(r.actionable_boundary_score),
                         'either_U':bool(q.either_U)})
            nU+=int(bool(q.either_U))
    return rows,{'n_actionable_pairs':int(len(cf)),'n_source_constraints':int(len(rows)),
                 'total_weight':float(sum(x['w'] for x in rows)),'n_U_constraints':int(nU)}

def run_task(task):
    t0=time.time()
    src=Path(str(REPRO_ROOT / '02_gse272769_sepsis/gse272769/frozen')) if task=='SEPSIS_GSE272769' else ROOT/'03_FROZEN_DATA'/task
    reference=ROOT/'04_reference'/task
    out=Path(str(REPRO_ROOT / '02_gse272769_sepsis/gse272769/selective_internal')) if task=='SEPSIS_GSE272769' else ROOT/'12_selective_INTERNAL'/task
    out.mkdir(parents=True,exist_ok=True)
    X=np.load(src/'X_development.npy').astype(float)
    y=np.load(src/'y_development.npy').astype(int)
    feat=pd.read_csv(src/('features_p1500.csv' if task=='SEPSIS_GSE272769' else 'features_p2000.csv'))
    if task=='SEPSIS_GSE272769': feat['feature_index']=np.arange(len(feat),dtype=int)
    splits_df=pd.read_csv(reference/'OUTER_SPLITS.csv')
    sa=pd.read_csv(reference/'reference_SELECTOR_AUDIT.csv')
    m0fold=pd.read_csv(reference/'reference_FOLD_RESULTS.csv')

    # Constraint map and audit.
    cons={}; ca=[]
    for fold in range(1,6):
      for sel in SELECTORS:
       for k in KGRID:
        cc,audit=build_constraints(task,fold,sel,k,feat)
        cons[(fold,sel,k)]=cc
        ca.append({'task':task,'fold':fold,'selector':sel,'k':k,**audit})
    pd.DataFrame(ca).to_csv(out/'selective_FOLD_LOCAL_CONSTRAINT_AUDIT.csv',index=False)

    inner_rows=[]; outer_rows=[]; selected_lam=[]; score_audit=[]
    for fold in range(1,6):
        tr_outer=splits_df[(splits_df.fold==fold)&(splits_df.role=='outer_train')].sample_index.to_numpy(int)
        va_outer=splits_df[(splits_df.fold==fold)&(splits_df.role=='outer_validation')].sample_index.to_numpy(int)
        ytr=y[tr_outer]
        inner=StratifiedKFold(n_splits=4,shuffle=True,random_state=SEED+100*fold)
        inner_splits=list(inner.split(X[tr_outer],ytr))
        for sel in SELECTORS:
            row=sa[(sa.fold==fold)&(sa.selector==sel)].iloc[0]
            C=None if sel=='SIS' else float(row.chosen_C)
            ratio=None if sel=='SIS' else float(row.chosen_l1_ratio)
            # Fit each inner score once, reuse across all k/lam.
            inner_scores=[]
            for ii,(itr_rel,iva_rel) in enumerate(inner_splits,1):
                itr=tr_outer[itr_rel]; iva=tr_outer[iva_rel]
                D,nit,conv,solver_name=selector_score(
                    X[itr],y[itr],sel,C,ratio,
                    SEED+100000*fold+1000*ii+(0 if sel=='LASSO' else 100 if sel=='ELASTICNET' else 200),
                    stage='inner')
                inner_scores.append((itr,iva,D,data_anchor(D),nit,conv))
                score_audit.append({'task':task,'outer_fold':fold,'selector':sel,'inner_fold':ii,
                                    'n_train':len(itr),'n_val':len(iva),'solver_iter':nit,'solver_converged':conv,
                                    'solver_name':solver_name,'stage':'inner'})
            # Outer data score exactly as original reference rule/seed. No fallback is allowed.
            Douter,nit_outer,conv_outer,solver_outer=selector_score(
                X[tr_outer],y[tr_outer],sel,C,ratio,SEED+fold,stage='outer')
            if not conv_outer:
                raise RuntimeError(f'OUTER_reference_SELECTOR_NONCONVERGENCE {task} fold={fold} selector={sel}')
            sDouter=data_anchor(Douter)
            score_audit.append({'task':task,'outer_fold':fold,'selector':sel,'inner_fold':0,
                                'n_train':len(tr_outer),'n_val':len(va_outer),'solver_iter':nit_outer,'solver_converged':conv_outer,
                                'solver_name':solver_outer,'stage':'outer'})
            for k in KGRID:
                cc=cons[(fold,sel,k)]
                # tune lam using only outer-training inner validation
                lam_perf=[]
                for lam in LAMS:
                    vals=[]
                    for ii,(itr,iva,Din,sDin,nit,conv) in enumerate(inner_scores,1):
                        z,ok,onit,W=optimize_selective(sDin,cc,lam)
                        cols=stable_topk(z,Din,k)
                        met=eval_selected(X,y,itr,iva,cols)
                        vals.append(met['auroc'])
                        inner_rows.append({'task':task,'outer_fold':fold,'selector':sel,'k':k,
                                           'lam':lam,'inner_fold':ii,'opt_success':ok,'opt_nit':onit,
                                           'constraint_weight_sum':W,**met,
                                           'selected_indices':'|'.join(map(str,cols.tolist()))})
                    lam_perf.append((lam,float(np.mean(vals)),float(np.std(vals,ddof=1))))
                best=max(x[1] for x in lam_perf)
                chosen=min(x[0] for x in lam_perf if np.isclose(x[1],best,atol=1e-12,rtol=0))
                chsd=next(x[2] for x in lam_perf if x[0]==chosen)
                selected_lam.append({'task':task,'outer_fold':fold,'selector':sel,'k':k,
                                     'chosen_lam':chosen,'best_inner_mean_auroc':best,'inner_sd_auroc':chsd,
                                     'n_source_constraints':len(cc),'total_constraint_weight':sum(x['w'] for x in cc)})
                # one outer validation evaluation after lam chosen
                z,ok,onit,W=optimize_selective(sDouter,cc,chosen)
                cols=stable_topk(z,Douter,k)
                met=eval_selected(X,y,tr_outer,va_outer,cols)
                ref=m0fold[(m0fold.fold==fold)&(m0fold.selector==sel)&(m0fold.k==k)].iloc[0]
                outer_rows.append({'task':task,'outer_fold':fold,'selector':sel,'k':k,'chosen_lam':chosen,
                                   'opt_success':ok,'opt_nit':onit,'constraint_weight_sum':W,**met,
                                   'reference_auroc':float(ref.auroc),'delta_auroc_vs_reference':float(met['auroc']-ref.auroc),
                                   'reference_macro_ap':float(ref.macro_ap),'delta_macro_ap_vs_reference':float(met['macro_ap']-ref.macro_ap),
                                   'selected_indices':'|'.join(map(str,cols.tolist())),
                                   'selected_genes':'|'.join(feat.iloc[cols].gene_symbol.astype(str).tolist())})
            print(task,'done outer fold',fold,'selector',sel,flush=True)

    pd.DataFrame(score_audit).to_csv(out/'selective_SELECTOR_SCORE_AUDIT.csv',index=False)
    if not pd.DataFrame(score_audit).solver_converged.all():
        raise RuntimeError(f'INNER_OR_OUTER_SELECTOR_NONCONVERGENCE {task}')
    innerdf=pd.DataFrame(inner_rows); innerdf.to_csv(out/'selective_INNER_ETA_RESULTS.csv',index=False)
    seldf=pd.DataFrame(selected_lam); seldf.to_csv(out/'selective_NESTED_SELECTED_ETA.csv',index=False)
    outerdf=pd.DataFrame(outer_rows); outerdf.to_csv(out/'selective_NESTED_OUTER_RESULTS.csv',index=False)
    agg=(outerdf.groupby(['selector','k'],as_index=False)
         .agg(mean_selective_auroc=('auroc','mean'),sd_selective_auroc=('auroc','std'),
              mean_reference_auroc=('reference_auroc','mean'),mean_delta_auroc=('delta_auroc_vs_reference','mean'),
              sd_delta_auroc=('delta_auroc_vs_reference','std'),
              mean_selective_macro_ap=('macro_ap','mean'),mean_reference_macro_ap=('reference_macro_ap','mean'),
              mean_delta_macro_ap=('delta_macro_ap_vs_reference','mean'),
              lam_zero_folds=('chosen_lam',lambda x:int(np.sum(np.asarray(x,float)==0))),
              lam_nonzero_folds=('chosen_lam',lambda x:int(np.sum(np.asarray(x,float)>0)))))
    agg.to_csv(out/'selective_NESTED_AGGREGATE.csv',index=False)
    sadf=pd.DataFrame(score_audit)
    status={'status':'PASS_NESTED_selective_DEVELOPMENT_ONLY_SOLVER_V2','task':task,'lambda_grid':LAMS,'k_grid':KGRID,
            'n_outer_rows':len(outerdf),'n_inner_rows':len(innerdf),
            'all_selector_fits_converged':True,
            'inner_sparse_solver':'SKGLM_WEIGHTED_PROXNEWTON_EXACT',
            'outer_sparse_solver':'ORIGINAL_reference_SAGA',
            'inner_exact_fit_count':int(((sadf.stage=='inner') & sadf.selector.isin(['LASSO','ELASTICNET'])).sum()),
            'outer_reference_saga_fit_count':int(((sadf.stage=='outer') & sadf.selector.isin(['LASSO','ELASTICNET'])).sum()),
            'all_optimizations_success':bool(outerdf.opt_success.all() and innerdf.opt_success.all()),
            'uses_sealed_validation':False,'runtime_sec':round(time.time()-t0,2)}
    (out/'selective_NESTED_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('\n',json.dumps(status,indent=2),flush=True)
    print(agg.to_string(index=False),flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=['SEPSIS_GSE272769'],default='SEPSIS_GSE272769')
    args=ap.parse_args()
    tasks=[args.task]
    for task in tasks:
        run_task(task)
