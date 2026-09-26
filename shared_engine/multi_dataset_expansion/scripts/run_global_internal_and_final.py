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
import argparse,json,time,hashlib,warnings,sys
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
sys.path.insert(0,str(ROOT/'scripts'))
import run_strict_nested_selective_routing as ROUTE

SEED=2026091901
LAMBDAS=[0,0.03,0.1,0.3,1,3,10]
KGRID=[10,20,30]
SELECTORS=['LASSO','ELASTICNET','SIS']
METHODS=['global','global_certainty']

ap=argparse.ArgumentParser()
ap.add_argument('--task',required=True)
ap.add_argument('--degree',type=int,choices=[20,10],default=20)
ap.add_argument('--guidance-path',default='')
ap.add_argument('--output-root',default='')
args=ap.parse_args()
TASK=args.task; DEG=args.degree

def sis_score(X,y):
    yc=y-y.mean(); xc=X-X.mean(axis=0)
    den=np.sqrt((xc*xc).sum(axis=0)*(yc*yc).sum())
    return np.nan_to_num(np.abs((xc*yc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0)

def sparse_outer_reference(X,y,C,ratio,seed):
    sc=StandardScaler().fit(X); Z=sc.transform(X)
    penalty='l1' if float(ratio)==1.0 else 'elasticnet'
    mod=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=penalty,
        l1_ratio=None if penalty=='l1' else float(ratio),max_iter=5000,tol=1e-4,
        random_state=int(seed),fit_intercept=True,n_jobs=1)
    mod.fit(Z,y)
    nit=int(mod.n_iter_[0]); conv=bool(nit<mod.max_iter)
    return np.abs(mod.coef_[0]),nit,conv

def selector_inner(X,y,sel,C,ratio,seed):
    if sel=='SIS': return sis_score(X,y),0,True,'SIS_PEARSON'
    D,ok,nnz=ROUTE.fit_sparse(X,y,float(C),float(ratio),int(seed))
    return D,-1,bool(ok),'EXACT_WEIGHTED_PROXNEWTON'

def selector_outer(X,y,sel,C,ratio,seed):
    if sel=='SIS': return sis_score(X,y),0,True,'SIS_PEARSON'
    D,nit,conv=sparse_outer_reference(X,y,C,ratio,seed)
    return D,nit,conv,'reference_SAGA'

def bounded_rank(D):
    r=rankdata(-D,method='average')
    return 2*(1-(r-1)/(len(D)-1))-1

def topk(F,k):
    return np.lexsort((np.arange(len(F)),-F))[:k]

def eval_cols(X,y,tr,va,cols):
    sc=StandardScaler().fit(X[tr][:,cols])
    Xt=sc.transform(X[tr][:,cols]); Xv=sc.transform(X[va][:,cols])
    mod=LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',class_weight='balanced',
                           max_iter=5000,random_state=SEED)
    mod.fit(Xt,y[tr]); p=mod.predict_proba(Xv)[:,1]
    app=average_precision_score(y[va],p); apn=average_precision_score(1-y[va],1-p)
    return {'auroc':float(roc_auc_score(y[va],p)),'ap_positive':float(app),'ap_negative':float(apn),
            'macro_ap':float((app+apn)/2),'balanced_accuracy':float(balanced_accuracy_score(y[va],p>=.5)),
            'validation_prevalence':float(y[va].mean())}

def parse_indices(s):
    return np.array([int(x) for x in str(s).split('|') if str(x)!=''],int)

def main():
    t0=time.time()
    src=ROOT/'03_FROZEN_DATA'/TASK
    reference=ROOT/'04_reference'/TASK
    guide=Path(args.guidance_path) if args.guidance_path else ROOT/'16_global_POSTPROCESS'/TASK/f'global_META_GUIDANCE_D{DEG}.csv'
    if not guide.exists(): raise RuntimeError('GUIDANCE_NOT_READY '+str(guide))
    out=(Path(args.output_root)/TASK if args.output_root else ROOT/'17_global_INTERNAL'/f'D{DEG}'/TASK)
    out.mkdir(parents=True,exist_ok=True)

    X=np.load(src/'X_development.npy').astype(float)
    y=np.load(src/'y_development.npy').astype(int)
    feat=pd.read_csv(src/'features_p2000.csv')
    G=pd.read_csv(guide)
    if 'h_global' not in G.columns and 'h_global_strict' in G.columns:
        G=G.rename(columns={'h_global_strict':'h_global','h_global_certainty_strict':'h_global_certainty'})
    G=G.set_index('feature_index').loc[feat.feature_index].reset_index()
    assert (G.gene_symbol.astype(str).values==feat.gene_symbol.astype(str).values).all()
    H={'global':G.h_global.to_numpy(float),'global_certainty':G.h_global_certainty.to_numpy(float)}
    splits=pd.read_csv(reference/'OUTER_SPLITS.csv')
    sa=pd.read_csv(reference/'reference_SELECTOR_AUDIT.csv')
    m0fr=pd.read_csv(reference/'reference_FOLD_RESULTS.csv')

    inner_rows=[]; nested_outer=[]; fixed_outer=[]; selected_nested=[]; score_audit=[]; gate=[]
    for fold in range(1,6):
        otr=splits[(splits.fold==fold)&(splits.role=='outer_train')].sample_index.to_numpy(int)
        ova=splits[(splits.fold==fold)&(splits.role=='outer_validation')].sample_index.to_numpy(int)
        inner=StratifiedKFold(n_splits=4,shuffle=True,random_state=SEED+100*fold)
        isplits=list(inner.split(X[otr],y[otr]))
        for sel in SELECTORS:
            row=sa[(sa.fold==fold)&(sa.selector==sel)].iloc[0]
            C=None if sel=='SIS' else float(row.chosen_C)
            ratio=None if sel=='SIS' else float(row.chosen_l1_ratio)
            ins=[]
            for ii,(itrrel,ivarel) in enumerate(isplits,1):
                itr=otr[itrrel]; iva=otr[ivarel]
                D,nit,conv,sname=selector_inner(X[itr],y[itr],sel,C,ratio,
                    SEED+100000*fold+1000*ii+(0 if sel=='LASSO' else 100 if sel=='ELASTICNET' else 200))
                if not conv: raise RuntimeError(f'INNER_SELECTOR_FAIL {TASK} {fold} {sel} {ii}')
                ins.append((itr,iva,D,bounded_rank(D)))
                score_audit.append({'task':TASK,'degree':DEG,'outer_fold':fold,'selector':sel,'inner_fold':ii,
                                    'stage':'inner','solver':sname,'solver_iter':nit,'converged':conv})
            Dout,nit,conv,sname=selector_outer(X[otr],y[otr],sel,C,ratio,SEED+fold)
            if not conv: raise RuntimeError(f'OUTER_SELECTOR_FAIL {TASK} {fold} {sel}')
            dout=bounded_rank(Dout)
            score_audit.append({'task':TASK,'degree':DEG,'outer_fold':fold,'selector':sel,'inner_fold':0,
                                'stage':'outer','solver':sname,'solver_iter':nit,'converged':conv})
            # lam=0 must match reference top-k once per selector/k (same for global/global_certainty)
            for k in KGRID:
                got=topk(dout,k)
                ref=m0fr[(m0fr.fold==fold)&(m0fr.selector==sel)&(m0fr.k==k)].iloc[0]
                exp=parse_indices(ref.selected_indices)
                exact=bool(np.array_equal(got,exp))
                gate.append({'task':TASK,'degree':DEG,'outer_fold':fold,'selector':sel,'k':k,'gamma0_exact_reference':exact})
                if not exact: raise RuntimeError(f'GAMMA0_reference_FAIL {TASK} {fold} {sel} {k}')

            for method in METHODS:
              h=H[method]
              for k in KGRID:
                perf=[]
                for lam in LAMBDAS:
                    vals=[]
                    for ii,(itr,iva,Din,din) in enumerate(ins,1):
                        cols=topk(din+lam*h,k)
                        met=eval_cols(X,y,itr,iva,cols); vals.append(met['auroc'])
                        inner_rows.append({'task':TASK,'degree':DEG,'outer_fold':fold,'selector':sel,'method':method,
                                           'k':k,'lam':lam,'inner_fold':ii,**met,
                                           'selected_indices':'|'.join(map(str,cols.tolist()))})
                    perf.append((lam,float(np.mean(vals)),float(np.std(vals,ddof=1))))
                best=max(x[1] for x in perf)
                chosen=min(x[0] for x in perf if np.isclose(x[1],best,atol=1e-12,rtol=0))
                selected_nested.append({'task':TASK,'degree':DEG,'outer_fold':fold,'selector':sel,'method':method,
                                        'k':k,'chosen_lam':chosen,'inner_mean_auroc':best,
                                        'inner_sd_auroc':next(x[2] for x in perf if x[0]==chosen)})
                # strict nested outer evaluation
                cols=topk(dout+chosen*h,k); met=eval_cols(X,y,otr,ova,cols)
                ref=m0fr[(m0fr.fold==fold)&(m0fr.selector==sel)&(m0fr.k==k)].iloc[0]
                nested_outer.append({'task':TASK,'degree':DEG,'outer_fold':fold,'selector':sel,'method':method,
                                     'k':k,'chosen_lam':chosen,**met,'reference_auroc':float(ref.auroc),
                                     'delta_auroc_vs_reference':float(met['auroc']-ref.auroc),
                                     'reference_macro_ap':float(ref.macro_ap),'delta_macro_ap_vs_reference':float(met['macro_ap']-ref.macro_ap),
                                     'selected_indices':'|'.join(map(str,cols.tolist()))})
                # fixed-lam outer curve for final development tuning
                for lam in LAMBDAS:
                    cols=topk(dout+lam*h,k); met=eval_cols(X,y,otr,ova,cols)
                    fixed_outer.append({'task':TASK,'degree':DEG,'outer_fold':fold,'selector':sel,'method':method,
                                        'k':k,'lam':lam,**met,'selected_indices':'|'.join(map(str,cols.tolist()))})
            print(TASK,'D',DEG,'fold',fold,'selector',sel,'done',flush=True)

    innerdf=pd.DataFrame(inner_rows); innerdf.to_csv(out/'global_INNER_GAMMA_RESULTS.csv',index=False)
    pd.DataFrame(selected_nested).to_csv(out/'global_NESTED_SELECTED_GAMMA.csv',index=False)
    nodf=pd.DataFrame(nested_outer); nodf.to_csv(out/'global_NESTED_OUTER_RESULTS.csv',index=False)
    nagg=(nodf.groupby(['method','selector','k'],as_index=False)
          .agg(mean_guided_auroc=('auroc','mean'),sd_guided_auroc=('auroc','std'),
               mean_reference_auroc=('reference_auroc','mean'),mean_delta_auroc=('delta_auroc_vs_reference','mean'),
               sd_delta_auroc=('delta_auroc_vs_reference','std'),mean_guided_macro_ap=('macro_ap','mean'),
               mean_delta_macro_ap=('delta_macro_ap_vs_reference','mean'),
               lam_zero_folds=('chosen_lam',lambda x:int((np.asarray(x,float)==0).sum())),
               lam_nonzero_folds=('chosen_lam',lambda x:int((np.asarray(x,float)>0).sum()))))
    nagg.to_csv(out/'global_NESTED_AGGREGATE.csv',index=False)

    fdf=pd.DataFrame(fixed_outer); fdf.to_csv(out/'global_FIXED_GAMMA_OUTER_FOLDS.csv',index=False)
    curve=(fdf.groupby(['method','selector','k','lam'],as_index=False)
           .agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),
                mean_macro_ap=('macro_ap','mean'),sd_macro_ap=('macro_ap','std')))
    curve.to_csv(out/'global_GLOBAL_GAMMA_CV_CURVE.csv',index=False)
    picks=[]
    for (method,sel,k),g in curve.groupby(['method','selector','k']):
        best=g.mean_auroc.max()
        chosen=float(g.loc[np.isclose(g.mean_auroc,best,atol=1e-12,rtol=0),'lam'].min())
        rr=g[g.lam==chosen].iloc[0]
        picks.append({'task':TASK,'degree':DEG,'method':method,'selector':sel,'k':int(k),
                      'chosen_lam':chosen,'cv_mean_auroc':float(rr.mean_auroc),
                      'cv_sd_auroc':float(rr.sd_auroc),'cv_mean_macro_ap':float(rr.mean_macro_ap)})
    picks=pd.DataFrame(picks); picks.to_csv(out/'global_FINAL_SELECTED_GAMMA.csv',index=False)
    pd.DataFrame(score_audit).to_csv(out/'global_SELECTOR_SCORE_AUDIT.csv',index=False)
    gdf=pd.DataFrame(gate); gdf.to_csv(out/'global_GAMMA0_reference_GATE.csv',index=False)

    # Full-development final top-k using already frozen full reference selector params.
    pars=pd.read_csv(ROOT/'13_FINAL_DEVELOPMENT_TUNING'/TASK/'FINAL_SELECTOR_PARAMS.csv')
    final_rows=[]; fgate=[]
    for sel in SELECTORS:
        rr=pars[pars.selector==sel].iloc[0]
        if sel=='SIS':
            D=sis_score(X,y); solver='SIS_PEARSON'; nit=0; conv=True
        else:
            D,nit,conv=sparse_outer_reference(X,y,float(rr.C),float(rr.l1_ratio),SEED)
            solver='reference_SAGA'
            if not conv: raise RuntimeError(f'FULL_SELECTOR_NONCONV {TASK} {sel}')
        d=bounded_rank(D)
        for method in METHODS:
            h=H[method]
            for k in KGRID:
                lam=float(picks[(picks.method==method)&(picks.selector==sel)&(picks.k==k)].iloc[0].chosen_lam)
                m0cols=topk(d,k); zero=topk(d+0*h,k)
                exact=bool(np.array_equal(m0cols,zero)); 
                if not exact: raise RuntimeError('FULL_GAMMA0_FAIL')
                fgate.append({'task':TASK,'degree':DEG,'method':method,'selector':sel,'k':k,
                              'gamma0_exact_final_reference':exact,'chosen_lam':lam})
                cols=topk(d+lam*h,k)
                final_rows.append({'task':TASK,'degree':DEG,'method':method,'selector':sel,'k':k,
                                   'chosen_lam':lam,'full_score_solver':solver,'full_score_iter':nit,
                                   'selected_indices':'|'.join(map(str,cols.tolist())),
                                   'selected_genes':'|'.join(feat.iloc[cols].gene_symbol.astype(str).tolist())})
    pd.DataFrame(final_rows).to_csv(out/'global_FINAL_SELECTED_FEATURES.csv',index=False)
    pd.DataFrame(fgate).to_csv(out/'global_FINAL_GAMMA0_GATE.csv',index=False)
    status={'status':'PASS_global_DEVELOPMENT_ONLY','task':TASK,'degree':DEG,'lambda_grid':LAMBDAS,
            'k_grid':KGRID,'methods':METHODS,'all_gamma0_outer_gates_pass':bool(gdf.gamma0_exact_reference.all()),
            'all_gamma0_final_gates_pass':bool(pd.DataFrame(fgate).gamma0_exact_final_reference.all()),
            'uses_sealed_validation':False,'runtime_sec':round(time.time()-t0,2)}
    (out/'global_INTERNAL_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))
    print(nagg.to_string(index=False))
    print('\nFINAL GAMMA\n',picks.to_string(index=False))

main()
