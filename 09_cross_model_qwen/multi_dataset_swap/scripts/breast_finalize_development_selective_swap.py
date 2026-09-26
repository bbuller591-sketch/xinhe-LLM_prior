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
import sys,json,time,hashlib,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
sys.path.insert(0,str(ROOT/'scripts'))
import breast_run_nested_selective_swap as selective
import run_strict_nested_selective_routing as ROUTE

LAMS=selective.LAMS; KGRID=selective.KGRID; SELECTORS=selective.SELECTORS
SEED=selective.SEED
MEAS=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/04_BREAST/selective_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B.csv')))

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def final_inner_tune(X,y,selector):
    """Frozen reference hyperparameter rule, applied to all development data."""
    from sklearn.model_selection import StratifiedKFold
    CGRID=[0.01,0.03,0.1,0.3,1.0,3.0]
    ALPHAS=[0.2,0.5,0.8]
    MAXK=30
    inner=StratifiedKFold(n_splits=4,shuffle=True,random_state=SEED)
    alphas=[1.0] if selector=='LASSO' else ALPHAS
    rec={(C,a):{'auc':[],'nnz':[],'converged':[]} for a in alphas for C in CGRID}
    for ii,(tr,va) in enumerate(inner.split(X,y),1):
        sc=StandardScaler().fit(X[tr])
        Xt=sc.transform(X[tr]); Xv=sc.transform(X[va])
        for a in alphas:
            penalty='l1' if selector=='LASSO' else 'elasticnet'
            mod=LogisticRegression(C=CGRID[0],solver='saga',class_weight='balanced',
                penalty=penalty,l1_ratio=None if penalty=='l1' else a,
                max_iter=5000,tol=1e-4,
                random_state=SEED+100*ii+int(a*10),
                fit_intercept=True,n_jobs=1,warm_start=True)
            for C in CGRID:
                mod.set_params(C=C)
                mod.fit(Xt,y[tr])
                pred=mod.predict_proba(Xv)[:,1]
                q=rec[(C,a)]
                q['auc'].append(float(selective.roc_auc_score(y[va],pred)))
                q['nnz'].append(int(np.sum(np.abs(mod.coef_[0])>1e-12)))
                q['converged'].append(bool(int(mod.n_iter_[0])<mod.max_iter))
    rows=[]
    for (C,a),q in rec.items():
        rows.append({'C':C,'l1_ratio':a,'mean_inner_auroc':float(np.mean(q['auc'])),
                     'sd_inner_auroc':float(np.std(q['auc'],ddof=1)),
                     'min_nnz':int(min(q['nnz'])),'median_nnz':float(np.median(q['nnz'])),
                     'max_nnz':int(max(q['nnz'])),'all_converged':bool(all(q['converged']))})
    tab=pd.DataFrame(rows)
    admiss=tab[tab.min_nnz>=MAXK].copy()
    fallback=False
    if len(admiss)==0:
        admiss=tab.copy(); fallback=True
        admiss=admiss.sort_values(['min_nnz','mean_inner_auroc','C','l1_ratio'],
                                  ascending=[False,False,True,False],kind='mergesort')
    else:
        admiss=admiss.sort_values(['mean_inner_auroc','C','l1_ratio'],
                                  ascending=[False,True,False],kind='mergesort')
    ch=admiss.iloc[0]
    return tab,{'C':float(ch.C),'l1_ratio':float(ch.l1_ratio),
                'inner_auroc':float(ch.mean_inner_auroc),'min_nnz':int(ch.min_nnz),
                'admissibility_fallback':bool(fallback)}

def final_full_sparse_score(X,y,C,ratio):
    sc=StandardScaler().fit(X); Z=sc.transform(X)
    penalty='l1' if float(ratio)==1.0 else 'elasticnet'
    mod=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=penalty,
        l1_ratio=None if penalty=='l1' else float(ratio),max_iter=5000,tol=1e-4,
        random_state=SEED,fit_intercept=True,n_jobs=1)
    mod.fit(Z,y)
    nit=int(mod.n_iter_[0]); conv=bool(nit<mod.max_iter)
    if conv:
        return np.abs(mod.coef_[0]),'reference_SAGA',nit,True
    D,ok,nnz=ROUTE.fit_sparse(X,y,float(C),float(ratio),SEED)
    return D,'EXACT_WEIGHTED_PROXNEWTON_FALLBACK',nit,bool(ok)

def final_constraints(task,selector,k,feat):
    score_by_pair={}
    orient={}
    for fold in range(1,6):
        fp=ROOT/'06_selective_ROUTING'/task/f'fold{fold}_{selector}'/f'K{k}_PAIR_CONFUSION.csv'
        cf=pd.read_csv(fp)
        foldmap={}
        if 'pair_type' in cf.columns and len(cf):
            a=cf[cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION']
            for r in a.itertuples():
                p='||'.join(sorted([str(r.feature_A),str(r.feature_B)]))
                foldmap[p]=float(r.actionable_boundary_score)
                orient[p]=tuple(sorted([str(r.feature_A),str(r.feature_B)]))
        for p in set(score_by_pair)|set(foldmap):
            pass
        # append per fold later after discovering union
        score_by_pair[fold]=foldmap
    union=sorted(set().union(*[set(v) for v in score_by_pair.values()]))
    name_to_idx={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}
    mt=MEAS[MEAS.task==task]
    mmap={(str(r.unordered_pair_id),str(r.arm)):r for r in mt.itertuples()}
    rows=[]; cons=[]
    for pair in union:
        vals=[score_by_pair[f].get(pair,0.0) for f in range(1,6)]
        acf=float(np.mean(vals))
        a,b=pair.split('||',1)
        avail=[q for (pp,arm),q in mmap.items() if pp==pair]
        m=len(avail)
        for q in avail:
            if str(q.gene_i)==a and str(q.gene_j)==b: yy=float(q.hard_y_i_over_j)
            elif str(q.gene_i)==b and str(q.gene_j)==a: yy=1.0-float(q.hard_y_i_over_j)
            else: raise RuntimeError('ORIENT '+pair)
            c=float(q.certainty_1_minus_H)
            w=acf*c/m if m else 0.0
            rec={'task':task,'selector':selector,'k':k,'unordered_pair_id':pair,
                 'gene_i':a,'gene_j':b,'arm':str(q.arm),'A_CF':acf,
                 'actionable_fold_count':int(sum(v>0 for v in vals)),
                 'A_data_fold1':vals[0],'A_data_fold2':vals[1],'A_data_fold3':vals[2],
                 'A_data_fold4':vals[3],'A_data_fold5':vals[4],
                 'y_i_over_j':yy,'certainty_1_minus_H':c,'weight':w,
                 'either_U':bool(q.either_U)}
            rows.append(rec)
            cons.append({'i':name_to_idx[a],'j':name_to_idx[b],'y':yy,'w':w,
                         'arm':str(q.arm),'pair':pair,'certainty':c,'actionable':acf,
                         'either_U':bool(q.either_U)})
    return pd.DataFrame(rows),cons

def run_task(task):
    t0=time.time()
    src=ROOT/'03_FROZEN_DATA'/task; reference=ROOT/'04_reference'/task
    out=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/04_BREAST/final_development_tuning'))/task; out.mkdir(parents=True,exist_ok=True)
    X=np.load(src/'X_development.npy').astype(float)
    y=np.load(src/'y_development.npy').astype(int)
    feat=pd.read_csv(src/'features_p2000.csv')
    splits=pd.read_csv(reference/'OUTER_SPLITS.csv')
    sa=pd.read_csv(reference/'reference_SELECTOR_AUDIT.csv')

    # 1) fixed-lam outer-fold CV curve for final lam tuning.
    curve_rows=[]
    for fold in range(1,6):
        tr=splits[(splits.fold==fold)&(splits.role=='outer_train')].sample_index.to_numpy(int)
        va=splits[(splits.fold==fold)&(splits.role=='outer_validation')].sample_index.to_numpy(int)
        for sel in SELECTORS:
            rr=sa[(sa.fold==fold)&(sa.selector==sel)].iloc[0]
            C=None if sel=='SIS' else float(rr.chosen_C)
            ratio=None if sel=='SIS' else float(rr.chosen_l1_ratio)
            D,nit,conv,solver=selective.selector_score(X[tr],y[tr],sel,C,ratio,SEED+fold,stage='outer')
            if not conv: raise RuntimeError(f'OUTER_SCORE_NONCONV {task} {fold} {sel}')
            sD=selective.data_anchor(D)
            for k in KGRID:
                cc,_=selective.build_constraints(task,fold,sel,k,feat)
                for lam in LAMS:
                    z,ok,onit,W=selective.optimize_selective(sD,cc,lam)
                    cols=selective.stable_topk(z,D,k)
                    met=selective.eval_selected(X,y,tr,va,cols)
                    curve_rows.append({'task':task,'fold':fold,'selector':sel,'k':k,'lam':lam,
                                       'solver':solver,'selector_iter':nit,'opt_success':ok,
                                       'constraint_weight_sum':W,**met,
                                       'selected_indices':'|'.join(map(str,cols.tolist()))})
    curve=pd.DataFrame(curve_rows)
    curve.to_csv(out/'FINAL_ETA_CV_FOLD_RESULTS.csv',index=False)
    agg=(curve.groupby(['selector','k','lam'],as_index=False)
         .agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),
              mean_macro_ap=('macro_ap','mean'),sd_macro_ap=('macro_ap','std')))
    agg.to_csv(out/'FINAL_ETA_CV_CURVE.csv',index=False)
    chosen=[]
    for (sel,k),g in agg.groupby(['selector','k']):
        best=float(g.mean_auroc.max())
        lam=float(g.loc[np.isclose(g.mean_auroc,best,atol=1e-12,rtol=0),'lam'].min())
        rr=g[g.lam==lam].iloc[0]
        chosen.append({'task':task,'selector':sel,'k':int(k),'chosen_lam':lam,
                       'cv_mean_auroc':float(rr.mean_auroc),'cv_sd_auroc':float(rr.sd_auroc),
                       'cv_mean_macro_ap':float(rr.mean_macro_ap)})
    chosen=pd.DataFrame(chosen)
    chosen.to_csv(out/'FINAL_SELECTED_ETA.csv',index=False)

    # 2) tune final selector hyperparameters on all development data using same reference 4-fold rule.
    param_rows=[]; full_scores={}
    for sel in SELECTORS:
        if sel=='SIS':
            D=selective.sis_score(X,y)
            full_scores[sel]=(D,'SIS_PEARSON',0,True)
            param_rows.append({'task':task,'selector':sel,'C':np.nan,'l1_ratio':np.nan,
                               'cv_mean_auroc':np.nan,'min_nnz':np.nan,
                               'admissibility_fallback':False,'full_score_solver':'SIS_PEARSON',
                               'full_score_iter':0,'full_score_converged':True})
        else:
            tab,params=final_inner_tune(X,y,sel)
            tab.insert(0,'selector',sel); tab.to_csv(out/f'FINAL_{sel}_HYPERPARAM_CV.csv',index=False)
            D,solver,nit,conv=final_full_sparse_score(X,y,params['C'],params['l1_ratio'])
            if not conv: raise RuntimeError(f'FINAL_FULL_SCORE_NONCONV {task} {sel}')
            full_scores[sel]=(D,solver,nit,conv)
            param_rows.append({'task':task,'selector':sel,'C':params['C'],'l1_ratio':params['l1_ratio'],
                               'cv_mean_auroc':params['inner_auroc'],'min_nnz':params['min_nnz'],
                               'admissibility_fallback':params['admissibility_fallback'],
                               'full_score_solver':solver,'full_score_iter':nit,'full_score_converged':conv})
    paramsdf=pd.DataFrame(param_rows)
    paramsdf.to_csv(out/'FINAL_SELECTOR_PARAMS.csv',index=False)

    # 3) cross-fitted final graph + train-all final rankings.
    graph_parts=[]; rank_rows=[]; gate=[]
    for sel in SELECTORS:
        D,solver,nit,conv=full_scores[sel]
        sD=selective.data_anchor(D)
        reference_order=np.lexsort((np.arange(len(D)),-D))
        for k in KGRID:
            gdf,cc=final_constraints(task,sel,k,feat)
            graph_parts.append(gdf)
            lam=float(chosen[(chosen.selector==sel)&(chosen.k==k)].iloc[0].chosen_lam)
            z0,ok0,n0,W0=selective.optimize_selective(sD,cc,0)
            top0=selective.stable_topk(z0,D,k)
            exact0=bool(np.array_equal(top0,reference_order[:k]))
            if not exact0: raise RuntimeError(f'FINAL_ETA0_GATE_FAIL {task} {sel} {k}')
            z,ok,onit,W=selective.optimize_selective(sD,cc,lam)
            top=selective.stable_topk(z,D,k)
            gate.append({'task':task,'selector':sel,'k':k,'eta0_exact_final_reference':exact0,
                         'chosen_lam':lam,'n_source_constraints':len(cc),'constraint_weight_sum':W})
            for rank,j in enumerate(top,1):
                rank_rows.append({'task':task,'selector':sel,'k':k,'chosen_lam':lam,'rank':rank,
                                  'feature_index':int(j),'gene_symbol':str(feat.iloc[j].gene_symbol),
                                  'selective_score_z':float(z[j]),'data_score_D':float(D[j]),
                                  'in_final_reference_topk':bool(j in set(reference_order[:k]))})
    graph=pd.concat(graph_parts,ignore_index=True) if graph_parts else pd.DataFrame()
    graph.to_csv(out/'FINAL_CROSSFITTED_selective_CONSTRAINTS.csv',index=False)
    pd.DataFrame(rank_rows).to_csv(out/'FINAL_selective_SELECTED_FEATURES.csv',index=False)
    pd.DataFrame(gate).to_csv(out/'FINAL_ETA0_GATE.csv',index=False)

    status={'status':'FINAL_DEVELOPMENT_TUNING_FROZEN','task':task,'uses_sealed_validation':False,
            'lambda_grid':LAMS,'k_grid':KGRID,'n_lam_curve_rows':len(curve),
            'n_crossfitted_source_constraints':len(graph),
            'all_eta0_final_gates_pass':bool(pd.DataFrame(gate).eta0_exact_final_reference.all()),
            'runtime_sec':round(time.time()-t0,2)}
    files=['FINAL_ETA_CV_FOLD_RESULTS.csv','FINAL_ETA_CV_CURVE.csv','FINAL_SELECTED_ETA.csv',
           'FINAL_SELECTOR_PARAMS.csv','FINAL_CROSSFITTED_selective_CONSTRAINTS.csv',
           'FINAL_selective_SELECTED_FEATURES.csv','FINAL_ETA0_GATE.csv']
    status['sha256']={f:sha(out/f) for f in files}
    (out/'FINAL_DEVELOPMENT_TUNING_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('\nTASK',task)
    print(chosen.to_string(index=False))
    print(paramsdf.to_string(index=False))
    print(json.dumps(status,indent=2))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682','all'],default='all')
    args=ap.parse_args()
    tasks=['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682'] if args.task=='all' else [args.task]
    for t in tasks:
        run_task(t)
