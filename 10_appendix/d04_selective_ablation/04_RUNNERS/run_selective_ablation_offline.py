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
import argparse,json,time,warnings,hashlib
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / '10_appendix/d04_selective_ablation'))
PKG=Path(str(REPRO_ROOT / 'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919'))
TASK='BREAST_GSE25055_GSE25065'
SEED=2026092401
LAMS=np.array([0,0.03,0.1,0.3,1,3,10.0],float)
SELECTORS=['LASSO','ELASTICNET','SIS']
KGRID=[10,20,30]
SRC=PKG/'DATA/FROZEN'/TASK
reference=PKG/'METHOD_ARTIFACTS/reference'
FINAL=PKG/'METHOD_ARTIFACTS/selective_FINAL'
MEASP=ROOT/'03_MEASUREMENTS/selective_ABLATION_PAIR_SOURCE_MEASUREMENTS_1834.csv'
SEALED=PKG/'RESULTS/SEALED_VALIDATION_PRIMARY_RESULTS.csv'
ROUTESETS=ROOT/'01_MANIFEST/BREAST_QB_Q_B_ROUTE_SETS.csv'
CAND=ROOT/'01_MANIFEST/BREAST_RANK_WINDOW_CANDIDATE_CELLS.csv'

ap=argparse.ArgumentParser()
ap.add_argument('--mode',choices=['identity','deterministic','random'],default='deterministic')
ap.add_argument('--n-reps',type=int,default=1000)
ap.add_argument('--start-rep',type=int,default=1)
ap.add_argument('--output-dir',default=str(ROOT/'06_E2_COMPONENT'))
args=ap.parse_args()
OUT=Path(args.output_dir); OUT.mkdir(parents=True,exist_ok=True)

X=np.load(SRC/'X_development.npy').astype(float); y=np.load(SRC/'y_development.npy').astype(int)
Xte=np.load(SRC/'X_sealed_validation.npy').astype(float); yte=np.load(SRC/'y_sealed_validation.npy').astype(int)
feat=pd.read_csv(SRC/'features_p2000.csv')
splits=pd.read_csv(reference/'OUTER_SPLITS.csv')
sa=pd.read_csv(reference/'reference_SELECTOR_AUDIT.csv')
finalpars=pd.read_csv(FINAL/'FINAL_SELECTOR_PARAMS.csv')
actual_lam=pd.read_csv(FINAL/'FINAL_SELECTED_ETA.csv')
actual_support=pd.read_csv(FINAL/'FINAL_selective_SELECTED_FEATURES.csv')
sealed_ref=pd.read_csv(SEALED)
sealed_ref=sealed_ref[(sealed_ref.task==TASK)&(sealed_ref.analysis=='PRIMARY')].copy()
reference_ref=sealed_ref[sealed_ref.method=='reference'].set_index(['selector','k'])
selective_ref=sealed_ref[sealed_ref.method=='selective'].set_index(['selector','k'])
name_to_idx={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}


def data_anchor(D):
    r=rankdata(-D,method='average'); u=1-(r-0.5)/len(D)
    return norm.ppf(np.clip(u,1e-6,1-1e-6))

def sis_score(A,b):
    bc=b-b.mean(); ac=A-A.mean(axis=0); den=np.sqrt((ac*ac).sum(axis=0)*(bc*bc).sum())
    return np.nan_to_num(np.abs((ac*bc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0)

def sparse_score(A,b,C,ratio,seed):
    sc=StandardScaler().fit(A); Z=sc.transform(A); pen='l1' if float(ratio)==1.0 else 'elasticnet'
    mod=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=pen,
        l1_ratio=None if pen=='l1' else float(ratio),max_iter=5000,tol=1e-4,
        random_state=int(seed),fit_intercept=True,n_jobs=1)
    mod.fit(Z,b)
    if int(mod.n_iter_[0])>=mod.max_iter: raise RuntimeError(f'SAGA_NONCONVERGENCE C={C} ratio={ratio} seed={seed}')
    return np.abs(mod.coef_[0])

def stable_topk(z,D,k):
    return np.lexsort((np.arange(len(z)),-D,-z))[:k]

eval_cache={}
def eval_support(tr,va,cols,tag):
    key=(tag,tuple(int(i) for i in cols))
    if key in eval_cache: return eval_cache[key]
    if tag=='SEALED': Atr=X[:,cols]; Ava=Xte[:,cols]; btr=y; bva=yte
    else: Atr=X[tr][:,cols]; Ava=X[va][:,cols]; btr=y[tr]; bva=y[va]
    sc=StandardScaler().fit(Atr); Ztr=sc.transform(Atr); Zva=sc.transform(Ava)
    mod=LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=2026091901)
    mod.fit(Ztr,btr); p=mod.predict_proba(Zva)[:,1]
    app=float(average_precision_score(bva,p)); apn=float(average_precision_score(1-bva,1-p))
    out={'auroc':float(roc_auc_score(bva,p)),'ap_positive':app,'macro_ap':0.5*(app+apn),'balanced_accuracy':float(balanced_accuracy_score(bva,p>=0.5))}
    eval_cache[key]=out; return out

outer_scores={}
for fold in range(1,6):
    tr=splits[(splits.fold==fold)&(splits.role=='outer_train')].sample_index.to_numpy(int)
    va=splits[(splits.fold==fold)&(splits.role=='outer_validation')].sample_index.to_numpy(int)
    for sel in SELECTORS:
        rr=sa[(sa.fold==fold)&(sa.selector==sel)].iloc[0]
        D=sis_score(X[tr],y[tr]) if sel=='SIS' else sparse_score(X[tr],y[tr],float(rr.chosen_C),float(rr.chosen_l1_ratio),2026091901+fold)
        outer_scores[(fold,sel)]={'tr':tr,'va':va,'D':D,'sD':data_anchor(D)}
full_scores={}
for sel in SELECTORS:
    rr=finalpars[finalpars.selector==sel].iloc[0]
    D=sis_score(X,y) if sel=='SIS' else sparse_score(X,y,float(rr.C),float(rr.l1_ratio),2026091901)
    full_scores[sel]={'D':D,'sD':data_anchor(D)}

meas=pd.read_csv(MEASP); meas=meas[meas.task==TASK].copy()
arms=sorted(meas.arm.astype(str).unique()); assert len(arms)==2
bundle={}
for pair,g in meas.groupby('unordered_pair_id'):
    a,b=str(pair).split('||',1); bundle[str(pair)]={}
    assert set(g.arm.astype(str))==set(arms) and len(g)==2
    for r in g.itertuples():
        if str(r.gene_i)==a and str(r.gene_j)==b: yy=float(r.hard_y_i_over_j)
        elif str(r.gene_i)==b and str(r.gene_j)==a: yy=1.0-float(r.hard_y_i_over_j)
        else: raise RuntimeError('MEAS_ORIENTATION '+str(pair))
        bundle[str(pair)][str(r.arm)]={'y':yy,'c':float(r.certainty_1_minus_H),'U':bool(r.either_U)}

routes=pd.read_csv(ROUTESETS)
cand=pd.read_csv(CAND)

def rows_from_df(df, score_col):
    rows=[]
    for r in df.itertuples():
        a,b=str(r.unordered_pair_id).split('||',1)
        rows.append((str(r.unordered_pair_id),name_to_idx[a],name_to_idx[b],float(getattr(r,score_col))))
    return rows

def build_fold_geom(arm):
    out={(fold,sel,k):[] for fold in range(1,6) for sel in SELECTORS for k in KGRID}
    if arm in ['QB_MAIN','Q_ONLY','B_ONLY']:
        score={'QB_MAIN':'QB','Q_ONLY':'Q','B_ONLY':'B'}[arm]
        for (fold,sel,k),g in routes[routes.arm==arm].groupby(['fold','selector','k']):
            out[(int(fold),str(sel),int(k))]=rows_from_df(g,score)
    else:
        raise ValueError(arm)
    return out

def build_final_geom(arm):
    if arm=='QB_MAIN':
        fc=pd.read_csv(FINAL/'FINAL_CROSSFITTED_selective_CONSTRAINTS.csv')
        out={}
        for (sel,k),g in fc.groupby(['selector','k']):
            q=g.groupby(['unordered_pair_id','gene_i','gene_j'],as_index=False).A_CF.first(); rows=[]
            for r in q.itertuples():
                a,b=str(r.unordered_pair_id).split('||',1); rows.append((str(r.unordered_pair_id),name_to_idx[a],name_to_idx[b],float(r.A_CF)))
            out[(str(sel),int(k))]=rows
        return out
    score={'Q_ONLY':'Q','B_ONLY':'B'}[arm]
    out={}
    for sel in SELECTORS:
      for k in KGRID:
        vals={}
        for fold in range(1,6):
            g=routes[(routes.arm==arm)&(routes.fold==fold)&(routes.selector==sel)&(routes.k==k)]
            for r in g.itertuples(): vals.setdefault(str(r.unordered_pair_id),[]).append(float(getattr(r,score)))
        rows=[]
        for pair,vs in vals.items():
            a,b=pair.split('||',1); rows.append((pair,name_to_idx[a],name_to_idx[b],float(sum(vs)/5.0)))
        out[(sel,k)]=rows
    return out

def build_random_fold_geom(rep):
    out={}
    for (fold,sel,k),g in cand.groupby(['fold','selector','k']):
        m=int(g.main_m_cell.iloc[0]); gg=g.sort_values(['unordered_pair_id']).reset_index(drop=True)
        ss=np.random.SeedSequence([SEED, int(rep), int(fold), SELECTORS.index(str(sel)), int(k)])
        rng=np.random.default_rng(ss)
        idx=np.sort(rng.choice(len(gg),size=m,replace=False)) if m>0 else []
        sub=gg.iloc[idx].copy() if m>0 else gg.iloc[[]].copy()
        out[(int(fold),str(sel),int(k))]=rows_from_df(sub,'QB')
    return out

def build_random_final_geom(fold_geom):
    out={}
    for sel in SELECTORS:
      for k in KGRID:
        vals={}
        for fold in range(1,6):
            for pair,i,j,A in fold_geom[(fold,sel,k)]: vals.setdefault(pair,[]).append(A)
        rows=[]
        for pair,vs in vals.items():
            a,b=pair.split('||',1); rows.append((pair,name_to_idx[a],name_to_idx[b],float(sum(vs)/5.0)))
        out[(sel,k)]=rows
    return out

def make_constraints(geom, use_certainty=True):
    out=[]
    for pair,i,j,A in geom:
        for arm in arms:
            q=bundle[pair][arm]; c=q['c'] if use_certainty else 1.0
            out.append((i,j,q['y'],A*c/2.0))
    return out

def optimize_active(sD,constraints,lam):
    if lam==0 or not constraints: return sD.copy()
    ii=np.fromiter((r[0] for r in constraints),dtype=int); jj=np.fromiter((r[1] for r in constraints),dtype=int)
    yy=np.fromiter((r[2] for r in constraints),dtype=float); w=np.fromiter((r[3] for r in constraints),dtype=float)
    W=float(w.sum())
    if W<=0: return sD.copy()
    active=np.unique(np.r_[ii,jj]); amap={int(v):q for q,v in enumerate(active)}
    ia=np.array([amap[int(v)] for v in ii],int); ja=np.array([amap[int(v)] for v in jj],int); s=sD[active].copy(); p=len(sD)
    def fg(x):
        dz=x[ia]-x[ja]; ce=np.logaddexp(0,dz)-yy*dz
        f=0.5*np.sum((x-s)**2)/p + float(lam)*np.dot(w,ce)/W
        grad=(x-s)/p; rr=float(lam)*(w/W)*(expit(dz)-yy)
        np.add.at(grad,ia,rr); np.add.at(grad,ja,-rr)
        return float(f),grad
    res=minimize(lambda z:fg(z),s.copy(),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not res.success:
        x0=res.x if np.all(np.isfinite(res.x)) else s.copy(); res=minimize(lambda z:fg(z),x0,jac=True,method='L-BFGS-B',options={'maxiter':5000,'maxls':200,'ftol':1e-13,'gtol':1e-8})
    if not res.success:
        x0=res.x if np.all(np.isfinite(res.x)) else s.copy(); res=minimize(lambda z:fg(z),x0,jac=True,method='BFGS',options={'maxiter':5000,'gtol':1e-8})
    if not res.success: raise RuntimeError('selective_OPT_FAIL '+str(res.message))
    z=sD.copy(); z[active]=res.x; return z

def run_arm(label, fold_geom, final_geom, use_certainty=True, replicate=0):
    curve_rows=[]; final_rows=[]
    for sel in SELECTORS:
      for k in KGRID:
        lam_means=[]
        for lam in LAMS:
            vals=[]
            for fold in range(1,6):
                os=outer_scores[(fold,sel)]
                cc=make_constraints(fold_geom[(fold,sel,k)],use_certainty)
                z=optimize_active(os['sD'],cc,float(lam)); cols=stable_topk(z,os['D'],k)
                met=eval_support(os['tr'],os['va'],cols,f'{label}_R{replicate}_F{fold}')
                vals.append(met['auroc'])
            lam_means.append((float(lam),float(np.mean(vals))))
            curve_rows.append({'arm':label,'replicate':replicate,'selector':sel,'k':k,'lam':float(lam),'dev_cv_mean_auroc':float(np.mean(vals))})
        best=max(v for _,v in lam_means); chosen=min(e for e,v in lam_means if np.isclose(v,best,atol=1e-12,rtol=0))
        fs=full_scores[sel]
        cc=make_constraints(final_geom[(sel,k)],use_certainty)
        z=optimize_active(fs['sD'],cc,chosen); cols=stable_topk(z,fs['D'],k)
        met=eval_support(None,None,cols,'SEALED')
        reference=float(reference_ref.loc[(sel,k),'auroc'])
        final_rows.append({'arm':label,'replicate':replicate,'selector':sel,'k':k,'chosen_lam':chosen,'dev_cv_best_mean_auroc':best,
                           'sealed_auroc':met['auroc'],'delta_auroc_vs_reference':met['auroc']-reference,'sealed_macro_ap':met['macro_ap'],
                           'selected_indices':'|'.join(map(str,cols.tolist()))})
    return pd.DataFrame(curve_rows),pd.DataFrame(final_rows)

def identity_gate():
    fg=build_fold_geom('QB_MAIN'); ffg=build_final_geom('QB_MAIN')
    c,f=run_arm('QB_MAIN_WITH_C_IDENTITY',fg,ffg,True,0)
    checks=[]
    for r in f.itertuples():
        lam=float(actual_lam[(actual_lam.selector==r.selector)&(actual_lam.k==r.k)].iloc[0].chosen_lam)
        g=actual_support[(actual_support.selector==r.selector)&(actual_support.k==r.k)].sort_values('rank')
        supp='|'.join(map(str,g.feature_index.astype(int).tolist()))
        auc=float(selective_ref.loc[(r.selector,r.k),'auroc'])
        checks.append({'selector':r.selector,'k':r.k,'lam_match':bool(np.isclose(r.chosen_lam,lam,atol=1e-12)),
                      'support_match':r.selected_indices==supp,'sealed_auroc_abs_diff':abs(r.sealed_auroc-auc)})
    idc=pd.DataFrame(checks); idc.to_csv(ROOT/'05_IDENTITY/IDENTITY_SANITY_CHECK.csv',index=False)
    status={'status':'PASS' if (idc.lam_match.all() and idc.support_match.all() and idc.sealed_auroc_abs_diff.max()<1e-12) else 'FAIL',
            'max_sealed_auroc_abs_diff':float(idc.sealed_auroc_abs_diff.max()),'all_lam_match':bool(idc.lam_match.all()),'all_support_match':bool(idc.support_match.all())}
    (ROOT/'05_IDENTITY/IDENTITY_SANITY_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    c.to_csv(ROOT/'05_IDENTITY/IDENTITY_ETA_CURVE.csv',index=False); f.to_csv(ROOT/'05_IDENTITY/IDENTITY_FINAL.csv',index=False)
    print('IDENTITY',json.dumps(status),flush=True)
    if status['status']!='PASS': raise SystemExit(3)

if args.mode=='identity':
    identity_gate(); raise SystemExit(0)

identity_gate()
if args.mode=='deterministic':
    arms_to_run=[('QB_MAIN_WITH_C','QB_MAIN',True),('Q_ONLY_WITH_C','Q_ONLY',True),('B_ONLY_WITH_C','B_ONLY',True),('QB_MAIN_NO_C','QB_MAIN',False)]
    curves=[]; finals=[]
    for label,arm,useC in arms_to_run:
        fg=build_fold_geom(arm); ffg=build_final_geom(arm)
        c,f=run_arm(label,fg,ffg,useC,0); curves.append(c); finals.append(f)
        print('DONE_ARM',label,flush=True)
    C=pd.concat(curves,ignore_index=True); F=pd.concat(finals,ignore_index=True)
    C.to_csv(OUT/'DETERMINISTIC_ABLATION_ETA_CURVES.csv',index=False)
    F.to_csv(OUT/'DETERMINISTIC_ABLATION_FINAL.csv',index=False)
    summary=F.groupby('arm').agg(mean_delta_auroc=('delta_auroc_vs_reference','mean'),median_delta_auroc=('delta_auroc_vs_reference','median'),mean_sealed_auroc=('sealed_auroc','mean'),mean_macro_ap=('sealed_macro_ap','mean'),positive_cells=('delta_auroc_vs_reference',lambda s:int((s>0).sum()))).reset_index()
    summary.to_csv(OUT/'DETERMINISTIC_ABLATION_SUMMARY.csv',index=False)
    print(summary.to_string(index=False),flush=True)

if args.mode=='random':
    cell_path=OUT/'RANDOM_ROUTING_REPLICATE_CELL_RESULTS.parquet'; curve_path=OUT/'RANDOM_ROUTING_REPLICATE_ETA_CURVES.parquet'; sum_path=OUT/'RANDOM_ROUTING_REPLICATE_SUMMARY.csv'
    oldf=pd.read_parquet(cell_path) if cell_path.exists() else pd.DataFrame()
    oldc=pd.read_parquet(curve_path) if curve_path.exists() else pd.DataFrame()
    olds=pd.read_csv(sum_path) if sum_path.exists() else pd.DataFrame()
    done=set(oldf.replicate.astype(int).unique()) if len(oldf) else set()
    allf=[oldf] if len(oldf) else []; allc=[oldc] if len(oldc) else []; alls=[olds] if len(olds) else []
    end=args.start_rep+args.n_reps-1; t0=time.time()
    for rep in range(args.start_rep,end+1):
        if rep in done: continue
        fg=build_random_fold_geom(rep); ffg=build_random_final_geom(fg)
        c,f=run_arm('RANDOM_QB_ROUTE_WITH_C',fg,ffg,True,rep)
        s={'replicate':rep,'mean_delta_auroc':float(f.delta_auroc_vs_reference.mean()),'median_delta_auroc':float(f.delta_auroc_vs_reference.median()),'positive_cells':int((f.delta_auroc_vs_reference>0).sum()),'zero_cells':int(np.isclose(f.delta_auroc_vs_reference,0,atol=1e-12).sum()),'negative_cells':int((f.delta_auroc_vs_reference<0).sum()),'mean_sealed_auroc':float(f.sealed_auroc.mean()),'runtime_sec':time.time()-t0}
        allc.append(c); allf.append(f); alls.append(pd.DataFrame([s]))
        if rep%10==0 or rep==end:
            C=pd.concat(allc,ignore_index=True); F=pd.concat(allf,ignore_index=True); S=pd.concat(alls,ignore_index=True)
            C.to_parquet(curve_path,index=False,compression='zstd'); F.to_parquet(cell_path,index=False,compression='zstd'); S.to_csv(sum_path,index=False)
            prog={'status':'RUNNING','completed_replicates':int(F.replicate.nunique()),'target_replicates':int(end),'last_replicate':rep,'elapsed_sec':time.time()-t0,'eval_cache_entries':len(eval_cache)}
            (OUT/'PROGRESS.json').write_text(json.dumps(prog,indent=2)+'\n')
            print(json.dumps(prog),flush=True)
    print('DONE_RANDOM',pd.concat(allf,ignore_index=True).replicate.nunique(),flush=True)
