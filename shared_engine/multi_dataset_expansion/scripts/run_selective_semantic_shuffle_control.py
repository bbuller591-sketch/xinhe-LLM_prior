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

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
TASK='BREAST_GSE25055_GSE25065'
SEED=2026091903
LAMS=np.array([0,0.03,0.1,0.3,1,3,10.0],float)
SELECTORS=['LASSO','ELASTICNET','SIS']
KGRID=[10,20,30]

ap=argparse.ArgumentParser()
ap.add_argument('--n-reps',type=int,default=1000)
ap.add_argument('--start-rep',type=int,default=1)
ap.add_argument('--output-dir',default=str(ROOT/'23_selective_SHUFFLE_CONTROL'))
ap.add_argument('--identity-only',action='store_true')
args=ap.parse_args()
OUT=Path(args.output_dir); OUT.mkdir(parents=True,exist_ok=True)

SRC=ROOT/'03_FROZEN_DATA'/TASK
reference=ROOT/'04_reference'/TASK
ROUTING=ROOT/'06_selective_ROUTING'/TASK
FINAL=ROOT/'13_FINAL_DEVELOPMENT_TUNING'/TASK
MEASP=ROOT/'11_selective_POSTPROCESS/selective_PAIR_SOURCE_MEASUREMENTS.csv'
SEALED=ROOT/'21_SEALED_VALIDATION/SEALED_VALIDATION_PRIMARY_RESULTS.csv'

X=np.load(SRC/'X_development.npy').astype(float)
y=np.load(SRC/'y_development.npy').astype(int)
Xte=np.load(SRC/'X_sealed_validation.npy').astype(float)
yte=np.load(SRC/'y_sealed_validation.npy').astype(int)
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

def data_anchor(D):
    r=rankdata(-D,method='average')
    u=1-(r-0.5)/len(D)
    return norm.ppf(np.clip(u,1e-6,1-1e-6))

def sis_score(A,b):
    bc=b-b.mean(); ac=A-A.mean(axis=0)
    den=np.sqrt((ac*ac).sum(axis=0)*(bc*bc).sum())
    return np.nan_to_num(np.abs((ac*bc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0)

def sparse_score(A,b,C,ratio,seed):
    sc=StandardScaler().fit(A); Z=sc.transform(A)
    pen='l1' if float(ratio)==1.0 else 'elasticnet'
    mod=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=pen,
        l1_ratio=None if pen=='l1' else float(ratio),max_iter=5000,tol=1e-4,
        random_state=int(seed),fit_intercept=True,n_jobs=1)
    mod.fit(Z,b)
    if int(mod.n_iter_[0])>=mod.max_iter:
        raise RuntimeError(f'SAGA_NONCONVERGENCE C={C} ratio={ratio} seed={seed}')
    return np.abs(mod.coef_[0])

def stable_topk(z,D,k):
    return np.lexsort((np.arange(len(z)),-D,-z))[:k]

# Evaluation cache: exact same downstream learner as frozen selective.
eval_cache={}
def eval_support(tr,va,cols,tag):
    key=(tag,tuple(int(i) for i in cols))
    if key in eval_cache: return eval_cache[key]
    A=X if tag!='SEALED' else X
    if tag=='SEALED':
        Atr=X[:,cols]; Ava=Xte[:,cols]; btr=y; bva=yte
    else:
        Atr=X[tr][:,cols]; Ava=X[va][:,cols]; btr=y[tr]; bva=y[va]
    sc=StandardScaler().fit(Atr)
    Ztr=sc.transform(Atr); Zva=sc.transform(Ava)
    mod=LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',class_weight='balanced',
                           max_iter=5000,random_state=2026091901)
    mod.fit(Ztr,btr)
    p=mod.predict_proba(Zva)[:,1]
    app=float(average_precision_score(bva,p)); apn=float(average_precision_score(1-bva,1-p))
    out={'auroc':float(roc_auc_score(bva,p)),'ap_positive':app,'macro_ap':0.5*(app+apn),
         'balanced_accuracy':float(balanced_accuracy_score(bva,p>=0.5))}
    eval_cache[key]=out
    return out

# Frozen data scores: outer fold and full-development.
outer_scores={}
for fold in range(1,6):
    tr=splits[(splits.fold==fold)&(splits.role=='outer_train')].sample_index.to_numpy(int)
    va=splits[(splits.fold==fold)&(splits.role=='outer_validation')].sample_index.to_numpy(int)
    for sel in SELECTORS:
        rr=sa[(sa.fold==fold)&(sa.selector==sel)].iloc[0]
        if sel=='SIS':
            D=sis_score(X[tr],y[tr])
        else:
            D=sparse_score(X[tr],y[tr],float(rr.chosen_C),float(rr.chosen_l1_ratio),2026091901+fold)
        outer_scores[(fold,sel)]={'tr':tr,'va':va,'D':D,'sD':data_anchor(D)}

full_scores={}
for sel in SELECTORS:
    rr=finalpars[finalpars.selector==sel].iloc[0]
    if sel=='SIS':
        D=sis_score(X,y)
    else:
        D=sparse_score(X,y,float(rr.C),float(rr.l1_ratio),2026091901)
    full_scores[sel]={'D':D,'sD':data_anchor(D)}

# Measurement bundle in canonical pair orientation.
meas=pd.read_csv(MEASP)
meas=meas[meas.task==TASK].copy()
pairs=sorted(meas.unordered_pair_id.astype(str).unique())
assert len(pairs)==1327
arms=sorted(meas.arm.astype(str).unique())
assert len(arms)==2 and len(meas)==2654
bundle={}
for pair,g in meas.groupby('unordered_pair_id'):
    a,b=str(pair).split('||',1)
    bundle[str(pair)]={}
    assert set(g.arm.astype(str))==set(arms) and len(g)==2
    for r in g.itertuples():
        if str(r.gene_i)==a and str(r.gene_j)==b:
            yy=float(r.hard_y_i_over_j)
        elif str(r.gene_i)==b and str(r.gene_j)==a:
            yy=1.0-float(r.hard_y_i_over_j)
        else:
            raise RuntimeError('MEAS_ORIENTATION '+str(pair))
        bundle[str(pair)][str(r.arm)]={'y':yy,'c':float(r.certainty_1_minus_H),'U':bool(r.either_U)}

pair_to_pos={p:i for i,p in enumerate(pairs)}
name_to_idx={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}

# Frozen route geometry per fold/selector/k.
route_geom={}
for fold in range(1,6):
  for sel in SELECTORS:
    for k in KGRID:
        cf=pd.read_csv(ROUTING/f'fold{fold}_{sel}'/f'K{k}_PAIR_CONFUSION.csv')
        if len(cf) and 'pair_type' in cf.columns:
            cf=cf[cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION'].copy()
        rows=[]
        for r in cf.itertuples():
            a,b=sorted([str(r.feature_A),str(r.feature_B)])
            pair=a+'||'+b
            assert pair in pair_to_pos
            rows.append((pair,name_to_idx[a],name_to_idx[b],float(r.actionable_boundary_score)))
        route_geom[(fold,sel,k)]=rows

# Frozen cross-fitted final route geometry (A_CF per pair).
fc=pd.read_csv(FINAL/'FINAL_CROSSFITTED_selective_CONSTRAINTS.csv')
final_geom={}
for (sel,k),g in fc.groupby(['selector','k']):
    q=g.groupby(['unordered_pair_id','gene_i','gene_j'],as_index=False).A_CF.first()
    rows=[]
    for r in q.itertuples():
        a,b=str(r.unordered_pair_id).split('||',1)
        rows.append((str(r.unordered_pair_id),name_to_idx[a],name_to_idx[b],float(r.A_CF)))
    final_geom[(str(sel),int(k))]=rows

def make_constraints(geom,perm_idx):
    # perm_idx maps target pair index -> donor pair index; the same donor's two-source bundle moves together.
    out=[]
    for pair,i,j,A in geom:
        donor=pairs[int(perm_idx[pair_to_pos[pair]])]
        for arm in arms:
            q=bundle[donor][arm]
            out.append((i,j,q['y'],A*q['c']/2.0))
    return out

def optimize_active(sD,constraints,lam):
    if lam==0 or not constraints:
        return sD.copy()
    ii=np.fromiter((r[0] for r in constraints),dtype=int)
    jj=np.fromiter((r[1] for r in constraints),dtype=int)
    yy=np.fromiter((r[2] for r in constraints),dtype=float)
    w=np.fromiter((r[3] for r in constraints),dtype=float)
    W=float(w.sum())
    if W<=0: return sD.copy()
    active=np.unique(np.r_[ii,jj])
    amap={int(v):q for q,v in enumerate(active)}
    ia=np.array([amap[int(v)] for v in ii],int)
    ja=np.array([amap[int(v)] for v in jj],int)
    s=sD[active].copy()
    p=len(sD)
    def fg(x):
        dz=x[ia]-x[ja]
        ce=np.logaddexp(0,dz)-yy*dz
        f=0.5*np.sum((x-s)**2)/p + float(lam)*np.dot(w,ce)/W
        grad=(x-s)/p
        rr=float(lam)*(w/W)*(expit(dz)-yy)
        np.add.at(grad,ia,rr); np.add.at(grad,ja,-rr)
        return float(f),grad
    res=minimize(lambda z:fg(z),s.copy(),jac=True,method='L-BFGS-B',
                 options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not res.success:
        x0=res.x if np.all(np.isfinite(res.x)) else s.copy()
        res=minimize(lambda z:fg(z),x0,jac=True,method='L-BFGS-B',
                     options={'maxiter':5000,'maxls':200,'ftol':1e-13,'gtol':1e-8})
    if not res.success:
        x0=res.x if np.all(np.isfinite(res.x)) else s.copy()
        res=minimize(lambda z:fg(z),x0,jac=True,method='BFGS',
                     options={'maxiter':5000,'gtol':1e-8})
    if not res.success:
        raise RuntimeError('selective_OPT_FAIL '+str(res.message))
    z=sD.copy(); z[active]=res.x
    return z

def run_one(rep,perm_idx):
    curve_rows=[]; final_rows=[]; rep_start=time.time()
    # Development fixed-lam curve, exactly matching final selective tuning logic.
    for sel in SELECTORS:
      for k in KGRID:
        lam_means=[]
        for lam in LAMS:
            vals=[]
            for fold in range(1,6):
                os=outer_scores[(fold,sel)]
                cc=make_constraints(route_geom[(fold,sel,k)],perm_idx)
                z=optimize_active(os['sD'],cc,float(lam))
                cols=stable_topk(z,os['D'],k)
                met=eval_support(os['tr'],os['va'],cols,f'F{fold}')
                vals.append(met['auroc'])
            lam_means.append((float(lam),float(np.mean(vals))))
            curve_rows.append({'replicate':rep,'selector':sel,'k':k,'lam':float(lam),
                               'dev_cv_mean_auroc':float(np.mean(vals))})
        best=max(v for _,v in lam_means)
        chosen=min(e for e,v in lam_means if np.isclose(v,best,atol=1e-12,rtol=0))
        fs=full_scores[sel]
        cc=make_constraints(final_geom[(sel,k)],perm_idx)
        z=optimize_active(fs['sD'],cc,chosen)
        cols=stable_topk(z,fs['D'],k)
        met=eval_support(None,None,cols,'SEALED')
        reference=float(reference_ref.loc[(sel,k),'auroc'])
        final_rows.append({'replicate':rep,'selector':sel,'k':k,'chosen_lam':chosen,
                           'dev_cv_best_mean_auroc':best,'sealed_auroc':met['auroc'],
                           'delta_auroc_vs_reference':met['auroc']-reference,
                           'sealed_macro_ap':met['macro_ap'],
                           'selected_indices':'|'.join(map(str,cols.tolist()))})
    f=pd.DataFrame(final_rows)
    summary={'replicate':rep,'mean_delta_auroc':float(f.delta_auroc_vs_reference.mean()),
             'median_delta_auroc':float(f.delta_auroc_vs_reference.median()),
             'positive_cells':int((f.delta_auroc_vs_reference>0).sum()),
             'zero_cells':int(np.isclose(f.delta_auroc_vs_reference,0,atol=1e-12).sum()),
             'negative_cells':int((f.delta_auroc_vs_reference<0).sum()),
             'mean_sealed_auroc':float(f.sealed_auroc.mean()),
             'runtime_sec':time.time()-rep_start}
    return pd.DataFrame(curve_rows),f,summary

# Identity sanity check: must reproduce the frozen LLM selective final tuning/support/sealed results.
identity=np.arange(len(pairs),dtype=int)
ic,ifin,isum=run_one(0,identity)
id_checks=[]
for r in ifin.itertuples():
    lam=float(actual_lam[(actual_lam.selector==r.selector)&(actual_lam.k==r.k)].iloc[0].chosen_lam)
    g=actual_support[(actual_support.selector==r.selector)&(actual_support.k==r.k)].sort_values('rank')
    supp='|'.join(map(str,g.feature_index.astype(int).tolist()))
    auc=float(selective_ref.loc[(r.selector,r.k),'auroc'])
    id_checks.append({'selector':r.selector,'k':r.k,
                      'lam_match':bool(np.isclose(r.chosen_lam,lam,atol=1e-12)),
                      'support_match':r.selected_indices==supp,
                      'sealed_auroc_abs_diff':abs(r.sealed_auroc-auc)})
idc=pd.DataFrame(id_checks)
idc.to_csv(OUT/'IDENTITY_SANITY_CHECK.csv',index=False)
id_status={'status':'PASS' if (idc.lam_match.all() and idc.support_match.all() and idc.sealed_auroc_abs_diff.max()<1e-12) else 'FAIL',
           'max_sealed_auroc_abs_diff':float(idc.sealed_auroc_abs_diff.max()),
           'all_lam_match':bool(idc.lam_match.all()),'all_support_match':bool(idc.support_match.all())}
(OUT/'IDENTITY_SANITY_STATUS.json').write_text(json.dumps(id_status,indent=2),encoding='utf-8')
print('IDENTITY',json.dumps(id_status),flush=True)
if id_status['status']!='PASS': raise SystemExit(3)
if args.identity_only: raise SystemExit(0)

# Resume-safe replication.
cell_path=OUT/'selective_SHUFFLE_REPLICATE_CELL_RESULTS.parquet'
curve_path=OUT/'selective_SHUFFLE_REPLICATE_ETA_CURVES.parquet'
sum_path=OUT/'selective_SHUFFLE_REPLICATE_SUMMARY.csv'
if cell_path.exists():
    oldf=pd.read_parquet(cell_path); oldc=pd.read_parquet(curve_path); olds=pd.read_csv(sum_path)
else:
    oldf=pd.DataFrame(); oldc=pd.DataFrame(); olds=pd.DataFrame()
done=set(oldf.replicate.astype(int).unique()) if len(oldf) else set()
allf=[oldf] if len(oldf) else []; allc=[oldc] if len(oldc) else []; alls=[olds] if len(olds) else []

t0=time.time()
end=args.start_rep+args.n_reps-1
for rep in range(args.start_rep,end+1):
    if rep in done: continue
    rng=np.random.default_rng(np.random.SeedSequence([SEED,rep]))
    perm=rng.permutation(len(pairs))
    c,f,s=run_one(rep,perm)
    allc.append(c); allf.append(f); alls.append(pd.DataFrame([s]))
    if rep%10==0 or rep==end:
        C=pd.concat(allc,ignore_index=True); F=pd.concat(allf,ignore_index=True); S=pd.concat(alls,ignore_index=True)
        C.to_parquet(curve_path,index=False,compression='zstd')
        F.to_parquet(cell_path,index=False,compression='zstd')
        S.to_csv(sum_path,index=False)
        prog={'status':'RUNNING','completed_replicates':int(F.replicate.nunique()),
              'target_replicates':int(end),'last_replicate':rep,
              'elapsed_sec':time.time()-t0,'eval_cache_entries':len(eval_cache)}
        (OUT/'PROGRESS.json').write_text(json.dumps(prog,indent=2),encoding='utf-8')
        print(json.dumps(prog),flush=True)

print('DONE_REPLICATES',pd.concat(allf,ignore_index=True).replicate.nunique(),flush=True)
