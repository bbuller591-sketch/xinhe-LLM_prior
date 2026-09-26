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
import argparse,json,math,time,warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from numba import float64
from skglm.datafits import Logistic
from skglm.penalties import L1_plus_L2
from skglm.solvers import ProxNewton
from skglm import GeneralizedLinearEstimator

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
FROZEN=ROOT/'03_FROZEN_DATA'
reference=ROOT/'04_reference'
OUTROOT=ROOT/'06_selective_ROUTING'
OUTROOT.mkdir(parents=True,exist_ok=True)

B=200
FRAC=0.80
WINDOW=5
ACT_THR=0.25
KGRID=[10,20,30]
BASE_SEED=2026091902

class WeightedLogistic(Logistic):
    def __init__(self, sample_weights):
        self.sample_weights=np.asarray(sample_weights,dtype=np.float64)
    def get_spec(self):
        return (('sample_weights', float64[:]),)
    def params_to_dict(self):
        return {'sample_weights':self.sample_weights}
    def raw_grad(self,y,Xw):
        sw=self.sample_weights
        return -sw*y/(1+np.exp(y*Xw))/sw.sum()
    def raw_hessian(self,y,Xw):
        sw=self.sample_weights
        z=np.exp(-y*Xw)
        return sw*z/(1+z)**2/sw.sum()
    def get_lipschitz(self,X,y):
        sw=self.sample_weights
        return (sw[:,None]*(X**2)).sum(axis=0)/(4*sw.sum())
    def value(self,y,w,Xw):
        sw=self.sample_weights
        return np.sum(sw*np.log(1+np.exp(-y*Xw)))/sw.sum()
    def gradient_scalar(self,X,y,w,Xw,j):
        sw=self.sample_weights
        return (-X[:,j]@(sw*y/(1+np.exp(y*Xw))))/sw.sum()
    def gradient(self,X,y,Xw):
        return X.T@self.raw_grad(y,Xw)
    def intercept_update_step(self,y,Xw):
        sw=self.sample_weights
        return np.sum(sw*(-y/(1+np.exp(y*Xw))))/sw.sum()/4

def fit_sparse(X,y,C,l1_ratio,seed):
    # Exact same convex penalized-logistic objective as the frozen sklearn/SAGA
    # selector, but solved to tight KKT tolerance with explicit balanced weights.
    sc=StandardScaler().fit(X)
    Xt=sc.transform(X)
    n=len(y); n1=int(np.sum(y==1)); n0=int(np.sum(y==0))
    sw=np.where(y==1,n/(2*n1),n/(2*n0)).astype(np.float64)
    alpha=1.0/(float(C)*float(sw.sum()))
    p0=50 if float(l1_ratio)>=0.999999 else 400
    est=GeneralizedLinearEstimator(
        datafit=WeightedLogistic(sw),
        penalty=L1_plus_L2(alpha,float(l1_ratio)),
        solver=ProxNewton(p0=p0,max_iter=50,max_pn_iter=5000,tol=1e-8,
                          fit_intercept=True,warm_start=False,verbose=0))
    est.fit(Xt,y)
    D=np.abs(np.asarray(est.coef_).reshape(-1))
    ok=bool(float(est.stop_crit_) <= 1e-7)
    return D,ok,int(np.sum(D>1e-12))

def sis_score(X,y):
    yc=y-y.mean(); xc=X-X.mean(axis=0)
    den=np.sqrt((xc*xc).sum(axis=0)*(yc*yc).sum())
    return np.nan_to_num(np.abs((xc*yc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0)

def resample_scores(X,y,selector,params,task_idx,fold):
    scores=np.zeros((B,X.shape[1]),dtype=np.float32)
    ranks=np.zeros((B,X.shape[1]),dtype=np.float32)
    nz=np.zeros(B,dtype=np.int32)
    conv=np.ones(B,dtype=np.int8)
    classes=np.unique(y)
    for b in range(B):
        rng=np.random.default_rng(BASE_SEED + task_idx*1000000 + fold*10000 + b)
        take=[]
        for c in classes:
            idx=np.where(y==c)[0]
            n=max(2,int(round(FRAC*len(idx))))
            n=min(n,len(idx))
            take.extend(rng.choice(idx,size=n,replace=False).tolist())
        take=np.array(sorted(take),dtype=int)
        if selector=='SIS':
            D=sis_score(X[take],y[take]); ok=True; nnz=int(np.sum(D>1e-12))
        else:
            D,ok,nnz=fit_sparse(X[take],y[take],params['C'],params['l1_ratio'],
                               BASE_SEED+task_idx*1000000+fold*10000+b)
        scores[b]=D.astype(np.float32)
        ranks[b]=rankdata(-D,method='average').astype(np.float32)
        nz[b]=nnz; conv[b]=int(ok)
        if (b+1)%25==0:
            print(f'{selector} fold={fold} resample={b+1}/{B}',flush=True)
    return scores,ranks,nz,conv

def confusion(scores,ranks,features,k):
    med=np.median(ranks,axis=0)
    q25=np.percentile(ranks,25,axis=0); q75=np.percentile(ranks,75,axis=0)
    pi=(ranks<=k).mean(axis=0)
    cand=np.where(med<=k+WINDOW)[0].tolist()
    near=set(np.where((med>=k-WINDOW)&(med<=k+WINDOW))[0].tolist())
    rows=[]
    for aa,ia in enumerate(cand):
        for ib in cand[aa+1:]:
            if ia not in near and ib not in near: continue
            ra,rb=ranks[:,ia],ranks[:,ib]
            sa,sb=scores[:,ia],scores[:,ib]
            ties=(ra==rb).sum()
            pab=float(((ra<rb).sum()+0.5*ties)/B)
            selA,selB=ra<=k,rb<=k
            Q=float((selA^selB).mean())
            bal=1-2*abs(pab-0.5)
            H=0.0
            if 0<pab<1:
                H=float(-(pab*math.log(pab,2)+(1-pab)*math.log(1-pab,2)))
            act=Q*bal
            rows.append({
                'feature_A':features[ia],'feature_B':features[ib],
                'feature_index_A':ia,'feature_index_B':ib,'k':k,
                'pi_A':float(pi[ia]),'pi_B':float(pi[ib]),
                'P_A_gt_B':pab,'P_B_gt_A':1-pab,
                'P_both_selected':float((selA&selB).mean()),
                'P_neither_selected':float((~selA&~selB).mean()),
                'P_exactly_one_selected':Q,
                'selection_disagreement_Q':Q,'order_balance_B':bal,
                'data_rank_entropy_H':H,'actionable_boundary_score':act,
                'median_rank_A':float(med[ia]),'median_rank_B':float(med[ib]),
                'rank_IQR_A':float(q75[ia]-q25[ia]),'rank_IQR_B':float(q75[ib]-q25[ib]),
                'both_zero_freq':float(((sa<=1e-12)&(sb<=1e-12)).mean()),
                'mean_score_gap':float(np.mean(np.abs(sa-sb))),
                'n_resamples':B,
                'pair_type':('ACTIONABLE_BOUNDARY_CONFUSION' if act>=ACT_THR else
                             'RANK_ORDER_ONLY' if bal>=0.5 else 'WEAK_OR_NONE')
            })
    return pd.DataFrame(rows)

def run(task,task_idx,selectors):
    src=FROZEN/task; reference=reference/task; out=OUTROOT/task; out.mkdir(parents=True,exist_ok=True)
    X=np.load(src/'X_development.npy').astype(float)
    y=np.load(src/'y_development.npy').astype(int)
    sm=pd.read_csv(src/'development_samples.csv')
    feat=pd.read_csv(src/'features_p2000.csv')
    features=feat.gene_symbol.astype(str).tolist()
    splits=pd.read_csv(reference/'OUTER_SPLITS.csv')
    sa=pd.read_csv(reference/'reference_SELECTOR_AUDIT.csv') if (reference/'reference_SELECTOR_AUDIT.csv').exists() else None
    if sa is None:
        raise RuntimeError(f'reference_SELECTOR_AUDIT missing for {task}')
    summaries=[]
    for fold in sorted(splits.fold.unique()):
        tr=splits[(splits.fold==fold)&(splits.role=='outer_train')].sample_index.to_numpy(int)
        Xtr=X[tr]; ytr=y[tr]
        for selector in selectors:
            sdir=out/f'fold{int(fold)}_{selector}'
            sdir.mkdir(parents=True,exist_ok=True)
            npz=sdir/'RESAMPLE_SCORES_RANKS.npz'
            params={}
            if selector!='SIS':
                row=sa[(sa.fold==fold)&(sa.selector==selector)].iloc[0]
                params={'C':float(row.chosen_C),'l1_ratio':float(row.chosen_l1_ratio)}
            if npz.exists():
                z=np.load(npz); scores=z['scores']; ranks=z['ranks']; nz=z['nz']; conv=z['converged']
                print('resume',task,fold,selector,flush=True)
            else:
                t=time.time()
                scores,ranks,nz,conv=resample_scores(Xtr,ytr,selector,params,task_idx,int(fold))
                np.savez_compressed(npz,scores=scores,ranks=ranks,nz=nz,converged=conv,
                                    features=np.array(features),train_indices=tr)
                (sdir/'RESAMPLE_STATUS.json').write_text(json.dumps({
                    'task':task,'fold':int(fold),'selector':selector,'B':B,'fraction':FRAC,
                    'outer_train_n':int(len(tr)),'positive_n':int(ytr.sum()),
                    'params':params,'all_converged':bool(np.all(conv==1)),
                    'solver':('SIS_PEARSON' if selector=='SIS' else 'SKGLM_WEIGHTED_PROXNEWTON_EXACT'),
                    'solver_tol':(None if selector=='SIS' else 1e-8),
                    'solver_p0':(None if selector=='SIS' else (50 if params['l1_ratio']>=0.999999 else 400)),
                    'mean_n_positive_scores':float(nz.mean()),'min_n_positive_scores':int(nz.min()),
                    'runtime_sec':round(time.time()-t,2),'uses_outer_validation':False,
                    'uses_sealed_validation':False,'uses_llm':False},indent=2),encoding='utf-8')
            for k in KGRID:
                fp=sdir/f'K{k}_PAIR_CONFUSION.csv'
                if fp.exists(): cf=pd.read_csv(fp)
                else:
                    cf=confusion(scores,ranks,features,k)
                    cf.insert(0,'selector',selector); cf.insert(0,'fold',int(fold)); cf.insert(0,'task',task)
                    cf.to_csv(fp,index=False)
                nact=int((cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION').sum()) if len(cf) else 0
                summaries.append({'task':task,'fold':int(fold),'selector':selector,'k':k,
                    'n_relevant_pairs':int(len(cf)),'n_actionable_pairs':nact,
                    'max_actionable':float(cf.actionable_boundary_score.max()) if len(cf) else 0.0,
                    'n_candidate_nodes':int(len(set(cf.feature_A)|set(cf.feature_B))) if len(cf) else 0,
                    'min_n_positive_scores':int(nz.min()),'all_converged':bool(np.all(conv==1))})
                print(task,'fold',fold,selector,'k',k,'relevant',len(cf),'actionable',nact,flush=True)
    smry=pd.DataFrame(summaries)
    smry.to_csv(out/'ROUTING_SUMMARY.csv',index=False)
    # union across fold-local actionables; union only for caching, not fold eligibility
    u=[]
    for r in summaries:
        fp=out/f"fold{r['fold']}_{r['selector']}"/f"K{r['k']}_PAIR_CONFUSION.csv"
        cf=pd.read_csv(fp)
        a=cf[cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION'].copy()
        if len(a): u.append(a)
    if u:
        U=pd.concat(u,ignore_index=True)
        U['unordered_pair']=U.apply(lambda r:'||'.join(sorted([str(r.feature_A),str(r.feature_B)])),axis=1)
        uu=(U.groupby('unordered_pair',as_index=False)
              .agg(feature_A=('feature_A','first'),feature_B=('feature_B','first'),
                   n_fold_selector_k_routes=('unordered_pair','size'),
                   n_outer_folds=('fold','nunique'),
                   max_actionable=('actionable_boundary_score','max'),
                   mean_actionable=('actionable_boundary_score','mean')))
    else:
        uu=pd.DataFrame(columns=['unordered_pair','feature_A','feature_B','n_fold_selector_k_routes','n_outer_folds','max_actionable','mean_actionable'])
    uu.to_csv(out/'ACTIONABLE_PAIR_UNION_FOR_CACHE.csv',index=False)
    status={'task':task,'status':'ROUTING_COMPLETE','B':B,'fraction':FRAC,'window':WINDOW,
            'actionable_threshold':ACT_THR,'k_grid':KGRID,'selectors':selectors,
            'n_unique_actionable_pairs_union':int(len(uu)),
            'uses_outer_validation_for_routing':False,'uses_sealed_validation':False,'uses_llm':False}
    (out/'ROUTING_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682','all'],default='all')
    ap.add_argument('--selectors',nargs='+',default=['LASSO','ELASTICNET','SIS'])
    args=ap.parse_args()
    tasks=['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682'] if args.task=='all' else [args.task]
    for t in tasks:
        run(t,0 if t.startswith('BREAST') else 1,args.selectors)
