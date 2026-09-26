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
import argparse, hashlib, json, math, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtri
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
DATA=Path(str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917'))
sys.path.insert(0,str(DATA/'scripts'))
import v2_core as V
from common import SEEDS, N_FOLDS

OUT=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/downstream_batch1'))
OUT.mkdir(parents=True,exist_ok=True)
GAMMA_GRID=[0.0,0.1,0.25,0.5,1.0,2.0,4.0]
ETA_GRID=[0.0,0.03,0.1,0.3,1.0,3.0,10.0]
K_GRID=[5,10]

def folds(y,g,seed,n_splits):
    return list(StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))

def metrics(y,p):
    return dict(auroc=float(roc_auc_score(y,p)),
                auprc=float(average_precision_score(y,p)),
                balanced_accuracy=float(balanced_accuracy_score(y,(p>=0.5).astype(int))))

def select_lam(Xr,s,y,g,grid,kind,seed,n_inner=4):
    best,best_lam=-np.inf,grid[0]
    table=[]
    for lam in grid:
        aucs=[]
        for tri,vai in folds(y,g,seed+int(round(lam*100)),n_inner):
            if len(np.unique(y[tri]))<2 or len(np.unique(y[vai]))<2: continue
            w,_,sc,_=V.fit_full(Xr[tri],s[tri],y[tri],lam,kind=kind)
            aucs.append(roc_auc_score(y[vai],V.predict_full(w,sc,Xr[vai],s[vai])))
        m=float(np.mean(aucs)) if aucs else -np.inf
        table.append((float(lam),m))
        if m>best:
            best,best_lam=m,lam
    return float(best_lam),float(best),table

def norm_rank(values,index):
    s=pd.Series(np.asarray(values,float),index=index)
    if s.nunique()<=1: return pd.Series(0.5,index=index)
    r=s.rank(method='average',ascending=True)
    return (r-1.0)/(len(s)-1.0)

def stable_topk(score,D,features,k):
    df=pd.DataFrame({'feature':features,'score':np.asarray(score,float),'D':np.asarray(D,float),'idx':np.arange(len(features))})
    df=df.sort_values(['score','D','idx'],ascending=[False,False,True],kind='mergesort')
    return tuple(df.feature.iloc[:k].tolist())

def data_topk(D,features,k):
    return stable_topk(D,D,features,k)

def llm_blend_topk(D,features,g_map,eligible,lam,k):
    dr=norm_rank(D,features)
    eg=[f for f in features if f in eligible]
    gv=pd.Series({f:float(g_map[f]) for f in eg})
    psi=norm_rank(gv.values,eg)
    term=pd.Series(0.0,index=features)
    term.loc[eg]=psi
    comb=dr+float(lam)*term
    return stable_topk(comb.values,D,features,k),comb

def selective_anchor(D,features):
    ranks=rankdata(-np.asarray(D,float),method='average')
    u=1.0-(ranks-0.5)/len(features)
    return pd.Series(ndtri(np.clip(u,1e-6,1-1e-6)),index=features)

def selective_solve(D,features,k,lam,pairk):
    if lam==0: return selective_anchor(D,features),True
    sd=selective_anchor(D,features); idx={f:i for i,f in enumerate(features)}
    q=pairk[(pairk.k==k)&(pairk.primary_weight>0)].copy()
    if len(q)==0 or float(q.primary_weight.sum())<=0: return sd,True
    pairs=[(idx[r.feature_A],idx[r.feature_B],float(r.primary_weight),float(r.hard_target_A)) for r in q.itertuples()]
    wsum=sum(x[2] for x in pairs)+1e-12
    x0=sd.values.copy(); p=len(features)
    def fg(x):
        loss=0.5/p*np.sum((x-x0)**2)
        grad=(x-x0)/p
        for i,j,w,y in pairs:
            d=x[i]-x[j]
            loss += float(lam)*w*(np.logaddexp(0.0,d)-y*d)/wsum
            rr=float(lam)*w*(expit(d)-y)/wsum
            grad[i]+=rr; grad[j]-=rr
        return float(loss),grad
    res=minimize(lambda x:fg(x)[0],x0,jac=lambda x:fg(x)[1],method='BFGS',options={'maxiter':2000,'gtol':1e-8})
    return pd.Series(res.x,index=features),bool(res.success)

def evaluate_set(Xfull,site,y,groups,tri,vai,features,all_features,seed):
    inds=[all_features.index(f) for f in features]
    Xtr=Xfull[tri][:,inds]; Xva=Xfull[vai][:,inds]
    lam,inner,lt=select_lam(Xtr,site[tri],y[tri],groups[tri],V.LAM_GRID,'l2',seed,n_inner=4)
    w,info,sc,_=V.fit_full(Xtr,site[tri],y[tri],lam,kind='l2')
    pv=V.predict_full(w,sc,Xva,site[vai])
    return {'predictor_l2_lam':lam,'predictor_inner_auc':inner,'predictor_n_nonzero_clinical':info['n_nonzero_clinical'],**metrics(y[vai],pv)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-folds',type=int,default=None)
    args=ap.parse_args()

    man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json'))
    features=list(man['selectable_features'])
    reg=pd.read_csv(ROOT/'01_TASK_AND_IDENTITIES/FROZEN_BROAD_SELECTOR_FEATURES_37.csv')
    frozen=reg.feature_name.astype(str).tolist()
    if set(features)!=set(frozen) or len(features)!=37: raise RuntimeError('FEATURE_UNIVERSE_MISMATCH')
    # Use data-pipeline frozen feature order as D order; record mapping.
    pd.DataFrame({'data_feature_order':features,'frozen_registry_index':[frozen.index(f) for f in features]}).to_csv(OUT/'FEATURE_ORDER_MAP.csv',index=False)

    full=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(
        pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),
        on=['patient_uid','site']).reset_index(drop=True)
    prim=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
    if len(prim)!=892 or prim.patient_uid.nunique()!=664: raise RuntimeError('PRIMARY_SCOPE_MISMATCH')
    X=prim[features].to_numpy(float); site=prim.site.to_numpy(); y=prim.y.to_numpy(int); groups=prim.patient_uid.to_numpy()

    # Frozen LLM priors.
    m1=pd.read_csv(ROOT/'08_LLM_MEASUREMENT/BT_ANALYSIS_V1_8_1/global_BT_SCORES_V1_8_1.csv')
    m2=pd.read_csv(ROOT/'08_LLM_MEASUREMENT/BT_ANALYSIS_V1_8_1/global_certainty_ENTROPY_WEIGHTED_BT_SCORES_V1_8_1.csv')
    g1=dict(zip(m1.feature.astype(str),m1.g_hat_global.astype(float)))
    g2=dict(zip(m2.feature.astype(str),m2.g_hat_global_certainty.astype(float)))
    state=pd.read_csv(ROOT/'07_BROAD37_EVIDENCE/V0_8_1_AUDITED_CORPUS_AND_PACKETS/BROAD37_FEATURE_EVIDENCE_STATE_V0_8_1.csv')
    eligible=set(state[state.bt_graph_eligible.astype(str).str.lower().isin(['true','1'])].feature.astype(str))
    if len(eligible)!=34: raise RuntimeError('ELIGIBLE_COUNT_MISMATCH')

    # selective frozen routed weights.
    targ=pd.read_csv(ROOT/'01_TASK_AND_IDENTITIES/FROZEN_ACTIONABLE_TARGETS_32.csv')
    meas=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/selective_SELECTIVE30_SHARED_MEASUREMENTS_QWEN3_32B.csv')))
    mm=meas[['pair_id_selective','p_selective_A','H_pair','c_pair']].rename(columns={'pair_id_selective':'pair_id'}).drop_duplicates('pair_id')
    pairk=targ.merge(mm,on='pair_id',how='left',validate='many_to_one')
    if pairk[['p_selective_A','c_pair']].isna().any().any(): raise RuntimeError('selective_MEASUREMENT_MISSING')
    pairk['evidence_gate']=pairk.apply(lambda r:int(str(r.feature_A) in eligible and str(r.feature_B) in eligible),axis=1)
    pairk['hard_target_A']=(pairk.p_selective_A>0.5).astype(float)
    pairk['primary_weight']=pairk.actionable_boundary_score*pairk.c_pair*pairk.evidence_gate
    pairk.to_csv(OUT/'selective_PAIRK_WEIGHTS_FROZEN.csv',index=False)

    outer=[]
    for seed in SEEDS:
        for fi,(tri,vai) in enumerate(folds(y,groups,seed,N_FOLDS)):
            outer.append((seed,fi,tri,vai))
    if args.max_folds is not None: outer=outer[:args.max_folds]

    selector_rows=[]; set_rows=[]; eval_rows=[]; invariants=[]
    for oi,(seed,fi,tri,vai) in enumerate(outer,1):
        print(f'OUTER {oi}/{len(outer)} seed={seed} fold={fi}',flush=True)
        l1lam,l1inner,_=select_lam(X[tri],site[tri],y[tri],groups[tri],V.LAM_GRID,'l1',seed+fi,n_inner=4)
        w,info,sc,nctx=V.fit_full(X[tri],site[tri],y[tri],l1lam,kind='l1')
        D=V.clinical_scores(w,nctx)
        nz=int((D>1e-12).sum())
        for f,dv in zip(features,D):
            selector_rows.append({'seed':seed,'fold':fi,'feature':f,'D':float(dv),'l1_lam':l1lam,'l1_inner_auc':l1inner,'n_nonzero':nz})

        candidates=[]
        for k in K_GRID:
            s0=data_topk(D,features,k); candidates.append(('reference',k,0.0,s0,True))
            for lam in GAMMA_GRID:
                s1,_=llm_blend_topk(D,features,g1,eligible,lam,k)
                s2,_=llm_blend_topk(D,features,g2,eligible,lam,k)
                candidates.append(('global',k,lam,s1,True)); candidates.append(('global_certainty',k,lam,s2,True))
            for lam in ETA_GRID:
                ss,ok=selective_solve(D,features,k,lam,pairk)
                s3=stable_topk(ss.values,D,features,k)
                candidates.append(('selective',k,lam,s3,ok))

            # Identity assertions.
            reference=s0
            m1z=[x[3] for x in candidates if x[0]=='global' and x[1]==k and x[2]==0.0][-1]
            m2z=[x[3] for x in candidates if x[0]=='global_certainty' and x[1]==k and x[2]==0.0][-1]
            m3z=[x[3] for x in candidates if x[0]=='selective' and x[1]==k and x[2]==0.0][-1]
            invariants.append({'seed':seed,'fold':fi,'k':k,'global_gamma0_eq_reference':m1z==reference,'global_certainty_gamma0_eq_reference':m2z==reference,'selective_eta0_eq_reference':m3z==reference})

        # Evaluate each unique set once in this outer fold.
        uniq={}
        for method,k,hp,ss,ok in candidates:
            key=tuple(ss)
            uniq.setdefault(key,[]).append((method,k,hp,ok))
        set_metric={}
        for ui,(ss,uses) in enumerate(uniq.items()):
            # Same inner predictor tuning seed for every set in this outer fold.
            met=evaluate_set(X,site,y,groups,tri,vai,list(ss),features,seed+fi)
            set_metric[ss]=met
            eval_rows.append({'seed':seed,'fold':fi,'selected_set':'|'.join(ss),'k':len(ss),'n_methods_using_set':len(uses),**met})
        for method,k,hp,ss,ok in candidates:
            met=set_metric[tuple(ss)]
            set_rows.append({'seed':seed,'fold':fi,'method':method,'k':k,'hyperparam':float(hp),
                             'selector_l1_lam':l1lam,'selector_n_nonzero':nz,'optimizer_success':bool(ok),
                             'selected_set':'|'.join(ss),**met})

    sel=pd.DataFrame(selector_rows); sets=pd.DataFrame(set_rows); ev=pd.DataFrame(eval_rows); inv=pd.DataFrame(invariants)
    sel.to_csv(OUT/'OUTER_SELECTOR_SCORES.csv',index=False)
    sets.to_csv(OUT/'METHOD_FOLD_RESULTS_ALL_HYPERPARAMS.csv',index=False)
    ev.to_csv(OUT/'UNIQUE_SELECTED_SET_EVALUATIONS.csv',index=False)
    inv.to_csv(OUT/'IDENTITY_INVARIANTS.csv',index=False)
    if not inv[['global_gamma0_eq_reference','global_certainty_gamma0_eq_reference','selective_eta0_eq_reference']].all().all(): raise RuntimeError('ZERO_GUIDANCE_IDENTITY_FAILURE')

    # Aggregate tuning results. reference has one row; global/2/3 hyperparams selected by mean AUROC.
    agg=sets.groupby(['method','k','hyperparam'],as_index=False).agg(
        n_folds=('auroc','size'),auroc_mean=('auroc','mean'),auroc_sd=('auroc','std'),
        auprc_mean=('auprc','mean'),auprc_sd=('auprc','std'),
        balanced_accuracy_mean=('balanced_accuracy','mean'),distinct_selected_sets=('selected_set','nunique'),
        mean_selector_nonzero=('selector_n_nonzero','mean'))
    agg.to_csv(OUT/'DEVELOPMENT_TUNING_CURVES.csv',index=False)
    chosen=[]
    for (method,k),g in agg.groupby(['method','k'],sort=False):
        gg=g.sort_values(['auroc_mean','hyperparam'],ascending=[False,True],kind='mergesort')
        chosen.append(gg.iloc[0].to_dict())
    ch=pd.DataFrame(chosen)
    ch.to_csv(OUT/'SELECTED_DEVELOPMENT_HYPERPARAMS.csv',index=False)

    # Fold-level rows at chosen HP.
    picked=[]
    for r in ch.itertuples():
        z=sets[(sets.method==r.method)&(sets.k==r.k)&(sets.hyperparam==r.hyperparam)].copy()
        picked.append(z)
    picked=pd.concat(picked,ignore_index=True)
    picked.to_csv(OUT/'DEVELOPMENT_RESULTS_SELECTED_HYPERPARAMS.csv',index=False)

    # Fit selector on full Batch1 and freeze final selected sets using selected hyperparameters.
    full_l1lam,full_l1inner,_=select_lam(X,site,y,groups,V.LAM_GRID,'l1',90919,n_inner=5)
    fw,finfo,fsc,fnctx=V.fit_full(X,site,y,full_l1lam,kind='l1')
    Dfull=V.clinical_scores(fw,fnctx)
    pd.DataFrame({'feature':features,'D_full_batch1':Dfull,'rank':rankdata(-Dfull,method='average')}).to_csv(OUT/'FINAL_FULL_BATCH1_DATA_SELECTOR_SCORES.csv',index=False)
    finalsets=[]
    for r in ch.itertuples():
        method=str(r.method); k=int(r.k); hp=float(r.hyperparam)
        if method=='reference': ss=data_topk(Dfull,features,k); ok=True
        elif method=='global': ss,_=llm_blend_topk(Dfull,features,g1,eligible,hp,k); ok=True
        elif method=='global_certainty': ss,_=llm_blend_topk(Dfull,features,g2,eligible,hp,k); ok=True
        else:
            latent,ok=selective_solve(Dfull,features,k,hp,pairk); ss=stable_topk(latent.values,Dfull,features,k)
        finalsets.append({'method':method,'k':k,'selected_hyperparam':hp,'selected_set':'|'.join(ss),
                          'selector_l1_lam_full_batch1':full_l1lam,'selector_l1_inner_auc_full_batch1':full_l1inner,
                          'selector_n_nonzero_full_batch1':finfo['n_nonzero_clinical'],'optimizer_success':bool(ok)})
    fs=pd.DataFrame(finalsets); fs.to_csv(OUT/'FINAL_BATCH1_SELECTED_SETS_PRE_BATCH2.csv',index=False)

    status={'status':'PASS','n_outer_folds':len(outer),'full_protocol_expected_folds':25,
            'primary_scope_rows':len(prim),'primary_scope_patients':int(prim.patient_uid.nunique()),
            'batch2_accessed':False,'identity_invariants_all_pass':bool(inv[['global_gamma0_eq_reference','global_certainty_gamma0_eq_reference','selective_eta0_eq_reference']].all().all()),
            'selector_nonzero_min':int(sel.n_nonzero.min()),'selector_nonzero_max':int(sel.n_nonzero.max()),
            'selector_nonzero_mean':float(sel[['seed','fold','n_nonzero']].drop_duplicates().n_nonzero.mean()),
            'n_unique_predictive_sets_evaluated':len(ev),'selected_hyperparams':[
                {'method':str(r.method),'k':int(r.k),'hyperparam':float(r.hyperparam),'development_mean_auroc':float(r.auroc_mean)}
                for r in ch.itertuples()],
            'final_full_batch1_l1_lam':full_l1lam,'final_full_batch1_nonzero':int(finfo['n_nonzero_clinical'])}
    (OUT/'BATCH1_DOWNSTREAM_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))
    print('\nTUNING CURVES\n',agg.to_string(index=False))
    print('\nFINAL SETS\n',fs.to_string(index=False))

if __name__=='__main__': main()
