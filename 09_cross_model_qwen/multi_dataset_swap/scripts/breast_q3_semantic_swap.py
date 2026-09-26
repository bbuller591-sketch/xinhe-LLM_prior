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
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time,warnings,math,hashlib
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(str(REPRO_ROOT / 'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919'))
OUT=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/08_Q2_Q5_DIAGNOSTICS/BREAST_Q3_SEMANTIC'));OUT.mkdir(parents=True,exist_ok=True)
TASK='BREAST_GSE25055_GSE25065';NULL_SEED=2026091903;NREP=1000
LAMS=np.array([0,.03,.1,.3,1,3,10.],float);SELECTORS=['SIS'];KGRID=[20]
SRC=ROOT/'DATA/FROZEN'/TASK;reference=ROOT/'METHOD_ARTIFACTS/reference';ROUTING=ROOT/'METHOD_ARTIFACTS/selective_ROUTING';FINAL=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/04_BREAST/final_development_tuning/BREAST_GSE25055_GSE25065'))
X=np.load(SRC/'X_development.npy').astype(float);y=np.load(SRC/'y_development.npy').astype(int)
Xte=np.load(SRC/'X_sealed_validation.npy').astype(float);yte=np.load(SRC/'y_sealed_validation.npy').astype(int)
feat=pd.read_csv(SRC/'features_p2000.csv');splits=pd.read_csv(reference/'OUTER_SPLITS.csv');sa=pd.read_csv(reference/'reference_SELECTOR_AUDIT.csv')
finalpars=pd.read_csv(FINAL/'FINAL_SELECTOR_PARAMS.csv');actual_lam=pd.read_csv(FINAL/'FINAL_SELECTED_ETA.csv');actual_support=pd.read_csv(FINAL/'FINAL_selective_SELECTED_FEATURES.csv')
sealed_ref=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/04_BREAST/sealed_validation/SEALED_reference_selective_RESULTS.csv')));sealed_ref=sealed_ref[sealed_ref.task==TASK]
reference_ref=sealed_ref[sealed_ref.method=='reference'].set_index(['selector','k']);selective_ref=sealed_ref[sealed_ref.method=='selective'].set_index(['selector','k'])

def data_anchor(D):
    r=rankdata(-D,method='average');u=1-(r-.5)/len(D);return norm.ppf(np.clip(u,1e-6,1-1e-6))
def sis_score(A,b):
    bc=b-b.mean();ac=A-A.mean(0);den=np.sqrt((ac*ac).sum(0)*(bc*bc).sum());return np.nan_to_num(np.abs((ac*bc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0.)
def sparse_score(A,b,C,ratio,seed):
    sc=StandardScaler().fit(A);Z=sc.transform(A);pen='l1' if float(ratio)==1 else 'elasticnet'
    mod=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=pen,l1_ratio=None if pen=='l1' else float(ratio),
        max_iter=5000,tol=1e-4,random_state=int(seed),fit_intercept=True,n_jobs=1);mod.fit(Z,b)
    if int(mod.n_iter_[0])>=mod.max_iter:raise RuntimeError('SAGA_NONCONVERGENCE')
    return np.abs(mod.coef_[0])
def stable_topk(z,D,k):return np.lexsort((np.arange(len(z)),-D,-z))[:k]
outer_scores={}
for fold in range(1,6):
    tr=splits[(splits.fold==fold)&(splits.role=='outer_train')].sample_index.to_numpy(int);va=splits[(splits.fold==fold)&(splits.role=='outer_validation')].sample_index.to_numpy(int)
    for sel in SELECTORS:
        rr=sa[(sa.fold==fold)&(sa.selector==sel)].iloc[0]
        D=sis_score(X[tr],y[tr]) if sel=='SIS' else sparse_score(X[tr],y[tr],rr.chosen_C,rr.chosen_l1_ratio,2026091901+fold)
        outer_scores[(fold,sel)]={'tr':tr,'va':va,'D':D,'sD':data_anchor(D)}
full_scores={}
for sel in SELECTORS:
    rr=finalpars[finalpars.selector==sel].iloc[0]
    D=sis_score(X,y) if sel=='SIS' else sparse_score(X,y,rr.C,rr.l1_ratio,2026091901)
    full_scores[sel]={'D':D,'sD':data_anchor(D)}

meas=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/04_BREAST/selective_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B.csv'))).copy()
pairs=sorted(meas.unordered_pair_id.astype(str).unique());arms=sorted(meas.arm.astype(str).unique());assert len(arms)==2
bundle={}
for pair,g in meas.groupby('unordered_pair_id'):
    a,b=str(pair).split('||',1);bundle[str(pair)]={}
    for r in g.itertuples():
        yy=float(r.hard_y_i_over_j) if str(r.gene_i)==a else 1-float(r.hard_y_i_over_j)
        bundle[str(pair)][str(r.arm)]={'y':yy,'c':float(r.certainty_1_minus_H)}
pair_to_pos={p:i for i,p in enumerate(pairs)};arm_to_pos={a:i for i,a in enumerate(arms)}
name_to_idx={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}
route_geom={}
for fold in range(1,6):
  for sel in SELECTORS:
    for k in KGRID:
        cf=pd.read_csv(ROUTING/f'fold{fold}_{sel}'/f'K{k}_PAIR_CONFUSION.csv')
        if len(cf) and 'pair_type' in cf.columns:cf=cf[cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION']
        rows=[]
        for r in cf.itertuples():
            a,b=sorted([str(r.feature_A),str(r.feature_B)]);pair=a+'||'+b;rows.append((pair,name_to_idx[a],name_to_idx[b],float(r.actionable_boundary_score)))
        route_geom[(fold,sel,k)]=rows
fc=pd.read_csv(FINAL/'FINAL_CROSSFITTED_selective_CONSTRAINTS.csv');final_geom={}
for (sel,k),g in fc.groupby(['selector','k']):
    q=g.groupby(['unordered_pair_id','gene_i','gene_j'],as_index=False).A_CF.first();rows=[]
    for r in q.itertuples():
        a,b=str(r.unordered_pair_id).split('||',1);rows.append((str(r.unordered_pair_id),name_to_idx[a],name_to_idx[b],float(r.A_CF)))
    final_geom[(str(sel),int(k))]=rows

def real_constraints(geom):
    out=[]
    for pair,i,j,A in geom:
        for arm in arms:
            if arm not in bundle.get(pair,{}): continue
            q=bundle[pair][arm];out.append((i,j,q['y'],A*q['c']/2.))
    return out
def random_constraints(geom,rand_y,rand_c):
    out=[]
    for pair,i,j,A in geom:
        pi=pair_to_pos[pair]
        for arm in arms:
            if arm not in bundle.get(pair,{}): continue
            ai=arm_to_pos[arm];out.append((i,j,float(rand_y[pi,ai]),A*float(rand_c[pi,ai])/2.))
    return out
def optimize_active(sD,cc,lam):
    if lam==0 or not cc:return sD.copy()
    ii=np.array([r[0] for r in cc],int);jj=np.array([r[1] for r in cc],int);yy=np.array([r[2] for r in cc]);w=np.array([r[3] for r in cc]);W=w.sum()
    if W<=0:return sD.copy()
    active=np.unique(np.r_[ii,jj]);am={v:q for q,v in enumerate(active)};li=np.array([am[v] for v in ii]);lj=np.array([am[v] for v in jj]);base=sD[active].copy();P=len(sD)
    def fg(v):
        d=v[li]-v[lj];ce=np.logaddexp(0,d)-yy*d;diff=v-base
        f=.5*np.dot(diff,diff)/P+lam*np.dot(w,ce)/W;g=diff/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
    r=minimize(lambda v:fg(v)[0],base,jac=lambda v:fg(v)[1],method='L-BFGS-B',options={'maxiter':1000,'maxls':50,'ftol':1e-12,'gtol':1e-8})
    if not r.success:
        x0=r.x if np.all(np.isfinite(r.x)) else base
        r=minimize(lambda v:fg(v)[0],x0,jac=lambda v:fg(v)[1],method='L-BFGS-B',options={'maxiter':3000,'maxls':300,'ftol':1e-13,'gtol':1e-7})
    if not r.success:
        x0=r.x if np.all(np.isfinite(r.x)) else base
        r=minimize(lambda v:fg(v)[0],x0,jac=lambda v:fg(v)[1],method='BFGS',options={'maxiter':3000,'gtol':1e-7})
    if (not r.success) and (not np.all(np.isfinite(r.x))):raise RuntimeError(str(r.message))
    z=sD.copy();z[active]=r.x;return z

def eval_support(fold,cols,cache,sealed=False):
    tag='SEALED' if sealed else f'F{fold}';key=(tag,tuple(int(i) for i in cols))
    if key in cache:return cache[key]
    if sealed:Atr=X[:,cols];Ava=Xte[:,cols];btr=y;bva=yte
    else:
        os=outer_scores[(fold,CURRENT_SEL)];Atr=X[os['tr']][:,cols];Ava=X[os['va']][:,cols];btr=y[os['tr']];bva=y[os['va']]
    sc=StandardScaler().fit(Atr);mod=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=2026091901)
    mod.fit(sc.transform(Atr),btr);pp=mod.predict_proba(sc.transform(Ava))[:,1];val=float(roc_auc_score(bva,pp));cache[key]=val;return val

# identity uses local helper without hidden global selector race
def evaluate_cc_provider(provider,cache):
    rows=[]
    global CURRENT_SEL
    for sel in SELECTORS:
      CURRENT_SEL=sel
      for k in KGRID:
        vals=[]
        for lam in LAMS:
            aa=[]
            for fold in range(1,6):
                os=outer_scores[(fold,sel)];z=optimize_active(os['sD'],provider(route_geom[(fold,sel,k)]),float(lam));cols=stable_topk(z,os['D'],k)
                aa.append(eval_support(fold,cols,cache,False))
            vals.append((float(lam),float(np.mean(aa))))
        best=max(v for _,v in vals);chosen=min(e for e,v in vals if np.isclose(v,best,atol=1e-12,rtol=0))
        fs=full_scores[sel];z=optimize_active(fs['sD'],provider(final_geom[(sel,k)]),chosen);cols=stable_topk(z,fs['D'],k)
        sealed=eval_support(0,cols,cache,True)
        rows.append({'selector':sel,'k':k,'chosen_lam':chosen,'best_dev_auc':best,'sealed_auc':sealed,'selected_indices':'|'.join(map(str,cols.tolist())),
                     'fixed_lam_auc':dict(vals)[float(actual_lam[(actual_lam.selector==sel)&(actual_lam.k==k)].iloc[0].chosen_lam)]})
    return pd.DataFrame(rows)

idres=evaluate_cc_provider(real_constraints,{})
idchecks=[]
for r in idres.itertuples():
    ae=float(actual_lam[(actual_lam.selector==r.selector)&(actual_lam.k==r.k)].iloc[0].chosen_lam)
    sup=actual_support[(actual_support.selector==r.selector)&(actual_support.k==r.k)].sort_values('rank')
    ss='|'.join(map(str,sup.feature_index.astype(int).tolist()));auc=float(selective_ref.loc[(r.selector,r.k),'auroc'])
    idchecks.append({'selector':r.selector,'k':r.k,'lam_match':np.isclose(r.chosen_lam,ae),'support_match':r.selected_indices==ss,'sealed_auc_diff':abs(r.sealed_auc-auc)})
idc=pd.DataFrame(idchecks);idc['pass']=idc.lam_match&idc.support_match&(idc.sealed_auc_diff<1e-12);idc.to_csv(OUT/'IDENTITY_REPRODUCTION.csv',index=False)
if not idc['pass'].all():raise RuntimeError('Breast identity gate failed')

obs={}
for r in idres.itertuples():
    base=float(reference_ref.loc[(r.selector,r.k),'auroc'])
    obs[(r.selector,r.k)]={'lam':r.chosen_lam,'dev':r.best_dev_auc,'sealed':r.sealed_auc,'sealed_reference':base}
# Development baseline for null comparison is lam=0 from identity curve; obtain from frozen selective tuning file via lam curves.
devcurve=pd.read_csv(FINAL/'FINAL_ETA_TUNING_CURVES.csv') if (FINAL/'FINAL_ETA_TUNING_CURVES.csv').exists() else None
if devcurve is None:
    # authoritative identity provider, lam=0 is data-only; compute once exactly.
    devref={}
    for sel in SELECTORS:
      CURRENT_SEL=sel
      for k in KGRID:
        aa=[]
        for fold in range(1,6):
            os=outer_scores[(fold,sel)];cols=stable_topk(os['sD'],os['D'],k);aa.append(eval_support(fold,cols,{},False))
        devref[(sel,k)]=float(np.mean(aa))
else:
    devref={(s,k):float(devcurve[(devcurve.selector==s)&(devcurve.k==k)&np.isclose(devcurve.lam,0)].iloc[0].cv_mean_auroc) for s in SELECTORS for k in KGRID}

def run_chunk(reps):
    global CURRENT_SEL
    cache={};rows=[]
    for rep in reps:
        rng=np.random.default_rng(np.random.SeedSequence([NULL_SEED,rep]))
        yhard=np.zeros((len(pairs),len(arms)),float);c=np.zeros((len(pairs),len(arms)),float);tokens=[]
        for arm in arms:
            ai=arm_to_pos[arm]; targets=[p for p in pairs if arm in bundle.get(p,{})]; perm=rng.permutation(len(targets));tokens.append(perm.tobytes())
            for ti,pair in enumerate(targets):
                donor=targets[int(perm[ti])];pi=pair_to_pos[pair];q=bundle[donor][arm];yhard[pi,ai]=float(q['y']);c[pi,ai]=float(q['c'])
        token=hashlib.sha256(b''.join(tokens)).hexdigest()
        def prov(geom):return random_constraints(geom,yhard,c)
        for sel in SELECTORS:
          CURRENT_SEL=sel
          for k in KGRID:
            vals=[]
            for lam in LAMS:
                aa=[]
                for fold in range(1,6):
                    os=outer_scores[(fold,sel)];z=optimize_active(os['sD'],prov(route_geom[(fold,sel,k)]),float(lam));cols=stable_topk(z,os['D'],k)
                    aa.append(eval_support(fold,cols,cache,False))
                vals.append((float(lam),float(np.mean(aa))))
            best=max(v for _,v in vals);chosen=min(e for e,v in vals if np.isclose(v,best,atol=1e-12,rtol=0));O=obs[(sel,k)]
            fs=full_scores[sel];z=optimize_active(fs['sD'],prov(final_geom[(sel,k)]),chosen);cols=stable_topk(z,fs['D'],k);sealed=eval_support(0,cols,cache,True)
            rows.append({'replicate':rep,'selector':sel,'k':k,'draw_sha256':token,'chosen_lam':chosen,'best_dev_auc':best,'dev_improvement':best-devref[(sel,k)],
                         'fixed_observed_lam':O['lam'],'fixed_lam_dev_auc':dict(vals)[O['lam']],'fixed_dev_improvement':dict(vals)[O['lam']]-devref[(sel,k)],
                         'sealed_auc_frozen_support':sealed,'sealed_delta_vs_reference':sealed-O['sealed_reference']})
    return rows

chunks=[list(range(i,NREP,4)) for i in range(4)];rows=[];t=time.time()
with ProcessPoolExecutor(max_workers=8) as ex:
    fs=[ex.submit(run_chunk,ch) for ch in chunks]
    for n,f in enumerate(as_completed(fs),1):
        rows.extend(f.result());pd.DataFrame(rows).to_csv(OUT/'PARTIAL.csv',index=False);print('chunks',n,'/4','rows',len(rows),'sec',round(time.time()-t,1),flush=True)
R=pd.DataFrame(rows).sort_values(['replicate','selector','k']);R.to_csv(OUT/'SEMANTIC_SHUFFLE_1000.csv',index=False)
ss=[]
for sel in SELECTORS:
  for k in KGRID:
    q=R[(R.selector==sel)&(R.k==k)];O=obs[(sel,k)];obs_delta=O['dev']-devref[(sel,k)];v=q.dev_improvement;fx=q.fixed_dev_improvement
    ss.append({'dataset':'Breast GSE25055->GSE25065','selector':sel,'k':k,'mode':'semantic_shuffle','n_replicates':NREP,'seed':NULL_SEED,
               'observed_lam':O['lam'],'reference_dev_auc':devref[(sel,k)],'observed_dev_auc':O['dev'],'observed_dev_improvement':obs_delta,
               'null_mean':float(v.mean()),'null_sd':float(v.std(ddof=1)),'null_q95':float(v.quantile(.95)),
               'observed_percentile':100*float(np.mean(v<=obs_delta)),'empirical_p_one_sided':float((1+int((v>=obs_delta-1e-12).sum()))/(NREP+1)),
               'fixed_null_mean':float(fx.mean()),'fixed_null_q95':float(fx.quantile(.95)),'fixed_empirical_p_one_sided':float((1+int((fx>=obs_delta-1e-12).sum()))/(NREP+1)),
               'observed_sealed_auc':O['sealed'],'null_sealed_auc_mean':float(q.sealed_auc_frozen_support.mean()),
               'lam_counts':json.dumps({str(kk):int(vv) for kk,vv in q.chosen_lam.value_counts().sort_index().items()})})
S=pd.DataFrame(ss);S.to_csv(OUT/'SUMMARY.csv',index=False)
(OUT/'SUMMARY.json').write_text(json.dumps({'design':'fixed routed identities/A_CF/evidence eligibility; within-source permutation of cached hard-direction/certainty bundles across usable pair-source cells; lam reselected development-only','new_llm_calls':0,'summary':ss},indent=2))
