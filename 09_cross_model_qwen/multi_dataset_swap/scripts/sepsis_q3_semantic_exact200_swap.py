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
import sys,json,math,hashlib,time,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata,norm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
PKG=Path(str(REPRO_ROOT / 'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919'));sys.path.insert(0,str(PKG/"07_CODE"));import run_strict_nested_selective_routing as route
OUT=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/08_Q2_Q5_DIAGNOSTICS/GSE272769_Q3_SEMANTIC_EXACT200'));OUT.mkdir(parents=True,exist_ok=True)
X=np.load(PKG/"01_FROZEN_DATA/X_development.npy").astype(float);y=np.load(PKG/"01_FROZEN_DATA/y_development.npy").astype(int);feat=pd.read_csv(PKG/"01_FROZEN_DATA/features_p1500.csv");feat["feature_index"]=np.arange(len(feat))
spl=pd.read_csv(PKG/"01_FROZEN_DATA/OUTER_SPLITS.csv");sa=pd.read_csv(PKG/"01_FROZEN_DATA/reference_SELECTOR_AUDIT.csv");real_outer=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/03_GSE272769/downstream/selective_NESTED_OUTER_RESULTS.csv')));ref50=real_outer[(real_outer.selector=="ELASTICNET")&(real_outer.k==50)].set_index("outer_fold")
MEAS=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/03_GSE272769/selective_PAIR_SOURCE_MEASUREMENTS_QWEN3_32B_K50.csv'))).reset_index(drop=True);SEED=2026091901;NREP=200;NULLSEED=2026091901;LAMS=[0,0.03,0.1,0.3,1,3,10];K=50;P=X.shape[1]
mid={(str(r.unordered_pair_id),str(r.arm)):i for i,r in MEAS.iterrows()}
def anchor(D):
 r=rankdata(-D,method="average");u=1-(r-.5)/len(D);return norm.ppf(np.clip(u,1e-6,1-1e-6))
def score_inner(X0,y0,C,ratio,seed): return route.fit_sparse(X0,y0,float(C),float(ratio),int(seed))[0]
def score_outer(X0,y0,C,ratio,seed):
 sc=StandardScaler().fit(X0);Z=sc.transform(X0);mod=LogisticRegression(C=float(C),solver="saga",class_weight="balanced",penalty="elasticnet",l1_ratio=float(ratio),max_iter=5000,tol=1e-4,random_state=seed,n_jobs=1).fit(Z,y0);assert mod.n_iter_[0]<mod.max_iter;return np.abs(mod.coef_[0])
def topk(z,D):return np.lexsort((np.arange(P),-D,-z))[:K]
def solve(a,cons,lam,canon_y,cert):
 if lam==0 or not cons:return a.copy()
 ii=np.array([c[0] for c in cons]);jj=np.array([c[1] for c in cons]);ori=np.array([c[3] for c in cons]);basew=np.array([c[4] for c in cons]);idx=np.array([c[2] for c in cons])
 # Restore donor canonical hard direction to each recipient measurement row orientation, then to actionable orientation.
 gi=MEAS.iloc[idx].gene_i.astype(str).to_numpy();gj=MEAS.iloc[idx].gene_j.astype(str).to_numpy();yc=canon_y[idx]
 y_meas=np.where(gi<gj,yc,1-yc);yy=np.where(ori>0,y_meas,1-y_meas).astype(float);w=basew*cert[idx];W=w.sum()
 if W<=0:return a.copy()
 active=np.unique(np.r_[ii,jj]);pos={v:t for t,v in enumerate(active)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj]);b=a[active].copy()
 def fg(v):
  d=v[li]-v[lj];df=v-b;ce=np.logaddexp(0,d)-yy*d;f=.5*np.dot(df,df)/P+lam*np.dot(w,ce)/W;g=df/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
 r=minimize(lambda v:fg(v)[0],b,jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":800,"ftol":1e-11,"gtol":1e-8})
 if not np.all(np.isfinite(r.x)):raise RuntimeError("nonfinite")
 z=a.copy();z[active]=r.x;return z
def eval_support(tr,va,ids):
 sc=StandardScaler().fit(X[tr][:,ids]);mod=LogisticRegression(C=1.,penalty="l2",solver="lbfgs",class_weight="balanced",max_iter=3000,random_state=SEED).fit(sc.transform(X[tr][:,ids]),y[tr]);p=mod.predict_proba(sc.transform(X[va][:,ids]))[:,1]
 app=average_precision_score(y[va],p);apn=average_precision_score(1-y[va],1-p);return float(roc_auc_score(y[va],p)),float(.5*(app+apn))
# precompute fold-local route templates and scores
F={}
for fold in range(1,6):
 tr=spl[(spl.fold==fold)&(spl.role=="outer_train")].sample_index.to_numpy(int);va=spl[(spl.fold==fold)&(spl.role=="outer_validation")].sample_index.to_numpy(int);row=sa[(sa.fold==fold)&(sa.selector=="ELASTICNET")].iloc[0];C=float(row.chosen_C);ratio=float(row.chosen_l1_ratio)
 ins=list(StratifiedKFold(n_splits=4,shuffle=True,random_state=SEED+100*fold).split(X[tr],y[tr]));inner=[]
 for q,(itr0,iva0) in enumerate(ins,1):
  itr=tr[itr0];iva=tr[iva0];D=score_inner(X[itr],y[itr],C,ratio,SEED+100000*fold+1000*q+100);inner.append((itr,iva,D,anchor(D)))
 Do=score_outer(X[tr],y[tr],C,ratio,SEED+fold);ao=anchor(Do)
 cf=pd.read_csv(PKG/f"04_ROUTING/fold{fold}_ELASTICNET/K50_PAIR_CONFUSION.csv");cf=cf[cf.pair_type=="ACTIONABLE_BOUNDARY_CONFUSION"]
 cons=[]
 for r in cf.itertuples():
  a,b=str(r.feature_A),str(r.feature_B);pair="||".join(sorted([a,b]));avs=[(arm,j) for (pp,arm),j in mid.items() if pp==pair];m=len(avs)
  for arm,jj0 in avs:
   mr=MEAS.iloc[jj0];ori=1 if (str(mr.gene_i)==a and str(mr.gene_j)==b) else -1;cons.append((int(feat.index[feat.gene_symbol.astype(str)==a][0]),int(feat.index[feat.gene_symbol.astype(str)==b][0]),int(jj0),ori,float(r.actionable_boundary_score)/m))
 F[fold]=(tr,va,inner,Do,ao,cons,float(ref50.loc[fold].reference_auroc),float(ref50.loc[fold].reference_macro_ap))
print("constraint counts",{f:len(F[f][5]) for f in F},flush=True)
def runrep(rep):
 rng=np.random.default_rng(2026091901+rep);base_y=np.where(MEAS.gene_i.astype(str).to_numpy()<MEAS.gene_j.astype(str).to_numpy(),MEAS.hard_y_i_over_j.astype(float).to_numpy(),1-MEAS.hard_y_i_over_j.astype(float).to_numpy());canon_y=base_y.copy();cert=MEAS.certainty_1_minus_H.astype(float).to_numpy().copy()
 for arm,idx0 in MEAS.groupby("arm").groups.items():
  idx=np.asarray(list(idx0),int);don=rng.permutation(idx);canon_y[idx]=base_y[don];cert[idx]=MEAS.certainty_1_minus_H.astype(float).to_numpy()[don]
 rows=[];cache={}
 for fold in range(1,6):
  tr,va,inner,Do,ao,cons,refa,refm=F[fold];perf=[]
  for lam in LAMS:
   vals=[]
   for itr,iva,D,a in inner:
    ids=topk(solve(a,cons,lam,canon_y,cert),D);key=(tuple(ids),tuple(iva))
    if key not in cache:cache[key]=eval_support(itr,iva,ids)[0]
    vals.append(cache[key])
   perf.append((lam,float(np.mean(vals))))
  best=max(v for _,v in perf);chosen=min(e for e,v in perf if np.isclose(v,best,atol=1e-12,rtol=0));ids=topk(solve(ao,cons,chosen,canon_y,cert),Do);au,ma=eval_support(tr,va,ids);rows.append((fold,chosen,au-refa,ma-refm,au,ma))
 a=np.array(rows,float);return {"replicate":rep,"draw_sha256":hashlib.sha256(canon_y.tobytes()+cert.tobytes()).hexdigest(),"mean_delta_auroc":float(a[:,2].mean()),"mean_delta_macro_ap":float(a[:,3].mean()),"mean_null_auroc":float(a[:,4].mean()),"mean_null_macro_ap":float(a[:,5].mean()),"chosen_lam_by_fold":"|".join(str(x[1]) for x in rows)}
if __name__=="__main__":
 t=time.time();rows=[]
 with ProcessPoolExecutor(max_workers=8) as ex:
  fs=[ex.submit(runrep,r) for r in range(1,NREP+1)]
  for n,f in enumerate(as_completed(fs),1):
   rows.append(f.result())
   if n%50==0:print("done",n,"sec",round(time.time()-t,1),flush=True);pd.DataFrame(rows).to_csv(OUT/"PARTIAL.csv",index=False)
 Z=pd.DataFrame(rows).sort_values("replicate");Z.to_csv(OUT/"REPLICATES_200.csv",index=False)
 real=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/03_GSE272769/downstream/selective_NESTED_AGGREGATE.csv')));o=real[(real.selector=="ELASTICNET")&(real.k==50)].iloc[0];oda=float(o.mean_delta_auroc);odm=float(o.mean_delta_macro_ap)
 def sm(v,obs):
  return {"mean":float(v.mean()),"sd":float(v.std(ddof=1)),"q95":float(v.quantile(.95)),"empirical_p":float((1+(v>=obs-1e-12).sum())/(len(v)+1)),"n_lower":int((v<obs-1e-12).sum()),"n_equal":int(np.isclose(v,obs,atol=1e-12,rtol=0).sum()),"n_greater":int((v>obs+1e-12).sum())}
 S={"dataset":"GSE272769","selector":"ElasticNet","k":50,"mode":"semantic_shuffle","n_replicates":NREP,"seed":NULLSEED,"new_llm_calls":0,"design":"fixed fold-local routing/source eligibility; within-source permutation of cached canonical hard-direction and certainty bundles; lam reselected inside each outer fold using inner development folds","observed_delta_auroc":oda,"observed_delta_macro_ap":odm,"auroc_null":sm(Z.mean_delta_auroc,oda),"macro_ap_null":sm(Z.mean_delta_macro_ap,odm)}
 (OUT/"SUMMARY.json").write_text(json.dumps(S,indent=2));pd.DataFrame([{"dataset":"GSE272769","selector":"ElasticNet","k":50,"n_replicates":NREP,"observed_delta_auroc":oda,"semantic_null_mean_delta_auroc":S["auroc_null"]["mean"],"semantic_p_auroc":S["auroc_null"]["empirical_p"],"observed_delta_macro_ap":odm,"semantic_null_mean_delta_macro_ap":S["macro_ap_null"]["mean"],"semantic_p_macro_ap":S["macro_ap_null"]["empirical_p"],"new_llm_calls":0}]).to_csv(OUT/"SUMMARY.csv",index=False);print(json.dumps(S,indent=2))
