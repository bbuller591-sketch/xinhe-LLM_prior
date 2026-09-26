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
import hashlib,json,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
R=Path(str(REPRO_ROOT / 'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919'));O=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion/artifacts/BREAST_Q2_MATCHED'));O.mkdir(parents=True,exist_ok=True)
TASK="BREAST_GSE25055_GSE25065";SEED=2026091903;LAMS=[0,0.03,0.1,0.3,1,3,10.0];SELECTORS=["LASSO","ELASTICNET","SIS"];KGRID=[10,20]
SRC=R/"DATA/FROZEN"/TASK;X=np.load(SRC/"X_development.npy").astype(float);y=np.load(SRC/"y_development.npy").astype(int);Xte=np.load(SRC/"X_sealed_validation.npy").astype(float);yte=np.load(SRC/"y_sealed_validation.npy").astype(int);feat=pd.read_csv(SRC/"features_p2000.csv");P=len(feat)
spl=pd.read_csv(R/"METHOD_ARTIFACTS/reference/OUTER_SPLITS.csv");sa=pd.read_csv(R/"METHOD_ARTIFACTS/reference/reference_SELECTOR_AUDIT.csv");finalpars=pd.read_csv(R/"METHOD_ARTIFACTS/selective_FINAL/FINAL_SELECTOR_PARAMS.csv")
sealed=pd.read_csv(R/"RESULTS/SEALED_VALIDATION_PRIMARY_RESULTS.csv");sealed=sealed[(sealed.task==TASK)&(sealed.analysis=="PRIMARY")];m0ref=sealed[sealed.method=="reference"].set_index(["selector","k"]);m3ref=sealed[sealed.method=="selective"].set_index(["selector","k"])
def anch(D):
 r=rankdata(-D,method="average");return norm.ppf(np.clip(1-(r-.5)/len(D),1e-6,1-1e-6))
def sis(A,b):
 bc=b-b.mean();ac=A-A.mean(0);den=np.sqrt((ac*ac).sum(0)*(bc*bc).sum());return np.nan_to_num(np.abs((ac*bc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0.)
def sparse(A,b,C,ratio,seed):
 sc=StandardScaler().fit(A);Z=sc.transform(A);pen="l1" if float(ratio)==1 else "elasticnet";m=LogisticRegression(C=float(C),solver="saga",class_weight="balanced",penalty=pen,l1_ratio=None if pen=="l1" else float(ratio),max_iter=5000,tol=1e-4,random_state=seed,n_jobs=1).fit(Z,b);assert m.n_iter_[0]<m.max_iter;return np.abs(m.coef_[0])
def top(z,D,k):return np.lexsort((np.arange(P),-D,-z))[:k]
def solve(a,rows,lam):
 if lam==0 or not rows:return a.copy()
 ii=np.array([x[1] for x in rows]);jj=np.array([x[2] for x in rows]);yy=np.array([x[3] for x in rows]);w=np.array([x[4] for x in rows]);W=w.sum();act=np.unique(np.r_[ii,jj]);pos={v:t for t,v in enumerate(act)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj]);b=a[act].copy()
 def fg(v):
  d=v[li]-v[lj];df=v-b;ce=np.logaddexp(0,d)-yy*d;f=.5*np.dot(df,df)/P+lam*np.dot(w,ce)/W;g=df/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
 rr=minimize(lambda v:fg(v)[0],b,jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":1500,"ftol":1e-12,"gtol":1e-8})
 if not rr.success:
  rr=minimize(lambda v:fg(v)[0],rr.x,jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":5000,"maxls":200,"ftol":1e-13,"gtol":1e-8})
 assert np.all(np.isfinite(rr.x));z=a.copy();z[act]=rr.x;return z
def evals(tr,va,ids,sealed_eval=False):
 Atr=X[:,ids] if sealed_eval else X[tr][:,ids];Ava=Xte[:,ids] if sealed_eval else X[va][:,ids];bt=y if sealed_eval else y[tr];bv=yte if sealed_eval else y[va]
 sc=StandardScaler().fit(Atr);m=LogisticRegression(C=1.,penalty="l2",solver="lbfgs",class_weight="balanced",max_iter=5000,random_state=2026091901).fit(sc.transform(Atr),bt);p=m.predict_proba(sc.transform(Ava))[:,1];ap=average_precision_score(bv,p);apn=average_precision_score(1-bv,1-p);return float(roc_auc_score(bv,p)),float(.5*(ap+apn))
# scores
OS={};FS={}
for f in range(1,6):
 tr=spl[(spl.fold==f)&(spl.role=="outer_train")].sample_index.to_numpy(int);va=spl[(spl.fold==f)&(spl.role=="outer_validation")].sample_index.to_numpy(int)
 for sel in SELECTORS:
  q=sa[(sa.fold==f)&(sa.selector==sel)].iloc[0];D=sis(X[tr],y[tr]) if sel=="SIS" else sparse(X[tr],y[tr],q.chosen_C,q.chosen_l1_ratio,2026091901+f);OS[(f,sel)]=(tr,va,D,anch(D))
for sel in SELECTORS:
 q=finalpars[finalpars.selector==sel].iloc[0];D=sis(X,y) if sel=="SIS" else sparse(X,y,q.C,q.l1_ratio,2026091901);FS[sel]=(D,anch(D))
# broad measurement dictionary in canonical orientation
B=pd.read_parquet(R/"MEASUREMENTS/global_BROAD/BROAD_D20_PAIR_NEUTRALIZED.parquet");arms=sorted(B.arm.astype(str).unique());assert len(arms)==2
bd={}
for pair,g in B.groupby("unordered_pair_id"):
 a,b=str(pair).split("||");bd[str(pair)]={}
 for rr in g.itertuples():
  yy=float(rr.hard_y_i_over_j) if (str(rr.gene_i)==a and str(rr.gene_j)==b) else 1-float(rr.hard_y_i_over_j);bd[str(pair)][str(rr.arm)]=(yy,float(rr.certainty_1_minus_H))
arm_pairs={arm:sorted(p for p,v in bd.items() if arm in v) for arm in arms};name={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}
# selective geometry provides only the target location/weight multiset
def selgeom_fold(f,sel,k):
 q=pd.read_csv(R/f"METHOD_ARTIFACTS/selective_ROUTING/fold{f}_{sel}/K{k}_PAIR_CONFUSION.csv");q=q[q.pair_type=="ACTIONABLE_BOUNDARY_CONFUSION"];return [(str(r.feature_A),str(r.feature_B),float(r.actionable_boundary_score)) for r in q.itertuples()]
fc=pd.read_csv(R/"METHOD_ARTIFACTS/selective_FINAL/FINAL_CROSSFITTED_selective_CONSTRAINTS.csv")
def selgeom_final(sel,k):
 g=fc[(fc.selector==sel)&(fc.k==k)].groupby(["unordered_pair_id","gene_i","gene_j"],as_index=False).A_CF.first();return [(str(r.gene_i),str(r.gene_j),float(r.A_CF)) for r in g.itertuples()]
def global_rows(target_geom,tag):
 n=len(target_geom)
 if n==0:return []
 weights=sorted([x[2] for x in target_geom],reverse=True)
 rows=[]
 for arm in arms:
  pool=arm_pairs[arm]
  if n>len(pool): raise ValueError(f'CACHE_INSUFFICIENT_SOURCE arm={arm} n={n} available={len(pool)}')
  sp=sorted(pool,key=lambda p:hashlib.sha256((tag+"|"+arm+"|"+p).encode()).hexdigest())[:n]
  sp=sorted(sp,key=lambda p:hashlib.sha256(("assign|"+tag+"|"+arm+"|"+p).encode()).hexdigest())
  for pair,A in zip(sp,weights):
   a,b=pair.split("||");i,j=name[a],name[b];yy,c=bd[pair][arm]
   rows.append((pair,i,j,yy,A*c/len(arms)))
 return rows
out=[]
for sel in SELECTORS:
 for k in KGRID:
  try:
   curve=[]
   for lam in LAMS:
    vv=[]
    for f in range(1,6):
     tr,va,D,a=OS[(f,sel)];rows=global_rows(selgeom_fold(f,sel,k),f"{f}|{sel}|{k}");ids=top(solve(a,rows,lam),D,k);vv.append(evals(tr,va,ids)[0])
    curve.append((lam,float(np.mean(vv))))
   best=max(v for _,v in curve);lam=min(e for e,v in curve if np.isclose(v,best,atol=1e-12,rtol=0));D,a=FS[sel];rows=global_rows(selgeom_final(sel,k),f"FINAL|{sel}|{k}");ids=top(solve(a,rows,lam),D,k);gau,gma=evals(None,None,ids,True)
   rr0=m0ref.loc[(sel,k)];rr3=m3ref.loc[(sel,k)]
   out.append({"dataset":"Breast","selector":sel,"k":k,"status":"COMPLETE_CACHE_ONLY_SOURCE_STRATIFIED","query_budget_semantic_pair_equivalents":len(rows)//2,"query_budget_pair_source_constraints":len(rows),"source_arms":len(arms),"global_selected_lam":lam,"global_dev_cv_auroc":best,"reference_sealed_auroc":float(rr0.auroc),"global_sealed_auroc":gau,"selective_selective_sealed_auroc":float(rr3.auroc),"global_delta_vs_reference":gau-float(rr0.auroc),"selective_delta_vs_reference":float(rr3.auroc)-float(rr0.auroc),"selective_minus_global":float(rr3.auroc)-gau,"reference_sealed_macro_ap":float(rr0.macro_ap),"global_sealed_macro_ap":gma,"selective_selective_sealed_macro_ap":float(rr3.macro_ap),"global_pair_rule":"within each frozen source arm, deterministic SHA256 sampling from the pre-existing D20 broad cache; source-specific constraint count matched to Selective; Selective actionable-weight multiset deterministically reassigned within each arm; same normalized-CE objective and lam grid","chronology":"post-hoc matched-Q2 baseline built from pre-existing cached LLM measurements; lam selected on development only; sealed outcome used only for final evaluation","new_llm_calls":0})
  except ValueError as e:
   out.append({"dataset":"Breast","selector":sel,"k":k,"status":"CACHE_INSUFFICIENT_FOR_SOURCE_STRATIFIED","notes":str(e),"new_llm_calls":0})
  print(sel,k,out[-1],flush=True)
pd.DataFrame(out).to_csv(O/"Q2_MATCHED_RESULTS.csv",index=False);(O/"Q2_MATCHED_RESULTS.json").write_text(json.dumps(out,indent=2))
