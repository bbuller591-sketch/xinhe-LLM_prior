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
import sys,json,math,hashlib,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,ndtri
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
DATA=Path(str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917'))
CANON=Path(str(REPRO_ROOT / 'hospital_osteoporosis_canonical_20260917'))
BASE=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4'))
OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion/artifacts/HOSPITAL_Q3_TEMPORAL_NULL'))
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(DATA/"scripts"));import v2_core as V
K=10;NULLSEED={"semantic":2026092203,"random":2026092204}
man=json.load(open(DATA/"03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json"));features=list(man["selectable_features"]);fidx={f:i for i,f in enumerate(features)}
full=pd.read_csv(DATA/"03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv").merge(pd.read_csv(DATA/"03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv",usecols=["patient_uid","site","y"]),on=["patient_uid","site"]).reset_index(drop=True)
dev=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True);Xdev=dev[features].to_numpy(float);sdev=dev.site.to_numpy();ydev=dev.y.to_numpy(int);gdev=dev.patient_uid.to_numpy()
Xall=pd.read_csv(CANON/"canonical/site_level_X.csv");Y=pd.read_csv(CANON/"canonical/site_level_y.csv");Xall=Xall.copy();Xall["y"]=pd.to_numeric(Y.iloc[:,0]).astype(int).to_numpy()
ext=Xall[Xall.cohort.astype(str).eq("batch2")].copy();ext=ext[ext.site.isin(V.SITE_PRIMARY)].reset_index(drop=True);yext=ext.y.to_numpy(int);sext=ext.site.to_numpy()
score=pd.read_csv(ROOT/"09_BATCH1_DOWNSTREAM_V1_9/FINAL_FULL_BATCH1_DATA_SELECTOR_SCORES.csv").set_index("feature").loc[features];D=score.D_full_batch1.to_numpy(float);P=len(D)
pairk=pd.read_csv(ROOT/"09_BATCH1_DOWNSTREAM_V1_9/selective_PAIRK_WEIGHTS_FROZEN.csv");q=pairk[(pairk.k==K)&(pairk.evidence_gate==1)].reset_index(drop=True)
pobs=q.p_selective_A.to_numpy(float);cobs=q.c_pair.to_numpy(float);Avec=q.actionable_boundary_score.to_numpy(float);ii=np.array([fidx[f] for f in q.feature_A]);jj=np.array([fidx[f] for f in q.feature_B])
active=np.unique(np.r_[ii,jj]);pos={v:k for k,v in enumerate(active)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj])
def certainty(p):
 p=np.clip(np.asarray(p,float),1e-12,1-1e-12);H=-(p*np.log(p)+(1-p)*np.log(1-p))/math.log(2);return 1-H
def anchor(D):
 r=rankdata(-D,method="average");u=1-(r-.5)/len(D);return ndtri(np.clip(u,1e-6,1-1e-6))
def solve(lam,p,c):
 sd=anchor(D)
 if lam==0:return sd
 w=Avec*c;W=w.sum()
 if W<=0:return sd
 yy=(p>.5).astype(float);b=sd[active].copy()
 def fg(v):
  d=v[li]-v[lj];df=v-b;ce=np.logaddexp(0,d)-yy*d;f=.5*np.dot(df,df)/P+lam*np.dot(w,ce)/W
  g=df/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
 r=minimize(lambda v:fg(v)[0],b,jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":1000,"maxls":60,"ftol":1e-12,"gtol":1e-8})
 if not r.success:
  x0=r.x if np.all(np.isfinite(r.x)) else b
  r=minimize(lambda v:fg(v)[0],x0,jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":3000,"maxls":300,"ftol":1e-13,"gtol":1e-7})
 if not r.success:
  x0=r.x if np.all(np.isfinite(r.x)) else b
  r=minimize(lambda v:fg(v)[0],x0,jac=lambda v:fg(v)[1],method="BFGS",options={"maxiter":3000,"gtol":1e-7})
 if (not r.success) and (not np.all(np.isfinite(r.x))): raise RuntimeError(r.message)
 z=sd.copy();z[active]=r.x;return z
def top_support(z):
 t=pd.DataFrame({"f":features,"z":z,"D":D,"idx":np.arange(P)}).sort_values(["z","D","idx"],ascending=[False,False,True],kind="mergesort")
 return tuple(t.f.iloc[:K])
def bundle(mode,rep):
 rng=np.random.default_rng(np.random.SeedSequence([NULLSEED[mode],rep]))
 if mode=="semantic":
  z=rng.permutation(len(pobs));return pobs[z],cobs[z]
 p=rng.uniform(0,1,len(pobs));return p,certainty(p)
def choose_lam(X,s,y,g):
 cv=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=91019).split(np.zeros(len(y)),y,g));best=(-1,None)
 for lam in V.LAM_GRID:
  aa=[]
  for tr,va in cv:
   w,_,sc,_=V.fit_full(X[tr],s[tr],y[tr],lam,kind="l2");aa.append(roc_auc_score(y[va],V.predict_full(w,sc,X[va],s[va])))
  m=float(np.mean(aa))
  if m>best[0]:best=(m,float(lam))
 return best[1],best[0]
def eval_support(supp):
 fs=supp.split("|");ids=[fidx[f] for f in fs];Xd=Xdev[:,ids]
 lam,cv=choose_lam(Xd,sdev,ydev,gdev);w,_,sc,_=V.fit_full(Xd,sdev,ydev,lam,kind="l2")
 Xe=ext[fs].copy()
 for c in fs:
  if c=="DXA_性别":Xe[c]=Xe[c].astype(str).str.strip().map({"女":1.0,"男":0.0})
  else:Xe[c]=pd.to_numeric(Xe[c],errors="coerce")
 p=V.predict_full(w,sc,Xe.to_numpy(float),sext)
 return supp,lam,cv,float(roc_auc_score(yext,p)),float(average_precision_score(yext,p)),float(balanced_accuracy_score(yext,p>=.5))
obsres=pd.read_csv(ROOT/"11_BATCH2_FINAL_EVALUATION_V2_0_1/BATCH2_PRIMARY_RESULTS.csv");obs=obsres[(obsres.method=="selective")&(obsres.k==K)].iloc[0];ref=obsres[(obsres.method=="reference")&(obsres.k==K)].iloc[0]
for mode,src in [("semantic",BASE/"01_q3_semantic_shuffle/HOSPITAL_OSTEOPOROSIS/REPLICATES_1000.csv"),("random",BASE/"02_q3_random_null/HOSPITAL_OSTEOPOROSIS/REPLICATES_1000.csv")]:
 R=pd.read_csv(src).sort_values("replicate");rr=[]
 for x in R.itertuples():
  p,c=bundle(mode,int(x.replicate));sp=top_support(solve(float(x.chosen_lam),p,c));rr.append({"replicate":int(x.replicate),"chosen_lam":float(x.chosen_lam),"support":"|".join(sp)})
 M=pd.DataFrame(rr);unique=M.support.unique().tolist();print(mode,"unique full supports",len(unique),flush=True)
 ev=[]
 with ProcessPoolExecutor(max_workers=12) as exx:
  fut=[exx.submit(eval_support,s) for s in unique]
  for n,f in enumerate(as_completed(fut),1):
   ev.append(f.result())
   if n%50==0:print(mode,"eval",n,"/",len(unique),flush=True)
 E=pd.DataFrame(ev,columns=["support","predictor_l2_lambda","batch1_cv_auc","external_auroc","external_auprc","external_balacc"])
 Z=M.merge(E,on="support",validate="many_to_one");Z.to_csv(OUT/f"{mode.upper()}_TEMPORAL_EXTERNAL_1000.csv",index=False)
 oe=float(obs.auroc);oap=float(obs.auprc);v=Z.external_auroc;vp=Z.external_auprc
 summ={"dataset":"Hospital Osteoporosis","configuration":"selective k=10","mode":mode,"n_replicates":len(Z),"new_llm_calls":0,
  "chronology":"post-hoc temporal external evaluation of nulls whose trust parameters were selected using Batch1 only; Batch2 never enters lam or predictor-lambda selection",
  "observed_external_auroc":oe,"reference_external_auroc":float(ref.auroc),"observed_external_delta_vs_ref":oe-float(ref.auroc),
  "null_external_mean_auroc":float(v.mean()),"null_external_sd_auroc":float(v.std(ddof=1)),"null_external_q025_auroc":float(v.quantile(.025)),"null_external_q975_auroc":float(v.quantile(.975)),
  "empirical_p_external_ge_observed":float((1+(v>=oe-1e-12).sum())/(len(v)+1)),"n_lower":int((v<oe-1e-12).sum()),"n_equal":int(np.isclose(v,oe,atol=1e-12,rtol=0).sum()),"n_greater":int((v>oe+1e-12).sum()),
  "null_external_mean_delta_vs_ref":float((v-float(ref.auroc)).mean()),"observed_external_auprc":oap,"null_external_mean_auprc":float(vp.mean()),"empirical_p_external_auprc_ge_observed":float((1+(vp>=oap-1e-12).sum())/(len(vp)+1)),"n_unique_final_supports":len(unique)}
 pd.DataFrame([summ]).to_csv(OUT/f"{mode.upper()}_SUMMARY.csv",index=False);(OUT/f"{mode.upper()}_SUMMARY.json").write_text(json.dumps(summ,indent=2,ensure_ascii=False));print(json.dumps(summ,indent=2,ensure_ascii=False),flush=True)
