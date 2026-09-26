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
import sys,hashlib,json,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,ndtri
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
R=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'));DATA=Path(str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917'));CAN=Path(str(REPRO_ROOT / 'hospital_osteoporosis_canonical_20260917'))
O=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2'));O.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(DATA/"scripts"));import v2_core as V;from common import SEEDS,N_FOLDS
LAMS=[0,.03,.1,.3,1,3,10];KGRID=[10]
man=json.load(open(DATA/"03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json"));features=list(man["selectable_features"]);fidx={f:i for i,f in enumerate(features)}
full=pd.read_csv(DATA/"03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv").merge(pd.read_csv(DATA/"03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv",usecols=["patient_uid","site","y"]),on=["patient_uid","site"]).reset_index(drop=True);dev=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True);X=dev[features].to_numpy(float);site=dev.site.to_numpy();y=dev.y.to_numpy(int);groups=dev.patient_uid.to_numpy()
def folds(y,g,seed,n):return list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))
outer=[(int(seed),int(fi),tr,va) for seed in SEEDS for fi,(tr,va) in enumerate(folds(y,groups,seed,N_FOLDS))]
scores=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/downstream_batch1/OUTER_SELECTOR_SCORES.csv')));Dmap={(s,fi):scores[(scores.seed==s)&(scores.fold==fi)].set_index("feature").loc[features].D.to_numpy(float) for s,fi,_,_ in outer}
Dfull=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/downstream_batch1/FINAL_FULL_BATCH1_DATA_SELECTOR_SCORES.csv'))).set_index("feature").loc[features].D_full_batch1.to_numpy(float);P=len(features)
selpair=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/downstream_batch1/selective_PAIRK_WEIGHTS_FROZEN.csv')));broad=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/08_Q2_Q5_DIAGNOSTICS/HOSPITAL_Q2/Q2_GLOBAL_PAIR_MEASUREMENTS_GPT.csv')))
obsdev=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/downstream_batch1/SELECTED_DEVELOPMENT_HYPERPARAMS.csv')));obsext=pd.read_csv(Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/06_HOSPITAL/final_temporal_eval/BATCH2_PRIMARY_RESULTS.csv')))
Xall=pd.read_csv(CAN/"canonical/site_level_X.csv");Y=pd.read_csv(CAN/"canonical/site_level_y.csv");Xall=Xall.copy();Xall["y"]=pd.to_numeric(Y.iloc[:,0]).astype(int);ext=Xall[(Xall.cohort.astype(str)=="batch2")&Xall.site.isin(V.SITE_PRIMARY)].reset_index(drop=True);ye=ext.y.to_numpy(int);se=ext.site.to_numpy()
def anch(D):
 r=rankdata(-D,method="average");return ndtri(np.clip(1-(r-.5)/len(D),1e-6,1-1e-6))
def top(z,D,k):return np.lexsort((np.arange(P),-D,-z))[:k]
def solve(D,rows,lam):
 a=anch(D)
 if lam==0:return a
 ii=np.array([x[0] for x in rows]);jj=np.array([x[1] for x in rows]);yy=np.array([x[2] for x in rows]);w=np.array([x[3] for x in rows]);W=w.sum();act=np.unique(np.r_[ii,jj]);pos={v:t for t,v in enumerate(act)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj]);b=a[act].copy()
 def fg(v):
  d=v[li]-v[lj];df=v-b;ce=np.logaddexp(0,d)-yy*d;f=.5*np.dot(df,df)/P+lam*np.dot(w,ce)/W;g=df/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
 rr=minimize(lambda v:fg(v)[0],b,jac=lambda v:fg(v)[1],method="L-BFGS-B",options={"maxiter":2000,"ftol":1e-12,"gtol":1e-8});z=a.copy();z[act]=rr.x;return z
def global_rows(k):
 q=selpair[(selpair.k==k)&(selpair.evidence_gate==1)];N=len(q);A=sorted(q.actionable_boundary_score.to_numpy(float),reverse=True);cand=broad.copy();cand["h"]=[hashlib.sha256((f"HOSP|K{k}|"+str(p)).encode()).hexdigest() for p in cand.pair_id];cand=cand.sort_values("h").head(N).copy();cand["ha"]=[hashlib.sha256(("ASSIGN|"+str(p)).encode()).hexdigest() for p in cand.pair_id];cand=cand.sort_values("ha")
 return [(fidx[str(r.feature_A)],fidx[str(r.feature_B)],1.0 if float(r.p_pair_semantic_A)>.5 else 0.0,float(a)*float(r.c_pair)) for r,a in zip(cand.itertuples(),A)]
def choose_lam(X0,s0,y0,g0,seed=91019,n=5):
 best=(-1,None)
 for lam in V.LAM_GRID:
  aa=[]
  for tr,va in folds(y0,g0,seed,n):
   w,_,sc,_=V.fit_full(X0[tr],s0[tr],y0[tr],lam,kind="l2");aa.append(roc_auc_score(y0[va],V.predict_full(w,sc,X0[va],s0[va])))
  m=float(np.mean(aa))
  if m>best[0]:best=(m,float(lam))
 return best[1],best[0]
def eval_outer(tr,va,ids,seed,fi):
 Xtr=X[tr][:,ids];Xva=X[va][:,ids];lam,_=choose_lam(Xtr,site[tr],y[tr],groups[tr],seed+fi,4);w,_,sc,_=V.fit_full(Xtr,site[tr],y[tr],lam,kind="l2");return float(roc_auc_score(y[va],V.predict_full(w,sc,Xva,site[va])))
def eval_ext(ids):
 fs=[features[i] for i in ids];lam,cv=choose_lam(X[:,ids],site,y,groups);w,_,sc,_=V.fit_full(X[:,ids],site,y,lam,kind="l2");Xe=ext[fs].copy()
 for c in fs:Xe[c]=Xe[c].astype(str).str.strip().map({"女":1.,"男":0.}) if c=="DXA_性别" else pd.to_numeric(Xe[c],errors="coerce")
 p=V.predict_full(w,sc,Xe.to_numpy(float),se);return float(roc_auc_score(ye,p)),float(average_precision_score(ye,p)),float(balanced_accuracy_score(ye,p>=.5)),lam,cv
out=[]
for k in KGRID:
 rows=global_rows(k);curve=[]
 for lam in LAMS:
  aa=[eval_outer(tr,va,top(solve(Dmap[(seed,fi)],rows,lam),Dmap[(seed,fi)],k),seed,fi) for seed,fi,tr,va in outer];curve.append((lam,float(np.mean(aa))))
 best=max(v for _,v in curve);lam=min(e for e,v in curve if np.isclose(v,best,atol=1e-12,rtol=0));ids=top(solve(Dfull,rows,lam),Dfull,k);gau,gap,gba,plam,pcv=eval_ext(ids)
 rr0=obsext[(obsext.method=="reference")&(obsext.k==k)].iloc[0];rr3=obsext[(obsext.method=="selective")&(obsext.k==k)].iloc[0]
 out.append({"dataset":"Hospital Osteoporosis","selector":"L1 logistic rank","k":k,"query_budget_pairs":len(rows),"global_selected_lam":lam,"global_dev_mean_auroc":best,"reference_temporal_auroc":float(rr0.auroc),"global_temporal_auroc":gau,"selective_selective_temporal_auroc":float(rr3.auroc),"global_delta_vs_reference":gau-float(rr0.auroc),"selective_delta_vs_reference":float(rr3.auroc)-float(rr0.auroc),"selective_minus_global":float(rr3.auroc)-gau,"global_temporal_auprc":gap,"selective_selective_temporal_auprc":float(rr3.auprc),"predictor_l2_lambda":plam,"global_pair_rule":"lowest SHA256 from frozen complete broad graph; pair count matched to Selective; Selective actionable-weight multiset deterministically reassigned; identical semantic probability/certainty format and normalized-CE objective","chronology":"post-hoc matched-Q2 baseline from cached pre-existing measurements; lam and predictor lambda selected on Batch1 only; Batch2 only final evaluation","new_llm_calls":22});print(out[-1],flush=True)
pd.DataFrame(out).to_csv(O/"Q2_MATCHED_RESULTS.csv",index=False);(O/"Q2_MATCHED_RESULTS.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
