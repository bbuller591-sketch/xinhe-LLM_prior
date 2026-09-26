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
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score

R=Path(str(REPRO_ROOT / 'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919'))
W=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap'))
O=W/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE'
TASK='BREAST_GSE25055_GSE25065';SEED=2026091903;LAMS=[0,.03,.1,.3,1,3,10.];SEL='SIS';K=20
SRC=R/'DATA/FROZEN'/TASK
X=np.load(SRC/'X_development.npy').astype(float);y=np.load(SRC/'y_development.npy').astype(int)
Xte=np.load(SRC/'X_sealed_validation.npy').astype(float);yte=np.load(SRC/'y_sealed_validation.npy').astype(int)
feat=pd.read_csv(SRC/'features_p2000.csv');P=len(feat);name={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}
spl=pd.read_csv(R/'METHOD_ARTIFACTS/reference/OUTER_SPLITS.csv')
sealed=pd.read_csv(W/'04_BREAST/sealed_validation/SEALED_reference_selective_RESULTS.csv')
ref=sealed[(sealed.selector==SEL)&(sealed.k==K)].set_index('method')
ME=pd.read_csv(O/'SOURCE_STRATIFIED_PAIR_MEASUREMENTS_GPT.csv')
mmap={(str(r.pair),str(r.arm)):r for r in ME.itertuples()}
selection=pd.read_csv(O/'GLOBAL_SOURCE_STRATIFIED_SELECTION_SIS20.csv')
arms=sorted(selection.arm.astype(str).unique())

def anch(D):
 r=rankdata(-D,method='average');return norm.ppf(np.clip(1-(r-.5)/len(D),1e-6,1-1e-6))
def sis(A,b):
 bc=b-b.mean();ac=A-A.mean(0);den=np.sqrt((ac*ac).sum(0)*(bc*bc).sum())
 return np.nan_to_num(np.abs((ac*bc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0.)
def top(z,D,k=K):return np.lexsort((np.arange(P),-D,-z))[:k]
def solve(a,rows,lam):
 if lam==0 or not rows:return a.copy()
 ii=np.array([x[0] for x in rows]);jj=np.array([x[1] for x in rows]);yy=np.array([x[2] for x in rows]);ww=np.array([x[3] for x in rows]);W0=ww.sum()
 if W0<=0:return a.copy()
 act=np.unique(np.r_[ii,jj]);pos={v:t for t,v in enumerate(act)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj]);b=a[act].copy()
 def fg(v):
  d=v[li]-v[lj];df=v-b;ce=np.logaddexp(0,d)-yy*d;f=.5*np.dot(df,df)/P+lam*np.dot(ww,ce)/W0
  g=df/P;rr=lam*(ww/W0)*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr);return float(f),g
 rr=minimize(lambda v:fg(v)[0],b,jac=lambda v:fg(v)[1],method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
 if not np.all(np.isfinite(rr.x)):raise RuntimeError('nonfinite')
 z=a.copy();z[act]=rr.x;return z
def evals(tr,va,ids,sealed_eval=False):
 Atr=X[:,ids] if sealed_eval else X[tr][:,ids];Ava=Xte[:,ids] if sealed_eval else X[va][:,ids];bt=y if sealed_eval else y[tr];bv=yte if sealed_eval else y[va]
 sc=StandardScaler().fit(Atr);m=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=2026091901).fit(sc.transform(Atr),bt)
 p=m.predict_proba(sc.transform(Ava))[:,1];ap=average_precision_score(bv,p);apn=average_precision_score(1-bv,1-p)
 return float(roc_auc_score(bv,p)),float((ap+apn)/2)
def geom_fold(f):
 q=pd.read_csv(R/f'METHOD_ARTIFACTS/selective_ROUTING/fold{f}_{SEL}/K{K}_PAIR_CONFUSION.csv');q=q[q.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION']
 return [float(r.actionable_boundary_score) for r in q.itertuples()]
def geom_final():
 fc=pd.read_csv(R/'METHOD_ARTIFACTS/selective_FINAL/FINAL_CROSSFITTED_selective_CONSTRAINTS.csv')
 g=fc[(fc.selector==SEL)&(fc.k==K)].groupby(['unordered_pair_id','gene_i','gene_j'],as_index=False).A_CF.first()
 return g.A_CF.astype(float).tolist()
def rows_for(tag,weights):
 rows=[];missing=[]
 for arm in arms:
  sp=selection[(selection.tag==tag)&(selection.arm==arm)].global_pair.astype(str).tolist()
  sp=sorted(sp,key=lambda p:hashlib.sha256(('assign|'+tag+'|'+arm+'|'+p).encode()).hexdigest())
  ww=sorted(weights,reverse=True);assert len(sp)==len(ww),(tag,arm,len(sp),len(ww))
  for pair,A in zip(sp,ww):
   mm=mmap.get((pair,arm))
   if mm is None:missing.append((pair,arm));continue
   a,b=pair.split('||');i,j=name[a],name[b]
   yy=float(mm.binary_choice_i)
   # mm gene_i/gene_j come from canonical query pair orientation; correct if needed.
   if str(mm.gene_i)==b and str(mm.gene_j)==a:yy=1-yy
   rows.append((i,j,yy,A*float(mm.certainty)/len(arms),pair,arm))
 return rows,missing

folds=[];curve=[]
for f in range(1,6):
 tr=spl[(spl.fold==f)&(spl.role=='outer_train')].sample_index.to_numpy(int);va=spl[(spl.fold==f)&(spl.role=='outer_validation')].sample_index.to_numpy(int)
 D=sis(X[tr],y[tr]);a=anch(D);rr,miss=rows_for(f'{f}|{SEL}|{K}',geom_fold(f));folds.append((f,tr,va,D,a,rr,miss))
for lam in LAMS:
 vals=[evals(tr,va,top(solve(a,rr,lam),D))[0] for f,tr,va,D,a,rr,miss in folds]
 curve.append((lam,float(np.mean(vals))))
best=max(v for _,v in curve);lam=min(e for e,v in curve if np.isclose(v,best,atol=1e-12,rtol=0))
D=sis(X,y);a=anch(D);rr,missf=rows_for(f'FINAL|{SEL}|{K}',geom_final());ids=top(solve(a,rr,lam),D);gau,gma=evals(None,None,ids,True)
r0=ref.loc['reference'];r3=ref.loc['selective']
summary={'dataset':'Breast','selector':SEL,'k':K,'status':'COMPLETE_GPT_SOURCE_STRATIFIED_MATCHED_GLOBAL',
 'global_selected_lam':lam,'global_dev_cv_auroc':best,'reference_sealed_auroc':float(r0.auroc),'global_sealed_auroc':gau,
 'selective_selective_sealed_auroc':float(r3.auroc),'global_delta_vs_reference':gau-float(r0.auroc),
 'selective_delta_vs_reference':float(r3.auroc)-float(r0.auroc),'selective_minus_global':float(r3.auroc)-gau,
 'reference_sealed_macro_ap':float(r0.macro_ap),'global_sealed_macro_ap':gma,'selective_selective_sealed_macro_ap':float(r3.macro_ap),
 'global_final_constraints':len(rr),'dropped_final_source_cells_top20':len(missf),
 'fold_dropped_source_cells_top20':{str(f):len(miss) for f,tr,va,D,a,rr0,miss in folds},
 'new_llm_calls_total_manifest':2366,'unique_swap_pair_source_measurements_used':int(len(ME)),
 'lam_curve':[{'lam':e,'mean_auroc':v} for e,v in curve]}
pd.DataFrame([{k:v for k,v in summary.items() if k not in ('lam_curve','fold_dropped_source_cells_top20')}]).to_csv(O/'Q2_MATCHED_RESULTS_GPT.csv',index=False)
(O/'Q2_MATCHED_RESULTS_GPT.json').write_text(json.dumps(summary,indent=2)+'\n')
pd.DataFrame({'rank':np.arange(1,K+1),'feature_index':ids,'gene_symbol':feat.iloc[ids].gene_symbol.astype(str).to_numpy()}).to_csv(O/'GLOBAL_FINAL_SUPPORT_SIS20.csv',index=False)
print(json.dumps(summary,indent=2))
