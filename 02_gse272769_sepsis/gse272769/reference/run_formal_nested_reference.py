from pathlib import Path
import json,time,warnings,platform
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
ROOT=Path('gse272769'); src=ROOT/'frozen'; out=ROOT/'reference_formal'; out.mkdir(exist_ok=True)
SEED=2026091901; OUTER_N=5; INNER_N=4; CGRID=[.01,.03,.1,.3,1.,3.]; ALPHAS=[.2,.5,.8]; KGRID=[10,20,30]; MAXK=30
X=np.load(src/'X_development.npy').astype(float); y=np.load(src/'y_development.npy').astype(int); feat=pd.read_csv(src/'features_p1500.csv'); samples=pd.read_csv(src/'development_samples.csv')
def order(s): return np.lexsort((np.arange(len(s)),-np.asarray(s,float)))
def tune(X,y,selector,fold):
 inn=StratifiedKFold(INNER_N,shuffle=True,random_state=SEED+100*fold); al=[1.] if selector=='LASSO' else ALPHAS; rows=[]
 for a in al:
  for C in CGRID:
   auc=[]; nn=[]
   for ii,(tr,va) in enumerate(inn.split(X,y),1):
    sc=StandardScaler().fit(X[tr]); xt=sc.transform(X[tr]); xv=sc.transform(X[va]); pen='l1' if selector=='LASSO' else 'elasticnet'
    m=LogisticRegression(C=C,solver='saga',class_weight='balanced',penalty=pen,l1_ratio=None if pen=='l1' else a,max_iter=5000,tol=1e-4,random_state=SEED+fold*100+ii,n_jobs=1).fit(xt,y[tr]); auc.append(roc_auc_score(y[va],m.predict_proba(xv)[:,1])); nn.append((abs(m.coef_[0])>1e-12).sum())
   rows.append((C,a,np.mean(auc),min(nn)))
 d=pd.DataFrame(rows,columns=['C','l1_ratio','mean_inner_auroc','min_nnz']); q=d[d.min_nnz>=MAXK]; fallback=q.empty; q=d if fallback else q; q=q.sort_values(['mean_inner_auroc','C','l1_ratio'],ascending=[False,True,False]); z=q.iloc[0]; return d,dict(C=float(z.C),l1_ratio=float(z.l1_ratio),fallback=fallback)
def score(X,y,sel,par,fold):
 if sel=='SIS':
  yc=y-y.mean(); xc=X-X.mean(0); den=np.sqrt((xc*xc).sum(0)*(yc*yc).sum()); return np.nan_to_num(abs((xc*yc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0)
 sc=StandardScaler().fit(X); xt=sc.transform(X); pen='l1' if sel=='LASSO' else 'elasticnet'; m=LogisticRegression(C=par['C'],solver='saga',class_weight='balanced',penalty=pen,l1_ratio=None if pen=='l1' else par['l1_ratio'],max_iter=5000,tol=1e-4,random_state=SEED+fold,n_jobs=1).fit(xt,y); return abs(m.coef_[0])
def evalsel(tr,va,cols):
 sc=StandardScaler().fit(X[tr][:,cols]); xt=sc.transform(X[tr][:,cols]); xv=sc.transform(X[va][:,cols]); m=LogisticRegression(C=1,solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=SEED).fit(xt,y[tr]); p=m.predict_proba(xv)[:,1]; return roc_auc_score(y[va],p),average_precision_score(y[va],p),balanced_accuracy_score(y[va],p>=.5)
t0=time.time(); folds=[]; tuning=[]; selected=[]; splits=[]
outer=StratifiedKFold(OUTER_N,shuffle=True,random_state=SEED)
for fi,(tr,va) in enumerate(outer.split(X,y),1):
 for ix in tr:splits.append([fi,'outer_train',ix,samples.iloc[ix].sample_id,int(y[ix])])
 for ix in va:splits.append([fi,'outer_validation',ix,samples.iloc[ix].sample_id,int(y[ix])])
 for sel in ['LASSO','ELASTICNET','SIS']:
  par=None
  if sel!='SIS':
   tab,par=tune(X[tr],y[tr],sel,fi); tab.insert(0,'fold',fi);tab.insert(1,'selector',sel);tuning.extend(tab.to_dict('records'))
  D=score(X[tr],y[tr],sel,par,fi); o=order(D)
  for k in KGRID:
   cols=o[:k]; auc,ap,ba=evalsel(tr,va,cols); folds.append([fi,sel,k,auc,ap,ba]);
   for rank,j in enumerate(cols,1):selected.append([fi,sel,k,rank,int(j),str(feat.iloc[j].gene_symbol),float(D[j])])
 print('done',fi,flush=True)
pd.DataFrame(splits,columns=['fold','role','sample_index','sample_id','y']).to_csv(out/'OUTER_SPLITS.csv',index=False); pd.DataFrame(tuning).to_csv(out/'reference_INNER_TUNING.csv',index=False); pd.DataFrame(selected,columns=['fold','selector','k','rank','feature_index','gene_symbol','data_score']).to_csv(out/'reference_SELECTED_FEATURES.csv',index=False)
df=pd.DataFrame(folds,columns=['fold','selector','k','auroc','ap_positive','balanced_accuracy']);df.to_csv(out/'reference_FOLD_RESULTS.csv',index=False); agg=df.groupby(['selector','k']).agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),mean_ap_positive=('ap_positive','mean'),mean_balanced_accuracy=('balanced_accuracy','mean')).reset_index();agg.to_csv(out/'reference_AGGREGATE.csv',index=False); best=agg.sort_values('mean_auroc',ascending=False).iloc[0]; status={'status':'reference_COMPLETE','task':'SEPSIS_GSE272769','candidate_p':1500,'n':len(y),'positive_n':int(y.sum()),'outer_folds':5,'inner_folds':4,'seed':SEED,'best_selector':best.selector,'best_k':int(best.k),'best_mean_auroc':float(best.mean_auroc),'gate':'PREFERRED_USABLE_ZONE' if .60<=best.mean_auroc<.90 else 'REVIEW','runtime_sec':round(time.time()-t0,1),'uses_external_evidence':False,'uses_llm':False};(out/'reference_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));print(agg.to_string(index=False))
