import numpy as np,pandas as pd,json
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
Xall=np.load('gse272769/audit/X_gene_ranked.npy')[:,:1500]
y=pd.read_csv('gse272769/audit/samples_y.csv').mort30.values
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=20260919)
rows=[]
for fold,(tr,va) in enumerate(cv.split(Xall,y)):
    # standardize using training only
    sc=StandardScaler().fit(Xall[tr]); Xt=sc.transform(Xall[tr]); Xv=sc.transform(Xall[va])
    # SIS score computed training-only
    F,_=f_classif(Xt,y[tr]); order=np.argsort(np.nan_to_num(F,nan=-np.inf))[::-1]
    for k in [10,20,30]:
        idx=order[:k]
        m=LogisticRegression(C=1.0,penalty='l2',solver='liblinear',max_iter=5000,class_weight='balanced').fit(Xt[:,idx],y[tr])
        pr=m.predict_proba(Xv[:,idx])[:,1]; pred=(pr>=.5).astype(int)
        rows.append([fold,'SIS',k,roc_auc_score(y[va],pr),average_precision_score(y[va],pr),balanced_accuracy_score(y[va],pred)])
    # sparse selectors; tune C within outer train is deliberately omitted here: diagnostic reference only
    for name,l1r in [('LASSO',1.0),('ELASTICNET',.5)]:
      for C in [.03,.1,.3]:
        m=LogisticRegression(C=C,penalty='elasticnet',l1_ratio=l1r,solver='saga',max_iter=10000,class_weight='balanced',random_state=fold).fit(Xt,y[tr])
        coef=np.abs(m.coef_[0]); order2=np.argsort(coef)[::-1]
        for k in [10,20,30]:
          idx=order2[:k]
          # refit top-k data-only predictive model
          r=LogisticRegression(C=1.0,penalty='l2',solver='liblinear',max_iter=5000,class_weight='balanced').fit(Xt[:,idx],y[tr])
          pr=r.predict_proba(Xv[:,idx])[:,1]; pred=(pr>=.5).astype(int)
          rows.append([fold,name+'_C'+str(C),k,roc_auc_score(y[va],pr),average_precision_score(y[va],pr),balanced_accuracy_score(y[va],pred)])
df=pd.DataFrame(rows,columns=['fold','selector','k','auroc','ap','balanced_accuracy']); df.to_csv('gse272769/reference/reference_DIAGNOSTIC_FOLDS.csv',index=False)
agg=df.groupby(['selector','k']).agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),mean_ap=('ap','mean'),mean_bal_acc=('balanced_accuracy','mean')).reset_index(); agg.to_csv('gse272769/reference/reference_DIAGNOSTIC_AGGREGATE.csv',index=False)
print(agg.sort_values('mean_auroc',ascending=False).head(20).to_string(index=False))
