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
import os
import sys, json, io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(str(REPRO_ROOT))
W=ROOT/'PURE_LLM_ALL_NON_JKP_20260926'
rows=[]
SCORE_VARIANT=os.environ.get('SCORE_VARIANT','LLM_SCORE')

def order_indices(ds):
    d=pd.read_csv(W/ds/SCORE_VARIANT/'LLM_SCORE.csv')
    return d['index'].astype(int).tolist()

# CREDIT-G
FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment",
"installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age",
"other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status",
"other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job",
"num_dependents","own_telephone","foreign_worker"]
NUM=[x for x in FEATURES if x not in CAT]
SRC=ROOT/'CREDIT_G_REPRO_PACKAGE_20260919'
X=pd.read_csv(SRC/'01_DATA_AND_SPLITS/X.csv');y=pd.read_csv(SRC/'01_DATA_AND_SPLITS/y.csv')['label'].to_numpy()
dev=np.load(SRC/'01_DATA_AND_SPLITS/modern_dev_indices_seed20260918.npy');hold=np.load(SRC/'01_DATA_AND_SPLITS/modern_holdout_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev];Xh=X.iloc[hold].reset_index(drop=True);yh=y[hold]
def prep(cols):
    num=[c for c in NUM if c in cols];cat=[c for c in CAT if c in cols];tr=[]
    if num:tr.append(('num',StandardScaler(),num))
    if cat:tr.append(('cat',Pipeline([('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),('scale',StandardScaler())]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)
idx=order_indices('CREDIT_G')[:10];sel=[FEATURES[i] for i in idx]
pipe=Pipeline([('prep',prep(sel)),('clf',LogisticRegression(penalty='l2',solver='lbfgs',C=.01,max_iter=5000,tol=1e-5))])
pipe.fit(Xd[sel],yd);p=pipe.predict_proba(Xh[sel])[:,1]
rows.append({'dataset':'CREDIT-G','k':10,'auroc':float(roc_auc_score(yh,p)),'selected_features':'|'.join(sel)})

# OSTEOPOROSIS
DATA=ROOT/'hospital_osteoporosis_dataonly_pilot_20260917';CANON=ROOT/'hospital_osteoporosis_canonical_20260917'
sys.path.insert(0,str(DATA/'scripts'));import v2_core as V
man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json'));all_features=list(man['selectable_features'])
full=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),on=['patient_uid','site']).reset_index(drop=True)
b1=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True);y1=b1.y.to_numpy(int);site1=b1.site.to_numpy();groups=b1.patient_uid.to_numpy()
Xall=pd.read_csv(CANON/'canonical/site_level_X.csv');Y=pd.read_csv(CANON/'canonical/site_level_y.csv');Xall=Xall.copy();Xall['y']=pd.to_numeric(Y.iloc[:,0]).astype(int).to_numpy()
b2=Xall[Xall.cohort.astype(str).eq('batch2') & Xall.site.isin(V.SITE_PRIMARY)].reset_index(drop=True);y2=b2.y.to_numpy(int);site2=b2.site.to_numpy()
idx=order_indices('OSTEOPOROSIS')[:10];sel=[all_features[i] for i in idx]
splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=91019).split(np.zeros(len(y1)),y1,groups))
best=(-1,None)
Xsel=b1[sel].to_numpy(float)
for lam in V.LAM_GRID:
    vals=[]
    for tr,va in splits:
        w,info,sc,nctx=V.fit_full(Xsel[tr],site1[tr],y1[tr],lam,kind='l2');pp=V.predict_full(w,sc,Xsel[va],site1[va]);vals.append(roc_auc_score(y1[va],pp))
    m=float(np.mean(vals))
    if m>best[0]:best=(m,float(lam))
w,info,sc,nctx=V.fit_full(Xsel,site1,y1,best[1],kind='l2')
zb2=pd.DataFrame(index=b2.index)
for f in sel:
    if f=='DXA_性别':
        zb2[f]=b2[f].astype(str).str.strip().map({'女':1.0,'男':0.0,'female':1.0,'male':0.0,'1.0':1.0,'0.0':0.0,'1':1.0,'0':0.0})
    else:
        zb2[f]=pd.to_numeric(b2[f],errors='coerce')
pp=V.predict_full(w,sc,zb2[sel].to_numpy(float),site2)
rows.append({'dataset':'Osteoporosis','k':10,'auroc':float(roc_auc_score(y2,pp)),'selected_features':'|'.join(sel)})

# Darmanis
B=ROOT/'gbm_canonical_recovery_20260917'
X=np.load(B/'canonical/X_candidate_aligned_log1p.npy').astype(float);y=pd.read_csv(B/'canonical/y_canonical.csv')['y'].to_numpy(int)
f=pd.read_csv(B/'canonical/candidate_features_canonical.csv');names=f.gene_symbol.fillna('').astype(str);names=names.where(names.str.len()>0,f.ensembl_gene_id_stable.astype(str)).tolist()
mp=pd.read_csv(B/'canonical/cell_sample_mapping_canonical.csv');ann=pd.read_csv(B/'metadata/darmanis_cell_annotation.csv')[['cell_key','plate']]
groups=mp.merge(ann,left_on='cell_id',right_on='cell_key',validate='one_to_one')['plate'].astype(str).to_numpy()
splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026091817).split(X,y,groups));order=order_indices('DARMANIS_GBM')
for k in [10,20]:
    cols=np.asarray(order[:k],int);vals=[]
    for fi,(tr,va) in enumerate(splits,1):
        sc=StandardScaler().fit(X[tr][:,cols]);m=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=2026091817).fit(sc.transform(X[tr][:,cols]),y[tr]);pp=m.predict_proba(sc.transform(X[va][:,cols]))[:,1];vals.append(roc_auc_score(y[va],pp))
    rows.append({'dataset':'Darmanis GBM','k':k,'auroc':float(np.mean(vals)),'selected_features':'|'.join(names[i] for i in cols)})

# Sepsis
B=ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/01_FROZEN_DATA'
X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);names=pd.read_csv(B/'features_p1500.csv').gene_symbol.astype(str).tolist();sp=pd.read_csv(B/'OUTER_SPLITS.csv')
order=order_indices('GSE272769_SEPSIS');cols=np.asarray(order[:50]);vals=[]
for fno in sorted(sp.fold.unique()):
    tr=sp[(sp.fold==fno)&(sp.role=='outer_train')].sample_index.astype(int).to_numpy();va=sp[(sp.fold==fno)&(sp.role=='outer_validation')].sample_index.astype(int).to_numpy()
    sc=StandardScaler().fit(X[tr][:,cols]);m=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=20260919+int(fno)).fit(sc.transform(X[tr][:,cols]),y[tr]);pp=m.predict_proba(sc.transform(X[va][:,cols]))[:,1];vals.append(roc_auc_score(y[va],pp))
rows.append({'dataset':'GSE272769 Sepsis','k':50,'auroc':float(np.mean(vals)),'selected_features':'|'.join(names[i] for i in cols)})

# Breast
B=ROOT/'GSE25055_GSE25065_FORMAL_REPRODUCIBILITY_PACKAGE/DATA/FROZEN/BREAST_GSE25055_GSE25065'
X=np.load(B/'X_development.npy').astype(float);y=np.load(B/'y_development.npy').astype(int);Xe=np.load(B/'X_sealed_validation.npy').astype(float);ye=np.load(B/'y_sealed_validation.npy').astype(int);names=pd.read_csv(B/'features_p2000.csv').gene_symbol.astype(str).tolist()
cols=np.asarray(order_indices('BREAST_PCR')[:20]);sc=StandardScaler().fit(X[:,cols]);m=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=2026091901).fit(sc.transform(X[:,cols]),y);pp=m.predict_proba(sc.transform(Xe[:,cols]))[:,1]
rows.append({'dataset':'Breast pCR','k':20,'auroc':float(roc_auc_score(ye,pp)),'selected_features':'|'.join(names[i] for i in cols)})

# Renal
cache=ROOT/'CROSS_DATASET_RECENT_LLM_FS_20260926/RENAL_TCMR/RENAL_FULL_MATRICES.npz';z=np.load(cache);X=z['X'];y=z['y'];Xe=z['Xe'];ye=z['ye'];names=pd.read_csv(ROOT/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR/CANDIDATE_UNIVERSE_FROZEN.csv').gene.astype(str).tolist()
cols=np.asarray(order_indices('RENAL_TCMR')[:50]);sc=StandardScaler().fit(X[:,cols]);m=LogisticRegression(C=1.,penalty='l2',solver='liblinear',class_weight='balanced',max_iter=5000,random_state=20261044).fit(sc.transform(X[:,cols]),y);pp=m.predict_proba(sc.transform(Xe[:,cols]))[:,1]
rows.append({'dataset':'Renal TCMR','k':50,'auroc':float(roc_auc_score(ye,pp)),'selected_features':'|'.join(names[i] for i in cols)})

out=pd.DataFrame(rows);out.to_csv(W/'SUMMARY'/f'PURE_LLM_STANDARDIZED_RESULTS_{SCORE_VARIANT}.csv',index=False)
print(out[['dataset','k','auroc']].to_string(index=False))
