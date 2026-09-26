

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
import json,re,sys
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'EXISTING_METHOD_COMPARISON_20260925'
DATA=ROOT/'hospital_osteoporosis_dataonly_pilot_20260917'
CANON=ROOT/'hospital_osteoporosis_canonical_20260917'
sys.path.insert(0,str(DATA/'scripts'))
import v2_core as V
keymap=json.loads((OUT/'osteoporosis_feature_key.json').read_text())
resp=json.loads((OUT/'OSTEOPOROSIS_DIRECT_LLM_RESPONSES.json').read_text())['tests'][0]['content']
order=re.findall(r'\[(f\d\d)\]',resp)
assert len(order)==37 and len(set(order))==37
features=[keymap[k] for k in order[:10]]
man=json.loads((DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json').read_text())
allf=man['selectable_features']
full=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(
    pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),
    on=['patient_uid','site']).reset_index(drop=True)
prim=full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
X=prim[features].to_numpy(float); site=prim.site.to_numpy(); y=prim.y.to_numpy(int); groups=prim.patient_uid.to_numpy()
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=91019)
splits=list(cv.split(np.zeros(len(y)),y,groups))
curve=[]; best=(-1,None)
for lam in V.LAM_GRID:
    aucs=[]
    for tri,vai in splits:
        w,info,sc,nctx=V.fit_full(X[tri],site[tri],y[tri],lam,kind='l2')
        p=V.predict_full(w,sc,X[vai],site[vai])
        aucs.append(float(roc_auc_score(y[vai],p)))
    m=float(np.mean(aucs)); curve.append({'lambda':float(lam),'mean_auroc':m,'fold_aurocs':'|'.join(map(str,aucs))})
    if m>best[0]:best=(m,float(lam))
lam=best[1]; w,info,sc,nctx=V.fit_full(X,site,y,lam,kind='l2')
Xall=pd.read_csv(CANON/'canonical/site_level_X.csv')
Y=pd.read_csv(CANON/'canonical/site_level_y.csv')
Xall=Xall.copy();Xall['y']=pd.to_numeric(Y.iloc[:,0]).astype(int).to_numpy()
b2=Xall[Xall.cohort.astype(str).eq('batch2')].copy()
te=b2[b2.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
z=te[features].copy()
for c in features:
    if c=='DXA_性别':z[c]=z[c].astype(str).str.strip().map({'女':1.0,'男':0.0})
    else:z[c]=pd.to_numeric(z[c],errors='coerce')
Xt=z.to_numpy(float); yt=te.y.to_numpy(int); st=te.site.to_numpy()
p=V.predict_full(w,sc,Xt,st)
res={'method':'Direct LLM-Rank (DeepSeek)','k':10,'selected_features':features,
     'predictor_l2_lambda':lam,'batch1_inner_cv_mean_auroc':best[0],
     'batch2_auroc':float(roc_auc_score(yt,p)),
     'batch2_auprc':float(average_precision_score(yt,p)),
     'batch2_balanced_accuracy':float(balanced_accuracy_score(yt,(p>=0.5).astype(int))),
     'reference_auroc':0.7294582610372083,'selective_auroc':0.7381723539618277,
     'note':'Post-hoc pure-LLM baseline; LLM selection does not use Batch1/Batch2 outcomes. Predictor lambda chosen on Batch1 only.'}
pd.DataFrame(curve).to_csv(OUT/'OSTEOPOROSIS_LLM_RANK_LAMBDA_CV.csv',index=False)
(OUT/'OSTEOPOROSIS_LLM_RANK_RESULT.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(res,ensure_ascii=False,indent=2))
