

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
import json,re
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'EXISTING_METHOD_COMPARISON_20260925'
DATA=ROOT/'CREDIT_G_REPRO_PACKAGE_20260919/01_DATA_AND_SPLITS'
keys=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
saved=json.loads((OUT/'CREDIT_G_SCORE_RANK_ORDERS.json').read_text())
scores=saved['scores']
rank=saved['rank_order']
score_order=saved['score_order']
seq=json.loads((OUT/'CREDIT_G_SEQ_ORDER.json').read_text())['seq_order']
assert len(rank)==20 and set(rank)==set(keys)
assert len(seq)==10 and len(set(seq))==10
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[k for k in keys if k not in CAT]
def prep(cols):
    tr=[]; num=[c for c in NUM if c in cols]; cat=[c for c in CAT if c in cols]
    if num:tr.append(('num',StandardScaler(),num))
    if cat:tr.append(('cat',Pipeline([('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),('scale',StandardScaler())]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)
X=pd.read_csv(DATA/'X.csv'); y=pd.read_csv(DATA/'y.csv')['label'].to_numpy()
dev=np.load(DATA/'modern_dev_indices_seed20260918.npy'); hold=np.load(DATA/'modern_holdout_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]; Xh=X.iloc[hold].reset_index(drop=True); yh=y[hold]
def ev(sel):
    pipe=Pipeline([('prep',prep(sel)),('clf',LogisticRegression(penalty='l2',solver='lbfgs',C=0.01,fit_intercept=True,class_weight=None,max_iter=5000,tol=1e-5))])
    pipe.fit(Xd[sel],yd); p=pipe.predict_proba(Xh[sel])[:,1]
    return float(roc_auc_score(yh,p)),float(average_precision_score(yh,p)),float(log_loss(yh,p,labels=[0,1]))
rows=[]
for name,order in [('Direct LLM-Score (DeepSeek; LLM-Select-style)',score_order),('Direct LLM-Rank (DeepSeek; LLM-Select-style)',rank),('LLM-Seq (DeepSeek; LLM-Select-style)',seq)]:
    sel=order[:10]; a,b,c=ev(sel)
    rows.append({'method':name,'k':10,'selected_features':'|'.join(sel),'holdout_auroc':a,'holdout_ap':b,'holdout_log_loss':c})
rows += [
 {'method':'Data-only Reference (GBM-permutation)','k':10,'selected_features':'frozen','holdout_auroc':0.780119,'holdout_ap':0.580819,'holdout_log_loss':0.512819},
 {'method':'Selective Correction (GBM-permutation, DeepSeek)','k':10,'selected_features':'frozen','holdout_auroc':0.793333,'holdout_ap':0.582880,'holdout_log_loss':0.508812},
 {'method':'All 20 features','k':20,'selected_features':'all','holdout_auroc':0.792024,'holdout_ap':0.622130,'holdout_log_loss':0.499551}]
df=pd.DataFrame(rows); df.to_csv(OUT/'CREDIT_G_SCORE_RANK_RESULTS.csv',index=False)
(OUT/'CREDIT_G_SCORE_RANK_ORDERS.json').write_text(json.dumps({'scores':scores,'score_order':score_order,'rank_order':rank,'seq_order':seq},indent=2))
print(df.to_string(index=False))
