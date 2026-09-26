

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
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
DATA=WS/'legacy_archive_snapshot/positive_results_package_20260918/credit_g/data'
TASK=WS/'01_TASK_FREEZE'
R=WS/'06_RESULTS'

FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[x for x in FEATURES if x not in CAT]

def prep(cols):
    cols=list(cols)
    num=[c for c in NUM if c in cols]
    cat=[c for c in CAT if c in cols]
    tr=[]
    if num:
        tr.append(('num',StandardScaler(),num))
    if cat:
        tr.append(('cat',Pipeline([
            ('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),
            ('scale',StandardScaler()),
        ]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)

X=pd.read_csv(DATA/'X.csv')
y=pd.read_csv(DATA/'y.csv')['label'].to_numpy()
dev=np.load(TASK/'modern_dev_indices_seed20260918.npy')
hold=np.load(TASK/'modern_holdout_indices_seed20260918.npy')
assert len(dev)==800 and len(hold)==200 and len(np.intersect1d(dev,hold))==0

Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]
Xh=X.iloc[hold].reset_index(drop=True); yh=y[hold]

sets=pd.read_csv(R/'FINAL_SELECTED_SETS_FREEZE.csv')
rows=[]
pred_rows=[]
for _,z in sets.iterrows():
    sel=str(z.selected_features).split('|')
    assert len(sel)==int(z.selected_feature_count)==int(z.k)
    pipe=Pipeline([
        ('prep',prep(sel)),
        ('clf',LogisticRegression(
            penalty='l2',solver='lbfgs',C=0.01,fit_intercept=True,
            class_weight=None,max_iter=5000,tol=1e-5
        )),
    ])
    pipe.fit(Xd[sel],yd)
    ph=pipe.predict_proba(Xh[sel])[:,1]
    auc=float(roc_auc_score(yh,ph))
    ap=float(average_precision_score(yh,ph))
    ll=float(log_loss(yh,ph,labels=[0,1]))
    rows.append({
        'selector':z.selector,'method':z.method,'lam':float(z.lam),
        'k':int(z.k),'selected_features':z.selected_features,
        'holdout_auroc':auc,'holdout_average_precision':ap,'holdout_log_loss':ll,
        'holdout_n':len(yh),'holdout_positive_n':int(np.sum(yh==1)),'holdout_negative_n':int(np.sum(yh==0)),
    })
    for local_i,(orig_i,yy,pp) in enumerate(zip(hold,yh,ph)):
        pred_rows.append({
            'selector':z.selector,'method':z.method,'holdout_local_index':local_i,
            'original_row_index':int(orig_i),'y_true':int(yy),'p_bad_credit':float(pp),
        })

res=pd.DataFrame(rows)
# Frozen-method deltas against same-selector reference.
res['delta_auroc_vs_selector_reference']=np.nan
res['delta_ap_vs_selector_reference']=np.nan
res['delta_logloss_vs_selector_reference']=np.nan
for s in ['L1','GBM_PERM','ELASTIC_NET']:
    q=res[res.selector==s]
    b=q[q.method=='reference'].iloc[0]
    mask=res.selector==s
    res.loc[mask,'delta_auroc_vs_selector_reference']=res.loc[mask,'holdout_auroc']-float(b.holdout_auroc)
    res.loc[mask,'delta_ap_vs_selector_reference']=res.loc[mask,'holdout_average_precision']-float(b.holdout_average_precision)
    res.loc[mask,'delta_logloss_vs_selector_reference']=res.loc[mask,'holdout_log_loss']-float(b.holdout_log_loss)

res.to_csv(R/'FINAL_HOLDOUT_RESULTS.csv',index=False)
pd.DataFrame(pred_rows).to_csv(R/'FINAL_HOLDOUT_PREDICTIONS.csv',index=False)

audit={
  'evaluation_date':'2026-09-18',
  'holdout_name':'modern predeclared holdout / historically non-sealed evaluation set',
  'holdout_n':int(len(yh)),
  'holdout_class_counts':{'0':int(np.sum(yh==0)),'1':int(np.sum(yh==1))},
  'development_n':int(len(yd)),
  'final_selected_sets_file':'FINAL_SELECTED_SETS_FREEZE.csv',
  'pre_holdout_freeze_file':'FINAL_PRE_HOLDOUT_FREEZE.md',
  'common_evaluator':'L2 logistic regression, C=0.01, corrected preprocessing',
  'primary_metric':'AUROC',
  'secondary_metrics':['average_precision','log_loss'],
  'n_frozen_configurations_evaluated':int(len(res)),
  'holdout_used_for_tuning':False,
  'evaluated_all_frozen_configurations_in_one_pass':True,
}
(R/'FINAL_HOLDOUT_EVALUATION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
print(res[['selector','method','lam','k','holdout_auroc','delta_auroc_vs_selector_reference','holdout_average_precision','holdout_log_loss']].to_string(index=False))
