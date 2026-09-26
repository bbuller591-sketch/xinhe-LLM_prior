

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
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
DATA=WS/'legacy_archive_snapshot/positive_results_package_20260918/credit_g/data'
TASK=WS/'01_TASK_FREEZE'; D=WS/'02_DATA_ONLY'; R=WS/'06_RESULTS'
FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[x for x in FEATURES if x not in CAT]
HP=json.loads((D/'reference_GLOBAL_HYPERPARAMETER_FREEZE.json').read_text())
GF=json.loads((R/'GAMMA_FREEZE.json').read_text())
K=10

def prep(cols=FEATURES):
    cols=list(cols); num=[c for c in NUM if c in cols]; cat=[c for c in CAT if c in cols]
    tr=[]
    if num: tr.append(('num',StandardScaler(),num))
    if cat: tr.append(('cat',Pipeline([('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),('scale',StandardScaler())]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)

def groups(p):
    names=list(p.get_feature_names_out()); g=defaultdict(list)
    for i,n in enumerate(names):
        if n.startswith('num__'): g[n[5:]].append(i)
        else:
            rest=n[5:]
            hits=[f for f in CAT if rest==f or rest.startswith(f+'_')]
            g[max(hits,key=len)].append(i)
    return g

def coef_scores(pipe):
    g=groups(pipe.named_steps['prep']); b=np.asarray(pipe.named_steps['clf'].coef_).reshape(-1)
    return {f:float(np.sqrt(np.mean(np.square(b[g[f]])))) for f in FEATURES}

def rank(scores):
    o=sorted(FEATURES,key=lambda f:(-scores[f],FEATURES.index(f)))
    return o,{f:i+1 for i,f in enumerate(o)}

X=pd.read_csv(DATA/'X.csv'); y=pd.read_csv(DATA/'y.csv')['label'].to_numpy()
dev=np.load(TASK/'modern_dev_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]

l1=Pipeline([('prep',prep()),('clf',LogisticRegression(penalty='l1',solver='saga',C=float(HP['L1']['C']),max_iter=5000,tol=1e-4,random_state=20260929))])
l1.fit(Xd,yd); s1=coef_scores(l1)

en=Pipeline([('prep',prep()),('clf',LogisticRegression(penalty='elasticnet',solver='saga',C=float(HP['ELASTIC_NET']['C']),l1_ratio=float(HP['ELASTIC_NET']['l1_ratio']),max_iter=5000,tol=1e-4,random_state=20260939))])
en.fit(Xd,yd); se=coef_scores(en)

inner=StratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=20260949)
itr,iva=next(inner.split(Xd,yd))
p=HP['GBM']
gb=Pipeline([('prep',prep()),('clf',GradientBoostingClassifier(
 learning_rate=float(p['learning_rate']),max_depth=int(p['max_depth']),min_samples_leaf=int(p['min_samples_leaf']),
 n_estimators=int(p['n_estimators']),subsample=float(p['subsample']),random_state=20260950))])
gb.fit(Xd.iloc[itr],yd[itr])
pi=permutation_importance(gb,Xd.iloc[iva],yd[iva],scoring='roc_auc',n_repeats=5,random_state=20260951,n_jobs=1)
sg={f:float(pi.importances_mean[i]) for i,f in enumerate(FEATURES)}

data_scores={'L1':s1,'GBM_PERM':sg,'ELASTIC_NET':se}
fg=pd.read_csv(R/'global_FEATURE_GUIDANCE.csv').set_index('feature')
h1=fg.bt_guidance_h.to_dict(); h2=(fg.bt_guidance_h*fg.entropy_trust_cj).to_dict()
selective=pd.read_csv(R/'selective_SELECTOR_GUIDANCE.csv')
h3={s:q.set_index('feature').h_selective.to_dict() for s,q in selective.groupby('selector')}

rows=[]; rankrows=[]
for s,scores in data_scores.items():
    order,rmap=rank(scores)
    for f in FEATURES:
        rankrows.append({'selector':s,'feature':f,'data_score':scores[f],'data_rank':rmap[f]})
    u={f:(20-rmap[f])/19 for f in FEATURES}
    methods={'reference':({f:0.0 for f in FEATURES},0.0),'global':(h1,float(GF['chosen_lam'][s]['global'])),'global_certainty':(h2,float(GF['chosen_lam'][s]['global_certainty'])),'selective':(h3[s],float(GF['chosen_lam'][s]['selective']))}
    for method,(h,lam) in methods.items():
        util={f:u[f]+lam*float(h.get(f,0)) for f in FEATURES}
        oo=sorted(FEATURES,key=lambda f:(-util[f],FEATURES.index(f)))
        rows.append({'selector':s,'method':method,'lam':lam,'k':K,'selected_features':'|'.join(oo[:K]),'selected_feature_count':K})

rows.append({'selector':'ALL20_REFERENCE','method':'ALL20','lam':0.0,'k':20,'selected_features':'|'.join(FEATURES),'selected_feature_count':20})
pd.DataFrame(rankrows).to_csv(R/'FINAL_DATA_ONLY_RANKINGS_FREEZE.csv',index=False)
sel=pd.DataFrame(rows)
sel.to_csv(R/'FINAL_SELECTED_SETS_FREEZE.csv',index=False)
print(sel.to_string(index=False))
