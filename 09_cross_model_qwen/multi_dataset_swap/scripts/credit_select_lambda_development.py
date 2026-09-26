

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
import json, math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
DATA=WS/'legacy_archive_snapshot/positive_results_package_20260918/credit_g/data'
TASK=WS/'01_TASK_FREEZE'; D=WS/'02_DATA_ONLY'; R=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/05_CREDIT_G/downstream'))
FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
CAT=["checking_status","credit_history","purpose","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
NUM=[x for x in FEATURES if x not in CAT]
LAMBDAS=[0,0.025,0.05,0.075,0.10,0.15,0.20,0.30]
K=10
SEED=20260918
C=0.01

def prep(cols):
    cols=list(cols); num=[c for c in NUM if c in cols]; cat=[c for c in CAT if c in cols]
    tr=[]
    if num: tr.append(('num',StandardScaler(),num))
    if cat: tr.append(('cat',Pipeline([('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),('scale',StandardScaler())]),cat))
    return ColumnTransformer(tr,remainder='drop',sparse_threshold=0.0)

def eval_set(sel,Xtr,ytr,Xev,yev):
    sel=list(sel)
    pipe=Pipeline([('prep',prep(sel)),('clf',LogisticRegression(penalty='l2',solver='lbfgs',C=C,max_iter=5000,tol=1e-5))])
    pipe.fit(Xtr[sel],ytr)
    return float(roc_auc_score(yev,pipe.predict_proba(Xev[sel])[:,1]))

X=pd.read_csv(DATA/'X.csv'); y=pd.read_csv(DATA/'y.csv')['label'].to_numpy()
dev=np.load(TASK/'modern_dev_indices_seed20260918.npy')
Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]
ranks=pd.read_csv(D/'reference_RESAMPLE_RANKS.csv')
m0res=pd.read_csv(D/'reference_RESAMPLE_RESULTS.csv')
fg=pd.read_csv(R/'global_FEATURE_GUIDANCE.csv').set_index('feature')
selective=pd.read_csv(R/'selective_SELECTOR_GUIDANCE.csv')

h_global=fg.bt_guidance_h.to_dict()
h_entropy=(fg.bt_guidance_h*fg.entropy_trust_cj).to_dict()
h_selective={s:q.set_index('feature').h_selective.to_dict() for s,q in selective.groupby('selector')}

outer=list(StratifiedShuffleSplit(n_splits=300,train_size=.8,test_size=.2,random_state=SEED).split(Xd,yd))
rows=[]
cache={}
baseline_check_fail=0
for rr,(itr,iev) in enumerate(outer):
    Xtr=Xd.iloc[itr].reset_index(drop=True); ytr=yd[itr]
    Xev=Xd.iloc[iev].reset_index(drop=True); yev=yd[iev]
    for s in ['L1','GBM_PERM','ELASTIC_NET']:
        q=ranks[(ranks['resample']==rr)&(ranks.selector==s)].set_index('feature')
        rmap=q['rank'].to_dict()
        u={f:(len(FEATURES)-rmap[f])/(len(FEATURES)-1) for f in FEATURES}
        methods={'global':h_global,'global_certainty':h_entropy,'selective':h_selective[s]}
        for method,h in methods.items():
            for lam in LAMBDAS:
                util={f:u[f]+lam*float(h.get(f,0.0)) for f in FEATURES}
                order=sorted(FEATURES,key=lambda f:(-util[f],FEATURES.index(f)))
                sel=tuple(sorted(order[:K],key=FEATURES.index))
                key=(rr,sel)
                if key not in cache:
                    cache[key]=eval_set(sel,Xtr,ytr,Xev,yev)
                auc=cache[key]
                rows.append({'resample':rr,'selector':s,'method':method,'lam':lam,'outer_auc':auc,'selected_features':'|'.join(sel)})
                if lam==0:
                    base=m0res[(m0res['resample']==rr)&(m0res.selector==s)&(m0res.k==K)].iloc[0]
                    if abs(float(base.outer_auc)-auc)>1e-10:
                        baseline_check_fail+=1

if baseline_check_fail:
    raise RuntimeError(f'gamma0 reproduction mismatches={baseline_check_fail}')

res=pd.DataFrame(rows)
res.to_csv(R/'GUIDED_DEVELOPMENT_GAMMA_RESULTS.csv',index=False)

summ=[]
freeze={}
rng=np.random.default_rng(SEED+77)
for s in ['L1','GBM_PERM','ELASTIC_NET']:
    freeze[s]={}
    for method in ['global','global_certainty','selective']:
        q=res[(res.selector==s)&(res.method==method)]
        stats=q.groupby('lam').outer_auc.agg(['mean','std','count']).reset_index()
        best_mean=float(stats['mean'].max())
        cand=stats[stats['mean']>=best_mean-1e-4].sort_values('lam')
        g=float(cand.iloc[0].lam)
        freeze[s][method]=g
        base=q[q.lam==0].set_index('resample').outer_auc
        chosen=q[q.lam==g].set_index('resample').outer_auc
        delta=(chosen-base).sort_index().to_numpy()
        boots=np.empty(3000)
        n=len(delta)
        for b in range(len(boots)):
            boots[b]=delta[rng.integers(0,n,n)].mean()
        ci=np.quantile(boots,[.025,.975])
        sel0=q[q.lam==0].set_index('resample').selected_features
        selg=q[q.lam==g].set_index('resample').selected_features
        jacs=[]
        changed=0
        for i in sel0.index:
            a=set(sel0.loc[i].split('|')); b=set(selg.loc[i].split('|'))
            jacs.append(len(a&b)/len(a|b))
            changed+=int(a!=b)
        for _,z in stats.iterrows():
            summ.append({'selector':s,'method':method,'lam':float(z.lam),'mean_auc':float(z['mean']),'sd_auc':float(z['std']),'n':int(z['count']),'selected_lam':float(z.lam)==g})
        # one extra diagnostic row saved separately below
        diag={
          'selector':s,'method':method,'selected_lam':g,
          'mean_delta_auc_vs_gamma0':float(delta.mean()),
          'sd_delta_auc':float(delta.std(ddof=1)),
          'bootstrap95_low':float(ci[0]),'bootstrap95_high':float(ci[1]),
          'fraction_harmed_gt_0p01':float(np.mean(delta < -0.01)),
          'mean_selected_set_jaccard_vs_gamma0':float(np.mean(jacs)),
          'fraction_resamples_selected_set_changed':float(changed/len(sel0)),
        }
        freeze[s][method+'_diagnostic']=diag

pd.DataFrame(summ).to_csv(R/'GAMMA_SELECTION_SUMMARY.csv',index=False)
freeze_obj={'lambda_grid':LAMBDAS,'tie_rule':'largest mean development AUROC; within 1e-4 choose smallest lam','chosen_lam':{s:{m:freeze[s][m] for m in ['global','global_certainty','selective']} for s in freeze},'diagnostics':[freeze[s][m+'_diagnostic'] for s in freeze for m in ['global','global_certainty','selective']],'modern_final_holdout_metrics_inspected':False}
(R/'GAMMA_FREEZE.json').write_text(json.dumps(freeze_obj,indent=2)+'\n')
print(json.dumps(freeze_obj,indent=2))
