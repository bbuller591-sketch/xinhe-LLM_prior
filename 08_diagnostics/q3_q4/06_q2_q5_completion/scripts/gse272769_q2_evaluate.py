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
import pandas as pd,numpy as np,json,math,warnings,sys
warnings.filterwarnings('ignore')
from scipy.special import expit
from scipy.stats import rankdata,norm
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

PKG=Path(str(REPRO_ROOT / 'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919'))
C=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion'))
A=C/'artifacts/GSE272769_Q2_MATCHED'
SUM=A/'measurement/summaries'
SEED=2026091901
LAMS=[0,0.03,0.1,0.3,1,3,10]
K=50
sys.path.insert(0,str(PKG/'07_CODE'))
import run_strict_nested_selective_routing as route

# Consolidate calls and verify exact completion.
fs=sorted(SUM.glob('CALLS_COMPLETE_*.csv'))
if len(fs)!=32:
    raise RuntimeError(f'need 32 worker files, got {len(fs)}')
calls=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
q=pd.read_csv(A/'Q2_GLOBAL_MATCHED_QUERIES.csv')
if len(calls)!=len(q) or calls.query_id.nunique()!=len(q):
    raise RuntimeError(f'call count mismatch {len(calls)} vs {len(q)}')
calls.to_csv(A/'Q2_GLOBAL_MATCHED_ALL_CALLS.csv',index=False)

# AB/BA neutralization.
rows=[]
for (pair,arm),g in calls.groupby(['unordered_pair_id','arm']):
    if len(g)!=2 or set(g.order.astype(str))!={'AB','BA'}:
        raise RuntimeError('ABBA '+str((pair,arm)))
    ab=g[g.order=='AB'].iloc[0]
    ba=g[g.order=='BA'].iloc[0]
    lab=float(ab.logp_A-ab.logp_B)
    lba=float(ba.logp_A-ba.logp_B)
    ell=(lab-lba)/2
    p=float(expit(ell))
    H=-(p*math.log(p,2)+(1-p)*math.log(1-p,2)) if 0<p<1 else 0.
    rows.append({
        'unordered_pair_id':pair,'arm':arm,
        'gene_i':str(ab.gene_A),'gene_j':str(ab.gene_B),
        'p_i_over_j':p,
        'hard_y_i_over_j':1. if p>.5 else 0. if p<.5 else .5,
        'certainty_1_minus_H':1-H,
        'either_U':bool((str(ab.response_token)=='U') or (str(ba.response_token)=='U')),
        'order_probability_gap':abs(float(ab.pA_vs_B)-(1-float(ba.pA_vs_B)))
    })
ME=pd.DataFrame(rows)
assert len(ME)==402
ME.to_csv(A/'Q2_GLOBAL_MATCHED_PAIR_SOURCE_MEASUREMENTS.csv',index=False)

# Expand the matched slot map to the original Selective fold-occurrence structure.
mp=pd.read_csv(A/'GLOBAL_SLOT_MAPPING.csv')
feat=pd.read_csv(PKG/'01_FROZEN_DATA/features_p1500.csv')
if 'feature_index' not in feat.columns: feat['feature_index']=np.arange(len(feat),dtype=int)
name={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}
mmap={(str(r.unordered_pair_id),str(r.arm)):r for r in ME.itertuples()}
cons={f:[] for f in range(1,6)}
for r in mp.itertuples():
    mm=mmap[(str(r.global_pair),str(r.arm))]
    folds=[int(x) for x in str(r.folds).split('|')]
    ad=[float(x) for x in str(r.Adivs).split('|')]
    assert len(folds)==len(ad)
    i=name[str(r.gene_i)]
    j=name[str(r.gene_j)]
    for f,a in zip(folds,ad):
        cons[f].append({
            'i':i,'j':j,'y':float(mm.hard_y_i_over_j),
            'w':a*float(mm.certainty_1_minus_H),
            'arm':str(r.arm),'pair':str(r.global_pair),
            'Adiv':a,'certainty':float(mm.certainty_1_minus_H)
        })
audit=[{
    'outer_fold':f,'n_constraints':len(cons[f]),
    'total_weight':sum(x['w'] for x in cons[f]),
    'by_arm':json.dumps(pd.Series([x['arm'] for x in cons[f]]).value_counts().to_dict())
} for f in range(1,6)]
pd.DataFrame(audit).to_csv(A/'Q2_GLOBAL_FOLD_CONSTRAINT_AUDIT.csv',index=False)
assert [len(cons[f]) for f in range(1,6)]==[85,88,93,0,143]

X=np.load(PKG/'01_FROZEN_DATA/X_development.npy').astype(float)
y=np.load(PKG/'01_FROZEN_DATA/y_development.npy').astype(int)
spl=pd.read_csv(PKG/'01_FROZEN_DATA/OUTER_SPLITS.csv')
sa=pd.read_csv(PKG/'01_FROZEN_DATA/reference_SELECTOR_AUDIT.csv')
m0fold=pd.read_csv(PKG/'01_FROZEN_DATA/reference_FOLD_RESULTS.csv')
P=X.shape[1]

def anchor(D):
    r=rankdata(-D,method='average')
    return norm.ppf(np.clip(1-(r-.5)/len(D),1e-6,1-1e-6))

def score_inner(A,b,C,ratio,seed):
    D,ok,nnz=route.fit_sparse(A,b,float(C),float(ratio),int(seed))
    if not ok:
        raise RuntimeError('inner sparse fit failed')
    return D

def score_outer(A,b,C,ratio,seed):
    sc=StandardScaler().fit(A)
    Z=sc.transform(A)
    m=LogisticRegression(
        C=float(C),solver='saga',class_weight='balanced',
        penalty='elasticnet',l1_ratio=float(ratio),
        max_iter=5000,tol=1e-4,random_state=int(seed),
        fit_intercept=True,n_jobs=1
    ).fit(Z,b)
    if int(m.n_iter_[0])>=m.max_iter:
        raise RuntimeError('outer sparse fit failed')
    return np.abs(m.coef_[0])

def solve(sd,cc,lam):
    if lam==0 or not cc:
        return sd.copy()
    ii=np.array([x['i'] for x in cc],int)
    jj=np.array([x['j'] for x in cc],int)
    yy=np.array([x['y'] for x in cc],float)
    w=np.array([x['w'] for x in cc],float)
    W=w.sum()
    if W<=0:
        return sd.copy()
    def fg(z):
        d=z[ii]-z[jj]
        df=z-sd
        ce=np.logaddexp(0,d)-yy*d
        f=.5*np.mean(df*df)+lam*np.dot(w,ce)/W
        g=df/P
        rr=lam*(w/W)*(expit(d)-yy)
        np.add.at(g,ii,rr)
        np.add.at(g,jj,-rr)
        return float(f),g
    rr=minimize(lambda z:fg(z),sd.copy(),jac=True,method='L-BFGS-B',
                options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not rr.success:
        rr=minimize(lambda z:fg(z),rr.x,jac=True,method='L-BFGS-B',
                    options={'maxiter':4000,'maxls':200,'ftol':1e-13,'gtol':1e-8})
    if not np.all(np.isfinite(rr.x)):
        raise RuntimeError('optimizer failed')
    return rr.x

def top(z,D):
    return np.lexsort((np.arange(P),-D,-z))[:K]

def evals(tr,va,ids):
    sc=StandardScaler().fit(X[tr][:,ids])
    m=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',
        class_weight='balanced',max_iter=5000,random_state=SEED)
    m.fit(sc.transform(X[tr][:,ids]),y[tr])
    p=m.predict_proba(sc.transform(X[va][:,ids]))[:,1]
    ap=average_precision_score(y[va],p)
    apn=average_precision_score(1-y[va],1-p)
    return float(roc_auc_score(y[va],p)),float(.5*(ap+apn)),float(balanced_accuracy_score(y[va],p>=.5))

out=[]
innerrows=[]
for fold in range(1,6):
    tr=spl[(spl.fold==fold)&(spl.role=='outer_train')].sample_index.to_numpy(int)
    va=spl[(spl.fold==fold)&(spl.role=='outer_validation')].sample_index.to_numpy(int)
    r=sa[(sa.fold==fold)&(sa.selector=='ELASTICNET')].iloc[0]
    C0=float(r.chosen_C)
    ratio=float(r.chosen_l1_ratio)
    inner=list(StratifiedKFold(n_splits=4,shuffle=True,random_state=SEED+100*fold).split(X[tr],y[tr]))
    curves=[]
    for lam in LAMS:
        vv=[]
        for ii,(itr0,iva0) in enumerate(inner,1):
            itr=tr[itr0]
            iva=tr[iva0]
            D=score_inner(X[itr],y[itr],C0,ratio,SEED+100000*fold+1000*ii+100)
            ids=top(solve(anchor(D),cons[fold],lam),D)
            au,ma,ba=evals(itr,iva,ids)
            vv.append(au)
        curves.append((lam,float(np.mean(vv))))
        innerrows.append({'outer_fold':fold,'lam':lam,'inner_mean_auroc':float(np.mean(vv))})
    best=max(v for _,v in curves)
    chosen=min(e for e,v in curves if np.isclose(v,best,atol=1e-12,rtol=0))
    D=score_outer(X[tr],y[tr],C0,ratio,SEED+fold)
    ids=top(solve(anchor(D),cons[fold],chosen),D)
    au,ma,ba=evals(tr,va,ids)
    rr=m0fold[(m0fold.fold==fold)&(m0fold.selector=='ELASTICNET')&(m0fold.k==K)]
    if len(rr):
        m0au=float(rr.iloc[0].auroc)
        m0ma=float(rr.iloc[0].macro_ap)
    else:
        ids0=top(anchor(D),D)
        m0au,m0ma,_=evals(tr,va,ids0)
    out.append({
        'outer_fold':fold,'chosen_lam':chosen,'best_inner_auroc':best,
        'global_auroc':au,'global_macro_ap':ma,'global_balacc':ba,
        'reference_auroc':m0au,'reference_macro_ap':m0ma,
        'delta_auroc_vs_reference':au-m0au,
        'delta_macro_ap_vs_reference':ma-m0ma,
        'selected_indices':'|'.join(map(str,ids))
    })

pd.DataFrame(innerrows).to_csv(A/'Q2_GLOBAL_INNER_ETA_CURVES.csv',index=False)
F=pd.DataFrame(out)
F.to_csv(A/'Q2_GLOBAL_NESTED_OUTER_RESULTS.csv',index=False)

sel=pd.read_csv(PKG/'05_REAL_RESULTS/selective_OUTER_RESULTS_TOP30_TOP50.csv')
s=sel[(sel.selector=='ELASTICNET')&(sel.k==50)]
obs_selective=float(s.auroc.mean())
obs_reference=float(s.reference_auroc.mean())
obs_m3ma=float(s.macro_ap.mean())
obs_m0ma=float(s.reference_macro_ap.mean())

summary={
    'dataset':'GSE272769','selector':'ElasticNet','k':50,
    'status':'COMPLETE_NEW_LLM_MATCHED_GLOBAL',
    'query_budget_unique_pair_source':402,
    'query_calls_ab_ba':804,
    'new_llm_calls':804,
    'global_mean_auroc':float(F.global_auroc.mean()),
    'reference_mean_auroc':obs_reference,
    'selective_selective_mean_auroc':obs_selective,
    'global_delta_vs_reference':float(F.global_auroc.mean()-obs_reference),
    'selective_delta_vs_reference':obs_selective-obs_reference,
    'selective_minus_global':obs_selective-float(F.global_auroc.mean()),
    'global_mean_macro_ap':float(F.global_macro_ap.mean()),
    'reference_mean_macro_ap':obs_m0ma,
    'selective_selective_mean_macro_ap':obs_m3ma,
    'global_macro_ap_delta_vs_reference':float(F.global_macro_ap.mean()-obs_m0ma),
    'selective_macro_ap_delta_vs_reference':obs_m3ma-obs_m0ma,
    'selective_minus_global_macro_ap':obs_m3ma-float(F.global_macro_ap.mean()),
    'chosen_lam_by_outer_fold':{str(int(r.outer_fold)):float(r.chosen_lam) for r in F.itertuples()},
    'design':'source-arm matched unique measurement budget; preserves Selective fold-occurrence/actionable-weight slots but replaces pair identities with data-confusion-independent source-eligible global pairs; same normalized CE objective and strict nested lam selection',
    'chronology':'Q2 matched-global completion performed after original GSE272769 result; pair identities and cost were frozen before these new LLM calls; outer-validation never used for query selection or lam tuning'
}
(A/'Q2_MATCHED_RESULT.json').write_text(json.dumps(summary,indent=2))
pd.DataFrame([summary]).to_csv(A/'Q2_MATCHED_RESULT.csv',index=False)
print(json.dumps(summary,indent=2))
