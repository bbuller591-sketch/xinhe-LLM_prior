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
import json,warnings,math
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

R=Path(str(REPRO_ROOT / 'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919'))
O=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion/artifacts/BREAST_Q4_FORMAL'))
O.mkdir(parents=True,exist_ok=True)
TASK='BREAST_GSE25055_GSE25065'; SELECTORS=['LASSO','ELASTICNET','SIS']; KGRID=[10,20]; P=2000
SRC=R/'DATA/FROZEN'/TASK
X=np.load(SRC/'X_development.npy').astype(float); y=np.load(SRC/'y_development.npy').astype(int)
feat=pd.read_csv(SRC/'features_p2000.csv'); assert len(feat)==P
pars=pd.read_csv(R/'METHOD_ARTIFACTS/selective_FINAL/FINAL_SELECTOR_PARAMS.csv')
lams=pd.read_csv(R/'METHOD_ARTIFACTS/selective_FINAL/FINAL_SELECTED_ETA.csv')
fc=pd.read_csv(R/'METHOD_ARTIFACTS/selective_FINAL/FINAL_CROSSFITTED_selective_CONSTRAINTS.csv')
actual=pd.read_csv(R/'METHOD_ARTIFACTS/selective_FINAL/FINAL_selective_SELECTED_FEATURES.csv')
name_to_idx={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}

def anchor(D):
    r=rankdata(-D,method='average'); return norm.ppf(np.clip(1-(r-.5)/len(D),1e-6,1-1e-6))
def sis(A,b):
    bc=b-b.mean();ac=A-A.mean(0);den=np.sqrt((ac*ac).sum(0)*(bc*bc).sum())
    return np.nan_to_num(np.abs((ac*bc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0.)
def sparse(A,b,C,ratio,seed=2026091901):
    sc=StandardScaler().fit(A);Z=sc.transform(A);pen='l1' if float(ratio)==1 else 'elasticnet'
    m=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=pen,l1_ratio=None if pen=='l1' else float(ratio),max_iter=5000,tol=1e-4,random_state=seed,n_jobs=1).fit(Z,b)
    if int(m.n_iter_[0])>=m.max_iter: raise RuntimeError('SAGA_NONCONVERGENCE')
    return np.abs(m.coef_[0])
def stable_top(z,D,k): return np.lexsort((np.arange(P),-D,-z))[:k]
def solve(a,rows,lam):
    if lam==0 or not rows:return a.copy()
    ii=np.array([r[0] for r in rows],int);jj=np.array([r[1] for r in rows],int);yy=np.array([r[2] for r in rows],float);w=np.array([r[3] for r in rows],float)
    W=float(w.sum()); act=np.unique(np.r_[ii,jj]);pos={v:q for q,v in enumerate(act)};li=np.array([pos[v] for v in ii]);lj=np.array([pos[v] for v in jj]);base=a[act].copy()
    def fg(v):
        d=v[li]-v[lj];df=v-base;ce=np.logaddexp(0,d)-yy*d
        f=.5*np.dot(df,df)/P+lam*np.dot(w,ce)/W
        g=df/P;rr=lam*(w/W)*(expit(d)-yy);np.add.at(g,li,rr);np.add.at(g,lj,-rr)
        return float(f),g
    rr=minimize(lambda v:fg(v)[0],base,jac=lambda v:fg(v)[1],method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
    if not rr.success:
        rr=minimize(lambda v:fg(v)[0],rr.x,jac=lambda v:fg(v)[1],method='L-BFGS-B',options={'maxiter':5000,'maxls':200,'ftol':1e-13,'gtol':1e-8})
    if not np.all(np.isfinite(rr.x)):raise RuntimeError('OPT_FAIL')
    z=a.copy();z[act]=rr.x;return z

rows=[]
for sel in SELECTORS:
    pr=pars[pars.selector==sel].iloc[0]
    D=sis(X,y) if sel=='SIS' else sparse(X,y,pr.C,pr.l1_ratio)
    a=anchor(D)
    for k in KGRID:
        lam=float(lams[(lams.selector==sel)&(lams.k==k)].iloc[0].chosen_lam)
        g=fc[(fc.selector==sel)&(fc.k==k)].copy()
        cons=[]
        for rr in g.itertuples():
            pair=str(rr.unordered_pair_id); aa,bb=pair.split('||',1)
            i,j=name_to_idx[aa],name_to_idx[bb]
            # file orientation gene_i/gene_j is canonical in this package; orient y to pair aa over bb
            yy=float(rr.y_i_over_j)
            if str(rr.gene_i)==bb and str(rr.gene_j)==aa: yy=1-yy
            elif not (str(rr.gene_i)==aa and str(rr.gene_j)==bb): raise RuntimeError('ORIENT')
            cons.append((i,j,yy,float(rr.weight)))
        z=solve(a,cons,lam)
        top0=set(stable_top(a,D,k).tolist()); top1=set(stable_top(z,D,k).tolist())
        # theorem-scale: objective .5/p ||z-a||^2 + lam * sum w CE/sum w
        W=sum(x[3] for x in cons)
        rho=np.zeros(P)
        if lam>0 and W>0:
            for i,j,yy,w in cons:
                tw=P*lam*w/W
                rho[i]+=tw;rho[j]+=tw
        disp=np.abs(z-a); ratio=np.divide(disp,rho,out=np.zeros_like(disp),where=rho>0)
        prot=den=0
        for i in range(P-1):
            da=np.abs(a[i]-a[i+1:]); rr=rho[i]+rho[i+1:]; nz=da>0
            den+=int(nz.sum()); prot+=int(((da[nz]-rr[nz])>0).sum())
        order0=stable_top(a,D,P); top=order0[:k];outside=order0[k:]
        cross=0;total=k*(P-k)
        for i in top:
            cross+=int((((a[i]-a[outside])-(rho[i]+rho[outside]))>0).sum())
        act_actual=actual[(actual.selector==sel)&(actual.k==k)].sort_values('rank').feature_index.astype(int).tolist()
        reproduced=stable_top(z,D,k).tolist()
        rec=dict(dataset='Breast GSE25055->GSE25065',selector=sel,k=k,lam=lam,n_constraints=len(cons),
                 rho_positive_n=int((rho>0).sum()),rho_positive_fraction=float(np.mean(rho>0)),
                 bound_violation_n=int((disp>rho+1e-9).sum()),
                 max_displacement_to_budget_ratio=float(ratio[rho>0].max()) if (rho>0).any() else 0.,
                 pairwise_protected_order_fraction=float(prot/den),
                 cross_boundary_protected_fraction=float(cross/total),
                 full_topk_certificate=bool(cross==total),
                 topk_jaccard=float(len(top0&top1)/len(top0|top1)),topk_changed_n=int(k-len(top0&top1)),
                 rho_sum_check=float(rho.sum()),expected_rho_sum=float(2*P*lam),
                 rho_sum_check_pass=bool(np.isclose(rho.sum(),2*P*lam,rtol=1e-10,atol=1e-9)),
                 frozen_support_reproduction_pass=bool(reproduced==act_actual),
                 objective_compatibility='EXACT_NORMALIZED_CE')
        rows.append(rec)
        print(rec,flush=True)
pd.DataFrame(rows).to_csv(O/'Q4_FORMAL_RESULTS.csv',index=False)
(O/'Q4_FORMAL_RESULTS.json').write_text(json.dumps(rows,indent=2))
