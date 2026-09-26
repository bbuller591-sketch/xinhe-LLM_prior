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
import json
import numpy as np, pandas as pd
from scipy.stats import rankdata,norm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(str(REPRO_ROOT / 'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919'))
TASK='BREAST_GSE25055_GSE25065'
OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/03_q4_protection/BREAST_GSE25055_GSE25065'))
OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'DATA/FROZEN'/TASK
reference=ROOT/'METHOD_ARTIFACTS/reference'
FINAL=ROOT/'METHOD_ARTIFACTS/selective_FINAL'
MEASP=ROOT/'MEASUREMENTS/selective/selective_PAIR_SOURCE_MEASUREMENTS.csv'

X=np.load(SRC/'X_development.npy').astype(float)
y=np.load(SRC/'y_development.npy').astype(int)
feat=pd.read_csv(SRC/'features_p2000.csv')
finalpars=pd.read_csv(FINAL/'FINAL_SELECTOR_PARAMS.csv')
actual_lam=pd.read_csv(FINAL/'FINAL_SELECTED_ETA.csv')
fc=pd.read_csv(FINAL/'FINAL_CROSSFITTED_selective_CONSTRAINTS.csv')
meas=pd.read_csv(MEASP); meas=meas[meas.task==TASK].copy()
p=X.shape[1]

def data_anchor(D):
    r=rankdata(-D,method='average'); u=1-(r-0.5)/len(D)
    return norm.ppf(np.clip(u,1e-6,1-1e-6))
def sis_score(A,b):
    bc=b-b.mean(); ac=A-A.mean(axis=0)
    den=np.sqrt((ac*ac).sum(axis=0)*(bc*bc).sum())
    return np.nan_to_num(np.abs((ac*bc[:,None]).sum(axis=0)/np.where(den==0,np.nan,den)),nan=0.0)
def sparse_score(A,b,C,ratio,seed):
    sc=StandardScaler().fit(A); Z=sc.transform(A)
    pen='l1' if float(ratio)==1.0 else 'elasticnet'
    mod=LogisticRegression(C=float(C),solver='saga',class_weight='balanced',penalty=pen,
        l1_ratio=None if pen=='l1' else float(ratio),max_iter=5000,tol=1e-4,
        random_state=int(seed),fit_intercept=True,n_jobs=1)
    mod.fit(Z,b)
    assert int(mod.n_iter_[0]) < mod.max_iter
    return np.abs(mod.coef_[0])
def stable_topk(z,D,k):
    return np.lexsort((np.arange(len(z)),-D,-z))[:k]

# canonical bundle by pair/source
arms=sorted(meas.arm.astype(str).unique())
bundle={}
for pair,g in meas.groupby('unordered_pair_id'):
    a,b=str(pair).split('||',1); bundle[str(pair)]={}
    for r in g.itertuples():
        if str(r.gene_i)==a and str(r.gene_j)==b: yy=float(r.hard_y_i_over_j)
        elif str(r.gene_i)==b and str(r.gene_j)==a: yy=1.0-float(r.hard_y_i_over_j)
        else: raise RuntimeError('orientation')
        bundle[str(pair)][str(r.arm)]=(yy,float(r.certainty_1_minus_H))
name_to_idx={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}

# final geometry
final_geom={}
for (sel,k),g in fc.groupby(['selector','k']):
    q=g.groupby(['unordered_pair_id','gene_i','gene_j'],as_index=False).A_CF.first()
    rows=[]
    for r in q.itertuples():
        a,b=str(r.unordered_pair_id).split('||',1)
        rows.append((str(r.unordered_pair_id),name_to_idx[a],name_to_idx[b],float(r.A_CF)))
    final_geom[(str(sel),int(k))]=rows

def make_constraints(geom):
    out=[]
    for pair,i,j,A in geom:
        for arm in arms:
            yy,c=bundle[pair][arm]
            out.append((i,j,yy,A*c/2.0,pair,arm))
    return out

def optimize(sD,constraints,lam):
    if lam==0 or not constraints: return sD.copy()
    ii=np.array([r[0] for r in constraints],int); jj=np.array([r[1] for r in constraints],int)
    yy=np.array([r[2] for r in constraints],float); w=np.array([r[3] for r in constraints],float)
    W=float(w.sum())
    if W<=0:return sD.copy()
    active=np.unique(np.r_[ii,jj]); amap={int(v):q for q,v in enumerate(active)}
    ia=np.array([amap[int(v)] for v in ii],int); ja=np.array([amap[int(v)] for v in jj],int)
    s=sD[active].copy()
    def fg(x):
        dz=x[ia]-x[ja]; ce=np.logaddexp(0,dz)-yy*dz
        f=0.5*np.sum((x-s)**2)/p+lam*np.dot(w,ce)/W
        grad=(x-s)/p; rr=lam*(w/W)*(expit(dz)-yy)
        np.add.at(grad,ia,rr); np.add.at(grad,ja,-rr)
        return float(f),grad
    res=minimize(lambda z:fg(z),s.copy(),jac=True,method='L-BFGS-B',options={'maxiter':5000,'ftol':1e-13,'gtol':1e-8})
    if not res.success: raise RuntimeError(res.message)
    z=sD.copy(); z[active]=res.x
    return z

rows=[]; budgets=[]; edges=[]
for sel,k in [('LASSO',10),('ELASTICNET',10),('SIS',10),('SIS',20)]:
    rr=finalpars[finalpars.selector==sel].iloc[0]
    D=sis_score(X,y) if sel=='SIS' else sparse_score(X,y,float(rr.C),float(rr.l1_ratio),2026091901)
    a=data_anchor(D)
    lam=float(actual_lam[(actual_lam.selector==sel)&(actual_lam.k==k)].iloc[0].chosen_lam)
    cc=make_constraints(final_geom[(sel,k)])
    s=optimize(a,cc,lam)
    W=sum(x[3] for x in cc)
    rho=np.zeros(p,float)
    for ei,e in enumerate(cc):
        tw=p*lam*e[3]/W if lam>0 and W>0 else 0.0
        rho[e[0]]+=tw; rho[e[1]]+=tw
        edges.append(dict(selector=sel,k=k,edge_index=ei,i=e[0],j=e[1],pair=e[4],arm=e[5],base_weight=e[3],theorem_weight=tw,lam=lam,total_weight=W))
    disp=np.abs(s-a); viol=disp-rho
    ref=stable_topk(a,D,k); corr=stable_topk(s,D,k)
    refset=set(map(int,ref)); corrset=set(map(int,corr))
    changed=k-len(refset&corrset); jacc=len(refset&corrset)/len(refset|corrset)
    order=np.argsort(-a,kind='mergesort'); prot=0; denom=0
    for pos in range(p-1):
        i=int(order[pos]); js=order[pos+1:]
        gap=a[i]-a[js]; mask=gap>0
        denom+=int(mask.sum()); prot+=int(np.sum(gap[mask]>(rho[i]+rho[js[mask]])))
    outside=np.array([j for j in range(p) if j not in refset],int)
    cross_total=0; cross_prot=0; full=True
    for i in ref:
        gaps=a[int(i)]-a[outside]; valid=gaps>0
        margins=gaps-(rho[int(i)]+rho[outside])
        cross_total+=int(valid.sum()); cross_prot+=int(np.sum(margins[valid]>0))
        if valid.any(): full=full and bool(np.all(margins[valid]>0))
    for j in range(p):
        budgets.append(dict(selector=sel,k=k,feature_index=j,gene_symbol=str(feat.iloc[j].gene_symbol),
          reference_score=float(a[j]),corrected_score=float(s[j]),displacement=float(disp[j]),rho=float(rho[j])))
    rows.append(dict(dataset='Breast GSE25055->GSE25065',selector=sel,k=k,lam=lam,n_constraints=len(cc),
      rho_positive_n=int(np.sum(rho>0)),rho_positive_fraction=float(np.mean(rho>0)),
      bound_violation_n=int(np.sum(viol>1e-8)),max_bound_violation=float(max(0,viol.max())),
      max_displacement_to_budget_ratio=float(np.max(np.divide(disp,rho,out=np.zeros_like(disp),where=rho>0))) if np.any(rho>0) else 0,
      pairwise_protected_order_fraction=float(prot/denom),protected_pairs=prot,pairwise_order_denominator=denom,
      cross_boundary_protected_fraction=float(cross_prot/cross_total),protected_cross_boundary_pairs=cross_prot,cross_boundary_pairs_total=cross_total,
      full_topk_certificate=bool(full),topk_jaccard=float(jacc),topk_changed_n=int(changed)))

df=pd.DataFrame(rows)
df.to_csv(OUT/'PROTECTION_SUMMARY.csv',index=False)
pd.DataFrame(budgets).to_csv(OUT/'FEATURE_PROTECTION_BUDGETS.csv',index=False)
pd.DataFrame(edges).to_csv(OUT/'THEOREM_SCALE_EDGE_WEIGHTS.csv',index=False)
summary={
 'configurations':rows,
 'weighted_pairwise_protected_order_fraction':float(df.protected_pairs.sum()/df.pairwise_order_denominator.sum()),
 'weighted_cross_boundary_protected_fraction':float(df.protected_cross_boundary_pairs.sum()/df.cross_boundary_pairs_total.sum()),
 'bound_violation_n_total':int(df.bound_violation_n.sum()),
 'mean_rho_positive_fraction':float(df.rho_positive_fraction.mean()),
 'mean_topk_jaccard':float(df.topk_jaccard.mean())
}
(OUT/'PROTECTION_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print(df.to_string(index=False))
