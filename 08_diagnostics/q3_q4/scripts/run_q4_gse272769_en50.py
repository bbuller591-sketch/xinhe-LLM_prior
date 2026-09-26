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
import sys, json, os
import numpy as np, pandas as pd

PKG=Path(str(REPRO_ROOT / 'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919'))
sys.path.insert(0,str(PKG/'07_CODE'))
import run_top30_top50_nested as m

TASK='SEPSIS_GSE272769'; SELECTOR='ELASTICNET'; K=50
OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/03_q4_protection/GSE272769_EN50'))
OUT.mkdir(parents=True,exist_ok=True)

X=np.load(PKG/'01_FROZEN_DATA/X_development.npy').astype(float)
y=np.load(PKG/'01_FROZEN_DATA/y_development.npy').astype(int)
feat=pd.read_csv(PKG/'01_FROZEN_DATA/features_p1500.csv')
feat['feature_index']=np.arange(len(feat),dtype=int)
splits=pd.read_csv(PKG/'01_FROZEN_DATA/OUTER_SPLITS.csv')
sa=pd.read_csv(PKG/'01_FROZEN_DATA/reference_SELECTOR_AUDIT.csv')
lams=pd.read_csv(PKG/'05_REAL_RESULTS/selective_SELECTED_ETA_TOP30_TOP50.csv')
p=X.shape[1]

fold_rows=[]; budgets=[]; edge_rows=[]
for fold in range(1,6):
    tr=splits[(splits.fold==fold)&(splits.role=='outer_train')].sample_index.to_numpy(int)
    row=sa[(sa.fold==fold)&(sa.selector==SELECTOR)].iloc[0]
    C=float(row.chosen_C); ratio=float(row.chosen_l1_ratio)
    D,nit,conv,solver=m.selector_score(X[tr],y[tr],SELECTOR,C,ratio,m.SEED+fold,stage='outer')
    assert conv
    a=m.data_anchor(D)
    cc,audit=m.build_constraints(TASK,fold,SELECTOR,K,feat)
    lam=float(lams[(lams.outer_fold==fold)&(lams.selector==SELECTOR)&(lams.k==K)].iloc[0].chosen_lam)
    s,ok,onit,W=m.optimize_selective(a,cc,lam)
    assert ok
    rho=np.zeros(p,float)
    if lam>0 and W>0 and len(cc):
        for ei,e in enumerate(cc):
            tw=p*lam*float(e['w'])/W
            rho[int(e['i'])]+=tw; rho[int(e['j'])]+=tw
            edge_rows.append(dict(fold=fold,edge_index=ei,i=int(e['i']),j=int(e['j']),pair=e['pair'],arm=e['arm'],
                                  base_weight=float(e['w']),theorem_weight=tw,lam=lam,total_weight=W))
    disp=np.abs(s-a)
    viol=disp-rho
    ref=m.stable_topk(a,D,K)
    corr=m.stable_topk(s,D,K)
    refset=set(map(int,ref)); corrset=set(map(int,corr))
    changed=K-len(refset&corrset); jacc=len(refset&corrset)/len(refset|corrset)

    # all unordered pairs sorted by reference score
    order=np.argsort(-a,kind='mergesort')
    prot=0; denom=0
    for pos in range(p-1):
        i=int(order[pos]); js=order[pos+1:]
        gap=a[i]-a[js]
        mask=gap>0
        if mask.any():
            denom+=int(mask.sum())
            prot+=int(np.sum(gap[mask] > (rho[i]+rho[js[mask]])))

    outside=np.array([j for j in range(p) if j not in refset],int)
    cross_total=0; cross_prot=0; min_cert=np.inf
    full=True
    for i in ref:
        gaps=a[int(i)]-a[outside]
        margins=gaps-(rho[int(i)]+rho[outside])
        valid=gaps>0
        cross_total+=int(valid.sum())
        cross_prot+=int(np.sum(margins[valid]>0))
        if valid.any():
            full=full and bool(np.all(margins[valid]>0))
            min_cert=min(min_cert,float(np.min(margins[valid])))
    if cross_total==0: full=False

    for j in range(p):
        budgets.append(dict(fold=fold,feature_index=j,gene_symbol=str(feat.iloc[j].gene_symbol),
                            reference_score=float(a[j]),corrected_score=float(s[j]),
                            displacement=float(disp[j]),rho=float(rho[j]),
                            protected_budget_slack=float(rho[j]-disp[j])))

    fold_rows.append(dict(
        dataset='GSE272769',selector=SELECTOR,k=K,outer_fold=fold,lam=lam,n_constraints=len(cc),sum_base_weight=W,
        rho_positive_n=int(np.sum(rho>0)),rho_positive_fraction=float(np.mean(rho>0)),
        rho_max=float(rho.max()),max_displacement=float(disp.max()),
        max_displacement_to_budget_ratio=float(np.max(np.divide(disp,rho,out=np.zeros_like(disp),where=rho>0))) if np.any(rho>0) else 0.0,
        bound_violation_n=int(np.sum(viol>1e-8)),
        max_bound_violation=float(max(0.0,viol.max())),
        pairwise_protected_order_fraction=float(prot/denom) if denom else float('nan'),
        protected_pairs=prot,pairwise_order_denominator=denom,
        cross_boundary_protected_fraction=float(cross_prot/cross_total) if cross_total else float('nan'),
        protected_cross_boundary_pairs=cross_prot,cross_boundary_pairs_total=cross_total,
        full_topk_certificate=bool(full),topk_jaccard=float(jacc),topk_changed_n=int(changed),
        minimum_cross_boundary_certified_margin=None if not np.isfinite(min_cert) else float(min_cert)
    ))

fdf=pd.DataFrame(fold_rows)
fdf.to_csv(OUT/'FOLD_PROTECTION_SUMMARY.csv',index=False)
pd.DataFrame(budgets).to_csv(OUT/'FEATURE_PROTECTION_BUDGETS.csv',index=False)
pd.DataFrame(edge_rows).to_csv(OUT/'THEOREM_SCALE_EDGE_WEIGHTS.csv',index=False)

def agg(df):
    totalp=df.protected_pairs.sum(); totald=df.pairwise_order_denominator.sum()
    totalc=df.protected_cross_boundary_pairs.sum(); totalct=df.cross_boundary_pairs_total.sum()
    return {
      'n_folds':int(len(df)),
      'lam_values':[float(x) for x in df.lam],
      'active_folds':int((df.lam>0).sum()),
      'rho_positive_fraction_mean':float(df.rho_positive_fraction.mean()),
      'rho_positive_fraction_range':[float(df.rho_positive_fraction.min()),float(df.rho_positive_fraction.max())],
      'bound_violation_n_total':int(df.bound_violation_n.sum()),
      'pairwise_protected_order_fraction_weighted':float(totalp/totald),
      'cross_boundary_protected_fraction_weighted':float(totalc/totalct),
      'full_topk_certificate_folds':int(df.full_topk_certificate.sum()),
      'topk_jaccard_mean':float(df.topk_jaccard.mean()),
      'topk_changed_n_mean':float(df.topk_changed_n.mean()),
      'topk_changed_n_by_fold':[int(x) for x in df.topk_changed_n]
    }
summary={'all_folds':agg(fdf),'active_folds_only':agg(fdf[fdf.lam>0])}
(OUT/'PROTECTION_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print(fdf.to_string(index=False))
