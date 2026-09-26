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
import json, math
import numpy as np,pandas as pd

OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/03_q4_protection'));OUT.mkdir(parents=True,exist_ok=True)

CFG=[
 dict(dataset='Renal GSE36059->GSE48581 TCMR',
      root=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR')),
      score='formal_outputs/02_downstream_dev_v3_2/CORRECTED_SCORES_FROZEN.csv',
      pair='formal_outputs/01_pre_llm_freeze_v3_2/FINAL/SELECTIVE_PAIRSET_FROZEN_V3_2.csv',
      meas='formal_outputs/measurements/v3_2/SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv',
      summary='formal_outputs/02_downstream_dev_v3_2/PRE_EXTERNAL_FREEZE_SUMMARY.json',
      gene='gene',pi='feature_i',pj='feature_j',mi='gene_i',mj='gene_j',pcol='p_e_gene_i_gt_gene_j',ccol='C_e',
      k=50,recommendation='Main',chronology='formal latest Selective selective; pre-external development freeze'),
 dict(dataset='CRC A03 GSE39582->GSE17536',
      root=Path(str(REPRO_ROOT / 'formal_experiments_20260921/A03_GSE39582_to_GSE17536_36m_recurrence')),
      score='development_outputs/downstream_dev/CORRECTED_SCORES_FROZEN.csv',
      pair='PAIRSET_U_FROZEN.csv',
      meas='formal_outputs/measurements/measurement/SHARED_LLM_PAIR_MEASUREMENTS_FROZEN.csv',
      summary='development_outputs/downstream_dev/PRE_EXTERNAL_selective_selective_no_certainty_FREEZE_SUMMARY.json',
      gene='gene_symbol',pi='feature_i',pj='feature_j',mi='feature_i',mj='feature_j',pcol='p_e_primary',ccol='C_e',
      k=100,recommendation='Supporting',chronology='latest normalized-CE chain; supporting certificate retains after-unseal chronology caveat for diagnostics added 2026-09-22'),
 dict(dataset='A05 GSE20685->GSE2990 5y distant recurrence',
      root=Path(str(REPRO_ROOT / 'formal_experiments_20260921/A05_GSE20685_to_GSE2990_5y_distant_recurrence')),
      score='development_outputs/downstream_dev/CORRECTED_SCORES_FROZEN.csv',
      pair='PAIRSET_U_FROZEN.csv',
      meas='measurements/deepseek_formal/measurement/SHARED_LLM_PAIR_MEASUREMENTS_FROZEN.csv',
      summary='development_outputs/downstream_dev/PRE_EXTERNAL_selective_selective_no_certainty_FREEZE_SUMMARY.json',
      gene='gene_symbol',pi='gene_i',pj='gene_j',mi='gene_i',mj='gene_j',pcol='p_e_primary',ccol='C_e',
      k=50,recommendation='Supplement',chronology='formal normalized-CE chain; external guided result negative')
]

def compute(cfg):
    root=cfg['root']; sc=pd.read_csv(root/cfg['score']); pairs=pd.read_csv(root/cfg['pair']); meas=pd.read_csv(root/cfg['meas']); sm=json.loads((root/cfg['summary']).read_text())
    lam=float(sm['selected_lambda']['selective']); p_dim=len(sc); k=int(cfg['k'])
    assert p_dim==int(sm.get('p',p_dim))
    # exact edge merge in frozen orientation
    keep=[cfg['mi'],cfg['mj'],cfg['pcol'],cfg['ccol']]
    mm=meas[keep].copy()
    e=pairs.merge(mm,left_on=[cfg['pi'],cfg['pj']],right_on=[cfg['mi'],cfg['mj']],validate='one_to_one')
    assert len(e)==len(pairs)
    e['w0']=e['U'].astype(float)*e[cfg['ccol']].astype(float)
    W=float(e.w0.sum()); assert W>0 and lam>0
    e['tilde_w']=p_dim*lam*e.w0/W
    # theorem-scale budget
    genes=sc[cfg['gene']].astype(str).tolist(); idx={g:i for i,g in enumerate(genes)}
    rho=np.zeros(p_dim,float)
    for r in e.itertuples():
        i=idx[str(getattr(r,cfg['pi']))]; j=idx[str(getattr(r,cfg['pj']))]; w=float(r.tilde_w)
        rho[i]+=w;rho[j]+=w
    a=sc['Reference'].to_numpy(float); s=sc['selective'].to_numpy(float);disp=np.abs(s-a)
    ratio=np.divide(disp,rho,out=np.zeros_like(disp),where=rho>0)
    tol=1e-9
    # Reference top-k uses same stable ordering encoded by corrected score table's candidate order as final tie-break.
    order=np.lexsort((sc['candidate_order'].to_numpy(int),-a))
    top=order[:k]; outside=order[k:]
    top_set=set(top.tolist())
    # all-pair protected fraction, orient by a
    prot=0; denom=0; min_all_margin=float('inf')
    for i in range(p_dim-1):
        da=a[i]-a[i+1:]; rr=rho[i]+rho[i+1:]
        nz=da!=0
        denom+=int(nz.sum())
        m=np.abs(da[nz])-rr[nz]
        prot+=int((m>0).sum())
        if m.size:min_all_margin=min(min_all_margin,float(m.min()))
    # cross-boundary margins: anchor top member must dominate outside
    cross_total=k*(p_dim-k);cross_prot=0; min_cross=float('inf');min_pos=float('inf')
    top_guaranteed=[];out_guaranteed_mask=np.ones(len(outside),dtype=bool)
    for i in top:
        m=(a[i]-a[outside])-(rho[i]+rho[outside])
        good=m>0
        cross_prot+=int(good.sum())
        if m.size:min_cross=min(min_cross,float(m.min()))
        if good.any():min_pos=min(min_pos,float(m[good].min()))
        if bool(good.all()):top_guaranteed.append(i)
        out_guaranteed_mask &= good
    outside_guaranteed=outside[out_guaranteed_mask]
    full=bool(cross_prot==cross_total)
    # corrected top-k consistency
    corr_order=np.lexsort((sc['candidate_order'].to_numpy(int),-a,-s))
    corr_top=corr_order[:k]
    feature=pd.DataFrame({
      'gene':genes,'reference_anchor_a':a,'corrected_score_s':s,'abs_displacement':disp,'rho':rho,
      'interval_lower':a-rho,'interval_upper':a+rho,'displacement_to_budget_ratio':np.where(rho>0,ratio,np.nan),
      'reference_topk':[i in top_set for i in range(p_dim)],'corrected_topk':[i in set(corr_top.tolist()) for i in range(p_dim)]
    })
    dsname=cfg['dataset'].replace(' ','_').replace('/','_').replace('>','').replace(':','')
    ddir=OUT/dsname;ddir.mkdir(parents=True,exist_ok=True)
    feature.to_csv(ddir/'FEATURE_PROTECTION_BUDGETS.csv',index=False)
    e.to_csv(ddir/'THEOREM_SCALE_EDGE_WEIGHTS.csv',index=False)
    # proof-scale consistency: sum rho = 2 * sum edge effective weights = 2 p lambda
    sum_check=float(rho.sum()); expected=2*p_dim*lam
    summary={
      'dataset':cfg['dataset'],'classification':'THEOREM_COMPATIBLE_AFTER_DETERMINISTIC_RESCALING',
      'objective':'(1/(2p))*||s-a||^2 + lambda * sum_e w0_e CE_e / sum_e w0_e',
      'rescaling':'multiply objective by p; tilde_w_e = p*lambda*w0_e/sum(w0), w0_e=U_e*C_e',
      'p':p_dim,'k':k,'selected_selective_trust_parameter':lam,'n_edges':len(e),'sum_w0':W,'sum_tilde_w':float(e.tilde_w.sum()),
      'sum_rho':sum_check,'expected_sum_rho_2p_lambda':expected,'rho_sum_check_pass':bool(np.isclose(sum_check,expected,rtol=1e-10,atol=1e-10)),
      'rho_positive_fraction':float(np.mean(rho>0)),'rho_positive_n':int((rho>0).sum()),
      'rho_median_positive':float(np.median(rho[rho>0])) if (rho>0).any() else 0.0,'rho_max':float(rho.max()),
      'max_displacement':float(disp.max()),'median_displacement_positive_budget':float(np.median(disp[rho>0])) if (rho>0).any() else 0.0,
      'max_displacement_to_budget_ratio':float(ratio[rho>0].max()) if (rho>0).any() else 0.0,
      'median_displacement_to_budget_ratio':float(np.median(ratio[rho>0])) if (rho>0).any() else 0.0,
      'bound_violation_n':int((disp>rho+tol).sum()),
      'pairwise_protected_order_fraction':float(prot/denom) if denom else 1.0,'pairwise_order_denominator':denom,
      'full_topk_certificate':full,'protected_cross_boundary_pair_fraction':float(cross_prot/cross_total),
      'protected_cross_boundary_pairs':cross_prot,'cross_boundary_pairs_total':cross_total,
      'guaranteed_unchanged_reference_topk_items':int(len(top_guaranteed)),
      'guaranteed_unchanged_outside_items':int(len(outside_guaranteed)),
      'guaranteed_unchanged_items_total':int(len(top_guaranteed)+len(outside_guaranteed)),
      'minimum_cross_boundary_certified_margin':None if min_cross==float('inf') else min_cross,
      'minimum_positive_certified_margin':None if min_pos==float('inf') else min_pos,
      'reference_vs_corrected_topk_jaccard':float(len(set(top)&set(corr_top))/len(set(top)|set(corr_top))),
      'reference_vs_corrected_topk_changed_n':int(k-len(set(top)&set(corr_top))),
      'recommendation':cfg['recommendation'],'chronology_note':cfg['chronology']
    }
    (ddir/'PROTECTION_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    return summary

rows=[compute(c) for c in CFG]
pd.DataFrame(rows).to_csv(OUT/'Q4_COMPATIBLE_EXPERIMENT_SUMMARY.csv',index=False)
(OUT/'Q4_COMPATIBLE_EXPERIMENT_SUMMARY.json').write_text(json.dumps(rows,indent=2))
print(pd.DataFrame(rows)[['dataset','selected_selective_trust_parameter','rho_positive_fraction','pairwise_protected_order_fraction','protected_cross_boundary_pair_fraction','full_topk_certificate','max_displacement_to_budget_ratio']].to_string(index=False))
