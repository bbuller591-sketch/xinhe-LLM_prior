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
import pandas as pd, numpy as np, json, math, hashlib
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
M=ROOT/'formal_outputs/measurements/v3_2'
d=pd.read_csv(M/'FORMAL_DEEPSEEK_V3_2.csv')
assert len(d)==400 and d.query_id.nunique()==400 and d.semantic_pair_id.nunique()==200
assert set(d.order)=={'AB','BA'} and not d.error.notna().any()
rows=[]
for pid,g in d.groupby('semantic_pair_id',sort=False):
    assert len(g)==2 and set(g.order)=={'AB','BA'}
    ab=g[g.order=='AB'].iloc[0];ba=g[g.order=='BA'].iloc[0]
    assert ab.gene_A==ba.gene_B and ab.gene_B==ba.gene_A
    assert ab.gene_i==ba.gene_i and ab.gene_j==ba.gene_j
    p_ab_A=float(ab.p_presented_A_cond_AB)
    p_ba_A=float(ba.p_presented_A_cond_AB)
    p_ba_B=1.0-p_ba_A
    # FROZEN KIDNEY FORMULA: simple probability average, not logit neutralization.
    p=0.5*(p_ab_A+p_ba_B)
    if p<=0 or p>=1:H=0.0
    else:H=-(p*math.log(p)+(1-p)*math.log(1-p))
    C=1.0-H/math.log(2.0)
    sem_ab=ab.gene_A if ab.first_token=='A' else ab.gene_B
    sem_ba=ba.gene_A if ba.first_token=='A' else ba.gene_B
    rows.append({
      'semantic_pair_id':pid,'arm':ab.arm,'arm_order':int(ab.arm_order),
      'gene_i':ab.gene_i,'gene_j':ab.gene_j,
      'AB_query_id':ab.query_id,'BA_query_id':ba.query_id,
      'AB_token':ab.first_token,'BA_token':ba.first_token,
      'AB_semantic_choice':sem_ab,'BA_semantic_choice':sem_ba,'semantic_choice_consistent':sem_ab==sem_ba,
      'p_AB_A':p_ab_A,'p_BA_A':p_ba_A,'p_BA_B':p_ba_B,
      'p_e_gene_i_gt_gene_j':p,'binary_entropy_nats':H,'C_e':C,
      'aggregation_formula':'0.5*(p_AB(A)+p_BA(B))'
    })
a=pd.DataFrame(rows).sort_values(['arm','arm_order'])
assert len(a)==200
a.to_csv(M/'FORMAL_PAIR_MEASUREMENTS_SIMPLE_ABBA_V3_2.csv',index=False)
sel=a[a.arm=='SELECTIVE'].copy();glob=a[a.arm=='GLOBAL'].copy()
assert len(sel)==100 and len(glob)==100
sel.to_csv(M/'SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv',index=False)
glob.to_csv(M/'GLOBAL_MEASUREMENTS_V3_2.csv',index=False)
summary={
 'status':'PASS','aggregation':'p_e=0.5*(p_AB(A)+p_BA(B)); C_e=1-H_binary(p_e)/ln2',
 'explicitly_not_used':'order-neutralized logit aggregation',
 'n_pairs':200,'n_selective':100,'n_global':100,
 'semantic_consistency_rate':float(a.semantic_choice_consistent.mean()),
 'mean_C_e':float(a.C_e.mean()),'median_C_e':float(a.C_e.median()),
 'mean_p_e':float(a.p_e_gene_i_gt_gene_j.mean()),
 'selective_sha256':hashlib.sha256((M/'SELECTIVE_SHARED_selective_selective_no_certainty_MEASUREMENTS_V3_2.csv').read_bytes()).hexdigest(),
 'global_sha256':hashlib.sha256((M/'GLOBAL_MEASUREMENTS_V3_2.csv').read_bytes()).hexdigest(),
 'selective_m4_requery_required':False
}
(M/'FORMAL_PAIR_MEASUREMENT_QA_V3_2.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
