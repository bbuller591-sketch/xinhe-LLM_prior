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
import pandas as pd
W=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap'))
O=W/'09_FINAL_COMPARISON';O.mkdir(exist_ok=True)
rows=[]
def one(dataset,selector,k,mode,series,obs,ref,scope,source):
    v=pd.Series(series,dtype=float)
    return dict(dataset=dataset,selector=selector,k=k,mode=mode,scope=scope,n_replicates=len(v),
      observed_auroc=float(obs),reference_auroc=float(ref),observed_delta=float(obs-ref),
      null_mean_auroc=float(v.mean()),null_mean_delta=float(v.mean()-ref),
      null_sd_auroc=float(v.std(ddof=1)),null_q025_auroc=float(v.quantile(.025)),null_q975_auroc=float(v.quantile(.975)),
      empirical_p_ge_observed=float((1+(v>=obs-1e-12).sum())/(len(v)+1)),
      n_lower=int((v<obs-1e-12).sum()),n_equal=int(((v-obs).abs()<=1e-12).sum()),n_greater=int((v>obs+1e-12).sum()),
      source_file=str(source))
# Breast SIS20 sealed external
obs=pd.read_csv(W/'04_BREAST/sealed_validation/SEALED_reference_selective_RESULTS.csv')
reference=obs[(obs.method=='reference')&(obs.selector=='SIS')&(obs.k==20)].iloc[0]
selective=obs[(obs.method=='selective')&(obs.selector=='SIS')&(obs.k==20)].iloc[0]
for mode,fn in [('semantic','SEMANTIC_SHUFFLE_1000.csv'),('random','RANDOM_PROBABILITY_NULL_1000.csv')]:
    p=W/f'08_Q2_Q5_DIAGNOSTICS/BREAST_Q3_{"SEMANTIC" if mode=="semantic" else "RANDOM"}/{fn}'
    d=pd.read_csv(p)
    rows.append(one('Breast','SIS',20,mode,d.sealed_auc_frozen_support,float(selective.auroc),float(reference.auroc),'sealed external GSE25065',p))
# Credit locked holdout, exact paper configurations
h=pd.read_csv(W/'05_CREDIT_G/downstream/FINAL_HOLDOUT_RESULTS.csv')
for sel_code,selector,method in [('GBM_PERM','GBM-permutation','GBM_PERM'),('ELASTIC_NET','Elastic Net','ELASTIC_NET')]:
    reference=h[(h.selector==method)&(h.method=='reference')].iloc[0]
    selective=h[(h.selector==method)&(h.method=='selective')].iloc[0]
    for mode,sub in [('semantic','01_q3_semantic_shuffle'),('random','02_q3_random_null')]:
        p=W/f'08_Q2_Q5_DIAGNOSTICS/CREDIT_Q3/{sub}/CREDIT_G/REPLICATES_1000.csv'
        d=pd.read_csv(p);d=d[d.selector==sel_code]
        rows.append(one('CREDIT-G',selector,10,mode,d.holdout_auc_frozen_support,float(selective.holdout_auroc),float(reference.holdout_auroc),'locked holdout',p))
D=pd.DataFrame(rows);D.to_csv(O/'EXTERNAL_Q3_PVALUES_GPT.csv',index=False)
(O/'EXTERNAL_Q3_PVALUES_GPT.json').write_text(json.dumps(rows,indent=2)+'\n')
print(D.to_string(index=False))
