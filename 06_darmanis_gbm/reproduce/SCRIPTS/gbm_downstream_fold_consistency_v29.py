

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
import pandas as pd, numpy as np, json
OUT=Path(str(REPRO_ROOT / 'GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6/DOWNSTREAM_V2_9'))
f12=pd.read_csv(OUT/"reference_global_FOLD_RESULTS_V2_9.csv")
s12=pd.read_csv(OUT/"reference_global_SELECTED_GAMMA_V2_9.csv")
f3=pd.read_csv(OUT/"selective_FOLD_RESULTS_V2_9.csv")
s3=pd.read_csv(OUT/"selective_SELECTED_ETA_V2_9.csv")
rows=[]
for sel in ["LASSO","ELASTICNET","SIS"]:
  for k in [10,20,30]:
    b=f12[(f12.selector==sel)&(f12.method=="reference")&(f12.k==k)][["fold","auroc","macro_ap","ap_periphery"]].sort_values("fold")
    for method in ["global","global_certainty"]:
      lam=float(s12[(s12.selector==sel)&(s12.method==method)&(s12.k==k)].lam.iloc[0])
      z=f12[(f12.selector==sel)&(f12.method==method)&(f12.k==k)&(f12.lam==lam)].sort_values("fold")
      d=z.auroc.to_numpy()-b.auroc.to_numpy()
      rows.append({"selector":sel,"k":k,"method":method,"chosen":lam,
                   "folds_auroc_better":int((d>1e-12).sum()),"folds_equal":int((abs(d)<=1e-12).sum()),
                   "folds_worse":int((d<-1e-12).sum()),"mean_delta":float(d.mean()),"min_delta":float(d.min()),"max_delta":float(d.max())})
    lam=float(s3[(s3.selector==sel)&(s3.k==k)].lam.iloc[0])
    z=f3[(f3.selector==sel)&(f3.k==k)&(f3.lam==lam)].sort_values("fold")
    d=z.auroc.to_numpy()-b.auroc.to_numpy()
    rows.append({"selector":sel,"k":k,"method":"selective","chosen":lam,
                 "folds_auroc_better":int((d>1e-12).sum()),"folds_equal":int((abs(d)<=1e-12).sum()),
                 "folds_worse":int((d<-1e-12).sum()),"mean_delta":float(d.mean()),"min_delta":float(d.min()),"max_delta":float(d.max())})
r=pd.DataFrame(rows);r.to_csv(OUT/"SELECTED_GUIDANCE_FOLD_CONSISTENCY_V2_9.csv",index=False)
print(r.to_string(index=False))
