

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
import json, pandas as pd, numpy as np
BASE=Path(str(REPRO_ROOT / 'GBM_EXTERNAL_EVIDENCE_20260917/10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6'))
OUT=BASE/"DOWNSTREAM_V2_9"

cur=pd.read_csv(OUT/"reference_global_GAMMA_CURVES_V2_9.csv")
sel12=pd.read_csv(OUT/"reference_global_SELECTED_GAMMA_V2_9.csv")
m3cur=pd.read_csv(OUT/"selective_ETA_CURVES_V2_9.csv")
sel3=pd.read_csv(OUT/"selective_SELECTED_ETA_V2_9.csv")

# Exact baseline reproduction audits
issues=[]
for selector in ["LASSO","ELASTICNET","SIS"]:
  for k in [10,20,30]:
    b=cur[(cur.selector==selector)&(cur.method=="reference")&(cur.k==k)].iloc[0]
    for meth in ["global","global_certainty"]:
      z=cur[(cur.selector==selector)&(cur.method==meth)&(cur.k==k)&(cur.lam==0)].iloc[0]
      for metric in ["mean_auroc","mean_macro_ap","mean_ap_periphery","mean_balanced_accuracy"]:
        if abs(float(b[metric])-float(z[metric]))>1e-12:
          issues.append(f"{selector} k{k} {meth} gamma0 mismatch {metric}")
    z=m3cur[(m3cur.selector==selector)&(m3cur.k==k)&(m3cur.lam==0)].iloc[0]
    for metric in ["mean_auroc","mean_macro_ap","mean_ap_periphery","mean_balanced_accuracy"]:
      if abs(float(b[metric])-float(z[metric]))>1e-12:
        issues.append(f"{selector} k{k} selective eta0 mismatch {metric}: {b[metric]} vs {z[metric]}")

# combined selected comparison
rows=[]
for selector in ["LASSO","ELASTICNET","SIS"]:
  for k in [10,20,30]:
    b=sel12[(sel12.selector==selector)&(sel12.method=="reference")&(sel12.k==k)].iloc[0]
    for meth in ["reference","global","global_certainty"]:
      z=sel12[(sel12.selector==selector)&(sel12.method==meth)&(sel12.k==k)].iloc[0]
      rows.append({"selector":selector,"k":k,"method":meth,"tuning_parameter":"lam",
                   "chosen_value":float(z.lam),"mean_auroc":float(z.mean_auroc),
                   "delta_auroc_vs_reference":float(z.mean_auroc-b.mean_auroc),
                   "mean_macro_ap":float(z.mean_macro_ap),"delta_macro_ap_vs_reference":float(z.mean_macro_ap-b.mean_macro_ap),
                   "mean_ap_periphery":float(z.mean_ap_periphery),"delta_ap_periphery_vs_reference":float(z.mean_ap_periphery-b.mean_ap_periphery)})
    z=sel3[(sel3.selector==selector)&(sel3.k==k)].iloc[0]
    rows.append({"selector":selector,"k":k,"method":"selective","tuning_parameter":"lam",
                 "chosen_value":float(z.lam),"mean_auroc":float(z.mean_auroc),
                 "delta_auroc_vs_reference":float(z.mean_auroc-b.mean_auroc),
                 "mean_macro_ap":float(z.mean_macro_ap),"delta_macro_ap_vs_reference":float(z.mean_macro_ap-b.mean_macro_ap),
                 "mean_ap_periphery":float(z.mean_ap_periphery),"delta_ap_periphery_vs_reference":float(z.mean_ap_periphery-b.mean_ap_periphery)})
comp=pd.DataFrame(rows)
comp.to_csv(OUT/"reference_global_selective_SELECTED_COMPARISON_V2_9.csv",index=False)

# Count which nonzero borrowing selected
nz=(comp[(comp.method!="reference") & (comp.chosen_value>0)])
status={"status":"PASS" if not issues else "FAIL",
        "baseline_reproduction_issues":issues,
        "n_configs":len(comp),
        "n_guided_configs":int((comp.method!="reference").sum()),
        "n_guided_nonzero_selected":int(len(nz)),
        "max_delta_auroc_vs_reference":float(comp.delta_auroc_vs_reference.max()),
        "min_delta_auroc_vs_reference":float(comp.delta_auroc_vs_reference.min())}
(OUT/"DOWNSTREAM_INTEGRITY_AUDIT_V2_9.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
print(json.dumps(status,indent=2))
print("\nCOMPARISON")
print(comp.to_string(index=False))
