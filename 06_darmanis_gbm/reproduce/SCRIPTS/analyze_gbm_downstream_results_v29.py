

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
cur12=pd.read_csv(OUT/"reference_global_GAMMA_CURVES_V2_9.csv")
cur3=pd.read_csv(OUT/"selective_ETA_CURVES_V2_9.csv")
sel12=pd.read_csv(OUT/"reference_global_SELECTED_GAMMA_V2_9.csv")
sel3=pd.read_csv(OUT/"selective_SELECTED_ETA_V2_9.csv")
set12=pd.read_csv(OUT/"reference_global_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv")
set3=pd.read_csv(OUT/"selective_FULL_DEVELOPMENT_SELECTED_SETS_V2_9.csv")

# overlap and swaps against reference for selected full-development sets
rows=[]; swaprows=[]
for sel in ["LASSO","ELASTICNET","SIS"]:
  for k in [10,20,30]:
    b=set(set12[(set12.selector==sel)&(set12.method=="reference")&(set12.k==k)].node.astype(int))
    for meth in ["global","global_certainty"]:
      z=set12[(set12.selector==sel)&(set12.method==meth)&(set12.k==k)]
      s=set(z.node.astype(int))
      rows.append({"selector":sel,"k":k,"method":meth,"overlap_n":len(b&s),"jaccard":len(b&s)/len(b|s),"n_changed":k-len(b&s)})
      out=z[~z.node.isin(b)][["rank","node","gene_symbol"]].copy()
      for r in out.itertuples():
        swaprows.append({"selector":sel,"k":k,"method":meth,"direction":"IN","rank":int(r.rank),"node":int(r.node),"gene_symbol":r.gene_symbol})
      bout=set12[(set12.selector==sel)&(set12.method=="reference")&(set12.k==k)]
      out2=bout[~bout.node.isin(s)][["rank","node","gene_symbol"]]
      for r in out2.itertuples():
        swaprows.append({"selector":sel,"k":k,"method":meth,"direction":"OUT","rank":int(r.rank),"node":int(r.node),"gene_symbol":r.gene_symbol})
    z=set3[(set3.selector==sel)&(set3.k==k)]
    s=set(z.node.astype(int))
    rows.append({"selector":sel,"k":k,"method":"selective","overlap_n":len(b&s),"jaccard":len(b&s)/len(b|s),"n_changed":k-len(b&s)})
    for r in z[~z.node.isin(b)][["rank","node","gene_symbol"]].itertuples():
      swaprows.append({"selector":sel,"k":k,"method":"selective","direction":"IN","rank":int(r.rank),"node":int(r.node),"gene_symbol":r.gene_symbol})
    bout=set12[(set12.selector==sel)&(set12.method=="reference")&(set12.k==k)]
    for r in bout[~bout.node.isin(s)][["rank","node","gene_symbol"]].itertuples():
      swaprows.append({"selector":sel,"k":k,"method":"selective","direction":"OUT","rank":int(r.rank),"node":int(r.node),"gene_symbol":r.gene_symbol})
ov=pd.DataFrame(rows); ov.to_csv(OUT/"SELECTED_SET_OVERLAP_V2_9.csv",index=False)
sw=pd.DataFrame(swaprows); sw.to_csv(OUT/"SELECTED_SET_SWAPS_V2_9.csv",index=False)

print("=== OVERLAP ===")
print(ov.to_string(index=False))
print("\n=== selective ETA CURVES ===")
for sel in ["LASSO","ELASTICNET","SIS"]:
  for k in [10,20,30]:
    z=cur3[(cur3.selector==sel)&(cur3.k==k)].sort_values("lam")
    print("\n",sel,"k",k)
    print(z[["lam","mean_auroc","mean_macro_ap","mean_ap_periphery"]].to_string(index=False))
print("\n=== global/global_certainty NONZERO GAMMA CURVES (configs with selected lam >0) ===")
for r in sel12[(sel12.method!="reference") & (sel12.lam>0)].itertuples():
    z=cur12[(cur12.selector==r.selector)&(cur12.method==r.method)&(cur12.k==r.k)].sort_values("lam")
    print("\n",r.selector,r.method,"k",r.k,"chosen",r.lam)
    print(z[["lam","mean_auroc","mean_macro_ap","mean_ap_periphery"]].to_string(index=False))
print("\n=== selective SWAPS ===")
for sel in ["LASSO","ELASTICNET","SIS"]:
  for k in [10,20,30]:
    z=sw[(sw.selector==sel)&(sw.k==k)&(sw.method=="selective")]
    print("\n",sel,"k",k,"changed",ov[(ov.selector==sel)&(ov.k==k)&(ov.method=="selective")].n_changed.iloc[0])
    print(z.to_string(index=False))
