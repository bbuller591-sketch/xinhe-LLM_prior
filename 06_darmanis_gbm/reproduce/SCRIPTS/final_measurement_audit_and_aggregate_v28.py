

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
import json, math, pandas as pd, numpy as np
ROOT=Path(str(REPRO_ROOT / 'GBM_EXTERNAL_EVIDENCE_20260917'))
RUN=ROOT/"10_LLM_MEASUREMENT/SCALEUP_V2_8"
PP=RUN/"POSTPROCESS_V2_8"
rows=[]
for cp in RUN.glob("*/cache/*.json"):
    try:
        rec=json.load(open(cp)); s=dict(rec.get("summary",{})); raw=rec.get("raw_response",{})
        if not s.get("query_id"): continue
        ch=(raw.get("choices") or [{}])[0]
        content=((ch.get("message") or {}).get("content") or "").strip()
        s["audit_contract_ok"]=content in {"A","B","U"}
        s["audit_content"]=content
        usage=raw.get("usage",{})
        s["audit_prompt_tokens"]=usage.get("prompt_tokens") or 0
        s["audit_completion_tokens"]=usage.get("completion_tokens") or 0
        s["audit_hit_tokens"]=usage.get("prompt_cache_hit_tokens") or 0
        s["audit_miss_tokens"]=usage.get("prompt_cache_miss_tokens") or 0
        rows.append(s)
    except Exception: pass
d=pd.DataFrame(rows).drop_duplicates("query_id",keep="last")
expected=120376
status={
 "status":"PASS" if len(d)==expected else "FAIL",
 "n_unique_queries":int(len(d)),"expected":expected,
 "n_content_nonconformant":int((~d.audit_contract_ok).sum()),
 "n_model_bad":int((d.provider_model!="deepseek-flash").sum()),
 "n_fingerprint_bad":int((d.system_fingerprint!="aeb56401ca74e127821c4f9126dcb669").sum()),
 "prompt_tokens":int(d.audit_prompt_tokens.sum()),
 "completion_tokens":int(d.audit_completion_tokens.sum()),
 "cache_hit_tokens":int(d.audit_hit_tokens.sum()),
 "cache_miss_tokens":int(d.audit_miss_tokens.sum()),
 "arms":{str(k):int(v) for k,v in d.arm.value_counts().to_dict().items()},
}
(PP/"FINAL_MEASUREMENT_INTEGRITY_AUDIT_V2_8.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
print(json.dumps(status,indent=2))
# Aggregate BT source ranks over actually available sources; no missing penalty.
srcs={"IVY":"IVY_BT_global_SCORES_V2_8.csv","G116":"G116_BT_global_SCORES_V2_8.csv","G132":"G132_BT_global_SCORES_V2_8.csv"}
base=pd.DataFrame({"node":np.arange(2000,dtype=int)})
for sn,fn in srcs.items():
    z=pd.read_csv(PP/fn)
    base=base.merge(z[["node","psi_global","psi_global_certainty"]].rename(columns={"psi_global":f"psi_global_{sn}","psi_global_certainty":f"psi_global_certainty_{sn}"}),on="node",how="left")
for m in ["global","global_certainty"]:
    cols=[f"psi_{m}_{s}" for s in srcs]
    base[f"n_sources_{m}"]=base[cols].notna().sum(axis=1)
    base[f"h_{m}"]=base[cols].mean(axis=1,skipna=True).fillna(0.0)
base["n_sources_available"]=base[[f"psi_global_{s}" for s in srcs]].notna().sum(axis=1)
base.to_csv(PP/"SOURCE_AGGREGATED_H_global_V2_8.csv",index=False)
agg={
 "status":"PASS","n_genes":len(base),
 "source_count_distribution":{str(k):int(v) for k,v in base.n_sources_available.value_counts().sort_index().to_dict().items()},
 "h_global_min":float(base.h_global.min()),"h_global_max":float(base.h_global.max()),
 "h_global_certainty_min":float(base.h_global_certainty.min()),"h_global_certainty_max":float(base.h_global_certainty.max()),
 "corr_h_global":float(base[["h_global","h_global_certainty"]].corr().iloc[0,1])
}
(PP/"SOURCE_AGGREGATION_STATUS_V2_8.json").write_text(json.dumps(agg,indent=2),encoding="utf-8")
print(json.dumps(agg,indent=2))
