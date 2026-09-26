

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
import json, math, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

ROOT=Path(str(REPRO_ROOT / 'GBM_EXTERNAL_EVIDENCE_20260917'))
RUN=ROOT/"10_LLM_MEASUREMENT/SCALEUP_V2_8"
BASE=ROOT/"10_LLM_MEASUREMENT/DUAL_METHOD_DESIGN_V2_6"
OUT=RUN/"POSTPROCESS_V2_8"; OUT.mkdir(exist_ok=True)
ARM="ARM_IVY"
prefix="BROAD_global_ARM_IVY_V2_8__C"

def parse_raw(cp):
    rec=json.load(open(cp)); s=rec["summary"]; raw=rec["raw_response"]
    ch=raw["choices"][0]
    pos=next((x for x in ((ch.get("logprobs") or {}).get("content") or []) if str(x.get("token","")).strip()),None)
    if pos is None:return None
    pools={k:[] for k in ["A","B","U"]}
    for x in pos.get("top_logprobs") or []:
        t=str(x.get("token","")).strip()
        if t in pools:pools[t].append(float(x["logprob"]))
    ft=str(pos.get("token","")).strip()
    if ft in pools and pos.get("logprob") is not None:pools[ft].append(float(pos["logprob"]))
    def lse(v):
        if not v:return None
        m=max(v);return m+math.log(sum(math.exp(x-m) for x in v))
    L={k:lse(v) for k,v in pools.items()}
    content=(ch["message"].get("content") or "").strip()
    return {"query_id":s["query_id"],"pair":s["unordered_pair_id"],"arm":s["arm"],"order":s["order"],
            "node_A":int(s["node_A"]),"node_B":int(s["node_B"]),"gene_A":s["gene_A"],"gene_B":s["gene_B"],
            "logp_A":L["A"],"logp_B":L["B"],"logp_U":L["U"],"sampled_content":content,
            "content_contract_ok":content in {"A","B","U"}}
rows=[]
for d in sorted(RUN.glob(prefix+"*")):
    for cp in (d/"cache").glob("*.json"):
        x=parse_raw(cp)
        if x:rows.append(x)
calls=pd.DataFrame(rows).drop_duplicates("query_id")
assert len(calls)==38860,(len(calls),calls.order.value_counts().to_dict())
assert not (calls.logp_A.isna()|calls.logp_B.isna()).any()
calls.to_csv(OUT/"IVY_BROAD_CALLS_REPARSED_V2_8.csv",index=False)

pairs=[]
for pair,g in calls.groupby("pair"):
    assert len(g)==2 and set(g.order)=={"AB","BA"}
    ab=g[g.order=="AB"].iloc[0];ba=g[g.order=="BA"].iloc[0]
    # canonical semantic nodes are AB orientation node_A/node_B
    lab=ab.logp_A-ab.logp_B
    lba=ba.logp_A-ba.logp_B
    ell=(lab-lba)/2; order=(lab+lba)/2
    p=expit(ell)
    H=-(p*math.log(p,2)+(1-p)*math.log(1-p,2)) if 0<p<1 else 0
    pairs.append({"pair":pair,"i":ab.node_A,"j":ab.node_B,"gene_i":ab.gene_A,"gene_j":ab.gene_B,
                  "ell":ell,"p_i":p,"H":H,"certainty":1-H,"order_component":order,
                  "hard_y":1 if p>0.5 else (0 if p<0.5 else np.nan),
                  "AB_contract_ok":ab.content_contract_ok,"BA_contract_ok":ba.content_contract_ok})
P=pd.DataFrame(pairs)
assert len(P)==19430
P.to_csv(OUT/"IVY_BROAD_PAIR_NEUTRALIZED_V2_8.csv",index=False)

nodes=sorted(set(P.i)|set(P.j)); idx={u:k for k,u in enumerate(nodes)}; n=len(nodes)
ii=P.i.map(idx).to_numpy(); jj=P.j.map(idx).to_numpy()
y=P.hard_y.to_numpy(float)
mask=~np.isnan(y); ii=ii[mask];jj=jj[mask];y=y[mask]
c=P.loc[mask,"certainty"].to_numpy(float)
lam=1e-3

def fit(w):
    def fg(x):
        z=x[ii]-x[jj]; p=expit(z)
        ce=np.logaddexp(0,z)-y*z
        f=float(np.dot(w,ce)+0.5*lam*np.dot(x,x))
        r=w*(p-y)
        g=np.zeros_like(x)
        np.add.at(g,ii,r);np.add.at(g,jj,-r)
        g += lam*x
        return f,g
    res=minimize(lambda x:fg(x),np.zeros(n),jac=True,method="L-BFGS-B",options={"maxiter":1000,"ftol":1e-12,"gtol":1e-8})
    x=res.x-res.x.mean()
    # descending average rank -> bounded +1 best, -1 worst
    r=rankdata(-x,method="average")
    psi=2*(1-(r-1)/(n-1))-1
    return res,x,psi

r1,g1,p1=fit(np.ones(len(y)))
r2,g2,p2=fit(c)
M=pd.DataFrame({"node":nodes,"g_global":g1,"psi_global":p1,"g_global_certainty":g2,"psi_global_certainty":p2})
M.to_csv(OUT/"IVY_BT_global_SCORES_V2_8.csv",index=False)
status={"status":"PASS_COMPLETED_SOURCE_IVY","n_calls":len(calls),"n_pairs":len(P),"n_nodes":n,
        "global_opt_success":bool(r1.success),"global_fun":float(r1.fun),"global_nit":int(r1.nit),
        "global_certainty_opt_success":bool(r2.success),"global_certainty_fun":float(r2.fun),"global_certainty_nit":int(r2.nit),
        "mean_H":float(P.H.mean()),"median_H":float(P.H.median()),
        "mean_abs_order":float(P.order_component.abs().mean()),
        "content_nonconformant_calls":int((~calls.content_contract_ok).sum()),
        "corr_global_psi":float(np.corrcoef(p1,p2)[0,1])}
(OUT/"IVY_POSTPROCESS_STATUS_V2_8.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
print(json.dumps(status,indent=2))

