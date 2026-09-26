

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
import numpy as np
import pandas as pd

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
OUT=WS/"04_METHOD_FREEZE"
OUT.mkdir(parents=True,exist_ok=True)
FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
SEED=20260918
rng=np.random.default_rng(SEED)
perm=np.array(FEATURES,dtype=object)
rng.shuffle(perm)
offsets=(1,4,7)
n=len(perm)
edges=set()
for i in range(n):
    for d in offsets:
        j=(i+d)%n
        a,b=sorted((str(perm[i]),str(perm[j])),key=lambda x:FEATURES.index(x))
        edges.add((a,b))
edges=sorted(edges,key=lambda z:(FEATURES.index(z[0]),FEATURES.index(z[1])))
assert len(edges)==60

A=np.zeros((n,n),dtype=float)
idx={str(f):i for i,f in enumerate(perm)}
for a,b in edges:
    i,j=idx[a],idx[b]
    A[i,j]=A[j,i]=1
deg=A.sum(1)
L=np.diag(deg)-A
evals=np.linalg.eigvalsh(L)
# simple connectivity check from spectral gap / BFS
seen={0}; stack=[0]
while stack:
    i=stack.pop()
    for j in np.flatnonzero(A[i]):
        j=int(j)
        if j not in seen:
            seen.add(j); stack.append(j)
connected=len(seen)==n
assert connected and np.all(deg==6)

rows=[]
for q,(a,b) in enumerate(edges,1):
    rows.append({
        "pair_id":f"BROAD_{q:03d}",
        "feature_a":a,
        "feature_b":b,
        "unordered_edge":True,
        "query_order_protocol":"AB_and_BA",
        "graph_source":"predeclared_reference_independent_seeded_6_regular_circulant",
        "construction_seed":SEED,
    })
pd.DataFrame(rows).to_csv(OUT/"global_BROAD_GRAPH_FREEZE.csv",index=False)

audit={
    "n_features":n,
    "degree":6,
    "n_unordered_edges":len(edges),
    "ordered_renderings_if_AB_BA":2*len(edges),
    "seed":SEED,
    "shuffled_feature_order":[str(x) for x in perm],
    "circulant_offsets":list(offsets),
    "connected":connected,
    "laplacian_eigenvalues":[float(x) for x in evals],
    "algebraic_connectivity_lambda2":float(evals[1]),
    "reference_independent":True,
    "legacy_exact_graph_claimed":False,
    "reason":"Historical graph was not exactly recoverable; modern broad graph is explicitly new, budget-controlled, connected, regular, and frozen independently of reference confusion."
}
(OUT/"BROAD_GRAPH_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps({k:audit[k] for k in ["n_features","degree","n_unordered_edges","ordered_renderings_if_AB_BA","connected","algebraic_connectivity_lambda2"]},indent=2))
