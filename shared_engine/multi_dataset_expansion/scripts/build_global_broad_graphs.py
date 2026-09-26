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
import json,hashlib
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
EV=ROOT/'05_EXTERNAL_EVIDENCE/gene_level_sources'
OUT=ROOT/'14_global_BROAD_FREEZE'
OUT.mkdir(parents=True,exist_ok=True)

ARMS=[
 ('BREAST_GSE25055_GSE25065','ARM_GSE41998','BREAST_GSE41998_HER2NEG_PCR',20260919031),
 ('BREAST_GSE25055_GSE25065','ARM_GSE32646','BREAST_GSE32646_HER2NEG_PCR',20260919032),
 ('SEPSIS_GSE65682','ARM_EMTAB4451','SEPSIS_EMTAB4451_28D_MORTALITY',20260919033),
 ('SEPSIS_GSE65682','ARM_EMTAB7581','SEPSIS_EMTAB7581_VANISH_28D_MORTALITY',20260919034),
]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def connected(n,edges):
    adj=[[] for _ in range(n)]
    for a,b in edges:
        adj[a].append(b); adj[b].append(a)
    seen={0}; stack=[0]
    while stack:
        u=stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    return len(seen)==n

def gap(n,edges,degree):
    rr=[]; cc=[]
    for a,b in edges:
        rr += [a,b]; cc += [b,a]
    A=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(n,n))
    L=sparse.eye(n,format='csr')-A/float(degree)
    vals=np.sort(np.real(eigsh(L,k=3,which='SM',return_eigenvectors=False,tol=1e-7,maxiter=20000)))
    return float(vals[1]),[float(x) for x in vals]

def generate_matchings(n,degree,seed):
    if n%2: raise ValueError('This frozen construction requires even node count')
    rng=np.random.default_rng(seed)
    existing=set(); matchings=[]
    for m in range(degree):
        ok=False
        for attempt in range(100000):
            perm=rng.permutation(n)
            pairs=[]
            good=True
            for q in range(0,n,2):
                a=int(perm[q]); b=int(perm[q+1])
                e=(a,b) if a<b else (b,a)
                if e in existing:
                    good=False; break
                pairs.append(e)
            if good:
                ok=True; break
        if not ok: raise RuntimeError(f'Could not generate matching {m}')
        matchings.append(pairs); existing.update(pairs)
    return matchings

summ=[]
for task,arm,prefix,base_seed in ARMS:
    g=pd.read_csv(EV/f'{prefix}_GENE_EVIDENCE.csv')
    g=g[g.available.fillna(False).astype(bool)].sort_values('feature_index',kind='mergesort')
    nodes=g[['feature_index','gene_symbol','GeneID']].drop_duplicates('feature_index').reset_index(drop=True)
    n=len(nodes)
    if n%2: raise RuntimeError(f'ODD NODE COUNT {arm} {n}')
    final=None
    for retry in range(50):
        seed=base_seed+retry
        ms=generate_matchings(n,20,seed)
        e10=[e for M in ms[:10] for e in M]
        e20=[e for M in ms for e in M]
        if connected(n,e10) and connected(n,e20):
            final=(seed,ms,e10,e20); break
    if final is None: raise RuntimeError(f'NO CONNECTED GRAPH {arm}')
    seed,ms,e10,e20=final
    # exact degree checks
    for d,edges in [(10,e10),(20,e20)]:
        deg=np.zeros(n,int)
        for a,b in edges: deg[a]+=1; deg[b]+=1
        assert deg.min()==d and deg.max()==d
        assert len(set(edges))==len(edges)
    assert set(e10).issubset(set(e20))
    gap10,eig10=gap(n,e10,10)
    gap20,eig20=gap(n,e20,20)
    adir=OUT/task/arm; adir.mkdir(parents=True,exist_ok=True)
    nodes.to_csv(adir/'NODES.csv',index=False)
    def save_edges(edges,d):
        rows=[]
        for a,b in sorted(edges):
            rows.append({'node_index_A':a,'node_index_B':b,
                         'feature_index_A':int(nodes.iloc[a].feature_index),
                         'feature_index_B':int(nodes.iloc[b].feature_index),
                         'gene_A':str(nodes.iloc[a].gene_symbol),
                         'gene_B':str(nodes.iloc[b].gene_symbol),
                         'unordered_pair_id':'||'.join(sorted([str(nodes.iloc[a].gene_symbol),str(nodes.iloc[b].gene_symbol)]))})
        pd.DataFrame(rows).to_csv(adir/f'EDGES_D{d}.csv',index=False)
    save_edges(e10,10); save_edges(e20,20)
    manifest={
      'task':task,'arm':arm,'node_count':n,'construction':'union_of_edge_disjoint_random_perfect_matchings',
      'seed':seed,'degree_primary':20,'degree_nested_sensitivity':10,
      'd20_edges':len(e20),'d10_edges':len(e10),'d10_is_subset_of_d20':True,
      'd20_connected':True,'d10_connected':True,
      'd20_normalized_laplacian_gap':gap20,'d10_normalized_laplacian_gap':gap10,
      'd20_smallest_laplacian_eigenvalues':eig20,'d10_smallest_laplacian_eigenvalues':eig10,
      'uses_y':False,'uses_data_scores':False,'uses_llm_outputs':False,'uses_sealed_validation':False
    }
    (adir/'GRAPH_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest['nodes_sha256']=sha(adir/'NODES.csv')
    manifest['d20_edges_sha256']=sha(adir/'EDGES_D20.csv')
    manifest['d10_edges_sha256']=sha(adir/'EDGES_D10.csv')
    (adir/'GRAPH_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    summ.append(manifest)
    print(json.dumps(manifest,indent=2))

pd.DataFrame(summ).to_csv(OUT/'BROAD_GRAPH_SUMMARY.csv',index=False)
