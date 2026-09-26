import pandas as pd,numpy as np,json,hashlib
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh
B=Path('gse272769'); OUT=B/'pre_llm/broad_graphs';OUT.mkdir(parents=True,exist_ok=True)
arms=[('ARM_MARS_GSE65682','GSE65682_MARS_28D_GENE_EVIDENCE.csv',20260919101),('ARM_VANISH_EMTAB7581','EMTAB7581_28D_GENE_EVIDENCE.csv',20260919102),('ARM_GSE95233_D01','GSE95233_D01_28D_GENE_EVIDENCE.csv',20260919103)]
cand=pd.read_csv(B/'audit/candidate_universe_p1500.csv'); rank=dict(zip(cand.gene_symbol,range(1500)))
def gen(n,d,seed):
 rng=np.random.default_rng(seed);ex=set();ms=[]
 for _ in range(d):
  for __ in range(100000):
   p=rng.permutation(n); pairs=[tuple(sorted((int(p[i]),int(p[i+1])))) for i in range(0,n,2)]
   if not any(e in ex for e in pairs):break
  ms.append(pairs);ex.update(pairs)
 return ms
def gap(n,e,d):
 rr=[];cc=[]
 for a,b in e:rr += [a,b];cc += [b,a]
 A=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(n,n));L=sparse.eye(n)-A/d;v=np.sort(np.real(eigsh(L,k=3,which='SM',return_eigenvectors=False)));return float(v[1])
for arm,f,seed in arms:
 d=pd.read_csv(B/'evidence'/f); nodes=d[d.available.fillna(False).astype(bool)][['gene_symbol']].copy();nodes['feature_index']=nodes.gene_symbol.map(rank);nodes=nodes.sort_values('feature_index').reset_index(drop=True); excluded=[]
 if len(nodes)%2: excluded=[nodes.iloc[-1].gene_symbol];nodes=nodes.iloc[:-1].copy().reset_index(drop=True)
 ms=gen(len(nodes),20,seed);e10=[e for m in ms[:10] for e in m];e20=[e for m in ms for e in m]; ad=OUT/arm;ad.mkdir(exist_ok=True);nodes.to_csv(ad/'NODES.csv',index=False)
 for deg,edges in [(10,e10),(20,e20)]:
  rows=[]
  for a,b in sorted(edges):rows.append({'node_index_A':a,'node_index_B':b,'feature_index_A':int(nodes.iloc[a].feature_index),'feature_index_B':int(nodes.iloc[b].feature_index),'gene_A':nodes.iloc[a].gene_symbol,'gene_B':nodes.iloc[b].gene_symbol,'unordered_pair_id':'||'.join(sorted([nodes.iloc[a].gene_symbol,nodes.iloc[b].gene_symbol]))})
  pd.DataFrame(rows).to_csv(ad/f'EDGES_D{deg}.csv',index=False)
 man={'arm':arm,'available_before_even_adjustment':int(len(nodes)+len(excluded)),'node_count':len(nodes),'excluded_for_even_matching':excluded,'exclusion_rule':'if odd, exclude highest candidate variance-rank index from broad graph only; candidate universe and selective unchanged','degree_primary':20,'degree_sensitivity':10,'d20_edges':len(e20),'d10_edges':len(e10),'d20_gap':gap(len(nodes),e20,20),'d10_gap':gap(len(nodes),e10,10),'uses_y':False,'uses_data_scores':False,'uses_llm':False};json.dump(man,open(ad/'GRAPH_MANIFEST.json','w'),indent=2);print(json.dumps(man,indent=2))
