import gzip,csv,re,json,collections,numpy as np,pandas as pd
from pathlib import Path
base=Path('gse272769'); matrix=base/'raw/GSE272769_series_matrix.txt.gz'; soft=base/'raw/platform/GPL17692_family.soft.txt'; out=base/'audit'; out.mkdir(exist_ok=True)
# expression
ids=[]; vals=[]; samples=None; begin=False
with gzip.open(matrix,'rt',errors='replace') as f:
    for line in f:
        if line.startswith('!series_matrix_table_begin'): begin=True; continue
        if line.startswith('!series_matrix_table_end'): break
        if not begin: continue
        r=next(csv.reader([line],delimiter='\t'))
        if samples is None: samples=[x.strip('"') for x in r[1:]]; continue
        ids.append(r[0].strip('"')); vals.append([float(x) if x not in ('','null','NA') else np.nan for x in r[1:]])
X=np.asarray(vals,dtype=np.float32) # features x samples
# annotation: only platform table, parse symbol as 2nd // field in each assignment component
ann={}; in_tab=False
with open(soft,'r',errors='replace') as f:
    header=None
    for line in f:
        line=line.rstrip('\r\n')
        if line=='!platform_table_begin': in_tab=True; continue
        if line=='!platform_table_end': break
        if not in_tab: continue
        r=line.split('\t')
        if header is None: header=r; idx={x:i for i,x in enumerate(header)}; continue
        if len(r)<=max(idx['ID'],idx['gene_assignment']): continue
        syms=[]
        for a in r[idx['gene_assignment']].split(' /// '):
            parts=a.split(' // ')
            if len(parts)>=2:
                s=parts[1].strip()
                if s and s!='---': syms.append(s)
        syms=list(dict.fromkeys(syms)); ann[r[idx['ID']]]=syms
# unambiguous gene symbols only; if several probesets map to same gene, representative=max X variance (X-only)
var=np.nanvar(X,axis=1); records=[]
for i,pid in enumerate(ids):
    syms=ann.get(pid,[])
    if len(syms)==1: records.append((syms[0],pid,i,float(var[i])))
by=collections.defaultdict(list)
for rec in records: by[rec[0]].append(rec)
chosen=[]
for sym,rr in by.items(): chosen.append(max(rr,key=lambda z:z[3]))
chosen.sort(key=lambda z:(-z[3],z[0]))
# mortality from titles
with gzip.open(matrix,'rt',errors='replace') as f:
    for line in f:
        if line.startswith('!Sample_title'):
            titles=[x.strip('"') for x in next(csv.reader([line],delimiter='\t'))[1:]]; break
y=np.array([1 if 'mort30_Yes' in t else 0 for t in titles],dtype=int)
assert len(y)==X.shape[1]==len(samples)
rows=[]
for rank,(sym,pid,i,v) in enumerate(chosen,1): rows.append({'rank':rank,'gene_symbol':sym,'probeset_id':pid,'variance':v,'n_probesets_for_gene':len(by[sym])})
pd.DataFrame(rows).to_csv(out/'gene_variance_rank.csv',index=False)
np.save(out/'X_gene_ranked.npy',X[[z[2] for z in chosen],:].T)
pd.DataFrame({'sample':samples,'title':titles,'mort30':y}).to_csv(out/'samples_y.csv',index=False)
summary={'n_samples':int(X.shape[1]),'p_probe':int(X.shape[0]),'missing_values':int(np.isnan(X).sum()),'unambiguous_probe_rows':len(records),'unique_gene_symbols':len(chosen),'mort30_yes':int(y.sum()),'mort30_no':int((1-y).sum()),'cutoffs':[x for x in [300,500,1000,2000,3000,5000] if x<=len(chosen)]}
json.dump(summary,open(out/'gene_matrix_summary.json','w'),indent=2); print(json.dumps(summary,indent=2))
