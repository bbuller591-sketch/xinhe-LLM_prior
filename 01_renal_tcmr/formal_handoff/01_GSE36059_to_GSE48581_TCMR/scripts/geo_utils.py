import csv, gzip, re
import pandas as pd
import numpy as np

def _split_tsv_line(line):
    return next(csv.reader([line.rstrip("\n")], delimiter="\t", quotechar='"'))

def _norm(s):
    return re.sub(r'[^a-z0-9]+','_',s.strip().lower()).strip('_')

def read_geo_metadata(path):
    ordinary=[]
    char_rows=[]
    with gzip.open(path,'rt',errors='replace') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
            if line.startswith('!Sample_'):
                parts=_split_tsv_line(line)
                key=parts[0][1:]
                vals=parts[1:]
                if key=='Sample_characteristics_ch1':
                    char_rows.append(vals)
                else:
                    ordinary.append((key,vals))
    n=max([len(v) for _,v in ordinary]+[len(v) for v in char_rows]+[0])
    data={'sample_index':list(range(n))}
    counts={}
    for key,vals in ordinary:
        vals=vals+['']*(n-len(vals))
        name=_norm(key.replace('Sample_',''))
        counts[name]=counts.get(name,0)+1
        if counts[name]>1: name=f'{name}_{counts[name]}'
        data[name]=vals
    # Parse every sample's characteristics independently to avoid row-shift artifacts.
    per=[{} for _ in range(n)]
    for vals in char_rows:
        vals=vals+['']*(n-len(vals))
        for i,v in enumerate(vals):
            v=v.strip()
            if not v: continue
            if ':' in v:
                a,b=v.split(':',1); key=_norm(a); val=b.strip()
            else:
                key='characteristic'; val=v
            if key in per[i]:
                # Preserve duplicates if present.
                j=2
                while f'{key}_{j}' in per[i]: j+=1
                key=f'{key}_{j}'
            per[i][key]=val
    for key in sorted(set().union(*(d.keys() for d in per)) if per else []):
        data[key]=[d.get(key,'') for d in per]
    return pd.DataFrame(data)

def read_geo_expression(path):
    df=pd.read_csv(path,sep='\t',comment='!',compression='gzip',low_memory=False)
    if df.empty:
        return df
    first=df.columns[0]
    df=df.rename(columns={first:'feature_id'})
    return df

def summarize_metadata(path):
    md=read_geo_metadata(path)
    out={'n_samples':len(md),'columns':list(md.columns)}
    vals={}
    for c in md.columns:
        if c=='sample_index': continue
        vc=md[c].astype(str).value_counts(dropna=False)
        if len(vc)<=20:
            vals[c]=vc.to_dict()
        else:
            vals[c]={'n_unique':int(md[c].nunique()),'examples':md[c].astype(str).head(5).tolist()}
    out['values']=vals
    return out
