import gzip,csv,re,pandas as pd,numpy as np
p='gse272769/raw/GSE272769_series_matrix.txt.gz'; meta=[]
with gzip.open(p,'rt',errors='replace') as f:
 for line in f:
  if line.startswith('!Sample_title') or line.startswith('!Sample_geo_accession') or line.startswith('!Sample_characteristics_ch1'):
   r=next(csv.reader([line],delimiter='\t'));meta.append((r[0],[x.strip('"') for x in r[1:]]))
  if line.startswith('!series_matrix_table_begin'):break
# identify rows by prefixes
D={}
for k,v in meta:
 if k=='!Sample_title': D['title']=v
 elif k=='!Sample_geo_accession':D['sample']=v
 else:
  pref=v[0].split(':',1)[0].strip().lower() if v else ''
  D[pref]=[x.split(':',1)[1].strip() if ':' in x else x for x in v]
df=pd.DataFrame({'sample':D['sample'],'title':D['title']});
for key in ['sex','age','shock','mort30','immunocompromised','bacteremia']:
 if key in D:df[key]=D[key]
print('columns',df.columns.tolist());print(df.groupby(['mort30','sex']).size().unstack(fill_value=0) if 'sex' in df else 'no sex');
if 'age' in df: print('age by mort30',df.assign(age_num=pd.to_numeric(df.age,errors='coerce')).groupby('mort30').age_num.agg(['count','mean','std']).to_string())
if 'shock' in df: print('shock by mort30\n',pd.crosstab(df.mort30,df.shock))
df.to_csv('gse272769/audit/clinical_metadata_audit.csv',index=False)
