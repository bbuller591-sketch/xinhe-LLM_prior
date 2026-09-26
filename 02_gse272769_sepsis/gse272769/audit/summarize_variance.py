import pandas as pd, numpy as np
p='gse272769/audit/gene_variance_rank.csv'; d=pd.read_csv(p)
qs=[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]
print('variance quantiles'); print(d.variance.quantile(qs).to_string())
print('\ncutoff table')
for k in [300,500,1000,2000,3000,5000]:
 v=d.iloc[k-1].variance
 print(k,'threshold',v,'ratio_to_top',v/d.iloc[0].variance,'ratio_to_median',v/d.variance.median())
print('\ntop20'); print(d.head(20).to_string(index=False))
