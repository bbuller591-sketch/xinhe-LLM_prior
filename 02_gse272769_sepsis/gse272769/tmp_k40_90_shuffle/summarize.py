import csv,glob,math,statistics
real={}
with open('gse272769/tmp_k40_90_results/selective_NESTED_AGGREGATE.csv') as f:
 for r in csv.DictReader(f): real[(r['selector'],int(r['k']))]=r
vals={}
files=glob.glob('gse272769/tmp_k40_90_shuffle/rep_*/selective_NESTED_AGGREGATE.csv')
for fn in files:
 with open(fn) as f:
  rows=list(csv.DictReader(f))
 assert len(rows)==18,(fn,len(rows))
 for r in rows:
  key=(r['selector'],int(r['k']))
  vals.setdefault(key,{'auroc':[],'ap':[]})
  vals[key]['auroc'].append(float(r['mean_delta_auroc']))
  vals[key]['ap'].append(float(r['mean_delta_macro_ap']))
def q(a,p):
 a=sorted(a); x=(len(a)-1)*p; lo=int(x); hi=min(lo+1,len(a)-1); return a[lo]+(a[hi]-a[lo])*(x-lo)
print('selector,k,real_dAUC,shuffle_mean,sd,q025,q975,percentile,p_upper,real_dAP,shuffle_AP_mean,AP_percentile,AP_p_upper')
for key in sorted(real,key=lambda x:(x[0],x[1])):
 r=real[key]; a=vals[key]['auroc']; b=vals[key]['ap']; ra=float(r['mean_delta_auroc']); rb=float(r['mean_delta_macro_ap'])
 print(f'{key[0]},{key[1]},{ra:.6f},{statistics.mean(a):.6f},{statistics.stdev(a):.6f},{q(a,.025):.6f},{q(a,.975):.6f},{100*sum(x<ra for x in a)/200:.1f},{(1+sum(x>=ra for x in a))/201:.4f},{rb:.6f},{statistics.mean(b):.6f},{100*sum(x<rb for x in b)/200:.1f},{(1+sum(x>=rb for x in b))/201:.4f}')