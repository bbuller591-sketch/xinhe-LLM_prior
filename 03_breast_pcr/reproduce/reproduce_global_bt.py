#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

PKG=Path(__file__).resolve().parents[1]
PAIR=pd.read_parquet(PKG/'MEASUREMENTS/global_BROAD/BROAD_D20_PAIR_NEUTRALIZED.parquet')
OUT=PKG/'REPRODUCE/OUTPUT'
OUT.mkdir(parents=True,exist_ok=True)
LAM=1e-3

def fit_bt(arm,degree):
    nodes=pd.read_csv(PKG/'METHOD_ARTIFACTS/global_GRAPHS'/arm/'NODES.csv',dtype={'GeneID':str})
    edges=pd.read_csv(PKG/'METHOD_ARTIFACTS/global_GRAPHS'/arm/f'EDGES_D{degree}.csv',dtype=str)
    q=PAIR[(PAIR.arm==arm)&PAIR.unordered_pair_id.isin(set(edges.unordered_pair_id))].copy()
    if len(q)!=len(edges):
        raise RuntimeError(f'edge/measurement mismatch {arm} d{degree}: {len(q)} vs {len(edges)}')
    genes=nodes.gene_symbol.astype(str).tolist(); idx={g:i for i,g in enumerate(genes)}
    ii=q.gene_i.map(idx).to_numpy(int); jj=q.gene_j.map(idx).to_numpy(int)
    y=q.hard_y_i_over_j.to_numpy(float)
    mask=np.isfinite(y); ii=ii[mask]; jj=jj[mask]; y=y[mask]
    cert=q.loc[mask,'certainty_1_minus_H'].to_numpy(float)

    def solve(w):
        def fg(x):
            z=x[ii]-x[jj]
            ce=np.logaddexp(0,z)-y*z
            f=float(np.dot(w,ce)+0.5*LAM*np.dot(x,x))
            rr=w*(expit(z)-y)
            g=np.zeros_like(x)
            np.add.at(g,ii,rr); np.add.at(g,jj,-rr)
            g+=LAM*x
            return f,g
        res=minimize(lambda x:fg(x),np.zeros(len(genes)),jac=True,method='L-BFGS-B',
                     options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
        if not res.success: raise RuntimeError(f'BT optimization failed {arm} d{degree}')
        x=res.x-res.x.mean()
        r=rankdata(-x,method='average')
        psi=2*(1-(r-1)/(len(x)-1))-1
        return x,psi,int(res.nit)

    g1,p1,n1=solve(np.ones(mask.sum()))
    g2,p2,n2=solve(cert)
    ref=pd.read_csv(PKG/'METHOD_ARTIFACTS/global_BT'/arm/f'BT_global_D{degree}_SCORES.csv')
    got=pd.DataFrame({'gene_symbol':genes,'g_global_repro':g1,'psi_global_repro':p1,'g_global_certainty_repro':g2,'psi_global_certainty_repro':p2})
    z=got.merge(ref[['gene_symbol','g_global','psi_global','g_global_certainty','psi_global_certainty']],on='gene_symbol',validate='one_to_one')
    return {
      'arm':arm,'degree':degree,'n_edges':len(edges),'n_nodes':len(nodes),'global_nit':n1,'global_certainty_nit':n2,
      'corr_psi_global':float(np.corrcoef(z.psi_global_repro,z.psi_global)[0,1]),
      'corr_psi_global_certainty':float(np.corrcoef(z.psi_global_certainty_repro,z.psi_global_certainty)[0,1]),
      'max_abs_psi_global_diff':float(np.max(np.abs(z.psi_global_repro-z.psi_global))),
      'max_abs_psi_global_certainty_diff':float(np.max(np.abs(z.psi_global_certainty_repro-z.psi_global_certainty))),
      'max_abs_g_global_diff':float(np.max(np.abs(z.g_global_repro-z.g_global))),
      'max_abs_g_global_certainty_diff':float(np.max(np.abs(z.g_global_certainty_repro-z.g_global_certainty)))
    }

rows=[]
for arm in ['ARM_GSE41998','ARM_GSE32646']:
    for d in [20,10]:
        rows.append(fit_bt(arm,d))
audit=pd.DataFrame(rows)
audit.to_csv(OUT/'BT_REPRODUCTION_AUDIT.csv',index=False)
ok=bool((audit.corr_psi_global>0.999999999).all() and (audit.corr_psi_global_certainty>0.999999999).all()
        and (audit.max_abs_psi_global_diff<1e-12).all() and (audit.max_abs_psi_global_certainty_diff<1e-12).all())
status={'status':'PASS' if ok else 'FAIL','lambda_BT':LAM,'rows':rows}
(OUT/'BT_REPRODUCTION_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(audit.to_string(index=False))
print(json.dumps({'status':status['status']},indent=2))
if not ok: raise SystemExit(2)
