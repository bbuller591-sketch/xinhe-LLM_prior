#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit

def logit(p):
    p=float(np.clip(p,1e-6,1-1e-6)); return math.log(p/(1-p))
def hb(p):
    p=float(np.clip(p,1e-12,1-1e-12)); return -(p*math.log2(p)+(1-p)*math.log2(1-p))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--year',type=int,required=True); ap.add_argument('--log',type=Path,required=True); ap.add_argument('--graph',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    calls=pd.DataFrame([json.loads(x) for x in a.log.read_text().splitlines() if x.strip()])
    calls=calls[calls.final_status.eq('SUCCESS')].copy()
    graph=pd.read_csv(a.graph)
    rows=[]
    for e in graph.to_dict('records'):
        eid=e['edge_id']
        arms=['DQ']+(['DQL'] if bool(e['DQL_call_required']) else [])
        armres={}
        for arm in arms:
            c=calls[(calls.edge_id==eid)&(calls.arm==arm)]
            logits=[]; ordergaps=[]
            for rep in [1,2,3]:
                ab=c[(c.order=='AB')&(c.measurement_repeat==rep)]
                ba=c[(c.order=='BA')&(c.measurement_repeat==rep)]
                if len(ab)!=1 or len(ba)!=1: continue
                ab=ab.iloc[0]; ba=ba.iloc[0]
                if ab.hard_token not in ('A','B') or ba.hard_token not in ('A','B'): continue
                if pd.isna(ab.pA_cond_AB) or pd.isna(ba.pA_cond_AB): continue
                la=logit(ab.pA_cond_AB)
                lb=-logit(ba.pA_cond_AB)
                logits.append((la+lb)/2); ordergaps.append(abs(la-lb))
            if len(logits)>=2:
                lm=float(np.mean(logits)); p=float(expit(lm)); valid=True
                H=hb(p); cert=1-H
                rsd=float(np.std(logits,ddof=1)) if len(logits)>1 else 0.0
            else:
                p=H=cert=rsd=np.nan; valid=False
            allc=c
            armres[arm]={
              'valid':valid,'valid_repeat_pairs':len(logits),'p_i_gt_j':p,'HL_bits':H,'certainty':cert,
              'order_gap_mean':float(np.mean(ordergaps)) if ordergaps else np.nan,'repeat_logit_sd':rsd,
              'hard_T_rate':float((allc.hard_token=='T').mean()) if len(allc) else np.nan,
              'hard_U_rate':float((allc.hard_token=='U').mean()) if len(allc) else np.nan,
            }
        if not bool(e['DQL_call_required']):
            armres['DQL']=dict(armres['DQ']); armres['DQL']['fallback_to_DQ_no_literature']=True
        row={k:e[k] for k in e}
        for arm in ['DQ','DQL']:
            for k,v in armres.get(arm,{}).items(): row[f'{arm}_{k}']=v
        rows.append(row)
    out=pd.DataFrame(rows); a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
    print(json.dumps({'year':a.year,'edges':len(out),'DQ_valid':int(out.DQ_valid.fillna(False).sum()),'DQL_valid':int(out.DQL_valid.fillna(False).sum())},indent=2))
if __name__=='__main__': main()
