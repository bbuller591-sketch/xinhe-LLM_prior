#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit

def lg(p):
    p=float(np.clip(p,1e-6,1-1e-6));return math.log(p/(1-p))
def hb(p):
    p=float(np.clip(p,1e-12,1-1e-12));return -(p*math.log2(p)+(1-p)*math.log2(1-p))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--year',type=int,required=True);ap.add_argument('--log',type=Path,required=True);ap.add_argument('--graph',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    calls=pd.DataFrame([json.loads(x) for x in a.log.read_text().splitlines() if x.strip()])
    # ensure exactly one terminal row per scheduled call in the production log
    calls=calls.drop_duplicates('measurement_call_id',keep='last')
    graph=pd.read_csv(a.graph)
    rows=[]
    for e in graph.to_dict('records'):
        eid=e['measurement_edge_id']; armres={}
        for arm in ['DQ']+(['DQL'] if bool(e['DQL_call_required']) else []):
            c=calls[(calls.edge_id==eid)&(calls.arm==arm)&(calls.final_status=='SUCCESS')]
            lr=[]; gaps=[]
            for rep in [1,2,3]:
                ab=c[(c.order=='AB')&(c.measurement_repeat==rep)]
                ba=c[(c.order=='BA')&(c.measurement_repeat==rep)]
                if len(ab)!=1 or len(ba)!=1: continue
                A=ab.iloc[0];B=ba.iloc[0]
                if A.hard_token not in ('A','B') or B.hard_token not in ('A','B'): continue
                if pd.isna(A.pA_cond_AB) or pd.isna(B.pA_cond_AB): continue
                lab=lg(A.pA_cond_AB); lba=-lg(B.pA_cond_AB)
                lr.append((lab+lba)/2);gaps.append(abs(lab-lba))
            valid=len(lr)>=2
            if valid:
                lm=float(np.mean(lr));p=float(expit(lm));H=hb(p);cert=1-H
                rsd=float(np.std(lr,ddof=1))
            else:
                p=H=cert=rsd=np.nan
            allc=calls[(calls.edge_id==eid)&(calls.arm==arm)]
            armres[arm]={
              'valid':valid,'valid_repeat_pairs':len(lr),'p_i_gt_j':p,'HL_bits':H,'certainty':cert,
              'order_gap_mean':float(np.mean(gaps)) if gaps else np.nan,'repeat_logit_sd':rsd,
              'hard_A_rate':float((allc.hard_token=='A').mean()) if len(allc) else np.nan,
              'hard_B_rate':float((allc.hard_token=='B').mean()) if len(allc) else np.nan,
              'hard_T_rate':float((allc.hard_token=='T').mean()) if len(allc) else np.nan,
              'hard_U_rate':float((allc.hard_token=='U').mean()) if len(allc) else np.nan,
              'successful_calls':int((allc.final_status=='SUCCESS').sum()),
              'expected_calls':6,
              'fallback_to_DQ_no_literature':False
            }
        if not bool(e['DQL_call_required']):
            armres['DQL']=dict(armres['DQ']);armres['DQL']['fallback_to_DQ_no_literature']=True
        row=dict(e)
        for arm in ['DQ','DQL']:
            for k,v in armres[arm].items():row[f'{arm}_{k}']=v
        rows.append(row)
    out=pd.DataFrame(rows);a.out.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.out,index=False)
    sm={'year':a.year,'call_rows':len(calls),'expected_schedule_calls':int(sum((1+(1 if bool(x) else 0))*6 for x in graph.DQL_call_required)),
        'call_status_counts':calls.final_status.value_counts(dropna=False).to_dict(),
        'edges':len(out),'DQ_valid_edges':int(out.DQ_valid.fillna(False).sum()),'DQL_valid_edges':int(out.DQL_valid.fillna(False).sum()),
        'broad_DQ_valid':int(out.loc[out.in_broad.astype(bool),'DQ_valid'].fillna(False).sum()),
        'selective_primary_DQ_valid':int(out.loc[out.selective_primary_H090.astype(bool),'DQ_valid'].fillna(False).sum())}
    Path(str(a.out)+'.summary.json').write_text(json.dumps(sm,indent=2)+'\n')
    print(json.dumps(sm,indent=2))
if __name__=='__main__':main()
