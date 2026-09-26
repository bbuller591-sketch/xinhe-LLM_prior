

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
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

HERE=Path(str(REPRO_ROOT / '07_jkp153/qwen25'))
PHASEB=Path(str(REPRO_ROOT / '07_jkp153/phaseB'))

def hb(p):
    p=float(np.clip(p,1e-15,1-1e-15))
    return -(p*math.log2(p)+(1-p)*math.log2(1-p))

ap=argparse.ArgumentParser()
ap.add_argument('--year',type=int,default=2024)
ap.add_argument('--log',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
graph=pd.read_csv(PHASEB/f'MEASUREMENT_UNION_GRAPH_{a.year}.csv')
calls=pd.DataFrame([json.loads(x) for x in a.log.read_text().splitlines() if x.strip()])
calls=calls.drop_duplicates('measurement_call_id',keep='last')
rows=[]
for e in graph.to_dict('records'):
    eid=e['measurement_edge_id']; armres={}
    for arm in ['DQ']+(['DQL'] if bool(e['DQL_call_required']) else []):
        c=calls[(calls.edge_id==eid)&(calls.arm==arm)&(calls.final_status=='SUCCESS')]
        ab=c[c.order.eq('AB')]; ba=c[c.order.eq('BA')]
        valid=(len(ab)==1 and len(ba)==1)
        if valid:
            A=ab.iloc[0]; B=ba.iloc[0]
            lab=float(A.logit_A-A.logit_B)
            lba=float(-(B.logit_A-B.logit_B))
            lm=(lab+lba)/2.0
            p=float(expit(lm)); H=hb(p); cert=1-H
            gap=abs(lab-lba)
            agree=(lab>=0)==(lba>=0)
            tu=float(((A.p4_T+A.p4_U)+(B.p4_T+B.p4_U))/2)
        else:
            p=H=cert=gap=tu=np.nan; agree=False
        armres[arm]={
            'valid':bool(valid),'valid_repeat_pairs':1 if valid else 0,
            'p_i_gt_j':p,'HL_bits':H,'certainty':cert,
            'order_gap_mean':gap,'repeat_logit_sd':0.0 if valid else np.nan,
            'order_direction_agree':bool(agree),'mean_TU_cond4':tu,
            'hard_A_rate':float((c.hard_token=='A').mean()) if len(c) else np.nan,
            'hard_B_rate':float((c.hard_token=='B').mean()) if len(c) else np.nan,
            'hard_T_rate':float((c.hard_token=='T').mean()) if len(c) else np.nan,
            'hard_U_rate':float((c.hard_token=='U').mean()) if len(c) else np.nan,
            'successful_calls':int(len(c)),'expected_calls':2,
            'fallback_to_DQ_no_literature':False}
    if not bool(e['DQL_call_required']):
        armres['DQL']=dict(armres['DQ'])
        armres['DQL']['fallback_to_DQ_no_literature']=True
    row=dict(e)
    for arm in ['DQ','DQL']:
        for k,v in armres[arm].items(): row[f'{arm}_{k}']=v
    rows.append(row)
out=pd.DataFrame(rows)
a.out.parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.out,index=False)

sm={
 'year':a.year,'call_rows':len(calls),'edges':len(out),
 'all_calls_success':bool((calls.final_status=='SUCCESS').all()),
 'DQ_valid_edges':int(out.DQ_valid.sum()),
 'DQL_valid_edges':int(out.DQL_valid.sum()),
 'broad_DQ_valid':int(out.loc[out.in_broad.astype(bool),'DQ_valid'].sum()),
 'selective_primary_DQ_valid':int(out.loc[out.selective_primary_H090.astype(bool),'DQ_valid'].sum()),
 'broad_DQ_order_agreement':float(out.loc[out.in_broad.astype(bool),'DQ_order_direction_agree'].mean()),
 'broad_DQL_order_agreement':float(out.loc[out.in_broad.astype(bool),'DQL_order_direction_agree'].mean()),
 'selective_H090_DQ_order_agreement':float(out.loc[out.selective_primary_H090.astype(bool),'DQ_order_direction_agree'].mean()),
 'selective_H090_DQL_order_agreement':float(out.loc[out.selective_primary_H090.astype(bool),'DQL_order_direction_agree'].mean())
}
Path(str(a.out)+'.summary.json').write_text(json.dumps(sm,indent=2)+'\n')
print(json.dumps(sm,indent=2))
