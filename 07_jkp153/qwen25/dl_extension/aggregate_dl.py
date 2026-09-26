

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
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
ROOT=Path(str(REPRO_ROOT))
PHASEB=ROOT/'finance_llm_prior/experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
def hb(p):
 p=float(np.clip(p,1e-15,1-1e-15)); return -(p*math.log2(p)+(1-p)*math.log2(1-p))
ap=argparse.ArgumentParser(); ap.add_argument('--year',type=int,required=True); ap.add_argument('--log',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
g=pd.read_csv(PHASEB/f'MEASUREMENT_UNION_GRAPH_{a.year}.csv')
calls=pd.DataFrame([json.loads(x) for x in a.log.read_text().splitlines() if x.strip()])
rows=[]
for e in g.to_dict('records'):
 c=calls[(calls.edge_id==e['measurement_edge_id'])&(calls.final_status=='SUCCESS')]
 ab=c[c.order.eq('AB')]; ba=c[c.order.eq('BA')]
 valid=(len(ab)==1 and len(ba)==1)
 if valid:
  A=ab.iloc[0]; B=ba.iloc[0]; lab=float(A.logit_A-A.logit_B); lba=float(-(B.logit_A-B.logit_B)); lm=(lab+lba)/2
  p=float(expit(lm)); H=hb(p); cert=1-H; gap=abs(lab-lba); agree=(lab>=0)==(lba>=0); tu=float(((A.p4_T+A.p4_U)+(B.p4_T+B.p4_U))/2)
 else: p=H=cert=gap=tu=np.nan; agree=False
 row=dict(e); row.update(DL_valid=bool(valid),DL_p_i_gt_j=p,DL_HL_bits=H,DL_certainty=cert,DL_order_gap=gap,DL_order_agree=bool(agree),DL_mean_TU_mass=tu,DL_successful_calls=int(len(c)))
 rows.append(row)
out=pd.DataFrame(rows); a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
print(json.dumps({'year':a.year,'calls':len(calls),'edges':len(out),'valid_edges':int(out.DL_valid.sum()),'broad_valid':int(out.loc[out.in_broad.astype(bool),'DL_valid'].sum()),'H090_valid':int(out.loc[out.selective_primary_H090.astype(bool),'DL_valid'].sum()),'order_agree_valid':float(out.loc[out.DL_valid,'DL_order_agree'].mean())},indent=2))
