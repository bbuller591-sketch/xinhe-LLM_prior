

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
from pathlib import Path
import pandas as pd,math,json
from scipy.special import expit
W=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap'))
O=W/'08_Q2_Q5_DIAGNOSTICS/BREAST_Q2_SOURCE'
d=pd.read_csv(O/'measurement/CALLS_COMPLETE.csv')
def ent(p):
    return 0.0 if p<=0 or p>=1 else -(p*math.log(p,2)+(1-p)*math.log(1-p,2))
rows=[];drop=[]
for (pair,arm),g in d.groupby(['unordered_pair_id','arm'],sort=False):
    if len(g)!=2 or set(g.order.astype(str))!={'AB','BA'}:
        drop.append((pair,arm,'orientation'));continue
    ab=g[g.order=='AB'].iloc[0];ba=g[g.order=='BA'].iloc[0]
    if not (str(ab.parse_status).startswith('PASS') and str(ba.parse_status).startswith('PASS')):
        drop.append((pair,arm,'top20'));continue
    if pd.isna(ab.logp_A) or pd.isna(ab.logp_B) or pd.isna(ba.logp_A) or pd.isna(ba.logp_B):
        drop.append((pair,arm,'missing'));continue
    ell=.5*((float(ab.logp_A)-float(ab.logp_B))-(float(ba.logp_A)-float(ba.logp_B)))
    p=float(expit(ell));H=ent(p)
    rows.append({'pair':pair,'arm':arm,'gene_i':ab.gene_A,'gene_j':ab.gene_B,
                 'binary_choice_i':1.0 if p>.5 else 0.0 if p<.5 else .5,
                 'p_i_over_j':p,'certainty':1-H,'entropy_bits':H,'symmetrized_logit':ell})
a=pd.DataFrame(rows);a.to_csv(O/'SOURCE_STRATIFIED_PAIR_MEASUREMENTS_GPT.csv',index=False)
(O/'AGGREGATION_AUDIT.json').write_text(json.dumps({'usable_pair_source_cells':len(a),'dropped':len(drop),'dropped_items':drop},ensure_ascii=False,indent=2)+'\n')
print('usable',len(a),'dropped',len(drop))
