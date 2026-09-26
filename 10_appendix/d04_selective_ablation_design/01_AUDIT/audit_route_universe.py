

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
import pandas as pd
import numpy as np
import glob, os, json

ROOT=Path(str(REPRO_ROOT))
TASKS={
 'BREAST':ROOT/'selective_multi_dataset_expansion_20260919/06_selective_ROUTING/BREAST_GSE25055_GSE25065',
 'SEPSIS65682':ROOT/'selective_multi_dataset_expansion_20260919/06_selective_ROUTING/SEPSIS_GSE65682',
 'GSE272769':ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/04_ROUTING'
}
def pid(r):
    return '||'.join(sorted([str(r.feature_A),str(r.feature_B)]))

def analyze(name,base):
    rows=[]; unions={x:set() for x in ['candidate','QB','Q','B']}
    files=sorted(glob.glob(str(base/'fold*_*'/'K*_PAIR_CONFUSION.csv')))
    for fp in files:
        d=pd.read_csv(fp)
        if 'pair_type' not in d.columns: continue
        bn=os.path.basename(os.path.dirname(fp))
        fold=int(bn.split('_')[0].replace('fold',''))
        sel='_'.join(bn.split('_')[1:])
        k=int(os.path.basename(fp).split('_')[0].replace('K',''))
        n=int((d.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION').sum())
        d=d.copy()
        d['pair']=d.apply(pid,axis=1)
        qb=set(d.loc[d.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION','pair'])
        q=set(d.sort_values(
            ['selection_disagreement_Q','feature_index_A','feature_index_B'],
            ascending=[False,True,True],kind='mergesort').head(n).pair) if n else set()
        b=set(d.sort_values(
            ['order_balance_B','feature_index_A','feature_index_B'],
            ascending=[False,True,True],kind='mergesort').head(n).pair) if n else set()
        cand=set(d.pair)
        unions['candidate'] |= cand
        unions['QB'] |= qb
        unions['Q'] |= q
        unions['B'] |= b
        rows.append({
          'dataset':name,'fold':fold,'selector':sel,'k':k,
          'candidate_n':len(cand),'matched_n':n,
          'QB_Q_overlap':len(qb&q),'QB_B_overlap':len(qb&b),
          'Q_B_overlap':len(q&b),
          'Q_new_vs_QB':len(q-qb),'B_new_vs_QB':len(b-qb)
        })
    summary={'dataset':name,'n_cells':len(rows)}
    for key,v in unions.items(): summary[f'{key}_union_n']=len(v)
    summary['Q_new_union_vs_QB']=len(unions['Q']-unions['QB'])
    summary['B_new_union_vs_QB']=len(unions['B']-unions['QB'])
    summary['QorB_new_union_vs_QB']=len((unions['Q']|unions['B'])-unions['QB'])
    return pd.DataFrame(rows),summary,unions
allrows=[]; summaries=[]; allunions={}
for name,base in TASKS.items():
    d,s,u=analyze(name,base)
    allrows.append(d); summaries.append(s); allunions[name]=u

cell=pd.concat(allrows,ignore_index=True)
sm=pd.DataFrame(summaries)
OUT=ROOT/'M3_ABLATION_DESIGN_20260924/01_AUDIT'
cell.to_csv(OUT/'ROUTING_COMPONENT_CELL_AUDIT.csv',index=False)
sm.to_csv(OUT/'ROUTING_COMPONENT_UNION_AUDIT.csv',index=False)

meas_b=pd.read_csv(ROOT/'selective_multi_dataset_expansion_20260919/11_selective_POSTPROCESS/selective_PAIR_SOURCE_MEASUREMENTS.csv')
meas_sets={
 'BREAST':set(meas_b.loc[meas_b.task=='BREAST_GSE25055_GSE25065','unordered_pair_id'].astype(str)),
 'SEPSIS65682':set(meas_b.loc[meas_b.task=='SEPSIS_GSE65682','unordered_pair_id'].astype(str))
}
g30=pd.read_csv(ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/03_MEASUREMENT/selective_PAIR_SOURCE_MEASUREMENTS_K30.csv')
g50=pd.read_csv(ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/03_MEASUREMENT/selective_PAIR_SOURCE_MEASUREMENTS_K50.csv')
meas_sets['GSE272769']=set(g30.unordered_pair_id.astype(str))|set(g50.unordered_pair_id.astype(str))

cov=[]
for name,u in allunions.items():
    m=meas_sets[name]
    for route in ['candidate','QB','Q','B']:
        v=u[route]
        cov.append({'dataset':name,'route':route,'union_n':len(v),
                    'measured_n':len(v&m),'missing_n':len(v-m),
                    'coverage':len(v&m)/len(v) if v else 1.0})
pd.DataFrame(cov).to_csv(OUT/'MEASUREMENT_PAIR_COVERAGE_AUDIT.csv',index=False)
print(sm.to_string(index=False))
print(pd.DataFrame(cov).to_string(index=False))
