#!/usr/bin/env python3

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
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(str(REPRO_ROOT / '10_appendix/d04_selective_ablation'))
PKG=Path(str(REPRO_ROOT / 'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919'))
TASK='BREAST_GSE25055_GSE25065'
SELECTORS=['LASSO','ELASTICNET','SIS']
KGRID=[10,20,30]
ROUTING=PKG/'METHOD_ARTIFACTS/selective_ROUTING'
MEASP=PKG/'MEASUREMENTS/selective/selective_PAIR_SOURCE_MEASUREMENTS.csv'
OUT=ROOT/'01_MANIFEST'
OUT.mkdir(parents=True,exist_ok=True)

all_rows=[]
route_rows=[]
for fold in range(1,6):
    for sel in SELECTORS:
        for k in KGRID:
            path=ROUTING/f'fold{fold}_{sel}'/f'K{k}_PAIR_CONFUSION.csv'
            cf=pd.read_csv(path)
            required=['feature_A','feature_B','selection_disagreement_Q','order_balance_B','actionable_boundary_score']
            miss=[c for c in required if c not in cf.columns]
            if miss:
                raise RuntimeError(f'{path} missing {miss}')
            cf=cf.copy()
            cf['feature_A']=cf['feature_A'].astype(str)
            cf['feature_B']=cf['feature_B'].astype(str)
            def pair_id(r):
                a,b=sorted([str(r.feature_A),str(r.feature_B)])
                return a+'||'+b
            cf['unordered_pair_id']=[pair_id(r) for r in cf.itertuples()]
            cf['Q']=cf['selection_disagreement_Q'].astype(float)
            cf['B']=cf['order_balance_B'].astype(float)
            cf['QB']=cf['actionable_boundary_score'].astype(float)
            if 'pair_type' in cf.columns:
                main=cf[cf['pair_type'].astype(str)=='ACTIONABLE_BOUNDARY_CONFUSION'].copy()
            else:
                main=cf[cf['QB']>=0.25].copy()
            m=len(main)
            cf['in_QB_main']=cf['unordered_pair_id'].isin(set(main['unordered_pair_id']))
            qtop=cf.sort_values(['Q','feature_A','feature_B'],ascending=[False,True,True]).head(m).copy()
            btop=cf.sort_values(['B','feature_A','feature_B'],ascending=[False,True,True]).head(m).copy()
            qbtop=cf.sort_values(['QB','feature_A','feature_B'],ascending=[False,True,True]).head(m).copy()
            arms=[('QB_MAIN',main,'QB'),('QB_TOPM',qbtop,'QB'),('Q_ONLY',qtop,'Q'),('B_ONLY',btop,'B')]
            for arm,df,score_col in arms:
                for rank,r in enumerate(df.itertuples(index=False),start=1):
                    route_rows.append({
                        'task':TASK,'fold':fold,'selector':sel,'k':k,'arm':arm,'rank_in_arm':rank,
                        'unordered_pair_id':r.unordered_pair_id,'feature_A':r.feature_A,'feature_B':r.feature_B,
                        'Q':float(r.Q),'B':float(r.B),'QB':float(r.QB),'route_score':float(getattr(r,score_col)),
                        'main_m_cell':m,'candidate_n':len(cf)
                    })
            for r in cf.itertuples(index=False):
                all_rows.append({
                    'task':TASK,'fold':fold,'selector':sel,'k':k,
                    'unordered_pair_id':r.unordered_pair_id,'feature_A':r.feature_A,'feature_B':r.feature_B,
                    'Q':float(r.Q),'B':float(r.B),'QB':float(r.QB),'in_QB_main':bool(r.in_QB_main),
                    'candidate_n_cell':len(cf),'main_m_cell':m,
                    'source_file':str(path)
                })

cand=pd.DataFrame(all_rows).drop_duplicates(['fold','selector','k','unordered_pair_id'])
routes=pd.DataFrame(route_rows)
union_pairs=pd.DataFrame({'unordered_pair_id':sorted(cand.unordered_pair_id.unique())})
union_pairs[['gene_a','gene_b']]=pd.DataFrame([x.split('||',1) for x in union_pairs['unordered_pair_id']], index=union_pairs.index)
meas=pd.read_csv(MEASP)
old_pairs=set(meas.unordered_pair_id.astype(str).unique())
union_pairs['covered_by_old_selective_measurement']=union_pairs.unordered_pair_id.isin(old_pairs)

cand.to_csv(OUT/'BREAST_RANK_WINDOW_CANDIDATE_CELLS.csv',index=False)
routes.to_csv(OUT/'BREAST_QB_Q_B_ROUTE_SETS.csv',index=False)
union_pairs.to_csv(OUT/'BREAST_CANDIDATE_PAIR_UNION_1834.csv',index=False)

summ=[]
for (fold,sel,k),g in cand.groupby(['fold','selector','k']):
    base=set(g.unordered_pair_id)
    m=int(g.main_m_cell.iloc[0])
    row={'fold':fold,'selector':sel,'k':k,'candidate_n':len(base),'m_cell':m}
    for arm in ['QB_MAIN','Q_ONLY','B_ONLY']:
        s=set(routes[(routes.fold==fold)&(routes.selector==sel)&(routes.k==k)&(routes.arm==arm)].unordered_pair_id)
        row[f'{arm}_n']=len(s)
        row[f'{arm}_old_measurement_coverage_n']=len(s & old_pairs)
        row[f'{arm}_missing_old_measurement_n']=len(s - old_pairs)
    summ.append(row)
summary=pd.DataFrame(summ)
summary.to_csv(OUT/'BREAST_ROUTE_CELL_COVERAGE_SUMMARY.csv',index=False)
coverage={
    'task':TASK,
    'candidate_cell_rows':int(len(cand)),
    'candidate_union_pairs':int(union_pairs.unordered_pair_id.nunique()),
    'old_measurement_pairs':int(len(old_pairs)),
    'candidate_union_old_covered':int(union_pairs.covered_by_old_selective_measurement.sum()),
    'candidate_union_missing_old':int((~union_pairs.covered_by_old_selective_measurement).sum()),
    'route_sets_rows':int(len(routes)),
    'sha256':{}
}
for p in [OUT/'BREAST_RANK_WINDOW_CANDIDATE_CELLS.csv',OUT/'BREAST_QB_Q_B_ROUTE_SETS.csv',OUT/'BREAST_CANDIDATE_PAIR_UNION_1834.csv',OUT/'BREAST_ROUTE_CELL_COVERAGE_SUMMARY.csv']:
    coverage['sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'02_QUERY_AUDIT'/'CANDIDATE_MEASUREMENT_COVERAGE.json').write_text(json.dumps(coverage,indent=2)+'\n')
print(json.dumps(coverage,indent=2))
