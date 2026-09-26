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
import pandas as pd, numpy as np, json
ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'finance_llm_prior/experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'

def key(a,b): return tuple(sorted((str(a),str(b))))
summ=[]
for year in [2024,2025]:
    b=pd.read_csv(OUT/f'BROAD_GRAPH_459_{year}.csv')
    s=pd.read_csv(OUT/f'QUERY_GRAPH_UNION_H085_{year}.csv')
    bd={key(a,c):r for a,c,r in zip(b.factor_i,b.factor_j,b.to_dict('records'))}
    sd={key(a,c):r for a,c,r in zip(s.factor_i,s.factor_j,s.to_dict('records'))}
    ks=sorted(set(bd)|set(sd))
    rows=[]
    for n,k in enumerate(ks,1):
        br=bd.get(k); sr=sd.get(k)
        src=sr if sr is not None else br
        rows.append({
          'measurement_edge_id':f'MU{year}_{n:04d}',
          'factor_i':k[0],'factor_j':k[1],'origin_year':year,
          'in_broad':br is not None,
          'broad_edge_id':br['edge_id'] if br is not None else '',
          'in_selective_union_H085':sr is not None,
          'selective_edge_id':sr['edge_id'] if sr is not None else '',
          'selective_primary_H090':bool(sr['primary_H_ge_0p90']) if sr is not None else False,
          'selective_strict_H095':bool(sr['strict_sensitivity_H_ge_0p95']) if sr is not None else False,
          'HD_bits':float(sr['HD_bits']) if sr is not None else np.nan,
          'raw_reference_rank_gap':float(sr['raw_reference_rank_gap']) if sr is not None else np.nan,
          'literature_eligible_i':bool(src['literature_eligible_i']),
          'literature_eligible_j':bool(src['literature_eligible_j']),
          'DQL_call_required':bool(src['DQL_call_required']),
        })
    u=pd.DataFrame(rows)
    u.to_csv(OUT/f'MEASUREMENT_UNION_GRAPH_{year}.csv',index=False)
    overlap=int((u.in_broad & u.in_selective_union_H085).sum())
    dql=int(u.DQL_call_required.sum())
    sm={'year':year,'broad_edges':459,'selective_union_edges':len(s),'pair_overlap':overlap,'unique_measurement_edges':len(u),
        'DQL_required_unique_edges':dql,'planned_calls_R3_ABBA':int((len(u)+dql)*6)}
    summ.append(sm)
    print(json.dumps(sm))
(OUT/'MEASUREMENT_UNION_SUMMARY.json').write_text(json.dumps(summ,indent=2)+'\n')
