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
import json
from pathlib import Path
import pandas as pd
ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'finance_llm_prior/experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
FREEZE=ROOT/'JKP153_FORMAL_MEASUREMENT_DESIGN_FREEZE_V0_4_20260913_201725'
OLD=FREEZE/'inputs/recovered_frozen/FULL153_PRODUCTION_GRAPH_REUSED.csv'

def elig(year):
    d={}
    for line in (FREEZE/f'inputs/FACTOR_EVIDENCE_PACKETS_{year}.jsonl').read_text().splitlines():
        if line.strip():
            o=json.loads(line); d[o['factor']['factor_id']]=o.get('llm_measurement_status')!='ABSTAIN_NO_APPROVED_ELIGIBLE_EVIDENCE'
    return d

g0=pd.read_csv(OLD)
if len(g0)!=459: raise RuntimeError(len(g0))
for year in [2024,2025]:
    e=elig(year); g=g0[['edge_id','factor_i','factor_j','graph_seed']].copy()
    g['origin_year']=year
    g['literature_eligible_i']=g.factor_i.map(e)
    g['literature_eligible_j']=g.factor_j.map(e)
    g['DQL_call_required']=g.literature_eligible_i|g.literature_eligible_j
    g.to_csv(OUT/f'BROAD_GRAPH_459_{year}.csv',index=False)
    print(year,'edges',len(g),'DQL_required',int(g.DQL_call_required.sum()),'calls_R3',int((len(g)+g.DQL_call_required.sum())*6))
