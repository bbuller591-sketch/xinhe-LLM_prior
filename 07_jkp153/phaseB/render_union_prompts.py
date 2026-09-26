#!/usr/bin/env python3
from __future__ import annotations

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
import hashlib, json
from pathlib import Path
import pandas as pd
import render_selective_prompts as R

ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'finance_llm_prior/experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
FREEZE=ROOT/'JKP153_FORMAL_MEASUREMENT_DESIGN_FREEZE_V0_4_20260913_201725'

def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def render_year(year):
    graph=pd.read_csv(OUT/f'MEASUREMENT_UNION_GRAPH_{year}.csv')
    meta=R.load_csv(FREEZE/'templates/D0_FACTOR_METADATA_REGISTRY.csv','factor_id')
    dq=R.load_csv(FREEZE/f'outputs/{year}/DQ_INPUT_{year}.csv','factor_id')
    pack=R.load_packets(FREEZE/f'inputs/FACTOR_EVIDENCE_PACKETS_{year}.jsonl')
    pdir=OUT/f'union_prompts/{year}'; pdir.mkdir(parents=True,exist_ok=True)
    base=[]
    for e in graph.to_dict('records'):
        fi,fj=e['factor_i'],e['factor_j']
        for arm in ['DQ','DQL']:
            if arm=='DQL' and not bool(e['DQL_call_required']): continue
            for order in ['AB','BA']:
                A,B=(fi,fj) if order=='AB' else (fj,fi)
                fa=R.build_factor(A,arm,meta,dq,pack); fb=R.build_factor(B,arm,meta,dq,pack)
                user=(f"{R.COMMON}\n\nSCIENTIFIC QUESTION\n{R.QUESTION}\n\nFACTOR A\n{fa}\n\nFACTOR B\n{fb}\n\n"
                      "Return exactly one token: A, B, T, or U.")
                full=R.SYSTEM+'\n\n'+user
                fn=pdir/f"{e['measurement_edge_id']}_{arm}_{order}.txt"
                fn.write_text(full+'\n',encoding='utf-8')
                base.append({
                    **{k:e[k] for k in e},
                    'edge_id':e['measurement_edge_id'],'arm':arm,'order':order,
                    'display_A':A,'display_B':B,
                    'prompt_file':str(fn.relative_to(OUT)),'prompt_sha256':sha(full)
                })
    b=pd.DataFrame(base)
    b.to_csv(OUT/f'UNION_BASE_PROMPT_SCHEDULE_{year}.csv',index=False)
    calls=[]
    for row in b.to_dict('records'):
        for rep in [1,2,3]:
            x=dict(row); x['measurement_repeat']=rep
            x['measurement_call_id']=f"{row['edge_id']}_{row['arm']}_{row['order']}_R{rep}"
            calls.append(x)
    c=pd.DataFrame(calls)
    seed=f'JKP153_PHASEB_UNION_EXEC_{year}_20260919'
    c['_rand']=c.measurement_call_id.map(lambda x:sha(seed+'|'+x))
    c=c.sort_values('_rand').drop(columns='_rand').reset_index(drop=True)
    c.insert(0,'execution_index',range(1,len(c)+1))
    c.to_csv(OUT/f'UNION_EXECUTION_SCHEDULE_{year}.csv',index=False)
    return {'year':year,'unique_edges':len(graph),'base_prompts':len(b),'calls':len(c),
            'DQ_calls':int((c.arm=='DQ').sum()),'DQL_calls':int((c.arm=='DQL').sum())}

if __name__=='__main__':
    x=[render_year(2024),render_year(2025)]
    (OUT/'UNION_PROMPT_RENDER_SUMMARY.json').write_text(json.dumps(x,indent=2)+'\n')
    print(json.dumps(x,indent=2))
