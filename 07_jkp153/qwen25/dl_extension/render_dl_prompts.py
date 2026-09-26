

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
import hashlib, json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(str(REPRO_ROOT))
PHASEB=ROOT/'finance_llm_prior/experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
FREEZE=ROOT/'JKP153_FORMAL_MEASUREMENT_DESIGN_FREEZE_V0_4_20260913_201725'
OUT=ROOT/'finance_llm_prior/experiments/jkp153_qwen25_local_formal_20260919/dl_extension'
sys.path.insert(0,str(PHASEB))
import render_selective_prompts as R

def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def render_year(year):
    graph=pd.read_csv(PHASEB/f'MEASUREMENT_UNION_GRAPH_{year}.csv')
    meta=R.load_csv(FREEZE/'templates/D0_FACTOR_METADATA_REGISTRY.csv','factor_id')
    pack=R.load_packets(FREEZE/f'inputs/FACTOR_EVIDENCE_PACKETS_{year}.jsonl')
    pdir=OUT/f'prompts/{year}'; pdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for e in graph.to_dict('records'):
        # No literature on either endpoint -> no DL measurement, zero downstream correction.
        if not bool(e['DQL_call_required']): continue
        fi,fj=str(e['factor_i']),str(e['factor_j'])
        for order in ['AB','BA']:
            A,B=(fi,fj) if order=='AB' else (fj,fi)
            fa=R.factor_base(meta[A])+'\n'+R.lblock(pack[A])
            fb=R.factor_base(meta[B])+'\n'+R.lblock(pack[B])
            user=(f"{R.COMMON}\n\nSCIENTIFIC QUESTION\n{R.QUESTION}\n\nFACTOR A\n{fa}\n\nFACTOR B\n{fb}\n\n"
                  "Return exactly one token: A, B, T, or U.")
            full=R.SYSTEM+'\n\n'+user
            fn=pdir/f"{e['measurement_edge_id']}_DL_{order}.txt"
            fn.write_text(full+'\n',encoding='utf-8')
            rows.append({**e,'arm':'DL','order':order,'display_A':A,'display_B':B,
                         'prompt_file':str(fn.relative_to(OUT)),'prompt_sha256':sha(full)})
    df=pd.DataFrame(rows).reset_index(drop=True)
    df.insert(0,'qwen_execution_index',range(1,len(df)+1))
    df['measurement_repeat']=1
    df['measurement_call_id']=[f'QWEN25_{year}_DL_{i:06d}' for i in range(1,len(df)+1)]
    df.to_csv(OUT/f'DL_EXECUTION_SCHEDULE_{year}.csv',index=False)
    return {'year':year,'calls':len(df),'edges':df.measurement_edge_id.nunique(),
            'AB':int((df.order=='AB').sum()),'BA':int((df.order=='BA').sum())}
if __name__=='__main__':
    s=[render_year(2024),render_year(2025)]
    (OUT/'DL_PROMPT_RENDER_SUMMARY.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(s,indent=2))
