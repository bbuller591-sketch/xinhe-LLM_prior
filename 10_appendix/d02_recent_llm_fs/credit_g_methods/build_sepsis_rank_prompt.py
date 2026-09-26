

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
import json,pandas as pd
ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'EXISTING_METHOD_COMPARISON_20260925'
d=pd.read_csv(ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/01_FROZEN_DATA/features_p1500.csv')
g=d.gene_symbol.astype(str).tolist(); assert len(g)==1500
key={f'g{i+1:04d}':x for i,x in enumerate(g)}
lst='\n'.join(f'{i+1}. {x} [g{i+1:04d}]' for i,x in enumerate(g))
p=f'''Select the 50 genes that are most likely to be useful for predicting 30-day mortality from adult ICU sepsis whole-blood transcriptomic profiles.

Use only your pretrained biomedical knowledge of sepsis pathophysiology, immune dysregulation, organ injury, and gene function. Do not use any training data, retrieved literature, tools, or web access. This is a pure LLM selection baseline.

Return ONLY 50 bracketed keys, one per line, ordered from most to least important. Do not output explanations and do not output more than 50 keys.

CANDIDATE GENES:
{lst}'''
(OUT/'sepsis_rank_prompt.json').write_text(json.dumps([{'id':'sepsis_direct_top50','prompt':p}],ensure_ascii=False))
(OUT/'sepsis_gene_key.json').write_text(json.dumps(key,indent=2))
print(len(g),len(p))
