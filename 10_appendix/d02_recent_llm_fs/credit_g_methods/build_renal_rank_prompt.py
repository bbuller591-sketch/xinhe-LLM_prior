

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
SRC=ROOT/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR/CANDIDATE_UNIVERSE_FROZEN.csv'
OUT=ROOT/'EXISTING_METHOD_COMPARISON_20260925'
g=pd.read_csv(SRC).gene.astype(str).tolist()
assert len(g)==2000
key={f'g{i+1:04d}':x for i,x in enumerate(g)}
lst='\n'.join(f'{i+1}. {x} [g{i+1:04d}]' for i,x in enumerate(g))
p=f'''Select the 50 genes that are most likely to be useful for predicting T-cell-mediated rejection (TCMR) from renal-allograft biopsy gene-expression profiles.

Use only your pretrained biomedical knowledge of kidney transplantation, rejection immunobiology, and gene function. Do not use any training data, retrieved literature, tools, or web access. This is a pure LLM selection baseline.

Return ONLY 50 bracketed keys, one per line, ordered from most to least important. Do not output explanations and do not output more than 50 keys.

CANDIDATE GENES:
{lst}'''
(OUT/'renal_rank_prompt.json').write_text(json.dumps([{'id':'renal_direct_top50','prompt':p}],ensure_ascii=False))
(OUT/'renal_gene_key.json').write_text(json.dumps(key,indent=2))
print(len(g),len(p))
