

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
import json
ROOT=Path(str(REPRO_ROOT))
man=json.loads((ROOT/'hospital_osteoporosis_dataonly_pilot_20260917/03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json').read_text())
fs=man['selectable_features']
lst='\n'.join(f'{i+1}. {f} [f{i+1:02d}]' for i,f in enumerate(fs))
p=f'''Given a list of clinical variables, rank them according to their importances for predicting whether a patient-site observation has low bone mineral density / a low T-score consistent with osteoporosis risk. The skeletal site itself is handled separately as context and should not be selected here.

The ranking should be in descending order, starting with the most important variable. Use only general medical knowledge; do not use any training data, retrieved literature, tools, or web access.

Only output the ranking. Do not output dialogue or explanations. Do not exclude any variables. Use each bracketed key exactly once.

VARIABLES:
{lst}'''
(Path(str(REPRO_ROOT / '10_appendix/d02_recent_llm_fs/credit_g_methods'))/'osteoporosis_rank_prompt.json').write_text(json.dumps([{'id':'osteoporosis_rank_all','prompt':p}],indent=2,ensure_ascii=False))
(Path(str(REPRO_ROOT / '10_appendix/d02_recent_llm_fs/credit_g_methods'))/'osteoporosis_feature_key.json').write_text(json.dumps({f'f{i+1:02d}':f for i,f in enumerate(fs)},indent=2,ensure_ascii=False))
print(len(fs))
