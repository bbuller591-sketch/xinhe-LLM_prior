

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
import json,re
OUT=Path(str(REPRO_ROOT / '10_appendix/d02_recent_llm_fs/credit_g_methods'))
key=json.loads((OUT/'renal_gene_key.json').read_text())
resp=json.loads((OUT/'RENAL_DIRECT_LLM_RESPONSES.json').read_text())['tests'][0]['content']
ks=re.findall(r'\[(g\d{4})\]',resp)
seen=[]; dups=[]
for k in ks:
    if k in seen:dups.append(k)
    else:seen.append(k)
need=50-len(seen)
remain=[k for k in key if k not in seen]
lst='\n'.join(f'{key[k]} [{k}]' for k in remain)
p=f'''Your previous pure-LLM renal TCMR top-50 selection contained duplicate keys, leaving {len(seen)} unique genes. Select exactly {need} additional genes from the remaining candidate list to complete the top-50 set.

Use only pretrained biomedical knowledge. Do not use tools, web, retrieved literature, or data. Do not choose any key already selected.

Return ONLY {need} bracketed keys, one per line, with no explanation.

ALREADY SELECTED KEYS:
{", ".join(seen)}

REMAINING CANDIDATES:
{lst}'''
(OUT/'renal_repair_prompt.json').write_text(json.dumps([{'id':'renal_repair','prompt':p}]))
(OUT/'renal_unique_initial.json').write_text(json.dumps({'initial_keys':ks,'unique_keys':seen,'duplicates':dups,'need':need},indent=2))
print(len(ks),len(seen),dups,need,len(p))
