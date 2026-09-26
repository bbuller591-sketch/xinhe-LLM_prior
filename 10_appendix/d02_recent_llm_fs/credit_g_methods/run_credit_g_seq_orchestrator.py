

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
import json,re,subprocess,shutil
OUT=Path(str(REPRO_ROOT / '10_appendix/d02_recent_llm_fs/credit_g_methods'))
F=[
('checking_status','Status of existing checking account'),('duration','Duration, in months'),
('credit_history','Credit history (credits taken, paid back duly, delays, critical accounts)'),
('purpose','Purpose of the credit (e.g., car, television, education)'),('credit_amount','Credit amount'),
('savings_status','Status of savings accounts/bonds, in Deutsche Mark'),('employment','Number of years spent in current employment'),
('installment_commitment','Installment rate in percentage of disposable income'),('personal_status','Sex and marital status'),
('other_parties','Other debtors/guarantors (none/co-applicant/guarantor)'),('residence_since','Number of years spent in current residence'),
('property_magnitude','Property (e.g., real estate, life insurance)'),('age','Age'),
('other_payment_plans','Other installment plans (bank/stores/none)'),('housing','Housing (rent/own/for free)'),
('existing_credits','Number of existing credits at the bank'),('job','Job'),
('num_dependents','Number of people being liable to provide maintenance for'),
('own_telephone',"Telephone (none/registered under customer's name)"),('foreign_worker','Is a foreign worker (yes/no)')]
D=dict(F); keys=[k for k,_ in F]; selected=[]; logs=[]
for step in range(1,11):
    cand=[k for k in keys if k not in selected]
    l='\n'.join(f'{i+1}. {D[k]} [{k}]' for i,k in enumerate(cand))
    s='\n'.join(f'- {D[k]} [{k}]' for k in selected) if selected else '(none)'
    p=f'''Given a list of features already selected and a list of candidate features available, your task is to output the next feature that should be included to maximally improve the performance in predicting whether an individual carries high credit risk.

Output ONLY one bracketed semantic key from the candidate list and nothing else.

ALREADY SELECTED:
{s}

CANDIDATE FEATURES:
{l}'''
    pf=OUT/f'seq_step_{step:02d}_prompt.json'
    pf.write_text(json.dumps([{'id':f'seq_step_{step:02d}','prompt':p}],indent=2))
    subprocess.run(['python',str(OUT/'direct_llm_driver.py'),str(pf)],check=True,cwd=OUT,stdout=subprocess.DEVNULL)
    resp=json.loads((OUT/'DIRECT_LLM_RESPONSES.json').read_text())
    z=resp['tests'][0]; content=z['content']
    found=[k for k in cand if f'[{k}]' in content or re.search(r'\b'+re.escape(k)+r'\b',content)]
    if not found: raise RuntimeError(f'parse failed step {step}: {content!r}')
    chosen=found[0]; selected.append(chosen)
    shutil.copy2(OUT/'DIRECT_LLM_RESPONSES.json',OUT/f'seq_step_{step:02d}_response.json')
    logs.append({'step':step,'chosen':chosen,'content':content})
(OUT/'CREDIT_G_SEQ_ORDER.json').write_text(json.dumps({'seq_order':selected,'logs':logs},indent=2))
print(json.dumps({'seq_order':selected},indent=2))
