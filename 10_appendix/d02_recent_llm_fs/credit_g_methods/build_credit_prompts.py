

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
OUT=Path(str(REPRO_ROOT / '10_appendix/d02_recent_llm_fs/credit_g_methods'))
features=[
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
rows=[]
for k,d in features:
    prompt=f'''For each feature input by the user, your task is to provide a feature importance score (between 0 and 1; larger value indicates greater importance) for predicting whether an individual carries high credit risk and a reasoning behind how the importance score was assigned.

Return ONLY a JSON object with exactly two keys: "reasoning" (string) and "score" (number between 0 and 1).

Provide a score and reasoning for "{d}" formatted according to the output schema above:'''
    rows.append({'id':'score__'+k,'prompt':prompt})
lst='\n'.join(f'{i+1}. {d} [{k}]' for i,(k,d) in enumerate(features))
rank=f'''Given a list of features, rank them according to their importances in predicting whether an individual carries high credit risk. The ranking should be in descending order, starting with the most important feature.

Only output the ranking. Do not output dialogue or explanations for the ranking. Do not exclude any features in the ranking. Use each bracketed semantic key exactly once.

FEATURES:
{lst}'''
rows.append({'id':'rank_all','prompt':rank})
(OUT/'credit_g_score_rank_prompts.json').write_text(json.dumps(rows,indent=2))
print(len(rows))
