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
import pandas as pd,json

O=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion/artifacts/SAFE_BORROWING_AUDIT'))
O.mkdir(parents=True,exist_ok=True)
rows=[
{'dataset':'Renal','configuration':'selective k=50','trust':'lambda=0.3','development_delta_auroc':0.0620,'final_delta_auroc':0.0260028,'final_scope':'independent external','semantic_null_p':0.251,'random_null_p':0.14685,'zero_trust_available':True,'warning':'development gain not separated from semantic/random null; external gain CI includes zero'},
{'dataset':'GSE272769','configuration':'ElasticNet k=50','trust':'outer-fold lam=[0,10,10,0,0.3]','development_delta_auroc':None,'final_delta_auroc':0.0266667,'final_scope':'strict nested outer CV','semantic_null_p':0.01990,'random_null_p':0.01998,'zero_trust_available':True,'warning':'strongest semantic/random separation; no independent external cohort'},
{'dataset':'Breast','configuration':'LASSO k=10','trust':'lam=0.03','development_delta_auroc':0.0059004,'final_delta_auroc':0.0287415,'final_scope':'sealed external','semantic_null_p':0.05495,'random_null_p':0.02498,'zero_trust_available':True,'warning':'positive final; semantic-shuffle separation borderline'},
{'dataset':'Breast','configuration':'LASSO k=20','trust':'lam=0.3','development_delta_auroc':0.0092288,'final_delta_auroc':-0.0069728,'final_scope':'sealed external','semantic_null_p':0.64735,'random_null_p':0.58042,'zero_trust_available':True,'warning':'development-selected borrowing harmed final AUROC'},
{'dataset':'Breast','configuration':'ElasticNet k=10','trust':'lam=0.1','development_delta_auroc':0.0236085,'final_delta_auroc':0.0069728,'final_scope':'sealed external','semantic_null_p':0.21778,'random_null_p':0.02597,'zero_trust_available':True,'warning':'beats random probabilities but not semantic reassignment'},
{'dataset':'Breast','configuration':'ElasticNet k=20','trust':'lam=0.03','development_delta_auroc':0.0064286,'final_delta_auroc':-0.0142857,'final_scope':'sealed external','semantic_null_p':0.97902,'random_null_p':0.05395,'zero_trust_available':True,'warning':'development-selected borrowing harmed final AUROC'},
{'dataset':'Breast','configuration':'SIS k=10','trust':'lam=0.03','development_delta_auroc':0.0068268,'final_delta_auroc':0.0192177,'final_scope':'sealed external','semantic_null_p':0.24176,'random_null_p':0.67932,'zero_trust_available':True,'warning':'positive final without null separation'},
{'dataset':'Breast','configuration':'SIS k=20','trust':'lam=3','development_delta_auroc':0.0094910,'final_delta_auroc':0.0147959,'final_scope':'sealed external','semantic_null_p':0.09890,'random_null_p':0.44056,'zero_trust_available':True,'warning':'positive final; large trust and weak null separation'},
{'dataset':'CREDIT-G','configuration':'GBM-permutation k=10 legacy selective','trust':'lam=0.3','development_delta_auroc':0.0006569,'final_delta_auroc':0.0132143,'final_scope':'locked holdout','semantic_null_p':1.0,'random_null_p':0.32168,'zero_trust_available':True,'warning':'one eligible actionable semantic pair; no semantic-null separation'},
{'dataset':'CREDIT-G','configuration':'Elastic Net k=10 legacy selective','trust':'lam=0.3','development_delta_auroc':0.0020824,'final_delta_auroc':0.0075,'final_scope':'locked holdout','semantic_null_p':0.68731,'random_null_p':0.46154,'zero_trust_available':True,'warning':'three eligible pairs; no semantic-null separation'},
{'dataset':'Hospital Osteoporosis','configuration':'selective k=5','trust':'lam=10','development_delta_auroc':0.0141,'final_delta_auroc':-0.0036630,'final_scope':'temporal external','semantic_null_p':None,'random_null_p':None,'zero_trust_available':True,'warning':'development-selected upper-grid borrowing harmed temporal AUROC/AUPRC'},
{'dataset':'Hospital Osteoporosis','configuration':'selective k=10','trust':'lam=0.03','development_delta_auroc':0.0020179,'final_delta_auroc':0.0087141,'final_scope':'temporal external','semantic_null_p':0.98202,'random_null_p':0.91409,'zero_trust_available':True,'warning':'positive final despite development nulls not separating; external post-hoc semantic-null p≈0.044'},
{'dataset':'Darmanis GBM','configuration':'SIS k=10','trust':'lam=1','development_delta_auroc':0.0340322,'final_delta_auroc':None,'final_scope':'internal plate-grouped only','semantic_null_p':0.19980,'random_null_p':0.17782,'zero_trust_available':True,'warning':'internal/leakage-sensitive routing; no independent final validation'},
{'dataset':'Darmanis GBM','configuration':'SIS k=20','trust':'lam=0.3','development_delta_auroc':0.0254739,'final_delta_auroc':None,'final_scope':'internal plate-grouped only','semantic_null_p':0.22877,'random_null_p':0.21279,'zero_trust_available':True,'warning':'internal/leakage-sensitive routing; no independent final validation'},
{'dataset':'Darmanis GBM','configuration':'Elastic Net k=10','trust':'lam=1','development_delta_auroc':0.0195657,'final_delta_auroc':None,'final_scope':'internal plate-grouped only','semantic_null_p':0.27872,'random_null_p':0.32068,'zero_trust_available':True,'warning':'internal/leakage-sensitive routing; no independent final validation'}
]
df=pd.DataFrame(rows)
df.to_csv(O/'SAFE_BORROWING_AUDIT.csv',index=False)
md=['# Development-only safe-borrowing audit','',
'This audit does **not** redefine selective. It checks whether the existing development-only trust selection (with zero borrowing available) is already sufficient to justify a universal safety claim. Final outcomes are used only descriptively, never to choose a new rule.','',
'## Result','',
'No universal safe-borrowing gate is justified by the current six-dataset evidence. Zero trust is available in the existing grids and is sometimes selected, but positive development utility does not reliably imply positive final utility. Breast LASSO-k20, Breast ElasticNet-k20 and Hospital-k5 are direct counterexamples. Conversely, several positive final results do not separate from semantic/random nulls.','',
'Therefore the current method should remain unchanged. Any new gate (for example requiring a minimum development gain, null separation, or a cap on trust) must be predeclared and validated on new development/external tasks rather than tuned on these already-observed final outcomes.','',
'## Configuration audit','',
'| Dataset | Configuration | Trust | Dev ΔAUROC | Final ΔAUROC | Final scope | Semantic p | Random p | Warning |',
'|---|---|---|---:|---:|---|---:|---:|---|']
for r in rows:
    def f(x):
        return '—' if x is None else f'{x:.4f}'
    md.append(f"| {r['dataset']} | {r['configuration']} | {r['trust']} | {f(r['development_delta_auroc'])} | {f(r['final_delta_auroc'])} | {r['final_scope']} | {f(r['semantic_null_p'])} | {f(r['random_null_p'])} | {r['warning']} |")
md += ['','## Audit boundary','',
'- The audit supports keeping **lam/lam = 0** in every trust grid and reporting when validation rejects borrowing.',
'- It does **not** support selecting a new threshold from these final outcomes.',
'- Large trust values at a grid boundary should be reported as a sensitivity warning rather than interpreted as stronger semantic evidence.',
'- Internal-CV results (especially Darmanis GBM) are not a substitute for patient-level or independent external validation.'
]
(O/'SAFE_BORROWING_AUDIT.md').write_text('\n'.join(md))
print(df.to_string(index=False))
