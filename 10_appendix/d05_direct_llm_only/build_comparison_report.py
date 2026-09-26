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
import pandas as pd
import numpy as np

ROOT=Path(str(REPRO_ROOT))
P=ROOT/'PURE_LLM_ALL_NON_JKP_20260926'
C=ROOT/'CROSS_DATASET_RECENT_LLM_FS_20260926/SUMMARY/ALL_AUROC_RESULTS.csv'
llm=pd.read_csv(P/'SUMMARY/PURE_LLM_ALL_NON_JKP_SUMMARY.csv')
allr=pd.read_csv(C)

rows=[]

def llmrow(dataset,k):
    return llm[(llm.dataset==dataset)&(llm.k==k)].iloc[0]

def add(dataset,config,k,reference,ours):
    r=llmrow(dataset,k)
    rows.append({
        'dataset':dataset,'configuration':config,'k':k,
        'llm_only_mean_3runs':float(r.auroc_mean_3runs),
        'llm_only_sd_3runs':float(r.auroc_sd_3runs),
        'llm_only_score_ensemble':float(r.auroc_score_ensemble),
        'reference_auroc':float(reference),
        'ours_auroc':float(ours),
        'ours_minus_llm_mean':float(ours-r.auroc_mean_3runs),
        'ours_minus_llm_ensemble':float(ours-r.auroc_score_ensemble),
        'ours_minus_reference':float(ours-reference),
    })

# Single-configuration datasets
for dataset,k,refname,oursname,config in [
    ('CREDIT-G',10,'Reference','Selective Correction (ours)','k=10'),
    ('Osteoporosis',10,'Reference','Selective Correction (ours)','k=10'),
    ('GSE272769 Sepsis',50,'Reference','Selective Correction (ours)','k=50'),
    ('Breast pCR',20,'Reference (SIS)','Selective Correction (SIS)','SIS-20'),
    ('Renal TCMR',50,'Reference','Selective Correction (ours)','k=50'),
]:
    rr=allr[(allr.dataset==dataset)&(allr.k==k)&(allr.method==refname)].iloc[0]
    oo=allr[(allr.dataset==dataset)&(allr.k==k)&(allr.method==oursname)].iloc[0]
    add(dataset,config,k,rr.auroc,oo.auroc)

# Darmanis: same pure LLM top-k baseline compared to every paper-facing selector configuration at that k.
for k in [10,20]:
    for sel in ['SIS','LASSO','ELASTICNET']:
        rr=allr[(allr.dataset=='Darmanis GBM')&(allr.k==k)&(allr.method==f'Reference ({sel})')].iloc[0]
        oo=allr[(allr.dataset=='Darmanis GBM')&(allr.k==k)&(allr.method==f'Selective Correction ({sel})')].iloc[0]
        add('Darmanis GBM',f'{sel}-{k}',k,rr.auroc,oo.auroc)

out=pd.DataFrame(rows)
out=out.sort_values(['dataset','k','configuration']).reset_index(drop=True)
out.to_csv(P/'SUMMARY/PURE_LLM_REFERENCE_OURS_COMPARISON.csv',index=False)

lines=[
'# LLM-only vs Reference vs Selective Correction',
'',
'Metric: AUROC.',
'',
'LLM-only = DeepSeek pointwise semantic scoring using only task description + feature identity, with no dataset values, fitted statistics, retrieval, web, or tools.',
'Three independent LLM scoring runs are shown through mean ± SD; score ensemble averages the three feature scores before selecting top-k.',
'',
'| Dataset | Configuration | LLM-only mean ± SD | LLM-only score ensemble | Reference | Ours | Ours − LLM ensemble | Ours − Reference |',
'|---|---|---:|---:|---:|---:|---:|---:|'
]
for r in out.itertuples():
    lines.append(f'| {r.dataset} | {r.configuration} | {r.llm_only_mean_3runs:.4f} ± {r.llm_only_sd_3runs:.4f} | {r.llm_only_score_ensemble:.4f} | {r.reference_auroc:.4f} | {r.ours_auroc:.4f} | {r.ours_minus_llm_ensemble:+.4f} | {r.ours_minus_reference:+.4f} |')
lines += [
'',
'Notes:',
'- For Darmanis, the LLM-only support depends on k but not on SIS/LASSO/ElasticNet, so the same LLM-only number is compared against each paper-facing selector configuration at the same k.',
'- JKP is intentionally excluded.',
'- Do not cherry-pick the best stochastic LLM-only run; use either mean ± SD or the pre-specified 3-run score ensemble.'
]
(P/'SUMMARY/COMPARISON_REPORT.md').write_text('\n'.join(lines)+'\n')
print(out.to_string(index=False))
