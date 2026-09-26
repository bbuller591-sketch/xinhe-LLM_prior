

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
import pandas as pd, hashlib

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
M=WS/'04_METHOD_FREEZE'; C=WS/'05_CONTROLS'
q=pd.read_csv(M/'global_QUERY_MANIFEST.csv')
sent=q[q.repeat_instability_sentinel.astype(bool)][['pair_id','canonical_feature_a','canonical_feature_b']].drop_duplicates().copy()

def h(s): return hashlib.sha256(s.encode()).hexdigest()

rows=[]
for _,r in sent.iterrows():
    a,b=r.canonical_feature_a,r.canonical_feature_b
    for arm in ['RAW_FEATURE_NAMES_GENERIC_TASK','EXPLICIT_BENCHMARK_CONTEXT']:
        for order in ['AB','BA']:
            left,right=(a,b) if order=='AB' else (b,a)
            if arm=='RAW_FEATURE_NAMES_GENERIC_TASK':
                user=f"""Task:
A lender observes applicant information at or before loan origination.
The outcome is whether the borrower later shows bad repayment performance / non-compliance with the credit contract.

Compare the two variable names below using your general pretrained knowledge about consumer-credit risk.
Do not infer a specific named dataset.

A: {left}
B: {right}

Choose A if A has stronger predictive relevance, B if B has stronger predictive relevance, or U if the names are insufficient for a meaningful directional choice.
Output exactly one token: A, B, or U."""
            else:
                user=f"""Diagnostic context:
These variables come from the German Credit / Statlog benchmark, whose task is good-versus-bad credit risk.

Compare the two benchmark variables below using your pretrained knowledge, including any benchmark-specific knowledge you may have.

A: {left}
B: {right}

Choose A if A has stronger predictive relevance in this benchmark, B if B has stronger predictive relevance, or U if you cannot make a meaningful directional choice.
Output exactly one token: A, B, or U."""
            rows.append({
                'query_id':f"CTRL_{arm}_{r.pair_id}_{order}_R0",
                'measurement_family':'POST_FINAL_BENCHMARK_RECALL_CONTROL',
                'arm':arm,'pair_id':r.pair_id,
                'canonical_feature_a':a,'canonical_feature_b':b,
                'presentation_order':order,
                'presented_A_feature':left,'presented_B_feature':right,
                'repeat_index':0,'primary_measurement':True,'repeat_instability_sentinel':False,
                'system_prompt_sha256':'15b989ddfeb369c0018b63d5c140331a8f778dc2d4a49f1338c2a65078477dfa',
                'user_prompt_sha256':h(user),'user_prompt':user,
            })
out=pd.DataFrame(rows)
out.to_csv(M/'POST_FINAL_BENCHMARK_CONTROL_QUERY_MANIFEST.csv',index=False)
doc=f"""# CREDIT-G Post-Final Benchmark Recall Control Freeze

Date: 2026-09-18
Status: frozen after final selected sets and holdout evaluation. This diagnostic cannot alter any reference-selective method, lam, selected set, or holdout result.

Pairs:
- exactly the six broad-graph sentinel pairs already chosen by SHA256 before primary measurement.

Arms:
1. RAW_FEATURE_NAMES_GENERIC_TASK: archive-style feature names, generic lending task, no dataset name.
2. EXPLICIT_BENCHMARK_CONTEXT: German Credit / Statlog named explicitly and archive-style feature names shown.

Comparator:
- the already completed primary construct-only/no-retrieved-evidence calls on the same six sentinel pairs.

Budget:
- 6 pairs × 2 orders × 2 new diagnostic arms = {len(out)} calls.

Purpose:
measure whether raw feature-name or explicit benchmark identity cues materially change pairwise judgments. This is a contamination/recall diagnostic only and is not part of method selection.
"""
(C/'POST_FINAL_BENCHMARK_CONTROL_FREEZE.md').write_text(doc)
print(out.groupby('arm').size())
