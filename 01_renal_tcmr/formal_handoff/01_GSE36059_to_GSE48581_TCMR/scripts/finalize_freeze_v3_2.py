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
import shutil,json,hashlib,pandas as pd
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
F=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2'
E=ROOT/'formal_outputs/evidence'
R=E/'research'
D=E/'prefreeze_v3_2_dossiers'
A=ROOT/'formal_outputs/recipient_audit'

mapping={
 ROOT/'formal_outputs/00_r200_refresh/PAIRSET_U_R200_FROZEN.csv':'SELECTIVE_PAIRSET_FROZEN_V3_2.csv',
 R/'V3_2_GLOBAL_PAIRSET_CLEAN_K2_M100.csv':'GLOBAL_PAIRSET_FROZEN_V3_2.csv',
 R/'V3_2_GLOBAL_RANDOM_STREAM_PREFIX.csv':'GLOBAL_RANDOM_STREAM_PREFIX_FROZEN_V3_2.csv',
 R/'PROPOSED_V3_2_SOURCE_MANIFEST_PRE_FREEZE.csv':'EVIDENCE_SOURCE_MANIFEST_FROZEN_V3_2.csv',
 R/'V3_AUGMENTED_GENE_EVIDENCE_LEDGER_LONG.csv':'GENE_EVIDENCE_LEDGER_FROZEN_V3_2.csv',
 E/'targeted_retrieval/V3_2_LITERATURE_SAFE_SHORTLIST.csv':'LITERATURE_CONTEXT_SHORTLIST_FROZEN_V3_2.csv',
 D/'V3_2_PRODUCTION_PAIR_DOSSIERS_PRE_FREEZE.jsonl':'PAIR_DOSSIERS_FROZEN_V3_2.jsonl',
 D/'V3_2_DEEPSEEK_QUERY_MANIFEST_PRE_FREEZE.csv':'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv',
 A/'RECIPIENT_ID_AUDIT.json':'RECIPIENT_ID_AUDIT_FROZEN.json',
 A/'RECIPIENT_ID_AUDIT.md':'RECIPIENT_ID_AUDIT_FROZEN.md',
 ROOT/'FORMAL_CONFIG.json':'ORIGINAL_FORMAL_CONFIG.json',
}
for src,name in mapping.items():
    if not src.exists():raise FileNotFoundError(src)
    shutil.copy2(src,F/name)

# Promote source manifest status wording only in frozen copy: no source content or evidence values changed.
sm=F/'EVIDENCE_SOURCE_MANIFEST_FROZEN_V3_2.csv'
sdf=pd.read_csv(sm,dtype=str,keep_default_na=False)
# Add freeze metadata columns without altering source rows.
sdf['freeze_version']='V3.2'
sdf['freeze_date']='2026-09-21'
sdf['formal_pair_eligibility_role']=sdf['status'].apply(
    lambda x:'PRIMARY_CLEAN_ELIGIBILITY' if x in ['CLEAN_PRIMARY_QUANTITATIVE_CANONICAL','CLEAN_PRIMARY_QUANTITATIVE']
    else 'SUPPORT_ONLY_OR_EXCLUDED')
sdf.to_csv(sm,index=False)

method={
 'freeze_version':'KIDNEY_TCMR_V3_2',
 'freeze_date':'2026-09-21',
 'task':'GSE36059 development TCMR vs non-TCMR -> sealed GSE48581 external evaluation',
 'candidate_universe':'2000 frozen X-only genes',
 'r200':{'R':200,'subsample_fraction':0.8,'stratification':'phenotype-row stratified','seed':20261044,'top_k':100,
         'recipient_grouping':'not available in public metadata; row-level by frozen contract'},
 'formal_pair_gate':'both genes measured in >=2 clean primary independent families',
 'clean_primary_families':['SARWAL_STANFORD_FAMILY','PITTSBURGH_FAMILY','TGC_MULTICENTER_ADULT_GPL13158'],
 'production_support_policy':{
   'same_family_replication':['GSE25902_PROJECT4_EXACT','GSE120495'],
   'small_direct_support':['GSE114712'],
   'large_targeted_panel_support':['GSE212160'],
   'significant_only_support':['EU_TRAIN_KTDINNOV_2024'],
   'broad_rejection_support':['GSE249451','GSE294632'],
   'target_aware_excluded_from_production_prompt':['GSE232825','GSE131179']
 },
 'arms':{
   'Reference':'frozen primary Elastic Net full-development fit; raw score abs(beta)+1e-8*SIS, max-normalized',
   'selective':'Selective frozen 100 pairs; weight lambda*U_e*C_e',
   'selective_no_certainty-NoC':'same exact Selective pairs/evidence/AB-BA calls/p_e/raw cache as selective; weight lambda*U_e',
   'Global':'frozen random 100 clean-K2 pairs from seed 20261044; weight lambda*C_e'
 },
 'llm':{
   'requested_model':'deepseek-flash','thinking':'disabled','temperature':1.0,'max_tokens':4,'logprobs':True,'top_logprobs':20,
   'calls_per_pair':2,'orders':['AB','BA'],'production_output_tokens':['A','B'],
   'probability_normalization':'normalize semantic token mass over A vs B within each call',
   'pair_probability':'p_e = 0.5 * [p_AB(A) + p_BA(B)]',
   'certainty':'C_e = 1 - H_binary(p_e)/ln(2)',
   'forbidden_aggregator':'order-neutralized/logit-neutralized aggregation from prior GBM experiment'
 },
 'correction':'latest paper Eq.(4), reference-anchored cross-entropy correction; no old Bradley-Terry score fusion',
 'lambda_grid':[0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1],
 'lambda_selection':'each arm separately, development-only corrected top-k; StandardScaler + L2 logistic C=1 class_weight=balanced; 5-fold stratified CV; mean AUROC; ties smaller lambda',
 'external_seal':'No GSE48581 outcome may affect evidence, routing, pair set, prompts, lambda, top-k, or any selection. External outcome evaluated only after all choices are fixed.',
 'recipient_id_limitation':'403 development biopsies arise from315 recipients and 300 external biopsies from264 patients, but public biopsy-to-recipient mapping is unavailable. Do not claim patient-grouped resampling/CV.',
 'semantic_shuffle':'500-1000 cached-measurement shuffles only if primary selective or selective_no_certainty selects nonzero lambda',
 'study_specific_llm_calls_at_freeze':0
}
(F/'METHOD_CONFIG_FROZEN_V3_2.json').write_text(json.dumps(method,indent=2)+'\n')

# Verify query/pair invariants from frozen copies.
q=pd.read_csv(F/'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv',dtype=str,keep_default_na=False)
sel=pd.read_csv(F/'SELECTIVE_PAIRSET_FROZEN_V3_2.csv')
glo=pd.read_csv(F/'GLOBAL_PAIRSET_FROZEN_V3_2.csv')
assert len(q)==400 and q.semantic_pair_id.nunique()==200
assert len(sel)==100 and len(glo)==100
assert q[q.arm=='SELECTIVE'].semantic_pair_id.nunique()==100
assert q[q.arm=='GLOBAL'].semantic_pair_id.nunique()==100
assert all(set(z.order)=={'AB','BA'} for _,z in q.groupby('semantic_pair_id'))
assert not q.prompt_text.str.contains('GSE36059|GSE48581|GSE21374|GSE98320|GSE124203|GSE275126',case=False,regex=True).any()

# Hash everything in freeze package except generated checksum/gate files.
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for p in sorted(F.iterdir()):
    if p.name in {'SHA256SUMS_FROZEN.csv','STOP_GATE.json'}:continue
    if p.is_file():rows.append({'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
pd.DataFrame(rows).to_csv(F/'SHA256SUMS_FROZEN.csv',index=False)

rec=json.loads((F/'RECIPIENT_ID_AUDIT_FROZEN.json').read_text())
struct=json.loads((F/'STRUCTURAL_PRE_LLM_AUDIT.json').read_text())
gate={
 'version':'STOP_GATE_V3_2',
 'status':'PASS_FOR_GENERIC_RUNTIME_SMOKE_ONLY',
 'study_specific_llm_calls_allowed':False,
 'generic_smoke_allowed':True,
 'structural_gate':bool(struct.get('structural_gate_pass')),
 'recipient_id_gate':rec.get('formal_decision'),
 'evidence_source_manifest_frozen':True,
 'pairsets_frozen':True,
 'query_manifest_frozen':True,
 'method_config_frozen':True,
 'sha_manifest_frozen':True,
 'runtime_model_fingerprint':'PENDING_GENERIC_SMOKE',
 'study_specific_llm_calls_so_far':0,
 'query_manifest_sha256':sha(F/'DEEPSEEK_QUERY_MANIFEST_FROZEN_V3_2.csv'),
 'selective_pairset_sha256':sha(F/'SELECTIVE_PAIRSET_FROZEN_V3_2.csv'),
 'global_pairset_sha256':sha(F/'GLOBAL_PAIRSET_FROZEN_V3_2.csv'),
 'notes':['Recipient dependence is a documented public-ID limitation, not silently assumed absent.',
          'Generic smoke contains no study genes/evidence and is the only permitted API action before final PASS.',
          'After smoke, freeze provider model/system fingerprint and regenerate this gate as PASS before formal calls.']
}
assert gate['structural_gate'] and gate['recipient_id_gate']=='PASS_WITH_EXPLICIT_PUBLIC_ID_LIMITATION'
(F/'STOP_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps(gate,indent=2))
print('\nFrozen files:')
print(pd.read_csv(F/'SHA256SUMS_FROZEN.csv').to_string(index=False))
