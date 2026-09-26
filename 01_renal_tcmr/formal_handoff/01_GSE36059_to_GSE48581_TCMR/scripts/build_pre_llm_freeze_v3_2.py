#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, json, hashlib, shutil, re, os

ROOT=Path('.')
E=ROOT/'formal_outputs/evidence'
R=E/'research'
D=E/'prefreeze_v3_2_dossiers'
F=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2'
F.mkdir(parents=True,exist_ok=True)

def sha(p):
    p=Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()

def cp(src,dstname=None):
    src=Path(src); dst=F/(dstname or src.name)
    shutil.copy2(src,dst); return dst

# Exact pre-freeze artifacts. Copying does not mutate source/frozen artifacts.
files=[
 ROOT/'formal_outputs/00_r200_refresh/PAIRSET_U_R200_FROZEN.csv',
 R/'V3_2_GLOBAL_PAIRSET_CLEAN_K2_M100.csv',
 R/'V3_2_GLOBAL_RANDOM_STREAM_PREFIX.csv',
 R/'PROPOSED_V3_2_SOURCE_MANIFEST_PRE_FREEZE.csv',
 R/'V3_AUGMENTED_GENE_EVIDENCE_LEDGER_LONG.csv',
 E/'targeted_retrieval/V3_2_LITERATURE_SAFE_SHORTLIST.csv',
 D/'V3_2_PRODUCTION_PAIR_DOSSIERS_PRE_FREEZE.jsonl',
 D/'V3_2_DEEPSEEK_QUERY_MANIFEST_PRE_FREEZE.csv',
 ROOT/'FORMAL_CONFIG.json',
]
for p in files:
    if not p.exists(): raise FileNotFoundError(p)
    cp(p)

q=pd.read_csv(D/'V3_2_DEEPSEEK_QUERY_MANIFEST_PRE_FREEZE.csv',dtype=str,keep_default_na=False)
sel=pd.read_csv(ROOT/'formal_outputs/00_r200_refresh/PAIRSET_U_R200_FROZEN.csv')
glob=pd.read_csv(R/'V3_2_GLOBAL_PAIRSET_CLEAN_K2_M100.csv')
reg=pd.read_csv(R/'PROPOSED_V3_EVIDENCE_REGISTRY_2000.csv')
dossiers=[json.loads(x) for x in (D/'V3_2_PRODUCTION_PAIR_DOSSIERS_PRE_FREEZE.jsonl').read_text().splitlines() if x.strip()]

checks={}
checks['query_rows_400']=len(q)==400
checks['semantic_pairs_200']=q.semantic_pair_id.nunique()==200
checks['two_queries_per_pair']=bool((q.groupby('semantic_pair_id').size()==2).all())
checks['orders_exact_ab_ba']=all(set(z.order)=={'AB','BA'} for _,z in q.groupby('semantic_pair_id'))
checks['provider_prompt_pair_reverse_consistent']=True
for pid,z in q.groupby('semantic_pair_id'):
    ab=z[z.order=='AB'].iloc[0];ba=z[z.order=='BA'].iloc[0]
    if not (ab.gene_A==ba.gene_B and ab.gene_B==ba.gene_A and ab.gene_i==ba.gene_i and ab.gene_j==ba.gene_j):
        checks['provider_prompt_pair_reverse_consistent']=False;break
checks['selective_queries_200']=int((q.arm=='SELECTIVE').sum())==200
checks['global_queries_200']=int((q.arm=='GLOBAL').sum())==200
checks['dossiers_200']=len(dossiers)==200
checks['all_dossiers_eligible']=all(bool(d.get('eligible')) for d in dossiers)
checks['selective_pair_rows_100']=len(sel)==100
checks['global_pair_rows_100']=len(glob)==100

sel_expected=[(str(a),str(b)) for a,b in zip(sel.feature_i,sel.feature_j)]
sel_manifest=[]
for pid,z in q[q.arm=='SELECTIVE'].groupby('semantic_pair_id',sort=False):
    r=z.iloc[0];sel_manifest.append((r.gene_i,r.gene_j))
glob_expected=[(str(a),str(b)) for a,b in zip(glob.feature_i,glob.feature_j)]
glob_manifest=[]
for pid,z in q[q.arm=='GLOBAL'].groupby('semantic_pair_id',sort=False):
    r=z.iloc[0];glob_manifest.append((r.gene_i,r.gene_j))
checks['selective_manifest_exact_pair_order']=sel_manifest==sel_expected
checks['global_manifest_exact_pair_order']=glob_manifest==glob_expected
checks['selective_global_no_overlap']=len({tuple(sorted(x)) for x in sel_expected}&{tuple(sorted(x)) for x in glob_expected})==0

# Clean k2 eligibility from the frozen V3 registry.
cov=dict(zip(reg.gene.astype(str),pd.to_numeric(reg.n_full_families_clean,errors='coerce').fillna(0).astype(int)))
checks['selective_all_clean_k2']=all(cov.get(a,0)>=2 and cov.get(b,0)>=2 for a,b in sel_expected)
checks['global_all_clean_k2']=all(cov.get(a,0)>=2 and cov.get(b,0)>=2 for a,b in glob_expected)

# Prompt contamination / hidden data audit.
forbidden_literal=[
 'GSE36059','GSE48581','GSE21374','GSE98320','GSE124203','GSE275126',
 'p_i_gt_j','incl_i','incl_j','reference rank','external auc','external outcome',
 'U_e','Q_e','B_e'
]
hits={t:int(q.prompt_text.str.contains(re.escape(t),case=False,regex=True).sum()) for t in forbidden_literal}
hits={k:v for k,v in hits.items() if v}
checks['prompt_forbidden_literal_zero']=not hits
checks['prompt_instructs_no_outside_knowledge']=bool(q.prompt_text.str.contains('Use ONLY the evidence below',regex=False).all())
checks['prompt_output_exact_ab']=bool(q.prompt_text.str.contains('exactly one token: A or B',regex=False).all())

# Formal measurement semantics: selective and selective_no_certainty use same SELECTIVE calls.
method={
  'version':'KIDNEY_TCMR_V3_2_PRE_LLM',
  'task':'GSE36059 development TCMR-vs-nonTCMR -> sealed GSE48581 evaluation',
  'formal_pair_gate':'both genes measured in >=2 clean primary independent families',
  'clean_primary_families':['SARWAL_STANFORD_FAMILY','PITTSBURGH_FAMILY','TGC_MULTICENTER_ADULT_GPL13158'],
  'selective_pairset':'PAIRSET_U_R200_FROZEN.csv',
  'global_pairset':'V3_2_GLOBAL_PAIRSET_CLEAN_K2_M100.csv',
  'selective_weight':'lambda * U_e * C_e',
  'm4_noc_weight':'lambda * U_e',
  'global_weight':'lambda * C_e',
  'shared_selective_measurement':'selective and selective_no_certainty-NoC share identical pair set, dossiers, AB/BA prompts, raw responses, p_e and cache; no requery',
  'llm_call_design':'two calls per semantic pair, AB and BA, same evidence with labels swapped',
  'call_probability':'For each call normalize semantic token probability only over A and B from top-logprob masses.',
  'pair_aggregation':'p_e = 0.5 * [ p_AB(A) + p_BA(B) ], where p_BA(B) is probability assigned to semantic gene_i when BA presents gene_j as A and gene_i as B.',
  'certainty':'C_e = 1 - H_binary(p_e)/ln(2), natural-log binary entropy; equivalent to 1-H2(p_e)',
  'prohibited_aggregator':'Do NOT use order-neutralized logit averaging from the GBM experiment.',
  'lambda_grid':[0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1],
  'lambda_selection':'development only: corrected top-k -> StandardScaler + L2 logistic regression C=1 class_weight=balanced -> 5-fold stratified CV mean AUROC; tie smaller lambda',
  'external_seal':'GSE48581 outcome unavailable to routing, evidence, prompts, lambda selection; evaluate only after all choices fixed',
  'recipient_grouping_gate':'PENDING recipient-ID audit before study-specific LLM calls',
  'study_specific_llm_calls_so_far':0
}
(F/'METHOD_FREEZE_PRE_LLM.json').write_text(json.dumps(method,indent=2)+'\n')

# Source data/stat file hashes that materially generate prompt evidence.
source_files=[
 R/'GSE72925_candidate2000_stats.csv',R/'GSE214703_candidate2000_stats.csv',R/'GSE76882_candidate2000_stats.csv',
 R/'GSE25902_PROJECT4_EXACT_candidate2000_stats.csv',R/'GSE120495_candidate2000_stats.csv',
 R/'GSE114712_candidate2000_stats.csv',R/'GSE212160_BHOT_candidate2000_stats.csv',
 R/'EUTRAIN_KTDINNOV_2024_candidate2000_stats.csv',R/'GSE249451_BROAD_REJECTION_candidate2000_stats.csv',
 R/'GSE294632_SCAR_candidate2000_stats.csv'
]
hash_rows=[]
for p in files+source_files+[R/'PROPOSED_V3_EVIDENCE_REGISTRY_2000.csv']:
    hash_rows.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
pd.DataFrame(hash_rows).to_csv(F/'INPUT_SHA256_MANIFEST.csv',index=False)

summary={
 'version':'V3.2_PRE_LLM_STRUCTURAL_AUDIT',
 'checks':checks,
 'failed_checks':[k for k,v in checks.items() if not v],
 'prompt_forbidden_hits':hits,
 'query_manifest_sha256':sha(D/'V3_2_DEEPSEEK_QUERY_MANIFEST_PRE_FREEZE.csv'),
 'selective_pairset_sha256':sha(ROOT/'formal_outputs/00_r200_refresh/PAIRSET_U_R200_FROZEN.csv'),
 'global_pairset_sha256':sha(R/'V3_2_GLOBAL_PAIRSET_CLEAN_K2_M100.csv'),
 'structural_gate_pass':all(checks.values()),
 'recipient_id_gate':'PENDING',
 'deepseek_runtime_gate':'NOT_RUN',
 'study_specific_llm_calls':0,
 'formal_stop_gate':'PENDING_RECIPIENT_ID_AUDIT'
}
(F/'STRUCTURAL_PRE_LLM_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
