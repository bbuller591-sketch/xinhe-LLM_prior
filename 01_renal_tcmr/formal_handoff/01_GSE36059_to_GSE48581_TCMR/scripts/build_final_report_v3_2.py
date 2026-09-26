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
import json,hashlib,pandas as pd,numpy as np
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
PRE=ROOT/'formal_outputs/04_pre_external_final_freeze_v3_2'
MEAS=ROOT/'formal_outputs/measurements/v3_2'
DOWN=ROOT/'formal_outputs/02_downstream_dev_v3_2'
SHUF=ROOT/'formal_outputs/03_semantic_shuffle_v3_2'
EXT=ROOT/'formal_outputs/05_external_evaluation_v3_2'
EVID=ROOT/'formal_outputs/01_pre_llm_freeze_v3_2/FINAL'
OUT=ROOT/'formal_outputs/06_final_report_v3_2';OUT.mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

mf=pd.read_csv(PRE/'PRE_EXTERNAL_FILE_HASHES.csv')
integrity=[]
for r in mf.itertuples():
    p=ROOT/r.path
    got=sha(p) if p.exists() else 'MISSING'
    integrity.append({'path':r.path,'expected_sha256':r.sha256,'current_sha256':got,'match':got==r.sha256})
idf=pd.DataFrame(integrity);idf.to_csv(OUT/'PRE_EXTERNAL_INTEGRITY_RECHECK.csv',index=False)
assert idf.match.all()

raw=[json.loads(line) for line in (MEAS/'FORMAL_DEEPSEEK_V3_2_RAW.jsonl').read_text().splitlines() if line.strip()]
assert len(raw)==400 and len({x['query_id'] for x in raw})==400
prompt_tokens=sum(int((x.get('usage') or {}).get('prompt_tokens',0)) for x in raw)
completion_tokens=sum(int((x.get('usage') or {}).get('completion_tokens',0)) for x in raw)
cache_hit=sum(int((x.get('usage') or {}).get('prompt_cache_hit_tokens',0)) for x in raw)
cache_miss=sum(int((x.get('usage') or {}).get('prompt_cache_miss_tokens',0)) for x in raw)
calls_by_arm=pd.Series([x['arm'] for x in raw]).value_counts().to_dict()
first_tok=pd.Series([x['first_token'] for x in raw]).value_counts().to_dict()
fps=sorted({x['system_fingerprint'] for x in raw});models=sorted({x['provider_model'] for x in raw})

pairs=pd.read_csv(MEAS/'FORMAL_PAIR_MEASUREMENTS_SIMPLE_ABBA_V3_2.csv')
pair_stats={}
for arm,z in pairs.groupby('arm'):
    pair_stats[arm]={
      'n_pairs':int(len(z)),
      'unique_genes':int(len(set(z.gene_i)|set(z.gene_j))),
      'semantic_choice_consistency_rate':float(z.semantic_choice_consistent.mean()),
      'mean_C_e':float(z.C_e.mean()),'median_C_e':float(z.C_e.median()),
      'C_e_q10':float(z.C_e.quantile(.1)),'C_e_q90':float(z.C_e.quantile(.9))
    }

dossiers=[json.loads(x) for x in (EVID/'PAIR_DOSSIERS_FROZEN_V3_2.jsonl').read_text().splitlines() if x.strip()]
assert len(dossiers)==200
source_manifest=pd.read_csv(EVID/'EVIDENCE_SOURCE_MANIFEST_FROZEN_V3_2.csv',dtype=str,keep_default_na=False)
lit=pd.read_csv(EVID/'LITERATURE_CONTEXT_SHORTLIST_FROZEN_V3_2.csv',dtype=str,keep_default_na=False)
evid_stats={
 'formal_pair_gate':'both endpoints measured in >=2 clean primary independence families',
 'all_200_pairs_pass_gate':bool(all(d['eligible'] for d in dossiers)),
 'clean_primary_families':['SARWAL_STANFORD_FAMILY','PITTSBURGH_FAMILY','TGC_MULTICENTER_ADULT_GPL13158'],
 'frozen_source_manifest_rows':int(len(source_manifest)),
 'source_status_counts':source_manifest.status.value_counts().to_dict(),
 'literature_context_rows':int(len(lit)),
 'literature_context_genes':int(lit.gene.nunique()),
 'literature_context_unique_pmids':int(lit.pmid.replace('',np.nan).dropna().nunique()),
 'target_aware_sources_excluded_from_production_prompt':['GSE232825','GSE131179'],
 'eligibility_not_based_on_significance_or_llm_certainty':True
}

dev=pd.read_csv(DOWN/'FOLDLOCAL_LAMBDA_CV_AGGREGATE.csv')
sel=json.loads((DOWN/'PRE_EXTERNAL_FREEZE_SUMMARY.json').read_text())
ext_ci=pd.read_csv(EXT/'EXTERNAL_METRICS_WITH_BIOPSY_BOOTSTRAP_CI.csv')
overlap=pd.read_csv(DOWN/'SUPPORT_OVERLAP.csv')
shuffle=json.loads((SHUF/'SEMANTIC_SHUFFLE_SUMMARY.json').read_text())

rows=[]
for m in ['Reference','Global','selective','selective_no_certainty-NoC']:
    lam=float(sel['selected_lambda'][m])
    drow=dev[(dev.method==m)&np.isclose(dev['lambda'],lam)].iloc[0]
    er=ext_ci[ext_ci.method==m].iloc[0]
    orow=overlap[overlap.method==m].iloc[0]
    row={
      'method':m,'selected_lambda':lam,'lambda_fallback_zero':bool(lam==0),
      'development_cv_auroc':float(drow.auroc),
      'development_cv_auprc':float(drow.auprc),
      'development_cv_balanced_accuracy':float(drow.balanced_accuracy),
      'external_auroc':float(er.auroc),
      'external_auroc_ci025_biopsy':float(er.auroc_ci025),
      'external_auroc_ci975_biopsy':float(er.auroc_ci975),
      'external_auprc':float(er.auprc),
      'external_auprc_ci025_biopsy':float(er.auprc_ci025),
      'external_auprc_ci975_biopsy':float(er.auprc_ci975),
      'external_balanced_accuracy':float(er.balanced_accuracy),
      'external_balanced_accuracy_ci025_biopsy':float(er.balanced_accuracy_ci025),
      'external_balanced_accuracy_ci975_biopsy':float(er.balanced_accuracy_ci975),
      'support_intersection_reference':int(orow.intersection_reference),
      'support_jaccard_reference':float(orow.jaccard_reference)
    }
    if m!='Reference':
        row.update({
          'external_delta_auroc_vs_reference':float(er.delta_auroc),
          'external_delta_auroc_ci025_biopsy':float(er.delta_auroc_ci025),
          'external_delta_auroc_ci975_biopsy':float(er.delta_auroc_ci975),
          'external_delta_auprc_vs_reference':float(er.delta_auprc),
          'external_delta_auprc_ci025_biopsy':float(er.delta_auprc_ci025),
          'external_delta_auprc_ci975_biopsy':float(er.delta_auprc_ci975),
        })
    rows.append(row)
final=pd.DataFrame(rows);final.to_csv(OUT/'FINAL_REFERENCE_GLOBAL_selective_selective_no_certainty_TABLE.csv',index=False)

ref=final[final.method=='Reference'].iloc[0]
gl=final[final.method=='Global'].iloc[0]
selective=final[final.method=='selective'].iloc[0]
m4=final[final.method=='selective_no_certainty-NoC'].iloc[0]
headline=(
  f"On sealed GSE48581, LLM-guided supports produced modest AUROC gains over the frozen Reference "
  f"(Reference {ref.external_auroc:.3f}; Global {gl.external_auroc:.3f}; selective {selective.external_auroc:.3f}; "
  f"selective_no_certainty-NoC {m4.external_auroc:.3f}), but paired biopsy-level bootstrap intervals for the gains included zero "
  f"and cached semantic-shuffle diagnostics did not show that the observed selective/selective_no_certainty development gains exceeded "
  f"random semantic reassignment (selective p={shuffle['selective']['empirical_p_retuned']:.3f}; "
  f"selective_no_certainty p={shuffle['selective_no_certainty-NoC']['empirical_p_retuned']:.3f}). The result supports a reproducible external signal "
  f"from the frozen guidance pipeline, not yet a specific claim that correct LLM semantics caused the gain."
)
(OUT/'PAPER_READY_HEADLINE.txt').write_text(headline+'\n')

audit={
 'version':'KIDNEY_TCMR_FINAL_AUDIT_V3_2','status':'COMPLETE',
 'formal_llm_calls':400,'additional_llm_calls_after_formal_measurement':0,
 'provider_models':models,'system_fingerprints':fps,
 'calls_by_arm':calls_by_arm,'first_token_counts':first_tok,
 'tokens':{'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,
           'total_tokens':prompt_tokens+completion_tokens,'prompt_cache_hit_tokens':cache_hit,'prompt_cache_miss_tokens':cache_miss},
 'pair_measurement_stats':pair_stats,'evidence_stats':evid_stats,
 'selected_lambda':sel['selected_lambda'],
 'guided_arms_fell_back_to_lambda_zero':{m:bool(float(sel['selected_lambda'][m])==0) for m in ['Global','selective','selective_no_certainty-NoC']},
 'semantic_shuffle':{
   'n':1000,'selective_empirical_p_retuned':shuffle['selective']['empirical_p_retuned'],
   'selective_no_certainty_empirical_p_retuned':shuffle['selective_no_certainty-NoC']['empirical_p_retuned'],
   'interpretation':'Observed development gains do not exceed the retuned semantic-shuffle null at conventional significance thresholds.'
 },
 'external':json.loads((EXT/'EXTERNAL_EVALUATION_SUMMARY.json').read_text()),
 'pre_external_integrity_recheck_all_match':bool(idf.match.all()),
 'post_external_tuning':False,
 'recipient_limitation':'Both studies contain repeat recipients but public biopsy-to-recipient mapping is unavailable. CV and bootstrap are biopsy-row-level, not patient-clustered.',
 'headline':headline
}
(OUT/'FINAL_INTEGRITY_AND_RESULTS_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')

md=f"""# Final Formal Experiment Report — GSE36059 → GSE48581 TCMR

## Status
COMPLETE. Formal DeepSeek measurement, development-only lambda selection, cached semantic-shuffle diagnostic, pre-external freeze, and one-time sealed GSE48581 evaluation are complete.

## Formal LLM measurement
- 200 semantic pairs: 100 Selective + 100 Global.
- 400 DeepSeek calls: one AB and one BA call per pair.
- Provider model: {models[0]}.
- Frozen system fingerprint: {fps[0]}.
- selective and selective_no_certainty-NoC use exactly the same Selective pair set, evidence, raw calls, p_e, C_e and cache.
- Pair aggregation: p_e = 0.5 * [p_AB(A) + p_BA(B)]; certainty C_e = 1 - H_binary(p_e)/log(2).
- No post-measurement LLM calls were issued.

## Evidence
- Formal eligibility requires both endpoints to be measured in at least two clean primary independence families.
- Clean primary families: Sarwal/Stanford, Pittsburgh, and TGC multicenter.
- All 200 formal pairs pass the frozen clean-K2 gate.
- Target-aware GSE232825/GSE131179 sources are excluded from production prompts.
- Literature context does not create eligibility and is not counted as an independent dataset vote.

## Development-only lambda selection
Fold-local 5-fold GSE36059 CV recomputes the Elastic-Net + SIS Reference anchor inside each training fold. Pair identities, U_e and cached LLM measurements remain fixed.

| Method | selected lambda | development CV AUROC | fallback to Reference? |
|---|---:|---:|---|
| Reference | 0 | {ref.development_cv_auroc:.4f} | yes by definition |
| Global | {gl.selected_lambda:g} | {gl.development_cv_auroc:.4f} | no |
| selective | {selective.selected_lambda:g} | {selective.development_cv_auroc:.4f} | no |
| selective_no_certainty-NoC | {m4.selected_lambda:g} | {m4.development_cv_auroc:.4f} | no |

## Cached semantic-shuffle diagnostic
1000 shuffles jointly permute cached (p_e, C_e) bundles over the fixed Selective edges while retaining edge identity and U_e; lambda is reselected on GSE36059 for every shuffle. No new LLM calls are made.

- selective observed development delta AUROC: {shuffle['selective']['observed_delta_vs_reference']:.4f}; retuned shuffle p = {shuffle['selective']['empirical_p_retuned']:.3f}.
- selective_no_certainty-NoC observed development delta AUROC: {shuffle['selective_no_certainty-NoC']['observed_delta_vs_reference']:.4f}; retuned shuffle p = {shuffle['selective_no_certainty-NoC']['empirical_p_retuned']:.3f}.
- The large development-CV improvements are not distinguishable from the semantic-reassignment null in this diagnostic.

## Sealed GSE48581 evaluation

| Method | lambda | AUROC | AUPRC | Balanced Acc. | Delta AUROC vs Ref. |
|---|---:|---:|---:|---:|---:|
| Reference | 0 | {ref.external_auroc:.4f} | {ref.external_auprc:.4f} | {ref.external_balanced_accuracy:.4f} | — |
| Global | {gl.selected_lambda:g} | {gl.external_auroc:.4f} | {gl.external_auprc:.4f} | {gl.external_balanced_accuracy:.4f} | {gl.external_delta_auroc_vs_reference:+.4f} |
| selective | {selective.selected_lambda:g} | {selective.external_auroc:.4f} | {selective.external_auprc:.4f} | {selective.external_balanced_accuracy:.4f} | {selective.external_delta_auroc_vs_reference:+.4f} |
| selective_no_certainty-NoC | {m4.selected_lambda:g} | {m4.external_auroc:.4f} | {m4.external_auprc:.4f} | {m4.external_balanced_accuracy:.4f} | {m4.external_delta_auroc_vs_reference:+.4f} |

The paired biopsy-level bootstrap 95% intervals for all guided-minus-Reference AUROC differences include zero. These intervals are diagnostic only because repeated recipients are known to exist and public biopsy-to-recipient identifiers are unavailable.

## Paper-ready headline
{headline}

## Integrity
- Pre-external hash recheck: all frozen files byte-identical after unsealing.
- External outcome first read: stage 05_external_evaluation_v3_2.
- No external result was used to change lambda, support, evidence, pair routing, prompts, or LLM measurements.
- No post-external tuning was performed.
"""
(OUT/'FINAL_EXPERIMENT_REPORT.md').write_text(md)

artifacts=[OUT/'FINAL_REFERENCE_GLOBAL_selective_selective_no_certainty_TABLE.csv',OUT/'PAPER_READY_HEADLINE.txt',
           OUT/'PRE_EXTERNAL_INTEGRITY_RECHECK.csv',OUT/'FINAL_INTEGRITY_AND_RESULTS_AUDIT.json',
           OUT/'FINAL_EXPERIMENT_REPORT.md']
with (OUT/'SHA256SUMS.txt').open('w') as h:
    for p in artifacts:h.write(sha(p)+'  '+p.name+'\n')
print(final.to_string(index=False))
print('\nHEADLINE\n'+headline)
print('\nQUERY_STATS '+json.dumps(audit['tokens']))
print('\nPAIR_STATS '+json.dumps(pair_stats))
print('\nINTEGRITY '+json.dumps(idf.match.value_counts().to_dict()))
