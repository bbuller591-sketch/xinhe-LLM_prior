

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
import pandas as pd, shutil, json, hashlib, os
src=Path(str(REPRO_ROOT / '02_gse272769_sepsis/gse272769'))
pkg=Path(str(REPRO_ROOT / 'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919'))
if pkg.exists(): shutil.rmtree(pkg)
for d in ['00_REPORT','01_FROZEN_DATA','02_EVIDENCE','03_MEASUREMENT','04_ROUTING','05_REAL_RESULTS','06_SHUFFLE_RESULTS','07_CODE','08_PROVENANCE']:
    (pkg/d).mkdir(parents=True,exist_ok=True)

# Frozen development data / features / splits
for f in (src/'frozen').iterdir():
    if f.is_file(): shutil.copy2(f,pkg/'01_FROZEN_DATA'/f.name)
for f in ['OUTER_SPLITS.csv','reference_SELECTOR_AUDIT.csv','reference_FOLD_RESULTS.csv','reference_AGGREGATE.csv','reference_STATUS.json']:
    p=src/'reference_formal'/f
    if not p.exists(): p=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion/04_reference/SEPSIS_GSE272769'))/f
    if p.exists(): shutil.copy2(p,pkg/'01_FROZEN_DATA'/f)

# Evidence: processed evidence + audits + metadata, avoid giant redundant raw matrices
evfiles=['EMTAB4451_28D_GENE_EVIDENCE.csv','EMTAB7581_28D_GENE_EVIDENCE.csv','GSE95233_D01_28D_GENE_EVIDENCE.csv',
'EVIDENCE_COVERAGE_MATRIX.csv','GENE_EVIDENCE_STATUS.csv','GSE272769_P1500_IDENTIFIER_REGISTRY.csv',
'SOURCE_CONCORDANCE_AUDIT.md','SOURCE_CONCORDANCE.csv','EVIDENCE_QUALITY_AUDIT_V1.md','EVIDENCE_SOURCE_AUDIT_CHECKPOINT.md',
'E-MTAB-4451.sdrf.txt','E-MTAB-7581.sdrf.txt']
for f in evfiles:
    p=src/'evidence'/f
    if p.exists(): shutil.copy2(p,pkg/'02_EVIDENCE'/f)
for f in ['SOURCE_AUDIT_REGISTRY_V1.tsv','GSE95233_AUDIT.md']:
    p=src/'evidence_search'/f
    if p.exists(): shutil.copy2(p,pkg/'02_EVIDENCE'/f)
# GSE95233 official metadata/summary
for f in ['GSE95233_official_metadata.csv','SUMMARY.json']:
    p=src/'evidence/gse95233'/f
    if p.exists(): shutil.copy2(p,pkg/'02_EVIDENCE'/('GSE95233_'+f if f=='SUMMARY.json' else f))

# Routing only K30/K50
for fold in range(1,6):
  for sel in ['LASSO','ELASTICNET','SIS']:
    dest=pkg/'04_ROUTING'/f'fold{fold}_{sel}'; dest.mkdir()
    shutil.copy2(Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion/06_selective_ROUTING/SEPSIS_GSE272769'))/f'fold{fold}_{sel}'/'K30_PAIR_CONFUSION.csv',dest/'K30_PAIR_CONFUSION.csv')
    shutil.copy2(src/'tmp_k40_90_routing/SEPSIS_GSE272769'/f'fold{fold}_{sel}'/'K50_PAIR_CONFUSION.csv',dest/'K50_PAIR_CONFUSION.csv')

# Determine needed pairs
pairs=set()
for p in (pkg/'04_ROUTING').glob('fold*/*PAIR_CONFUSION.csv'):
    d=pd.read_csv(p)
    if 'pair_type' in d: d=d[d.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION']
    for r in d.itertuples():
        pairs.add('||'.join(sorted([str(r.feature_A),str(r.feature_B)])))
meas=pd.read_csv(src/'tmp_k40_90_llm/measurement/selective_PAIR_SOURCE_MEASUREMENTS_MERGED.csv')
mf=meas[meas.unordered_pair_id.astype(str).isin(pairs)].copy()
mf.to_csv(pkg/'03_MEASUREMENT/selective_PAIR_SOURCE_MEASUREMENTS_TOP30_TOP50.csv',index=False)

# Measurement provenance/freeze
for p in [src/'pre_llm/PRE_LLM_FREEZE_REPORT_FINAL.md',src/'pre_llm/LLM_RUNTIME_CONFIG_FROZEN.json',
          src/'pre_llm/PROVIDER_FINGERPRINT_LOCK.json',src/'pre_llm/LLM_EXECUTION_APPROVAL.json',
          src/'measurement/MEASUREMENT_COMPLETION_AUDIT.json',src/'tmp_k40_90_llm/FREEZE_MANIFEST.json']:
    if p.exists(): shutil.copy2(p,pkg/'08_PROVENANCE'/p.name)

# Real results: only k30/k50, all selectors/folds
frames=[]
for path,k in [(src/'selective_internal/selective_NESTED_OUTER_RESULTS.csv',30),(src/'tmp_k40_90_results/selective_NESTED_OUTER_RESULTS.csv',50)]:
    d=pd.read_csv(path); frames.append(d[d.k==k])
pd.concat(frames).to_csv(pkg/'05_REAL_RESULTS/selective_OUTER_RESULTS_TOP30_TOP50.csv',index=False)
frames=[]
for path,k in [(src/'selective_internal/selective_NESTED_AGGREGATE.csv',30),(src/'tmp_k40_90_results/selective_NESTED_AGGREGATE.csv',50)]:
    d=pd.read_csv(path); frames.append(d[d.k==k])
realagg=pd.concat(frames); realagg.to_csv(pkg/'05_REAL_RESULTS/selective_AGGREGATE_TOP30_TOP50.csv',index=False)
frames=[]
for path,k in [(src/'selective_internal/selective_NESTED_SELECTED_ETA.csv',30),(src/'tmp_k40_90_results/selective_NESTED_SELECTED_ETA.csv',50)]:
    d=pd.read_csv(path); frames.append(d[d.k==k])
pd.concat(frames).to_csv(pkg/'05_REAL_RESULTS/selective_SELECTED_ETA_TOP30_TOP50.csv',index=False)

# Shuffle consolidate 200 reps for k30 and k50
agg=[]; outer=[]; lam=[]
for rep in range(1,201):
  for root,k in [(src/'selective_shuffle',30),(src/'tmp_k40_90_shuffle',50)]:
    rd=root/f'rep_{rep:03d}'
    a=pd.read_csv(rd/'selective_NESTED_AGGREGATE.csv'); a=a[a.k==k].copy(); a.insert(0,'replicate',rep); agg.append(a)
    o=pd.read_csv(rd/'selective_NESTED_OUTER_RESULTS.csv'); o=o[o.k==k].copy(); o.insert(0,'replicate',rep); outer.append(o)
    e=pd.read_csv(rd/'selective_NESTED_SELECTED_ETA.csv'); e=e[e.k==k].copy(); e.insert(0,'replicate',rep); lam.append(e)
shagg=pd.concat(agg,ignore_index=True); shagg.to_csv(pkg/'06_SHUFFLE_RESULTS/SHUFFLE_200_AGGREGATES_TOP30_TOP50.csv',index=False)
pd.concat(outer,ignore_index=True).to_csv(pkg/'06_SHUFFLE_RESULTS/SHUFFLE_200_OUTER_FOLDS_TOP30_TOP50.csv',index=False)
pd.concat(lam,ignore_index=True).to_csv(pkg/'06_SHUFFLE_RESULTS/SHUFFLE_200_SELECTED_ETA_TOP30_TOP50.csv',index=False)

# shuffle summary
rows=[]
for r in realagg.itertuples():
    a=shagg[(shagg.selector==r.selector)&(shagg.k==r.k)].mean_delta_auroc.to_numpy(float)
    b=shagg[(shagg.selector==r.selector)&(shagg.k==r.k)].mean_delta_macro_ap.to_numpy(float)
    ra=float(r.mean_delta_auroc); rb=float(r.mean_delta_macro_ap)
    rows.append(dict(selector=r.selector,k=int(r.k),real_delta_auroc=ra,shuffle_mean_delta_auroc=a.mean(),
      shuffle_sd_delta_auroc=a.std(ddof=1),shuffle_q025_auroc=float(pd.Series(a).quantile(.025)),shuffle_q975_auroc=float(pd.Series(a).quantile(.975)),
      real_percentile_auroc=100*(a<ra).mean(),empirical_p_upper_auroc=(1+(a>=ra).sum())/201,
      real_delta_macro_ap=rb,shuffle_mean_delta_macro_ap=b.mean(),real_percentile_macro_ap=100*(b<rb).mean(),empirical_p_upper_macro_ap=(1+(b>=rb).sum())/201))
summary=pd.DataFrame(rows); summary.to_csv(pkg/'06_SHUFFLE_RESULTS/SHUFFLE_SUMMARY_TOP30_TOP50.csv',index=False)

# code
for p in [src/'tmp_k40_90_llm/run_nested_k40_90.py',src/'tmp_k40_90_shuffle/run_nested_shufflebase.py',src/'tmp_k40_90_shuffle/run_worker.py',
          Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion/scripts/run_strict_nested_selective_routing.py'))]:
    if p.exists(): shutil.copy2(p,pkg/'07_CODE'/p.name)
print('PKG',pkg)
print(realagg[['selector','k','mean_reference_auroc','mean_selective_auroc','mean_delta_auroc','mean_delta_macro_ap']].to_string(index=False))
print(summary.to_string(index=False))
