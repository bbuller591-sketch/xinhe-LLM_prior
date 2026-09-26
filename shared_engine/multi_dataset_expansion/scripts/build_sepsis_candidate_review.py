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
import pandas as pd, numpy as np, pyreadr, json
import matplotlib.pyplot as plt

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion/02_GSE65682'))
RAW=ROOT/'raw'; PROC=ROOT/'processed'; OUT=ROOT/'candidate_review'; OUT.mkdir(exist_ok=True)

X=pyreadr.read_r(str(RAW/'GSE65682_raw.rds'))[None]
ph=pyreadr.read_r(str(RAW/'GSE65682_phenotype.rds'))[None]
off=pd.read_csv(PROC/'GSE65682_official_metadata.csv')
# raw matrix columns exactly define the 479 endotype cohort samples
ids=list(X.columns)
ph=ph.set_index('Sample').loc[ids].reset_index()
off479=off.set_index('Sample_ID').loc[ids].reset_index()

# numeric-equivalent audit of mortality/time fields between official GEO and curated phenotype
audit={}
for c in ['mortality_event_28days','time_to_event_28days']:
    a=pd.to_numeric(ph[c],errors='coerce')
    b=pd.to_numeric(off479[c],errors='coerce')
    audit[c+'_numeric_mismatches']=int((~((a.eq(b)) | (a.isna()&b.isna()))).sum())
for c in ['gender','age','endotype_cohort','endotype_class']:
    a=ph[c].astype(str); b=off479[c].astype(str)
    audit[c+'_string_mismatches']=int((a.ne(b)).sum())

dev_ids=ph.loc[ph.endotype_cohort.eq('discovery'),'Sample'].tolist()
val_ids=ph.loc[ph.endotype_cohort.eq('validation'),'Sample'].tolist()
assert len(dev_ids)==263 and len(val_ids)==216 and set(dev_ids).isdisjoint(val_ids)
devvar=X[dev_ids].var(axis=1,ddof=0).sort_values(ascending=False)
valvar=X[val_ids].var(axis=1,ddof=0).sort_values(ascending=False)
fullvar=X.var(axis=1,ddof=0).sort_values(ascending=False)

# Optional protein-coding annotation from an already-local GDC gene registry, not used in primary ranking.
gdc_path=Path(str(REPRO_ROOT / 'GBM_CANONICAL_HANDOFF_V2_20260917/metadata/gdc_genes_by_ensembl.json'))
protein=set()
if gdc_path.exists():
    d=json.loads(gdc_path.read_text())
    protein={v.get('symbol') for v in d.values() if v.get('biotype')=='protein_coding' and v.get('symbol')}

tab=pd.DataFrame({'gene':devvar.index,'variance_discovery':devvar.values})
tab['rank_discovery']=np.arange(1,len(tab)+1)
tab['variance_validation']=tab.gene.map(valvar)
tab['rank_validation']=tab.gene.map(pd.Series(np.arange(1,len(valvar)+1),index=valvar.index))
tab['variance_all479']=tab.gene.map(fullvar)
tab['protein_coding_by_local_gdc_registry']=tab.gene.isin(protein)
tab.to_csv(OUT/'GSE65682_GENE_VARIANCE_RANKS_DISCOVERY_ONLY.csv',index=False)

rows=[]
for universe in ['all_genes','protein_coding']:
    d=tab if universe=='all_genes' else tab[tab.protein_coding_by_local_gdc_registry]
    d=d.sort_values('variance_discovery',ascending=False)
    total=d.variance_discovery.sum()
    for k in [500,1000,2000,3000,5000]:
        top=d.head(k)
        rows.append({
          'universe':universe,'k':k,'available_genes':len(d),'selected':len(top),
          'validation_coverage':int(top.variance_validation.notna().sum()),
          'variance_at_cutoff':float(top.variance_discovery.iloc[-1]),
          'median_variance_topk':float(top.variance_discovery.median()),
          'fraction_total_gene_variance_topk':float(top.variance_discovery.sum()/total)
        })
        top.to_csv(OUT/f'GSE65682_CANDIDATE_{universe}_TOP{k}.csv',index=False)
pd.DataFrame(rows).to_csv(OUT/'GSE65682_CANDIDATE_CUTOFF_SUMMARY.csv',index=False)

stab=[]
for k in [500,1000,2000,3000,5000]:
    a=set(devvar.head(k).index); b=set(valvar.head(k).index)
    stab.append({'k':k,'overlap_discovery_validation_topk':len(a&b),'jaccard':len(a&b)/len(a|b)})
pd.DataFrame(stab).to_csv(OUT/'GSE65682_DISCOVERY_VALIDATION_VARIANCE_RANK_STABILITY.csv',index=False)

fig,ax=plt.subplots(figsize=(8,5))
vals=devvar.values
ax.plot(np.arange(1,len(vals)+1),vals)
ax.set_yscale('log')
ax.set_xlabel('Gene variance rank (discovery X only)')
ax.set_ylabel('Variance across discovery samples (log scale)')
ax.set_title('GSE65682: discovery-only gene variance rank curve')
for k in [500,1000,2000,3000,5000]:
    ax.axvline(k,linestyle='--',linewidth=0.8)
fig.tight_layout(); fig.savefig(OUT/'GSE65682_DISCOVERY_VARIANCE_RANK_CURVE.png',dpi=180); plt.close(fig)

# Save task metadata and matrix in easy downstream formats.
ph.to_csv(PROC/'GSE65682_TASK479_PHENOTYPE.csv',index=False)
X.reset_index(names='Gene_Symbol').to_parquet(PROC/'GSE65682_raw_gene_expression_11518x479.parquet',index=False)

summary={
  'raw_gene_matrix_shape':list(X.shape),
  'official_geo_samples':int(len(off)),
  'task_samples_479':int(len(ph)),
  'development_discovery_n':int(len(dev_ids)),
  'sealed_validation_n':int(len(val_ids)),
  'development_deaths':int(pd.to_numeric(ph.loc[ph.endotype_cohort.eq('discovery'),'mortality_event_28days']).sum()),
  'validation_deaths':int(pd.to_numeric(ph.loc[ph.endotype_cohort.eq('validation'),'mortality_event_28days']).sum()),
  'total_deaths':int(pd.to_numeric(ph.mortality_event_28days).sum()),
  'total_survivors':int((pd.to_numeric(ph.mortality_event_28days)==0).sum()),
  'raw_missing_values':int(X.isna().sum().sum()),
  'unique_gene_symbols':int(X.index.nunique()),
  'protein_coding_by_local_gdc_registry':int(sum(pd.Index(X.index).isin(protein))),
  'official_vs_phenotype_audit':audit
}
(OUT/'GSE65682_CANDIDATE_REVIEW_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('\ncutoffs')
print(pd.DataFrame(rows).to_string(index=False))
print('\nstability')
print(pd.DataFrame(stab).to_string(index=False))
