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
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import json

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion/01_GSE25055_25065'))
RAW=ROOT/'raw'; PROC=ROOT/'processed'; OUT=ROOT/'candidate_review'; OUT.mkdir(exist_ok=True)

ann=pd.read_csv(PROC/'GPL96_annotation.csv',dtype=str).fillna('')
ann=ann[ann['ID'].ne('ID')].copy()
sym=ann['Gene symbol'].str.strip()
clean=(sym.ne('')) & (~sym.str.contains('///',regex=False)) & (~sym.str.contains(';',regex=False)) & (~sym.str.contains(',',regex=False))
a=ann.loc[clean,['ID','Gene symbol','Gene ID','Gene title']].copy().drop_duplicates('ID')
a.to_csv(OUT/'GPL96_CLEAN_SINGLE_GENE_PROBES.csv',index=False)

cur=pd.read_csv(RAW/'GSE25055.tsv.gz',sep='\t',usecols=['HGNC_Symbol','Gene_Biotype'],dtype=str)
bio=(cur.dropna().groupby('HGNC_Symbol')['Gene_Biotype'].agg(lambda x:'|'.join(sorted(set(x)))).rename('curated_biotypes'))
protein=set(bio[bio.eq('protein_coding')].index)

summary={}
for ds in ['GSE25055','GSE25065']:
    meta=pd.read_csv(PROC/f'{ds}_official_metadata.csv')
    expr=pd.read_parquet(PROC/f'{ds}_official_probe_expression.parquet').set_index('ID_REF')
    eligible=meta.loc[meta['pathologic_response_pcr_rd'].isin(['pCR','RD']),'Sample_ID'].tolist()
    common_probe=expr.index.intersection(a.ID)
    Xp=expr.loc[common_probe,eligible].astype(float)
    amap=a.set_index('ID').loc[common_probe]
    Xm=Xp.copy(); Xm['Gene symbol']=amap['Gene symbol'].values
    Xm=Xm.groupby('Gene symbol',sort=True).mean()
    pv=Xp.var(axis=1,ddof=0)
    tmp=pd.DataFrame({'probe':pv.index,'variance':pv.values,'gene':amap.loc[pv.index,'Gene symbol'].values})
    chosen=tmp.sort_values(['gene','variance','probe'],ascending=[True,False,True]).drop_duplicates('gene')
    Xv=Xp.loc[chosen.probe].copy(); Xv.index=chosen.gene.values
    mvar=Xm.var(axis=1,ddof=0).rename('variance_mean_collapse').sort_values(ascending=False)
    vvar=Xv.var(axis=1,ddof=0).rename('variance_maxvar_probe').sort_values(ascending=False)
    tab=pd.DataFrame({'gene':mvar.index,'variance_mean_collapse':mvar.values})
    tab['rank_mean']=np.arange(1,len(tab)+1)
    tab=tab.merge(vvar.rename_axis('gene').reset_index(),on='gene',how='left')
    tab['rank_maxvar']=tab['variance_maxvar_probe'].rank(method='first',ascending=False).astype(int)
    tab['protein_coding_by_curated_registry']=tab.gene.isin(protein)
    tab=tab.merge(bio.rename_axis('gene').reset_index(),on='gene',how='left')
    tab.to_csv(OUT/f'{ds}_GENE_VARIANCE_RANKS.csv',index=False)
    Xm.reset_index().to_parquet(PROC/f'{ds}_official_gene_mean_expression.parquet',index=False)
    chosen.to_csv(OUT/f'{ds}_MAXVAR_PROBE_PER_GENE.csv',index=False)
    summary[ds]={
      'n_official_samples':int(len(meta)),
      'n_evaluable_pcr_rd':int(len(eligible)),
      'p_clean_single_gene_probes':int(Xp.shape[0]),
      'p_unique_gene_symbols':int(Xm.shape[0]),
      'p_protein_coding_registry':int(sum(Xm.index.isin(protein))),
      'n_zero_variance_genes':int((mvar==0).sum()),
      'top500_mean_vs_maxvar_overlap':int(len(set(mvar.head(500).index)&set(vvar.head(500).index))),
      'top1000_mean_vs_maxvar_overlap':int(len(set(mvar.head(1000).index)&set(vvar.head(1000).index))),
      'top2000_mean_vs_maxvar_overlap':int(len(set(mvar.head(2000).index)&set(vvar.head(2000).index))),
      'top3000_mean_vs_maxvar_overlap':int(len(set(mvar.head(3000).index)&set(vvar.head(3000).index))),
      'top5000_mean_vs_maxvar_overlap':int(len(set(mvar.head(5000).index)&set(vvar.head(5000).index))),
    }
    fig,ax=plt.subplots(figsize=(8,5))
    vals=mvar.values
    ax.plot(np.arange(1,len(vals)+1),vals)
    ax.set_yscale('log')
    ax.set_xlabel('Gene variance rank (X-only)')
    ax.set_ylabel('Variance across evaluable samples (log scale)')
    ax.set_title(f'{ds}: gene-level variance rank curve (mean probe collapse)')
    for k in [500,1000,2000,3000,5000]:
        if k<=len(vals): ax.axvline(k,linestyle='--',linewidth=0.8)
    fig.tight_layout(); fig.savefig(OUT/f'{ds}_VARIANCE_RANK_CURVE.png',dpi=180); plt.close(fig)

dev=pd.read_csv(OUT/'GSE25055_GENE_VARIANCE_RANKS.csv')
valgenes=set(pd.read_parquet(PROC/'GSE25065_official_gene_mean_expression.parquet')['Gene symbol'])
rows=[]
for universe in ['all_clean','protein_coding']:
    d=dev if universe=='all_clean' else dev[dev.protein_coding_by_curated_registry]
    d=d.sort_values('variance_mean_collapse',ascending=False)
    for k in [500,1000,2000,3000,5000]:
        top=d.head(k)
        rows.append({
          'universe':universe,'k':k,'available_genes':len(d),'selected':len(top),
          'validation_coverage':sum(top.gene.isin(valgenes)),
          'variance_at_cutoff':float(top.variance_mean_collapse.iloc[-1]) if len(top) else np.nan,
          'median_variance_topk':float(top.variance_mean_collapse.median()) if len(top) else np.nan,
          'fraction_total_gene_variance_topk':float(top.variance_mean_collapse.sum()/d.variance_mean_collapse.sum())
        })
        top.to_csv(OUT/f'GSE25055_CANDIDATE_{universe}_TOP{k}.csv',index=False)
pd.DataFrame(rows).to_csv(OUT/'GSE25055_CANDIDATE_CUTOFF_SUMMARY.csv',index=False)

meta=pd.read_csv(PROC/'GSE25055_official_metadata.csv')
X=pd.read_parquet(PROC/'GSE25055_official_gene_mean_expression.parquet').set_index('Gene symbol')
out=[]
full_rank=dev.sort_values('variance_mean_collapse',ascending=False)
for source in ['MDACC','ISPY']:
    ids=meta.loc[(meta.source==source)&meta.pathologic_response_pcr_rd.isin(['pCR','RD']),'Sample_ID'].tolist()
    vv=X[ids].var(axis=1,ddof=0).sort_values(ascending=False)
    for k in [500,1000,2000,3000,5000]:
        full=set(full_rank.head(k).gene); ss=set(vv.head(k).index)
        out.append({'source':source,'n':len(ids),'k':k,'overlap_with_full_topk':len(full&ss),'jaccard':len(full&ss)/len(full|ss)})
pd.DataFrame(out).to_csv(OUT/'GSE25055_SOURCE_VARIANCE_RANK_STABILITY.csv',index=False)
(OUT/'BREAST_PREPROCESSING_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('\ncutoffs')
print(pd.DataFrame(rows).to_string(index=False))
print('\nsource stability')
print(pd.DataFrame(out).to_string(index=False))
