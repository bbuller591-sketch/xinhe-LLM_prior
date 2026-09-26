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
import pandas as pd, numpy as np, pyreadr, json, shutil
import matplotlib.pyplot as plt

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
GI=ROOT/'02_GSE65682/raw/Homo_sapiens.gene_info.gz'
geneinfo=pd.read_csv(GI,sep='\t',dtype=str)
by_id=geneinfo.set_index('GeneID')
by_symbol={r.Symbol:r for _,r in geneinfo.iterrows()}
syn={}
for _,r in geneinfo[['GeneID','Symbol','Synonyms','type_of_gene']].iterrows():
    if isinstance(r.Synonyms,str) and r.Synonyms!='-':
        for x in r.Synonyms.split('|'):
            syn.setdefault(x,[]).append((r.GeneID,r.Symbol,r.type_of_gene))

def symbol_map(x):
    if x in by_symbol:
        r=by_symbol[x]; return ('EXACT_SYMBOL',r.GeneID,r.Symbol,r.type_of_gene)
    cand=syn.get(x,[])
    ids={z[0] for z in cand}
    if len(ids)==1:
        z=cand[0]; return ('UNIQUE_SYNONYM',z[0],z[1],z[2])
    if len(ids)>1: return ('AMBIGUOUS_SYNONYM','','','')
    return ('UNMAPPED','','','')

# ---------- Breast ----------
B=ROOT/'01_GSE25055_25065'; BP=B/'processed'; BO=B/'candidate_review_v2'; BO.mkdir(exist_ok=True)
ann=pd.read_csv(BP/'GPL96_annotation.csv',dtype=str).fillna('')
ann=ann[ann.ID.ne('ID')].copy()
sym=ann['Gene symbol'].str.strip(); gid=ann['Gene ID'].str.strip()
single=(sym.ne('')) & (~sym.str.contains('///',regex=False)) & gid.str.fullmatch(r'\d+')
a=ann.loc[single,['ID','Gene symbol','Gene ID','Gene title']].copy()
reg=[]
for _,r in a.iterrows():
    if r['Gene ID'] in by_id.index:
        gi=by_id.loc[r['Gene ID']]
        if isinstance(gi,pd.DataFrame): gi=gi.iloc[0]
        reg.append((r.ID,r['Gene symbol'],r['Gene ID'],'GENEID_EXACT',gi.Symbol,gi.type_of_gene))
    else:
        reg.append((r.ID,r['Gene symbol'],r['Gene ID'],'GENEID_NOT_IN_CURRENT_NCBI','',''))
breg=pd.DataFrame(reg,columns=['probe_id','platform_symbol','platform_gene_id','mapping_status','current_symbol','type_of_gene'])
breg.to_csv(BO/'GPL96_PROBE_IDENTIFIER_REGISTRY.csv',index=False)
pc=breg[(breg.mapping_status=='GENEID_EXACT')&(breg.type_of_gene=='protein-coding')&(breg.current_symbol!='')].copy()

breast_summary={'mapping':{
    'platform_rows':int(len(ann)),
    'single_symbol_single_numeric_geneid_probes':int(len(a)),
    'protein_coding_mapped_probes':int(len(pc)),
    'protein_coding_unique_geneids':int(pc.platform_gene_id.nunique()),
    'protein_coding_unique_current_symbols':int(pc.current_symbol.nunique())
}}
for ds in ['GSE25055','GSE25065']:
    meta=pd.read_csv(BP/f'{ds}_official_metadata.csv')
    expr=pd.read_parquet(BP/f'{ds}_official_probe_expression.parquet').set_index('ID_REF')
    eligible=meta.loc[meta.pathologic_response_pcr_rd.isin(['pCR','RD']),'Sample_ID'].tolist()
    probes=[p for p in pc.probe_id if p in expr.index]
    xp=expr.loc[probes,eligible].astype(float)
    rr=pc.set_index('probe_id').loc[probes]
    # Primary technical collapse: mean all unambiguous probes mapping to same current NCBI GeneID.
    tmp=xp.copy(); tmp['GeneID']=rr.platform_gene_id.values
    xg=tmp.groupby('GeneID',sort=True).mean()
    id2sym=rr.drop_duplicates('platform_gene_id').set_index('platform_gene_id').current_symbol
    xg.index=[id2sym.loc[i] for i in xg.index]
    vv=xg.var(axis=1,ddof=0).sort_values(ascending=False)
    xg.reset_index(names='Gene_Symbol').to_parquet(BP/f'{ds}_protein_coding_gene_mean_expression.parquet',index=False)
    tab=pd.DataFrame({'gene':vv.index,'variance':vv.values,'rank':np.arange(1,len(vv)+1)})
    tab.to_csv(BO/f'{ds}_PROTEIN_CODING_VARIANCE_RANKS.csv',index=False)
    breast_summary[ds]={
      'official_n':int(len(meta)),'evaluable_n':int(len(eligible)),
      'protein_coding_gene_features':int(len(xg)),'zero_variance':int((vv==0).sum())
    }
# Candidate selection only from GSE25055 development.
dev=pd.read_csv(BO/'GSE25055_PROTEIN_CODING_VARIANCE_RANKS.csv')
cut=[]
for k in [500,1000,2000,3000,5000]:
    top=dev.head(k)
    cut.append({'k':k,'available_genes':len(dev),'selected':len(top),
                'variance_at_cutoff':float(top.variance.iloc[-1]),
                'median_variance_topk':float(top.variance.median()),
                'fraction_total_gene_variance_topk':float(top.variance.sum()/dev.variance.sum()),
                'p_over_n_evaluable':float(k/breast_summary['GSE25055']['evaluable_n'])})
    top.to_csv(BO/f'GSE25055_PRIMARY_CANDIDATE_TOP{k}.csv',index=False)
pd.DataFrame(cut).to_csv(BO/'GSE25055_PRIMARY_CUTOFF_SUMMARY.csv',index=False)
fig,ax=plt.subplots(figsize=(8,5)); ax.plot(np.arange(1,len(dev)+1),dev.variance); ax.set_yscale('log')
ax.set_xlabel('Protein-coding gene variance rank (development X only)'); ax.set_ylabel('Variance (log scale)')
ax.set_title('GSE25055: development-only candidate variance curve')
for k in [500,1000,2000,3000,5000]: ax.axvline(k,linestyle='--',linewidth=0.8)
fig.tight_layout(); fig.savefig(BO/'GSE25055_PRIMARY_VARIANCE_RANK_CURVE.png',dpi=180); plt.close(fig)
(BO/'BREAST_V2_SUMMARY.json').write_text(json.dumps(breast_summary,indent=2),encoding='utf-8')

# ---------- Sepsis ----------
S=ROOT/'02_GSE65682'; SP=S/'processed'; SO=S/'candidate_review_v2'; SO.mkdir(exist_ok=True)
X=pyreadr.read_r(str(S/'raw/GSE65682_raw.rds'))[None]
ph=pyreadr.read_r(str(S/'raw/GSE65682_phenotype.rds'))[None]
reg=[]
for old in X.index.astype(str):
    st,gid,cur,typ=symbol_map(old)
    reg.append((old,st,gid,cur,typ))
sreg=pd.DataFrame(reg,columns=['original_symbol','mapping_status','GeneID','current_symbol','type_of_gene'])
sreg.to_csv(SO/'GSE65682_IDENTIFIER_REGISTRY.csv',index=False)
spc=sreg[(sreg.type_of_gene=='protein-coding') & sreg.GeneID.ne('')].copy()
# collapse aliases that now map to same current GeneID, X-only mean
sub=X.loc[spc.original_symbol].copy()
sub['GeneID']=spc.set_index('original_symbol').loc[sub.index,'GeneID'].values
xg=sub.groupby('GeneID',sort=True).mean()
id2sym=spc.drop_duplicates('GeneID').set_index('GeneID').current_symbol
xg.index=[id2sym.loc[i] for i in xg.index]
ph=ph.set_index('Sample')
dev_ids=ph.index[ph.endotype_cohort.eq('discovery')].tolist()
val_ids=ph.index[ph.endotype_cohort.eq('validation')].tolist()
devvar=xg[dev_ids].var(axis=1,ddof=0).sort_values(ascending=False)
stab_meta={
 'n_raw_symbols':int(len(X)),
 'mapping_status_counts':sreg.mapping_status.value_counts().to_dict(),
 'protein_coding_original_symbols':int(len(spc)),
 'protein_coding_current_gene_features_after_collapse':int(len(xg)),
 'development_n':int(len(dev_ids)),
 'development_deaths':int(pd.to_numeric(ph.loc[dev_ids,'mortality_event_28days']).sum()),
 'sealed_validation_n':int(len(val_ids)),
 'sealed_validation_deaths':int(pd.to_numeric(ph.loc[val_ids,'mortality_event_28days']).sum()),
}
xg.reset_index(names='Gene_Symbol').to_parquet(SP/'GSE65682_protein_coding_gene_expression_479.parquet',index=False)
stab=pd.DataFrame({'gene':devvar.index,'variance':devvar.values,'rank':np.arange(1,len(devvar)+1)})
stab.to_csv(SO/'GSE65682_DISCOVERY_PROTEIN_CODING_VARIANCE_RANKS.csv',index=False)
scut=[]
for k in [500,1000,2000,3000,5000]:
    top=stab.head(k)
    scut.append({'k':k,'available_genes':len(stab),'selected':len(top),
                 'variance_at_cutoff':float(top.variance.iloc[-1]),
                 'median_variance_topk':float(top.variance.median()),
                 'fraction_total_gene_variance_topk':float(top.variance.sum()/stab.variance.sum()),
                 'p_over_n_development':float(k/len(dev_ids))})
    top.to_csv(SO/f'GSE65682_PRIMARY_CANDIDATE_TOP{k}.csv',index=False)
pd.DataFrame(scut).to_csv(SO/'GSE65682_PRIMARY_CUTOFF_SUMMARY.csv',index=False)
fig,ax=plt.subplots(figsize=(8,5)); ax.plot(np.arange(1,len(stab)+1),stab.variance); ax.set_yscale('log')
ax.set_xlabel('Protein-coding gene variance rank (discovery X only)'); ax.set_ylabel('Variance (log scale)')
ax.set_title('GSE65682: discovery-only candidate variance curve')
for k in [500,1000,2000,3000,5000]: ax.axvline(k,linestyle='--',linewidth=0.8)
fig.tight_layout(); fig.savefig(SO/'GSE65682_PRIMARY_VARIANCE_RANK_CURVE.png',dpi=180); plt.close(fig)
(SO/'GSE65682_V2_SUMMARY.json').write_text(json.dumps(stab_meta,indent=2),encoding='utf-8')

# Quarantine earlier validation-X rank diagnostics so they cannot be used for p/collapse freeze.
for parent,names in [
    (B/'candidate_review',['GSE25065_GENE_VARIANCE_RANKS.csv','GSE25065_MAXVAR_PROBE_PER_GENE.csv','GSE25065_VARIANCE_RANK_CURVE.png']),
    (S/'candidate_review',['GSE65682_DISCOVERY_VALIDATION_VARIANCE_RANK_STABILITY.csv'])
]:
    q=parent/'QUARANTINED_VALIDATION_X_DIAGNOSTICS'; q.mkdir(exist_ok=True)
    for name in names:
        src=parent/name
        if src.exists(): shutil.move(str(src),str(q/name))
    (q/'DO_NOT_USE_FOR_FREEZE.txt').write_text('These diagnostics inspect sealed-validation X values and are quarantined. They must not influence preprocessing, candidate-universe size, selector/k, or any pre-test choice.\n',encoding='utf-8')

print('BREAST SUMMARY'); print(json.dumps(breast_summary,indent=2))
print('\nBREAST CUTOFFS'); print(pd.DataFrame(cut).to_string(index=False))
print('\nSEPSIS SUMMARY'); print(json.dumps(stab_meta,indent=2))
print('\nSEPSIS CUTOFFS'); print(pd.DataFrame(scut).to_string(index=False))
