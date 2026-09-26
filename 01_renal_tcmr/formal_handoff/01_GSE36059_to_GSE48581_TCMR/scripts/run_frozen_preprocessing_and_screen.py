

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
import sys, json, io
import numpy as np, pandas as pd
sys.path.insert(0,str(REPRO_ROOT / '01_renal_tcmr/src'))
from geo_utils import read_geo_metadata, read_geo_expression
from screen_utils import run_screen
base=str(REPRO_ROOT / 'dataset_screening_20260921')

# GPL570 probe -> gene symbol map.
txt=open(base+'/data/GPL570_full.txt',errors='replace').read().replace('\r','')
block=txt.split('!platform_table_begin\n',1)[1].split('!platform_table_end',1)[0]
ann=pd.read_csv(io.StringIO(block),sep='\t',dtype=str)
ann=ann[['ID','Gene Symbol']].dropna()
ann=ann[~ann['Gene Symbol'].isin(['---',''])]
ann['Gene Symbol']=ann['Gene Symbol'].str.split(' /// ').str[0].str.strip()
probe2gene=dict(zip(ann.ID.astype(str),ann['Gene Symbol'].astype(str)))

def load(g):
    m=read_geo_metadata(f'{base}/data/{g}/{g}_series_matrix.txt.gz')
    e=read_geo_expression(f'{base}/data/{g}/{g}_series_matrix.txt.gz').set_index('feature_id')
    if g=='GSE36059':
        lab=m['diagnosis_tcmr_abmr_mixed_non_rejecting'].astype(str)
        keep=lab.isin(['TCMR','ABMR','MIXED','non-rejecting'])
        m=m[keep].copy()
        m['y']=(m['diagnosis_tcmr_abmr_mixed_non_rejecting']=='TCMR').astype(int)
    else:
        lab=m['diagnosis_tcmr_non_tcmr_nephrectomies'].astype(str)
        keep=lab.isin(['TCMR','non-TCMR'])
        m=m[keep].copy()
        m['y']=(m['diagnosis_tcmr_non_tcmr_nephrectomies']=='TCMR').astype(int)
    sam=[s for s in m.geo_accession if s in e.columns]
    m=m.set_index('geo_accession').loc[sam]
    sub=e[sam].copy()
    sub['gene']=[probe2gene.get(str(i),'') for i in sub.index]
    sub=sub[sub.gene!='']
    ge=sub.groupby('gene',sort=False).median(numeric_only=True)
    X=ge[sam].T.to_numpy(dtype=float)
    y=m.y.to_numpy(dtype=int)
    return X,y,ge.index.astype(str).to_numpy()

X,y,f=load('GSE36059')
Xe,ye,fe=load('GSE48581')
common,ia,ib=np.intersect1d(f,fe,return_indices=True)
out=base+'/results/GSE36059_to_GSE48581_TCMR_gene'
summary=run_screen(X[:,ia],y,common,out,p_candidate=2000,X_external=Xe[:,ib],y_external=ye)
summary['task']='renal allograft biopsy expression -> molecular TCMR vs non-TCMR'
summary['development']='GSE36059 molecular diagnosis: pure TCMR positive; ABMR/MIXED/non-rejecting negative; nephrectomy excluded'
summary['external']='GSE48581 INTERCOM molecular diagnosis: TCMR vs non-TCMR; nephrectomy excluded'
summary['preprocessing']='RMA submitter matrices; GPL570 Gene Symbol mapping; median probe-to-gene collapse; common genes; development-X-only variance top 2000'
summary['common_gene_count_before_variance']=int(len(common))
open(out+'/summary.json','w').write(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
