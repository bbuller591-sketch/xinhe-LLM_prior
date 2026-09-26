

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
import sys,io,json,re
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'EXISTING_METHOD_COMPARISON_20260925'
TASK=ROOT/'dataset_screening_20260921/formal_handoff/01_GSE36059_to_GSE48581_TCMR'
BASE=ROOT/'dataset_screening_20260921'
key=json.loads((OUT/'renal_gene_key.json').read_text())
init=json.loads((OUT/'renal_unique_initial.json').read_text())['unique_keys']
rep=json.loads((OUT/'RENAL_REPAIR_RESPONSES.json').read_text())['tests'][0]['content']
add=re.findall(r'\[(g\d{4})\]',rep)
ks=init+add
assert len(ks)==50 and len(set(ks))==50
sel=[key[k] for k in ks]
sys.path.insert(0,str(BASE/'src'))
from geo_utils import read_geo_metadata,read_geo_expression
txt=(BASE/'data/GPL570_full.txt').read_text(errors='replace').replace('\r','')
block=txt.split('!platform_table_begin\n',1)[1].split('!platform_table_end',1)[0]
ann=pd.read_csv(io.StringIO(block),sep='\t',dtype=str)[['ID','Gene Symbol']].dropna()
ann=ann[~ann['Gene Symbol'].isin(['---',''])]
ann['Gene Symbol']=ann['Gene Symbol'].str.split(' /// ').str[0].str.strip()
p2g=dict(zip(ann.ID.astype(str),ann['Gene Symbol'].astype(str)))
def load(acc):
    m=read_geo_metadata(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz'))
    e=read_geo_expression(str(BASE/f'data/{acc}/{acc}_series_matrix.txt.gz')).set_index('feature_id')
    if acc=='GSE36059':
        lab=m['diagnosis_tcmr_abmr_mixed_non_rejecting'].astype(str)
        m=m[lab.isin(['TCMR','ABMR','MIXED','non-rejecting'])].copy()
        m['y']=(m['diagnosis_tcmr_abmr_mixed_non_rejecting']=='TCMR').astype(int)
    else:
        lab=m['diagnosis_tcmr_non_tcmr_nephrectomies'].astype(str)
        m=m[lab.isin(['TCMR','non-TCMR'])].copy()
        m['y']=(m['diagnosis_tcmr_non_tcmr_nephrectomies']=='TCMR').astype(int)
    sam=[s for s in m.geo_accession if s in e.columns]
    m=m.set_index('geo_accession').loc[sam]
    sub=e[sam].copy();sub['gene']=[p2g.get(str(i),'') for i in sub.index];sub=sub[sub.gene!='']
    ge=sub.groupby('gene',sort=False).median(numeric_only=True)
    return m,ge,sam
md,gd,sd=load('GSE36059'); me,ge,se=load('GSE48581')
assert all(g in gd.index and g in ge.index for g in sel)
Xd=gd.loc[sel,sd].T.to_numpy(float); yd=md.y.to_numpy(int)
Xe=ge.loc[sel,se].T.to_numpy(float); ye=me.y.to_numpy(int)
med=np.nanmedian(Xd,axis=0)
for A in [Xd,Xe]:
    rr,cc=np.where(~np.isfinite(A)); A[rr,cc]=med[cc]
sc=StandardScaler().fit(Xd)
mdl=LogisticRegression(C=1.,penalty='l2',solver='liblinear',class_weight='balanced',max_iter=3000,random_state=20261044).fit(sc.transform(Xd),yd)
p=mdl.predict_proba(sc.transform(Xe))[:,1]
res={'method':'Direct LLM top-50 ranking (DeepSeek)','k':50,'selected_genes':sel,
     'external_auroc':float(roc_auc_score(ye,p)),
     'external_auprc':float(average_precision_score(ye,p)),
     'external_balanced_accuracy':float(balanced_accuracy_score(ye,(p>=.5).astype(int))),
     'reference_auroc':0.7855643656716418,'selective_auroc':0.8115671641791046,
     'note':'Pure LLM selection from the frozen 2000-gene universe; downstream evaluator exactly matches the frozen Renal external evaluator. This is a post-hoc additional baseline.'}
(OUT/'RENAL_DIRECT_LLM_RANK_RESULT.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
