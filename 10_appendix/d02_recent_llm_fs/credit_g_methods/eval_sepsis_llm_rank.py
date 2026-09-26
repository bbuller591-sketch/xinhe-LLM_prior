

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
import json,re
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'EXISTING_METHOD_COMPARISON_20260925'
PKG=ROOT/'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919/01_FROZEN_DATA'
key=json.loads((OUT/'sepsis_gene_key.json').read_text())
resp=json.loads((OUT/'SEPSIS_DIRECT_LLM_RESPONSES.json').read_text())['tests'][0]['content']
ks=re.findall(r'\[(g\d{4})\]',resp)
assert len(ks)==50 and len(set(ks))==50
sel=[key[k] for k in ks]
feat=pd.read_csv(PKG/'features_p1500.csv').gene_symbol.astype(str).tolist()
idx=[feat.index(g) for g in sel]
X=np.load(PKG/'X_development.npy'); y=np.load(PKG/'y_development.npy').astype(int)
sp=pd.read_csv(PKG/'OUTER_SPLITS.csv')
rows=[]
for fold in sorted(sp.fold.unique()):
    tr=sp[(sp.fold==fold)&(sp.role=='outer_train')].sample_index.astype(int).to_numpy()
    va=sp[(sp.fold==fold)&(sp.role=='outer_validation')].sample_index.astype(int).to_numpy()
    sc=StandardScaler().fit(X[tr][:,idx])
    mdl=LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=3000,random_state=20260919+int(fold))
    mdl.fit(sc.transform(X[tr][:,idx]),y[tr])
    p=mdl.predict_proba(sc.transform(X[va][:,idx]))[:,1]
    app=float(average_precision_score(y[va],p)); apn=float(average_precision_score(1-y[va],1-p))
    rows.append({'fold':int(fold),'auroc':float(roc_auc_score(y[va],p)),'ap_positive':app,'macro_ap':(app+apn)/2,
                 'balanced_accuracy':float(balanced_accuracy_score(y[va],(p>=.5).astype(int))),'n_val':len(va)})
df=pd.DataFrame(rows)
res={'method':'Direct LLM top-50 ranking (DeepSeek)','k':50,'selected_genes':sel,
     'mean_auroc':float(df.auroc.mean()),'sd_auroc':float(df.auroc.std(ddof=1)),
     'mean_macro_ap':float(df.macro_ap.mean()),'mean_ap_positive':float(df.ap_positive.mean()),
     'mean_balanced_accuracy':float(df.balanced_accuracy.mean()),
     'reference_auroc':0.6287301587301587,'selective_auroc':0.6553968253968254,
     'reference_macro_ap':0.6641437011392607,'selective_macro_ap':0.68588484652087,
     'note':'Post-hoc pure-LLM top-50 support; evaluated on the exact frozen five outer splits with the same L2 balanced logistic evaluator used in the nested package.'}
df.to_csv(OUT/'SEPSIS_DIRECT_LLM_FOLD_RESULTS.csv',index=False)
(OUT/'SEPSIS_DIRECT_LLM_RANK_RESULT.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
print(df.to_string(index=False))
