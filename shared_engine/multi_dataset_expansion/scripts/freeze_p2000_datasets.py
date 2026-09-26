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
import hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
OUT=ROOT/"03_FROZEN_DATA"
OUT.mkdir(parents=True,exist_ok=True)

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def save_task(name,Xdev,ydev,devmeta,Xval,yval,valmeta,features,notes):
    d=OUT/name
    d.mkdir(parents=True,exist_ok=True)
    assert Xdev.shape==(len(ydev),2000)
    assert Xval.shape==(len(yval),2000)
    assert len(features)==2000 and features.gene_symbol.is_unique
    np.save(d/"X_development.npy",Xdev.astype(np.float64))
    np.save(d/"y_development.npy",np.asarray(ydev,dtype=np.int8))
    np.save(d/"X_sealed_validation.npy",Xval.astype(np.float64))
    np.save(d/"y_sealed_validation.npy",np.asarray(yval,dtype=np.int8))
    devmeta.to_csv(d/"development_samples.csv",index=False)
    valmeta.to_csv(d/"sealed_validation_samples.csv",index=False)
    features.to_csv(d/"features_p2000.csv",index=False)
    files=["X_development.npy","y_development.npy","X_sealed_validation.npy","y_sealed_validation.npy",
           "development_samples.csv","sealed_validation_samples.csv","features_p2000.csv"]
    manifest={
      "task":name,"candidate_p":2000,
      "development_n":int(len(ydev)),"development_positive":int(np.sum(ydev)),
      "sealed_validation_n":int(len(yval)),"sealed_validation_positive":int(np.sum(yval)),
      "feature_order":"development-X-only variance descending; deterministic rank from approved candidate audit",
      "sealed_validation_policy":"materialized only for schema/integrity; MUST NOT be loaded by reference/nested tuning/routing code before final freeze",
      "notes":notes,
      "sha256":{f:sha256(d/f) for f in files},
    }
    (d/"FREEZE_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest

# Breast
B=ROOT/"01_GSE25055_25065"
cand=pd.read_csv(B/"candidate_review_v2/GSE25055_PRIMARY_CANDIDATE_TOP2000.csv")
genes=cand.gene.astype(str).tolist()
devx=pd.read_parquet(B/"processed/GSE25055_protein_coding_gene_mean_expression.parquet").set_index("Gene_Symbol")
valx=pd.read_parquet(B/"processed/GSE25065_protein_coding_gene_mean_expression.parquet").set_index("Gene_Symbol")
dm=pd.read_csv(B/"processed/GSE25055_official_metadata.csv")
vm=pd.read_csv(B/"processed/GSE25065_official_metadata.csv")
dm=dm[dm.pathologic_response_pcr_rd.isin(["pCR","RD"])].copy()
vm=vm[vm.pathologic_response_pcr_rd.isin(["pCR","RD"])].copy()
dev_ids=dm.Sample_ID.astype(str).tolist(); val_ids=vm.Sample_ID.astype(str).tolist()
assert set(genes).issubset(devx.index) and set(genes).issubset(valx.index)
Xdev=devx.loc[genes,dev_ids].T.to_numpy(float)
Xval=valx.loc[genes,val_ids].T.to_numpy(float)
ydev=(dm.pathologic_response_pcr_rd.eq("pCR")).astype(int).to_numpy()
yval=(vm.pathologic_response_pcr_rd.eq("pCR")).astype(int).to_numpy()
reg=pd.read_csv(B/"candidate_review_v2/GPL96_PROBE_IDENTIFIER_REGISTRY.csv",dtype=str)
m=reg[(reg.mapping_status=="GENEID_EXACT")&(reg.type_of_gene=="protein-coding")].drop_duplicates(["current_symbol","platform_gene_id"])
sym2gid=m.drop_duplicates("current_symbol").set_index("current_symbol").platform_gene_id
feat=pd.DataFrame({"feature_index":np.arange(2000),"gene_symbol":genes,
                   "GeneID":[sym2gid.get(g,"") for g in genes],
                   "development_variance":cand.variance.to_numpy(float),
                   "development_variance_rank":cand["rank"].to_numpy(int)})
dmeta=pd.DataFrame({"sample_index":np.arange(len(dm)),"sample_id":dev_ids,"y":ydev,
                    "source":dm["source"].astype(str).to_numpy()})
vmeta=pd.DataFrame({"sample_index":np.arange(len(vm)),"sample_id":val_ids,"y":yval,
                    "source":vm["source"].astype(str).to_numpy()})
mb=save_task("BREAST_GSE25055_GSE25065",Xdev,ydev,dmeta,Xval,yval,vmeta,feat,
             ["GSE25055 development task n=306; GSE25065 sealed validation n=182",
              "Official deposited MAS5/log2/reference-scaled expression; no RMA reprocessing",
              "GPL96 unambiguous GeneID -> current NCBI protein-coding; mean-collapse probes per GeneID",
              "p=2000 explicitly user-approved on 2026-09-19"])

# Sepsis
S=ROOT/"02_GSE65682"
cand=pd.read_csv(S/"candidate_review_v2/GSE65682_PRIMARY_CANDIDATE_TOP2000.csv")
genes=cand.gene.astype(str).tolist()
x=pd.read_parquet(S/"processed/GSE65682_protein_coding_gene_expression_479.parquet").set_index("Gene_Symbol")
ph=pd.read_csv(S/"processed/GSE65682_TASK479_PHENOTYPE.csv")
dev=ph[ph.endotype_cohort.eq("discovery")].copy()
val=ph[ph.endotype_cohort.eq("validation")].copy()
dev_ids=dev.Sample.astype(str).tolist(); val_ids=val.Sample.astype(str).tolist()
assert set(genes).issubset(x.index)
Xdev=x.loc[genes,dev_ids].T.to_numpy(float); Xval=x.loc[genes,val_ids].T.to_numpy(float)
ydev=pd.to_numeric(dev.mortality_event_28days).astype(int).to_numpy()
yval=pd.to_numeric(val.mortality_event_28days).astype(int).to_numpy()
reg=pd.read_csv(S/"candidate_review_v2/GSE65682_IDENTIFIER_REGISTRY.csv",dtype=str).fillna("")
m=reg[(reg.type_of_gene=="protein-coding")&(reg.GeneID.ne(""))].drop_duplicates(["current_symbol","GeneID"])
sym2gid=m.drop_duplicates("current_symbol").set_index("current_symbol").GeneID
feat=pd.DataFrame({"feature_index":np.arange(2000),"gene_symbol":genes,
                   "GeneID":[sym2gid.get(g,"") for g in genes],
                   "development_variance":cand.variance.to_numpy(float),
                   "development_variance_rank":cand["rank"].to_numpy(int)})
dmeta=pd.DataFrame({"sample_index":np.arange(len(dev)),"sample_id":dev_ids,"y":ydev,
                    "endotype_cohort":dev.endotype_cohort.astype(str).to_numpy()})
vmeta=pd.DataFrame({"sample_index":np.arange(len(val)),"sample_id":val_ids,"y":yval,
                    "endotype_cohort":val.endotype_cohort.astype(str).to_numpy()})
ms=save_task("SEPSIS_GSE65682",Xdev,ydev,dmeta,Xval,yval,vmeta,feat,
             ["GSE65682 discovery n=263; sealed validation n=216",
              "Target is official mortality_event_28days; development 69 deaths, validation 45",
              "GEO-derived 11,518 gene-level resource; exact/unique-synonym NCBI mapping; protein-coding; alias mean-collapse",
              "p=2000 explicitly user-approved on 2026-09-19"])

(OUT/"FREEZE_SUMMARY.json").write_text(json.dumps({"breast":mb,"sepsis":ms},indent=2),encoding="utf-8")
print(json.dumps({"breast":{k:v for k,v in mb.items() if k!="sha256"},
                  "sepsis":{k:v for k,v in ms.items() if k!="sha256"}},indent=2))
