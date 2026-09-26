#!/usr/bin/env python3
"""GBM s4 - build the canonical candidate universe.

Rule (X-only, no y at any point):
  raw 632 x 23,257 counts (already library-size normalised upstream)
  -> technical QC: drop constant columns (none expected)
  -> variance-stabilising transform: log1p   (the only X-only transform applied)
  -> per-gene variance (ddof=0) on log1p(X)
  -> descending variance, tie-break = ascending original column index
  -> top 2000

Gene identity: versioned Ensembl id (kept), stable Ensembl id, symbol, entrez
(reused from the symbol->entrez table assembled for the TCGA task), biotype and
Ensembl cross-check from the GDC /genes API.
"""
import json, os, time
from collections import Counter
import numpy as np
import pandas as pd
import requests

B = "./gbm_canonical_recovery_20260917"
REC = "./tcga_gene_mapping_recovery_20260916"
DL_RAW = "./high_dim_dataset_recovery_20260916/gbm_scrna/raw/rna2.csv"
GDC_CACHE = f"{B}/metadata/gdc_genes_by_ensembl.json"
K = 2000
t0 = time.time()

dl = pd.read_csv(DL_RAW)
y = dl.iloc[:, -1].to_numpy().astype(int)
X = dl.iloc[:, :-1].to_numpy(dtype=np.float64)
gene_ids = [str(c).strip('"') for c in dl.columns[:-1]]
assert X.shape == (632, 23257), X.shape
print(f"raw X {X.shape}  y {dict(Counter(y))}")

# ---------------------------------------------------------------- technical QC
var_raw = X.var(axis=0, ddof=0)
const = np.where(var_raw == 0)[0]
nz = (X != 0).mean(axis=0)
print(f"constant columns: {len(const)}   all-zero columns: {int((nz == 0).sum())}")
print(f"library sizes (row sums): min {X.sum(1).min():.0f} max {X.sum(1).max():.0f}")

# ---------------------------------------------------------------- transform
Xlog = np.log1p(X)
var_log = Xlog.var(axis=0, ddof=0)
# total order: variance desc, tie-break original index asc
order = np.lexsort((np.arange(len(gene_ids)), -var_log))
cand_idx = np.sort(order[:K])                    # ascending original index (frozen convention)
cand_rank = np.empty(len(gene_ids), dtype=int)
cand_rank[order] = np.arange(1, len(gene_ids) + 1)
print(f"log1p variance: top-2000 selected; "
      f"var[rank1] = {var_log[order[0]]:.6f}, var[rank2000] = {var_log[order[K-1]]:.6f}, "
      f"var[rank2001] = {var_log[order[K]]:.6f}")
overlap_raw = len(set(cand_idx.tolist()) & set(np.sort(np.argsort(-var_raw)[:K]).tolist()))
print(f"overlap with the raw(no-log) variance top-2000: {overlap_raw} / {K}")

# ---------------------------------------------------------------- gene identity
stable = [g.split(".")[0] for g in gene_ids]
ver = [g.split(".")[1] if "." in g else "" for g in gene_ids]
print(f"\nversioned ids: {sum(1 for v in ver if v)} / {len(ver)}  "
      f"(version range {min(int(v) for v in ver if v)}..{max(int(v) for v in ver if v)})")

if os.path.exists(GDC_CACHE):
    gdc = json.load(open(GDC_CACHE))
else:
    print("querying GDC /genes for symbol/biotype (batched) ...", flush=True)
    gdc = {}
    uniq = sorted(set(stable))
    for k in range(0, len(uniq), 200):
        chunk = uniq[k:k + 200]
        body = {"filters": {"op": "in", "content": {"field": "gene_id", "value": chunk}},
                "fields": "gene_id,symbol,biotype,chromosome", "size": str(len(chunk) * 2),
                "format": "JSON"}
        for a in range(4):
            try:
                r = requests.post("https://api.gdc.cancer.gov/genes", json=body, timeout=90)
                r.raise_for_status()
                for h in r.json()["data"]["hits"]:
                    gdc[h["gene_id"]] = {"symbol": h.get("symbol", ""), "biotype": h.get("biotype", "")}
                break
            except Exception as e:
                print(f"   retry {a}: {type(e).__name__}", flush=True)
                time.sleep(4)
        if k % 2000 == 0:
            print(f"   {k}/{len(uniq)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(gdc, open(GDC_CACHE, "w"))
print(f"GDC annotation available for {len(gdc)} Ensembl ids")
sym2ent = {}
for s, e in json.load(open(f"{REC}/metadata/gdc_symbol_to_ensembl.json")).items():
    pass
# symbol -> entrez from the official PanCanAtlas SYMBOL|ENTREZ column (reused table)
gm = pd.read_csv(f"{REC}/full_gene_mapping.csv", dtype=str, keep_default_na=False)
sym2ent = {s: e for s, e in zip(gm.gene_symbol, gm.entrez_id) if s and e}
print(f"symbol->entrez table (from the TCGA task): {len(sym2ent)} entries")

rows = []
sym_map = {g: (gdc.get(g.split('.')[0], {}) or {}).get("symbol", "") for g in gene_ids}
for i, g in enumerate(gene_ids):
    s = sym_map[g]
    rows.append({
        "original_gene_index": i,
        "original_gene_id": g,
        "ensembl_gene_id_versioned": g,
        "ensembl_gene_id_stable": stable[i],
        "gene_symbol": s,
        "entrez_id": sym2ent.get(s, ""),
        "biotype": (gdc.get(stable[i], {}) or {}).get("biotype", ""),
        "variance_log1p": var_log[i],
        "variance_raw": var_raw[i],
        "mean_log1p": Xlog[:, i].mean(),
        "nonzero_fraction": nz[i],
        "x_only_rank": int(cand_rank[i]),
        "retained": bool(cand_rank[i] <= K),
        "mapping_status": ("RESOLVED" if (gdc.get(stable[i]) and sym_map[g]) else
                           ("ENSEMBL_ONLY_NOT_IN_GDC" if not gdc.get(stable[i]) else "SYMBOL_EMPTY")),
    })
fullmap = pd.DataFrame(rows)
fullmap.to_csv(f"{B}/metadata/full_gene_mapping_all23257.csv", index=False)
print(f"\nfull gene mapping rows {len(fullmap)}")
print("  mapping_status:", dict(fullmap.mapping_status.value_counts()))
print("  symbols with entrez:", int((fullmap.entrez_id != "").sum()))

# ---------------------------------------------------------------- candidate table
sel = np.argsort(cand_rank)[:K]                  # canonical order = rank order
cand = fullmap.iloc[sel].reset_index(drop=True)
cand.insert(0, "candidate_col_index", np.arange(K))
cand = cand.rename(columns={"original_gene_index": "original_uci_feature_index"})
cand.to_csv(f"{B}/metadata/candidate_features_all2000.csv", index=False)
print(f"\ncandidate features {cand.shape} (canonical order = variance rank order)")
print(cand.head(5)[["candidate_col_index", "original_uci_feature_index", "original_gene_id",
                    "gene_symbol", "entrez_id", "variance_log1p", "x_only_rank"]].to_string(index=False))

# ---------------------------------------------------------------- cross-check vs frozen
fz = pd.read_csv("./high_dim_dataset_recovery_20260916/gbm_scrna/"
                 "candidate_universe/candidate_features.csv", keep_default_na=False)
mine = set(cand.original_uci_feature_index.tolist())
froz = set(fz.feature_index.astype(int).tolist())
print(f"\ncross-check vs the frozen GBM universe: identical set = {mine == froz} "
      f"(|sym diff| = {len(mine ^ froz)})")
json.dump({"p_raw": int(X.shape[1]), "n_cells": int(X.shape[0]), "k": K,
           "constant_columns": int(len(const)), "all_zero_columns": int((nz == 0).sum()),
           "overlap_with_raw_variance_top2000": int(overlap_raw),
           "identical_to_frozen_universe": bool(mine == froz),
           "var_rank1": float(var_log[order[0]]), "var_rank2000": float(var_log[order[K-1]]),
           "var_rank2001": float(var_log[order[K]])},
          open(f"{B}/metadata/s4_candidate_rule.json", "w"), indent=2)
print(f"done {time.time()-t0:.0f}s")
