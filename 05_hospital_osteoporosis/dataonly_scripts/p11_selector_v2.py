#!/usr/bin/env python3
"""p11 - v2 baseline verification + primary (lumbar+hip) selector resampling.

Uses ONLY the v2 masked solver: intercept and `site` unpenalized, clinical block penalized.
  * baseline : L2-penalized clinical block, nested grouped CV (5 seeds x 5 folds), lam by inner
               grouped CV; plus forced-context-only reference, a permuted-label null, and an
               all-sites sensitivity.
  * selector : L1-penalized clinical block, B = 200 patient-group subsamples at 80 % of patients,
               imputation/scaling/lam-selection/fit all re-done inside every draw.
Outputs FEATURE_STABILITY_V2.csv, TOPK_SET_STABILITY_V2.csv, resample_scores_v2.npz,
BASELINE_V2.json, ALL_SITES_SENSITIVITY_V2.csv.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from common import D, SEEDS, N_FOLDS, N_RESAMPLES, RESAMPLE_FRACTION, json_dump
import v2_core as V

FISTA_TOL = 1e-8          # the active-set polish fixes the final accuracy, so FISTA can stop early
FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


man = json.load(open(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_manifest.json"))
F = man["selectable_features"]
P = len(F)
full = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv").merge(
    pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_y_and_context.csv",
                usecols=["patient_uid", "site", "y"]), on=["patient_uid", "site"])
full = full.reset_index(drop=True)
prim = full[full.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
check("primary scope = lumbar+hip (892 rows / 664 patients)",
      len(prim) == 892 and prim.patient_uid.nunique() == 664,
      f"{len(prim)}/{prim.patient_uid.nunique()}")


def folds(y, g, seed, n_splits):
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed).split(np.zeros(len(y)), y, g))


def metrics(y, p):
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, (p >= 0.5).astype(int))))


def select_lam(Xr, s, y, g, grid, kind, seed, n_inner=4):
    best, best_lam = -np.inf, grid[0]
    for lam in grid:
        aucs = []
        for tri, vai in folds(y, g, seed + int(round(lam * 100)), n_inner):
            if len(np.unique(y[tri])) < 2 or len(np.unique(y[vai])) < 2:
                continue
            w, _, sc, _ = V.fit_full(Xr[tri], s[tri], y[tri], lam, kind=kind)
            aucs.append(roc_auc_score(y[vai], V.predict_full(w, sc, Xr[vai], s[vai])))
        m = float(np.mean(aucs)) if aucs else -np.inf
        if m > best:
            best, best_lam = m, lam
    return best_lam, best


def nested_cv(df, grid, kind, seeds, n_splits=N_FOLDS):
    Xr, s, y, g = df[F].to_numpy(float), df.site.to_numpy(), df.y.to_numpy(int), \
        df.patient_uid.to_numpy()
    rows = []
    for seed in seeds:
        for fi, (tri, vai) in enumerate(folds(y, g, seed, n_splits)):
            lam, inner = select_lam(Xr[tri], s[tri], y[tri], g[tri], grid, kind, seed + fi)
            w, _, sc, _ = V.fit_full(Xr[tri], s[tri], y[tri], lam, kind=kind)
            pv = V.predict_full(w, sc, Xr[vai], s[vai])
            rows.append(dict(seed=seed, fold=fi, lam=lam, inner_auc=inner,
                             n_train=len(tri), n_valid=len(vai),
                             **metrics(y[vai], pv)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ baseline (primary scope)
BASE_GRID = V.LAM_GRID
base = nested_cv(prim, BASE_GRID, "l2", SEEDS)
check("baseline produced 25 fold evaluations", len(base) == 25, f"{len(base)}")

# forced-context-only reference
Xr, s, y, g = prim[F].to_numpy(float), prim.site.to_numpy(), prim.y.to_numpy(int), \
    prim.patient_uid.to_numpy()
ref_rows = []
for seed in SEEDS:
    for fi, (tri, vai) in enumerate(folds(y, g, seed, N_FOLDS)):
        Xd, _ = V.design(np.zeros((len(tri), P)), s[tri], include_site=True)
        Xv, _ = V.design(np.zeros((len(vai), P)), s[vai], include_site=True)
        # zeroed clinical block -> the fit can only use intercept + site
        w, _ = V.fit_masked(np.ascontiguousarray(Xd), y[tri], np.ones(Xd.shape[1]), 1e-6, kind="l2")
        ref_rows.append(dict(seed=seed, fold=fi,
                             **metrics(y[vai], V.sigmoid(Xv @ w))))
ref = pd.DataFrame(ref_rows)

# permuted-label null (labels permuted within site stratum, masked L2 pipeline)
rng = np.random.default_rng(20260918)
null_rows = []
for rep in range(2):
    yp = y.copy()
    for st in np.unique(s):
        idx = np.where(s == st)[0]
        yp[idx] = rng.permutation(yp[idx])
    for seed in SEEDS[:2]:
        for fi, (tri, vai) in enumerate(folds(yp, g, seed, N_FOLDS)):
            if len(np.unique(yp[tri])) < 2 or len(np.unique(yp[vai])) < 2:
                continue
            lam, _ = select_lam(Xr[tri], s[tri], yp[tri], g[tri], BASE_GRID, "l2", seed, 3)
            w, _, sc, _ = V.fit_full(Xr[tri], s[tri], yp[tri], lam, kind="l2")
            null_rows.append(dict(repeat=rep, seed=seed, fold=fi,
                                 **metrics(yp[vai], V.predict_full(w, sc, Xr[vai], s[vai]))))
nul = pd.DataFrame(null_rows)

# all-sites sensitivity
allsites = nested_cv(full, BASE_GRID, "l2", [SEEDS[0], SEEDS[1]])
sens = pd.DataFrame({
    "scope": ["lumbar_hip_primary", "all_sites_sensitivity", "forced_context_only_primary",
              "permuted_label_null_primary"],
    "n_rows": [len(prim), len(full), len(prim), len(prim)],
    "n_patients": [prim.patient_uid.nunique(), full.patient_uid.nunique(),
                   prim.patient_uid.nunique(), prim.patient_uid.nunique()],
    "n_evaluations": [len(base), len(allsites), len(ref), len(nul)],
    "auroc_mean": [base.auroc.mean(), allsites.auroc.mean(), ref.auroc.mean(), nul.auroc.mean()],
    "auroc_sd": [base.auroc.std(), allsites.auroc.std(), ref.auroc.std(), nul.auroc.std()],
    "auprc_mean": [base.auprc.mean(), allsites.auprc.mean(), ref.auprc.mean(), nul.auprc.mean()],
    "bal_acc_mean": [base.balanced_accuracy.mean(), allsites.balanced_accuracy.mean(),
                     ref.balanced_accuracy.mean(), nul.balanced_accuracy.mean()],
}).round(6)
sens.to_csv(f"{D['04_DATA_ONLY_BASELINE']}/ALL_SITES_SENSITIVITY_V2.csv", index=False,
            encoding="utf-8-sig")
print(f"\nbaseline (lumbar+hip, unpenalized site context): AUROC {base.auroc.mean():.4f} "
      f"±{base.auroc.std():.4f}  AUPRC {base.auprc.mean():.4f}  balacc {base.balanced_accuracy.mean():.4f}")
print(f"forced-context-only {ref.auroc.mean():.4f} ±{ref.auroc.std():.4f} | "
      f"null {nul.auroc.mean():.4f} ±{nul.auroc.std():.4f} | "
      f"all-sites sensitivity {allsites.auroc.mean():.4f} ±{allsites.auroc.std():.4f}")
check("signal is still present with the true unpenalized site context",
      base.auroc.mean() - nul.auroc.mean() > 3 * max(nul.auroc.std(), 0.01),
      f"{base.auroc.mean():.4f} vs null {nul.auroc.mean():.4f}")
check("the task is still not saturated", base.auroc.mean() < 0.95,
      f"AUROC {base.auroc.mean():.4f}")
check("lam was not pinned to the grid edge for a majority of folds",
      float((base.lam == BASE_GRID[-1]).mean()) < 0.9,
      f"lam=min(grid) in {int((base.lam == BASE_GRID[-1]).sum())}/{len(base)} folds; "
      f"selected: {base.lam.value_counts().to_dict()}")

# ------------------------------------------------------------------ L1 selector resampling
uids = np.unique(g)
scores = np.zeros((N_RESAMPLES, P))
ranks = np.zeros((N_RESAMPLES, P))
lams = np.zeros(N_RESAMPLES)
nz = np.zeros(N_RESAMPLES, int)
nz_all = np.zeros(N_RESAMPLES, int)
pat_per_draw = np.zeros(N_RESAMPLES, int)
row_per_draw = np.zeros(N_RESAMPLES, int)
for b in range(N_RESAMPLES):
    rngb = np.random.default_rng(700000 + b)
    for _ in range(50):
        take = rngb.choice(len(uids), size=int(round(RESAMPLE_FRACTION * len(uids))), replace=False)
        sel = np.isin(g, uids[take])
        if len(np.unique(y[sel])) == 2 and y[sel].mean() > 0.05:
            break
    idx = np.where(sel)[0]
    lam_b, _ = select_lam(Xr[idx], s[idx], y[idx], g[idx], V.LAM_GRID, "l1", 900 + b)
    w, info, _, n_ctx = V.fit_full(Xr[idx], s[idx], y[idx], lam_b, kind="l1")
    sc_b = V.clinical_scores(w, n_ctx)
    scores[b] = sc_b
    ranks[b] = rankdata(-sc_b, method="average")
    lams[b] = lam_b
    nz[b] = int(np.sum(sc_b > 1e-12))
    nz_all[b] = info["n_nonzero"]
    pat_per_draw[b] = int(len(np.unique(g[idx])))
    row_per_draw[b] = int(len(idx))
    if (b + 1) % 50 == 0:
        print(f"  resample {b + 1}/{N_RESAMPLES}")

np.savez_compressed(f"{D['05_TOPK_STABILITY']}/resample_scores_v2.npz",
                    scores=scores, ranks=ranks, lams=lams, nz=nz, nz_all=nz_all,
                    features=np.array(F),
                    n_patients_per_draw=pat_per_draw, n_rows_per_draw=row_per_draw)
print(f"\nresampling: mean {pat_per_draw.mean():.0f} patients / {row_per_draw.mean():.0f} rows per draw")
print(f"\nL1 selector: lam selected {dict(zip(*np.unique(lams, return_counts=True)))}")
print(f"non-zero clinical coefficients per draw: mean {nz.mean():.1f}, range [{nz.min()}, {nz.max()}] of {P}")
print(f"zero-coefficient clinical features per draw: mean {(P - nz).mean():.1f}, max {(P - nz).max()}")

# ------------------------------------------------------------------ feature stability table
K_GRID = [5, 10, 15, 20]
st = pd.DataFrame({
    "feature_name": F,
    "feature_category": [("DEMOGRAPHIC" if f in ("DXA_性别", "DXA_年龄")
                          else "ANTHROPOMETRIC" if f in ("身高_cm", "体重_kg") else "LAB") for f in F],
    "derived_feature_flag": [f in ("AST/ALT", "白球比", "尿素：肌酐", "估算肾小球滤过率") for f in F],
    "median_rank": np.median(ranks, axis=0), "mean_rank": ranks.mean(axis=0),
    "rank_iqr": np.percentile(ranks, 75, axis=0) - np.percentile(ranks, 25, axis=0),
    "rank_sd": ranks.std(axis=0), "rank_min": ranks.min(axis=0), "rank_max": ranks.max(axis=0),
    "rank_range": ranks.max(axis=0) - ranks.min(axis=0),
    "mean_abs_score": scores.mean(axis=0), "median_abs_score": np.median(scores, axis=0),
    "sd_abs_score": scores.std(axis=0), "zero_score_freq": (scores <= 1e-12).mean(axis=0),
})
for k in K_GRID:
    st[f"pi_top{k}"] = (ranks <= k).mean(axis=0)
    st[f"boundary_presence_top{k}"] = ((ranks >= k - 1) & (ranks <= k + 1)).mean(axis=0)
st["most_used_k"] = st[[f"pi_top{k}" for k in K_GRID]].idxmax(axis=1).str.replace("pi_top", "k")
st = st.sort_values("pi_top10", ascending=False).reset_index(drop=True)
st.to_csv(f"{D['05_TOPK_STABILITY']}/FEATURE_STABILITY_V2.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ top-k set stability
from collections import Counter
rows = []
for k in K_GRID:
    valid = nz >= k
    sets = [frozenset(np.where(ranks[b] <= k)[0]) for b in range(N_RESAMPLES)]
    js = np.array([len(sets[a] & sets[c]) / max(len(sets[a] | sets[c]), 1)
                   for a in range(N_RESAMPLES) for c in range(a + 1, N_RESAMPLES)])
    sets_v = [sets[b] for b in range(N_RESAMPLES) if valid[b]]
    js_v = np.array([len(sets_v[a] & sets_v[c]) / max(len(sets_v[a] | sets_v[c]), 1)
                     for a in range(len(sets_v)) for c in range(a + 1, len(sets_v))]) \
        if len(sets_v) > 1 else np.array([1.0])
    cnt = Counter(tuple(sorted(s)) for s in sets)
    cnt_v = Counter(tuple(sorted(s)) for s in sets_v)
    pi = (ranks <= k).mean(axis=0)
    rows.append(dict(
        k=k, n_resamples=N_RESAMPLES,
        n_valid_resamples=int(valid.sum()), fraction_valid=round(float(valid.mean()), 4),
        n_invalid_resamples=int((~valid).sum()),
        min_nonzero=int(nz.min()), max_nonzero=int(nz.max()),
        jaccard_mean_all=round(float(js.mean()), 4), jaccard_median_all=round(float(np.median(js)), 4),
        jaccard_p05_all=round(float(np.percentile(js, 5)), 4),
        jaccard_mean_valid=round(float(js_v.mean()), 4),
        jaccard_median_valid=round(float(np.median(js_v)), 4),
        jaccard_p05_valid=round(float(np.percentile(js_v, 5)), 4),
        n_distinct_sets_all=len(cnt), n_distinct_sets_valid=len(cnt_v),
        modal_set_frequency_all=round(cnt.most_common(1)[0][1] / N_RESAMPLES, 4),
        modal_set_frequency_valid=round(cnt_v.most_common(1)[0][1] / max(len(sets_v), 1), 4),
        mean_set_size=round(float(np.mean([len(x) for x in sets])), 3),
        n_features_pi_ge_095=int((pi >= 0.95).sum()),
        n_features_borderline_010_090=int(((pi > 0.10) & (pi < 0.90)).sum()),
        n_features_never=int((pi == 0).sum()), n_features_always=int((pi == 1).sum()),
    ))
setstab = pd.DataFrame(rows)
setstab.to_csv(f"{D['05_TOPK_STABILITY']}/TOPK_SET_STABILITY_V2.csv", index=False,
               encoding="utf-8-sig")
print("\n", setstab[["k", "fraction_valid", "jaccard_mean_all", "jaccard_mean_valid",
                     "n_distinct_sets_all", "mean_set_size", "n_features_pi_ge_095",
                     "n_features_borderline_010_090"]].to_string(index=False))
check("ranking instability still exists under v2 (no k is near-fixed)",
      setstab.jaccard_mean_valid.max() < 0.9, f"best mean Jaccard {setstab.jaccard_mean_valid.max():.4f}")

json_dump({"primary_scope_rows": int(len(prim)), "primary_scope_patients": int(prim.patient_uid.nunique()),
           "prevalence": round(float(y.mean()), 6),
           "baseline": {"auroc_mean": round(float(base.auroc.mean()), 4),
                        "auroc_sd": round(float(base.auroc.std()), 4),
                        "auprc_mean": round(float(base.auprc.mean()), 4),
                        "bal_acc_mean": round(float(base.balanced_accuracy.mean()), 4),
                        "lam_selected_counts": {str(k): int(v) for k, v in
                                                base.lam.value_counts().items()}},
           "forced_context_only": {"auroc_mean": round(float(ref.auroc.mean()), 4),
                                   "auroc_sd": round(float(ref.auroc.std()), 4)},
           "null": {"auroc_mean": round(float(nul.auroc.mean()), 4),
                    "auroc_sd": round(float(nul.auroc.std()), 4)},
           "all_sites_sensitivity": {"auroc_mean": round(float(allsites.auroc.mean()), 4),
                                     "auroc_sd": round(float(allsites.auroc.std()), 4),
                                     "auprc_mean": round(float(allsites.auprc.mean()), 4)},
           "selector": {"lam_grid": V.LAM_GRID,
                        "lam_selected_counts": {str(k): int(v) for k, v in
                                                zip(*np.unique(lams, return_counts=True))},
                        "nonzero_mean": round(float(nz.mean()), 2),
                        "nonzero_range": [int(nz.min()), int(nz.max())],
                        "zero_features_max": int(P - nz.min())},
           "fista_tol": FISTA_TOL, "B": N_RESAMPLES, "fraction": RESAMPLE_FRACTION,
           "batch2_used": False, "llm_used": False, "literature_used": False},
          f"{D['04_DATA_ONLY_BASELINE']}/BASELINE_V2.json")
print(f"\nP11: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
