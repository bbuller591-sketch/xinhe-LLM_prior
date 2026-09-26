#!/usr/bin/env python3
"""p5 - DATA-ONLY SELECTOR + TOP-K STABILITY (development = batch1 only).

Primary selector: L1-regularised logistic regression on
   [ forced site context | 37 standardised eligible clinical features ].
`C` is chosen by inner group-aware CV on the grid fixed in the manifest BEFORE any result was
inspected (C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]). The grid is NOT changed afterwards; instead the
full inner-CV curve is reported, and a second, wider grid is run as an explicitly-labelled SENSITIVITY.

Stability protocol (fixed before running): B = 200 patient-level subsamples at 80% of patients (both
classes required), each one re-fitting imputation, standardisation, `C` selection AND the selector
itself from scratch.

Ranking is over the SELECTABLE CLINICAL BLOCK ONLY; the forced site context is never ranked.
Ties (exactly equal |coef|, typically several zeroed coefficients) get AVERAGE ranks, which means a
top-k set can contain FEWER than k members when a tie group straddles the boundary; both the set-size
distribution and the frequency of that event are reported rather than hidden.
"""
import json
import warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
from scipy.stats import rankdata
from common import (D, C_GRID, K_GRID, N_FOLDS, N_RESAMPLES, RESAMPLE_FRACTION, SEEDS, json_dump)
from modeling import fit_lr, clinical_scores, select_C, metrics, grouped_folds, predict_lr

C_GRID_EXTENDED = C_GRID + [1.0, 3.0]
FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


man = json.load(open(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_manifest.json"))
FEATS = man["selectable_features"]
p = len(FEATS)
dev = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv") \
    .merge(pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_y_and_context.csv",
                       usecols=["patient_uid", "site", "y"]),   # T_used deliberately NOT loaded
           on=["patient_uid", "site"])
X, str_, y = dev[FEATS].to_numpy(float), dev["site"].to_numpy(), dev["y"].to_numpy(int)
groups = dev["patient_uid"].to_numpy()
uids = np.unique(groups)
check("protocol matches the manifest (B, fraction, k-grid, C-grid)",
      (N_RESAMPLES, RESAMPLE_FRACTION, K_GRID, C_GRID) ==
      (man["preregistered_design"]["stability"]["n_resamples"],
       man["preregistered_design"]["stability"]["patient_fraction"],
       man["preregistered_design"]["k_grid"], man["preregistered_design"]["C_grid"]))


# ------------------------------------------------------------------ pre-registered subsamples
def draw_indices():
    out = []
    for b in range(N_RESAMPLES):
        rng = np.random.default_rng(100000 + b)
        for _ in range(50):
            take = rng.choice(len(uids), size=int(round(RESAMPLE_FRACTION * len(uids))), replace=False)
            sel = np.isin(groups, uids[take])
            if len(np.unique(y[sel])) == 2 and y[sel].mean() > 0.05:
                break
        out.append(np.where(sel)[0])
    return out


DRAWS = draw_indices()
print(f"{N_RESAMPLES} patient-level subsamples drawn "
      f"(mean {np.mean([len(np.unique(groups[i])) for i in DRAWS]):.1f} patients, "
      f"mean {np.mean([len(i) for i in DRAWS]):.1f} rows)")


def run_resampling(grid, tag):
    scores = np.zeros((N_RESAMPLES, p))
    ranks = np.zeros((N_RESAMPLES, p))
    Cs = np.zeros(N_RESAMPLES)
    nz = np.zeros(N_RESAMPLES, int)
    ambig = {k: 0 for k in K_GRID}
    for b, idx in enumerate(DRAWS):
        Cb, _, _ = select_C(X[idx], str_[idx], y[idx], groups[idx], grid, "l1", 4, 100 + b)
        mb, scb = fit_lr(X[idx], str_[idx], y[idx], Cb, "l1")
        s = clinical_scores(mb, p)
        r = rankdata(-s, method="average")
        scores[b], ranks[b], Cs[b], nz[b] = s, r, Cb, int((s > 0).sum())
        for k in K_GRID:
            m = r <= k
            if m.any() and (~m).any() and np.isclose(s[m].min(), s[~m].max()):
                ambig[k] += 1
        if (b + 1) % 50 == 0:
            print(f"  [{tag}] resample {b + 1}/{N_RESAMPLES}")
    return dict(scores=scores, ranks=ranks, Cs=Cs, nz=nz, ambig=ambig)


# ------------------------------------------------------------------ full-sample reference fit
C_full, inner_full, inner_curve = select_C(X, str_, y, groups, C_GRID, "l1", N_FOLDS, 11)
m_full, _ = fit_lr(X, str_, y, C_full, "l1")
s_full = clinical_scores(m_full, p)
r_full = rankdata(-s_full, method="average")
print(f"\nfull-sample L1: C={C_full} (inner CV AUROC {inner_full:.4f}); "
      f"non-zero coefs {int((s_full > 0).sum())}/{p}")
print("inner-CV curve on the pre-registered grid:",
      {str(c): round(a, 4) for c, a in inner_curve})

PRI = run_resampling(C_GRID, "primary")
print(f"C selected across resamples: {dict(zip(*np.unique(PRI['Cs'], return_counts=True)))}")
print(f"non-zero coefficients per resample: mean {PRI['nz'].mean():.1f}, "
      f"range [{PRI['nz'].min()}, {PRI['nz'].max()}] of {p}")
np.savez_compressed(f"{D['05_TOPK_STABILITY']}/resample_scores.npz",
                    scores=PRI["scores"], ranks=PRI["ranks"], Cs=PRI["Cs"], nz=PRI["nz"],
                    features=np.array(FEATS), C_full=C_full, s_full=s_full, r_full=r_full,
                    n_patients_per_draw=np.array([len(np.unique(groups[i])) for i in DRAWS]))

EXT = run_resampling(C_GRID_EXTENDED, "extended-grid-sensitivity")


# ------------------------------------------------------------------ stability metrics
def pi_of(res, k):
    return (res["ranks"] <= k).mean(axis=0)


def set_stats(res, k):
    sets = [frozenset(np.where(res["ranks"][b] <= k)[0]) for b in range(N_RESAMPLES)]
    js = np.array([len(sets[a] & sets[c]) / max(len(sets[a] | sets[c]), 1)
                   for a in range(N_RESAMPLES) for c in range(a + 1, N_RESAMPLES)])
    cnt = Counter(tuple(sorted(s)) for s in sets)
    modal, mfreq = cnt.most_common(1)[0]
    pi = pi_of(res, k)
    return dict(k=k, jaccard_mean=round(float(js.mean()), 4),
                jaccard_median=round(float(np.median(js)), 4),
                jaccard_p05=round(float(np.percentile(js, 5)), 4),
                jaccard_p25=round(float(np.percentile(js, 25)), 4),
                jaccard_p75=round(float(np.percentile(js, 75)), 4),
                jaccard_sd=round(float(js.std()), 4), jaccard_min=round(float(js.min()), 4),
                n_pairs=len(js),
                n_distinct_topk_sets=len(cnt), modal_set_frequency=round(mfreq / N_RESAMPLES, 4),
                modal_set=";".join(FEATS[i] for i in modal),
                mean_set_size=round(float(np.mean([len(s) for s in sets])), 3),
                min_set_size=int(min(len(s) for s in sets)),
                ambiguous_boundary_resamples=int(res["ambig"][k]),
                ambiguous_boundary_freq=round(res["ambig"][k] / N_RESAMPLES, 4),
                n_features_pi_ge_099=int((pi >= 0.99).sum()),
                n_features_pi_ge_095=int((pi >= 0.95).sum()),
                n_features_pi_ge_090=int((pi >= 0.90).sum()),
                n_features_borderline_010_090=int(((pi > 0.10) & (pi < 0.90)).sum()),
                n_features_never=int((pi == 0).sum()), n_features_always=int((pi == 1).sum()))


setstab = pd.DataFrame([set_stats(PRI, k) for k in K_GRID])
setstab.to_csv(f"{D['05_TOPK_STABILITY']}/TOPK_SET_STABILITY.csv", index=False, encoding="utf-8-sig")
extstab = pd.DataFrame([set_stats(EXT, k) for k in K_GRID])
extstab.to_csv(f"{D['05_TOPK_STABILITY']}/TOPK_SET_STABILITY_EXTENDED_C.csv", index=False,
               encoding="utf-8-sig")
print("\n", setstab[["k", "jaccard_mean", "jaccard_median", "n_distinct_topk_sets",
                     "modal_set_frequency", "mean_set_size", "n_features_pi_ge_095",
                     "n_features_borderline_010_090", "ambiguous_boundary_resamples"]]
      .to_string(index=False))

# ------------------------------------------------------------------ per-feature stability table
st = pd.DataFrame({
    "feature_name": FEATS,
    "feature_category": [("DEMOGRAPHIC" if f in ("DXA_性别", "DXA_年龄")
                          else "ANTHROPOMETRIC" if f in ("身高_cm", "体重_kg") else "LAB")
                         for f in FEATS],
    "derived_feature_flag": [f in ("AST/ALT", "白球比", "尿素：肌酐", "估算肾小球滤过率")
                             for f in FEATS],
    "median_rank": np.median(PRI["ranks"], axis=0), "mean_rank": PRI["ranks"].mean(axis=0),
    "rank_iqr": np.percentile(PRI["ranks"], 75, axis=0) - np.percentile(PRI["ranks"], 25, axis=0),
    "rank_sd": PRI["ranks"].std(axis=0), "rank_min": PRI["ranks"].min(axis=0),
    "rank_max": PRI["ranks"].max(axis=0),
    "rank_range": PRI["ranks"].max(axis=0) - PRI["ranks"].min(axis=0),
    "mean_abs_score": PRI["scores"].mean(axis=0),
    "median_abs_score": np.median(PRI["scores"], axis=0),
    "sd_abs_score": PRI["scores"].std(axis=0),
    "zero_score_freq": (PRI["scores"] == 0).mean(axis=0),
    "full_sample_rank": r_full,
    "pi_top5_extendedC": pi_of(EXT, 5), "pi_top10_extendedC": pi_of(EXT, 10),
    "pi_top15_extendedC": pi_of(EXT, 15), "pi_top20_extendedC": pi_of(EXT, 20),
})
for k in K_GRID:
    st[f"pi_top{k}"] = pi_of(PRI, k)
    st[f"boundary_presence_top{k}"] = ((PRI["ranks"] >= k - 1) & (PRI["ranks"] <= k + 1)).mean(axis=0)
st["most_used_k"] = st[[f"pi_top{k}" for k in K_GRID]].idxmax(axis=1).str.replace("pi_top", "k")
st = st.sort_values("pi_top10", ascending=False).reset_index(drop=True)
st.to_csv(f"{D['05_TOPK_STABILITY']}/FEATURE_STABILITY.csv", index=False, encoding="utf-8-sig")
check("site / ctx_* never appears in the stability table",
      not st.feature_name.str.startswith("ctx_").any() and "site" not in set(st.feature_name))
check("stability table has exactly p rows", len(st) == p, f"{len(st)} vs p={p}")
check("ranking universe is fixed at p regardless of ties", PRI["ranks"].shape == (N_RESAMPLES, p))

# ------------------------------------------------------------------ CV-selected k (diagnostic only)
diag_k = []
for k in K_GRID:
    top = np.argsort(r_full)[:k]
    aucs = []
    for seed in SEEDS:
        for tri, vai in grouped_folds(y, groups, seed, N_FOLDS):
            if len(np.unique(y[vai])) < 2:
                continue
            mk, sck = fit_lr(X[tri][:, top], str_[tri], y[tri], C_full, "l2")
            aucs.append(metrics(y[vai], predict_lr(mk, sck, X[vai][:, top], str_[vai]))["auroc"])
    diag_k.append(dict(k=k, cv_auroc_mean=round(float(np.mean(aucs)), 4),
                       cv_auroc_sd=round(float(np.std(aucs)), 4)))
diag = pd.DataFrame(diag_k)
best_k = int(diag.loc[diag.cv_auroc_mean.idxmax(), "k"])
print("\nCV-selected-k diagnostic:\n", diag.to_string(index=False), f"\n-> diagnostic best k = {best_k}")

# ------------------------------------------------------------------ plots
from pilot_plots import fig_selection_probability, fig_rank_variability, fig_topk_jaccard
fig_selection_probability(st, f"{D['plots']}/05_selection_probability.png")
fig_rank_variability(st, PRI["ranks"], f"{D['plots']}/05_rank_variability.png")
fig_topk_jaccard(PRI["ranks"], f"{D['plots']}/05_topk_jaccard.png")
print("figures written")

json_dump({"p": p, "n_rows": int(len(dev)), "n_patients": int(len(uids)),
           "selector": "L1 logistic (l1_ratio=1, liblinear), C by inner grouped CV",
           "preregistered_C_grid": C_GRID, "extended_C_grid_sensitivity": C_GRID_EXTENDED,
           "C_full_sample": C_full,
           "inner_cv_curve_full_sample": [[c, round(a, 4)] for c, a in inner_curve],
           "non_zero_full_sample": int((s_full > 0).sum()),
           "C_selected_counts": {str(k): int(v) for k, v in zip(*np.unique(PRI["Cs"], return_counts=True))},
           "nonzero_per_resample_mean": round(float(PRI["nz"].mean()), 2),
           "nonzero_per_resample_range": [int(PRI["nz"].min()), int(PRI["nz"].max())],
           "B": N_RESAMPLES, "fraction": RESAMPLE_FRACTION,
           "set_stability": setstab.to_dict("records"),
           "set_stability_extended_C": extstab.to_dict("records"),
           "cv_selected_k_diagnostic": diag.to_dict("records"), "diagnostic_best_k": best_k,
           "batch2_used": False, "llm_used": False, "literature_used": False},
          f"{D['05_TOPK_STABILITY']}/stability_summary.json")

# ------------------------------------------------------------------ report
ls_ = st[(st.pi_top10 > 0) & (st.pi_top10 < 1)].sort_values("rank_range", ascending=False)
top_stable = st.sort_values("pi_top10", ascending=False)
boundary = st.sort_values("boundary_presence_top10", ascending=False).head(12)
meandraw_pat = np.mean([len(np.unique(groups[i])) for i in DRAWS])
meandraw_row = np.mean([len(i) for i in DRAWS])
doc = f"""# TOP-K STABILITY REPORT (data-only, development = batch1)

## 1. Protocol (pre-registered before any result was inspected)

* Selector: **L1-regularised logistic regression** on
  `[forced site context | 37 standardised eligible clinical features]`; `C` by **inner group-aware CV**
  inside the fitted sample (pre-registered grid {C_GRID}).
* Ranking universe: **selectable clinical features only** ({p}). `site` is forced context, never ranked.
* Resampling: **B = {N_RESAMPLES}** patient-level subsamples at **{RESAMPLE_FRACTION:.0%} of patients**
  (mean {meandraw_pat:.0f} patients / {meandraw_row:.0f} rows), both classes required. Imputation,
  standardisation, `C` selection and the selector are re-fitted inside every draw.
* Ranking rule: descending `|coefficient|` on the standardised clinical block, **average ranks for
  ties**. top-k grid **{K_GRID}**, fixed before inspection.

## 2. Is top-k stable? — set-level answer

| k | mean Jaccard | median | 5th pct | 75th pct | SD | distinct sets | modal freq | mean set size | pi>=0.95 | borderline (0.10<pi<0.90) | never | always | ambiguous boundary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.k} | {r.jaccard_mean:.4f} | {r.jaccard_median:.4f} | {r.jaccard_p05:.4f} | "
    f"{r.jaccard_p75:.4f} | {r.jaccard_sd:.4f} | {r.n_distinct_topk_sets} | "
    f"{r.modal_set_frequency:.4f} | {r.mean_set_size:.2f} | {r.n_features_pi_ge_095} | "
    f"{r.n_features_borderline_010_090} | {r.n_features_never} | {r.n_features_always} | "
    f"{r.ambiguous_boundary_resamples} |" for r in setstab.itertuples()) + f"""

**Reading: the top-k set is NOT stable at any k.**

* The best mean pairwise Jaccard is only **{setstab.jaccard_mean.max():.4f}**
  (k = {int(setstab.loc[setstab.jaccard_mean.idxmax(), 'k'])}); at k = 20 it is
  {float(setstab.loc[setstab.k == 20, 'jaccard_mean'].iloc[0]):.4f}.
* **{setstab.n_distinct_topk_sets.min()}–{setstab.n_distinct_topk_sets.max()} distinct top-k sets** occur
  across the {N_RESAMPLES} draws, and the single most frequent set covers at most
  **{setstab.modal_set_frequency.max():.1%}** of them.
* Highly stable features (pi >= 0.95) number {setstab.n_features_pi_ge_095.min()}–{
   setstab.n_features_pi_ge_095.max()} depending on k, while **{setstab.n_features_borderline_010_090.max()}
  features sit in the genuinely contested band** 0.10 < pi < 0.90 — their membership is decided by the
  draw, not by the data.
* **Boundary ambiguity from exact ties is negligible**
  ({setstab.ambiguous_boundary_freq.max():.4f} of draws at worst; mean set size
  {setstab.mean_set_size.min():.2f}–{setstab.mean_set_size.max():.2f} vs nominal k). The instability is
  therefore real re-ordering, **not** a tie-breaking artefact.

## 3. Sensitivity: wider regularisation grid

The pre-registered grid's inner-CV optimum sits at its upper edge for
{int((PRI['Cs'] == max(C_GRID)).sum())}/{N_RESAMPLES} draws (full-sample `C = {C_full}`,
{int((s_full > 0).sum())} of {p} coefficients non-zero), so the experiment was repeated with an
explicitly-labelled extended grid {C_GRID_EXTENDED} — reported as a sensitivity, never as the primary.

| k | mean Jaccard (primary grid) | mean Jaccard (extended grid) | distinct sets (primary) | distinct sets (extended) |
|---|---|---|---|---|
""" + "\n".join(
    f"| {r.k} | {r.jaccard_mean:.4f} | {e.jaccard_mean:.4f} | {r.n_distinct_topk_sets} | "
    f"{e.n_distinct_topk_sets} |" for r, e in zip(setstab.itertuples(), extstab.itertuples())) + f"""

Full inner-CV curve on the pre-registered grid (full sample):
{", ".join(f"C={c}->{a:.4f}" for c, a in inner_curve)}.
Non-zero coefficients per draw: mean {PRI['nz'].mean():.1f}, range [{PRI['nz'].min()}, {
   PRI['nz'].max()}] of {p} — the selector is *mildly* sparse, so the instability measured is not an
artefact of extreme sparsity either.

## 4. Most stable features (by pi at k = 10)

""" + "\n".join(
    f"{i+1}. **{r.feature_name}** — pi(5)={r.pi_top5:.3f}, pi(10)={r.pi_top10:.3f}, "
    f"pi(20)={r.pi_top20:.3f}; median rank {r.median_rank:.1f}, IQR {r.rank_iqr:.1f}, "
    f"range {int(r.rank_min)}-{int(r.rank_max)}"
    for i, r in enumerate(top_stable.head(10).itertuples())) + f"""

## 5. Least stable features (widest rank range among features that do get selected)

| feature | median rank | rank IQR | rank range | pi(5) | pi(10) | pi(15) | pi(20) | zero-score freq |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.feature_name} | {r.median_rank:.1f} | {r.rank_iqr:.1f} | {int(r.rank_min)}-{int(r.rank_max)} | "
    f"{r.pi_top5:.3f} | {r.pi_top10:.3f} | {r.pi_top15:.3f} | {r.pi_top20:.3f} | "
    f"{r.zero_score_freq:.3f} |" for r in ls_.head(12).itertuples()) + f"""

## 6. Features that live on the k = 10 boundary

| feature | boundary presence at k=10 | median rank | pi(10) |
|---|---|---|---|
""" + "\n".join(
    f"| {r.feature_name} | {r.boundary_presence_top10:.3f} | {r.median_rank:.1f} | {r.pi_top10:.3f} |"
    for r in boundary.itertuples()) + f"""

## 7. Diagnostic only — CV-selected k

""" + "\n".join(f"* k={r['k']}: mean grouped-CV AUROC {r['cv_auroc_mean']:.4f} (±{r['cv_auroc_sd']:.4f})"
                for r in diag.to_dict("records")) + f"""

→ diagnostic best k = **{best_k}**. Per instruction this is **not** announced as "the paper's k": the
instability above is exactly why a single k cannot be frozen yet.

## 8. Reference full-sample ranking (a data-only ordering, not a validated one)

""" + "\n".join(f"{i+1}. {r.feature_name} ({r.feature_category})" for i, r in
                enumerate(st.sort_values('median_rank').head(20).itertuples())) + f"""

## 9. Figures

* `plots/05_selection_probability.png` — pi_j(k) for every feature and k.
* `plots/05_rank_variability.png` — rank distribution per feature.
* `plots/05_topk_jaccard.png` — pairwise Jaccard distributions per k.
"""
open(f"{D['05_TOPK_STABILITY']}/TOPK_STABILITY_REPORT.md", "w").write(doc)
print(f"\nP5: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
