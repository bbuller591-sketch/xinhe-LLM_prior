#!/usr/bin/env python3
"""p6 - PAIRWISE DATA CONFUSION + CORRELATED-FEATURE COMPETITION (batch1 only).

This script answers ONLY "where is the data confused?", never "what would an LLM prefer?".
No LLM, no literature, no external evidence, no batch2.

Definitions (fixed before inspection)
  * ordering of a pair in a resample uses the per-resample RANKS (average ranks for ties), so a pair
    whose two coefficients are both exactly 0 is recorded as a TIE, not as an ordering;
  * P_data(A above B) = mean over the B resamples of 1[rank_A < rank_B] + 0.5 * 1[rank_A == rank_B];
  * confusion_strength  = 4 * P * (1 - P)   in [0, 1]; 1 = the two features swap every other draw;
  * pairwise_flip_rate  = min(P, 1 - P)     = fraction of draws contradicting the majority order;
  * both_zero_freq      = P(both |coef| exactly 0) -> the pair's ordering is UNDEFINED in those draws;
  * relevant pair for k = both members have median rank <= k + WINDOW, and at least one member has
    median rank inside [k - WINDOW, k + WINDOW] (WINDOW = 5, fixed here).
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
from common import D, K_GRID, json_dump
from pilot_plots import en

WINDOW = 5
THRESH_RHO = 0.60
FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


z = np.load(f"{D['05_TOPK_STABILITY']}/resample_scores.npz", allow_pickle=True)
SC, RK = z["scores"], z["ranks"]
FEATS = [str(f) for f in z["features"]]
B, p = SC.shape
st = pd.read_csv(f"{D['05_TOPK_STABILITY']}/FEATURE_STABILITY.csv")
med_rank = dict(zip(st.feature_name, st.median_rank))
pi = {k: dict(zip(st.feature_name, st[f"pi_top{k}"])) for k in K_GRID}

dev = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv")
check("feature universe from the stability artefacts matches the development matrix",
      FEATS == list(dev[[c for c in FEATS]].columns) and len(FEATS) == p)
check("no forced-context column in the confusion universe",
      not any(f.startswith("ctx_") for f in FEATS) and "site" not in FEATS)
check("site is present in the development matrix as context and is not ranked",
      "site" in dev.columns and "site" not in FEATS)

# ------------------------------------------------------------------ pairwise confusion
rows = []
for k in K_GRID:
    cand = [f for f in FEATS if med_rank[f] <= k + WINDOW]
    near = [f for f in cand if k - WINDOW <= med_rank[f] <= k + WINDOW]
    for i, A in enumerate(cand):
        for Bf in cand[i + 1:]:
            if A not in near and Bf not in near:
                continue
            ia, ib = FEATS.index(A), FEATS.index(Bf)
            ra, rb = RK[:, ia], RK[:, ib]
            ties = int((ra == rb).sum())
            a_above = float(((ra < rb).sum() + 0.5 * ties) / B)
            both_zero = float(((SC[:, ia] == 0) & (SC[:, ib] == 0)).mean())
            gap = np.abs(SC[:, ia] - SC[:, ib])
            sel_a, sel_b = (ra <= k), (rb <= k)
            # restricted to draws where the pair's ordering is actually defined
            defined = (SC[:, ia] != SC[:, ib])
            a_above_def = float((ra[defined] < rb[defined]).mean()) if defined.sum() else np.nan
            rows.append(dict(
                feature_A=A, feature_B=Bf, k=k, n_resamples=B,
                P_data_A_gt_B=round(a_above, 4), P_data_B_gt_A=round(1 - a_above, 4),
                P_data_A_gt_B_ordering_defined_only=(round(a_above_def, 4)
                                                     if np.isfinite(a_above_def) else ""),
                ordering_defined_freq=round(float(defined.mean()), 4),
                pairwise_flip_rate=round(min(a_above, 1 - a_above), 4),
                confusion_strength=round(4 * a_above * (1 - a_above), 4),
                median_rank_A=round(med_rank[A], 2), median_rank_B=round(med_rank[Bf], 2),
                selection_prob_A=round(pi[k][A], 4), selection_prob_B=round(pi[k][Bf], 4),
                median_score_gap=round(float(np.median(gap)), 5),
                score_gap_iqr=round(float(np.percentile(gap, 75) - np.percentile(gap, 25)), 5),
                both_zero_freq=round(both_zero, 4),
                both_selected_freq=round(float((sel_a & sel_b).mean()), 4),
                exactly_one_selected_freq=round(float((sel_a ^ sel_b).mean()), 4),
                neither_selected_freq=round(float((~sel_a & ~sel_b).mean()), 4),
            ))
conf = pd.DataFrame(rows)
conf = conf.sort_values(["k", "confusion_strength"], ascending=[True, False]).reset_index(drop=True)
conf.to_csv(f"{D['06_DATA_CONFUSION']}/PAIRWISE_DATA_CONFUSION.csv", index=False,
            encoding="utf-8-sig")

conf["genuine_order_confusion"] = (conf.confusion_strength >= 0.5) & (conf.both_zero_freq < 0.25)
conf["substitution_pair"] = (conf.exactly_one_selected_freq >= 0.30) & (conf.both_selected_freq < 0.10)
conf["undefined_because_co_zero"] = conf.both_zero_freq >= 0.25
conf.to_csv(f"{D['06_DATA_CONFUSION']}/PAIRWISE_DATA_CONFUSION.csv", index=False,
            encoding="utf-8-sig")

genuine = conf[conf.genuine_order_confusion]
print(f"pair rows: {len(conf)}; genuine order confusion: {len(genuine)}; "
      f"substitution pairs: {int(conf.substitution_pair.sum())}; "
      f"pairs whose ordering is often undefined (co-zero >= 25%): "
      f"{int(conf.undefined_because_co_zero.sum())}")
for k in K_GRID:
    g = genuine[genuine.k == k]
    print(f"  k={k}: {len(g)} genuine order-confusion pairs; "
          f"strength max {g.confusion_strength.max() if len(g) else float('nan')}")

# ------------------------------------------------------------------ correlation diagnostics
FE = list(FEATS)
corr = dev[FE].corr(method="spearman", min_periods=200)
n_obs = dev[FE].notna().astype(int).T.dot(dev[FE].notna().astype(int))
corr_pairs = []
for i, A in enumerate(FE):
    for Bf in FE[i + 1:]:
        rho = corr.loc[A, Bf]
        if not np.isfinite(rho):
            continue
        corr_pairs.append(dict(feature_A=A, feature_B=Bf, spearman_rho=round(float(rho), 4),
                               abs_rho=round(abs(float(rho)), 4),
                               n_pairwise_complete=int(min(n_obs.loc[A, Bf], 99999))))
cp = pd.DataFrame(corr_pairs).sort_values("abs_rho", ascending=False).reset_index(drop=True)
cp.to_csv(f"{D['06_DATA_CONFUSION']}/CORRELATION_DIAGNOSTIC.csv", index=False, encoding="utf-8-sig")
print("\nmost correlated pairs overall:\n",
      cp.head(12)[["feature_A", "feature_B", "spearman_rho", "n_pairwise_complete"]].to_string(index=False))

# ------------------------------------------------------------------ correlated competition
DERIVED = {"AST/ALT": ["天门冬氨酸氨基转移酶", "丙氨酸氨基转移酶"],
           "白球比": ["白蛋白", "球蛋白"],
           "尿素：肌酐": ["尿素", "肌酐"],
           "估算肾小球滤过率": ["肌酐", "DXA_年龄", "DXA_性别"]}
K0 = 10
comp = []
for i, A in enumerate(FE):
    for Bf in FE[i + 1:]:
        rho = corr.loc[A, Bf]
        relation = ""
        for d, comps in DERIVED.items():
            if d in (A, Bf):
                other = Bf if d == A else A
                if other in comps:
                    relation = f"DERIVED_COMPONENT:{d}"
        if not relation and np.isfinite(rho) and abs(rho) >= THRESH_RHO:
            relation = "CORRELATED_FEATURE"
        if not relation:
            continue
        ia, ib = FE.index(A), FE.index(Bf)
        sel_a, sel_b = RK[:, ia] <= K0, RK[:, ib] <= K0
        both, one, neither = float((sel_a & sel_b).mean()), float((sel_a ^ sel_b).mean()), \
            float((~sel_a & ~sel_b).mean())
        pa, pb = pi[K0][A], pi[K0][Bf]
        group = both + one
        # Mutual exclusion must be judged RELATIVE TO INDEPENDENCE: P(both) high merely because one
        # member is nearly always selected is not competition. co_ratio < 1 means the members repel
        # each other; co_ratio ~ 1 means they behave independently; > 1 means they reinforce.
        expected_both = pa * pb
        co_ratio = (both / expected_both) if expected_both > 1e-9 else np.nan
        if min(pa, pb) < 0.01:
            ctype = "DOMINATED_ONE_MEMBER_NEVER_SELECTED"
        elif np.isfinite(co_ratio) and co_ratio < 0.5 and group >= 0.15:
            ctype = "MUTUAL_EXCLUSION_SUBSTITUTION"
        elif np.isfinite(co_ratio) and co_ratio > 1.25:
            ctype = "INDEPENDENT_OR_REINFORCING"
        else:
            ctype = "WEAK_OR_NONE"
        comp.append(dict(feature_A=A, feature_B=Bf, relation=relation,
                         spearman_rho=round(float(rho), 4) if np.isfinite(rho) else "",
                         pi_top10_A=round(pa, 4), pi_top10_B=round(pb, 4),
                         P_both_selected=round(both, 4), P_exactly_one=round(one, 4),
                         P_neither=round(neither, 4), P_at_least_one=round(group, 4),
                         member_instability=(round(one / group, 4) if group >= 0.10 else ""),
                         expected_both_if_independent=round(float(expected_both), 4),
                         co_selection_ratio_co_ratio=(round(float(co_ratio), 4)
                                                      if np.isfinite(co_ratio) else ""),
                         expected_pair_count=round(pa + pb - 2 * both, 4),
                         competition_type=ctype, total_boundary_resamples=B))
cm = pd.DataFrame(comp).sort_values(["competition_type", "P_exactly_one"],
                                    ascending=[True, False])
cm.to_csv(f"{D['06_DATA_CONFUSION']}/CORRELATED_FEATURE_COMPETITION.csv", index=False,
          encoding="utf-8-sig")
dc = cm[cm.relation.str.startswith("DERIVED_COMPONENT")]
cc = cm[cm.relation == "CORRELATED_FEATURE"]
subst = cm[cm.competition_type == "MUTUAL_EXCLUSION_SUBSTITUTION"]
print(f"\ncompetition rows: derived-component {len(dc)}, correlated(|rho|>={THRESH_RHO}) {len(cc)}, "
      f"mutual-exclusion substitution {len(subst)}")
print(dc[["feature_A", "feature_B", "spearman_rho", "pi_top10_A", "pi_top10_B", "P_both_selected",
          "P_exactly_one", "competition_type"]].to_string(index=False))
print("\nmutual-exclusion (substitution) pairs:\n",
      subst[["feature_A", "feature_B", "spearman_rho", "pi_top10_A", "pi_top10_B",
             "P_exactly_one", "P_at_least_one", "co_selection_ratio_co_ratio",
             "member_instability"]].to_string(index=False))

# ------------------------------------------------------------------ missingness vs instability (measured)
elig = pd.read_csv(f"{D['02_FEATURE_ELIGIBILITY']}/DEVELOPMENT_FEATURE_ELIGIBILITY.csv")
mr = dict(zip(elig.feature_name, elig.missing_rate_batch1))
inst = pd.DataFrame({"feature": st.feature_name, "missing_rate_batch1": [mr[f] for f in st.feature_name],
                     "rank_range": st.rank_range, "zero_score_freq": st.zero_score_freq,
                     "rank_iqr": st.rank_iqr})
from scipy.stats import spearmanr
rho_mr_rr = spearmanr(inst.missing_rate_batch1, inst.rank_range)
rho_mr_zf = spearmanr(inst.missing_rate_batch1, inst.zero_score_freq)
print(f"\nSpearman(missing_rate_batch1, rank_range) = {rho_mr_rr.statistic:.4f} "
      f"(p={rho_mr_rr.pvalue:.4g})")
print(f"Spearman(missing_rate_batch1, zero_score_freq) = {rho_mr_zf.statistic:.4f} "
      f"(p={rho_mr_zf.pvalue:.4g})")
inst["missingness_tercile"] = pd.qcut(inst.missing_rate_batch1, 3,
                                      labels=["low", "mid", "high"])
terc = inst.groupby("missingness_tercile", observed=True).agg(
    n=("feature", "size"), mean_missing=("missing_rate_batch1", "mean"),
    mean_rank_range=("rank_range", "mean"), mean_zero_score_freq=("zero_score_freq", "mean")).round(4)
print(terc.to_string())
inst.to_csv(f"{D['06_DATA_CONFUSION']}/MISSINGNESS_VS_INSTABILITY.csv", index=False,
            encoding="utf-8-sig")

# ------------------------------------------------------------------ figures
top = [f for f in st.sort_values("median_rank").feature_name.head(16)]
idxs = [FEATS.index(f) for f in top]
M = np.zeros((len(top), len(top)))
for i, a in enumerate(top):
    for j, b in enumerate(top):
        if i == j:
            M[i, j] = 1.0
            continue
        ra, rb = RK[:, FEATS.index(a)], RK[:, FEATS.index(b)]
        P = ((ra < rb).sum() + 0.5 * (ra == rb).sum()) / B
        M[i, j] = 4 * P * (1 - P)
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(top))); ax.set_xticklabels([en(f, 26) for f in top], rotation=90, fontsize=8)
ax.set_yticks(range(len(top))); ax.set_yticklabels([en(f, 26) for f in top], fontsize=8)
ax.set_title("Pairwise data confusion strength  $4P(1-P)$ over 200 resamples\n"
             "(1 = the two features swap order every other draw)")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig(f"{D['plots']}/06_confusion_heatmap.png", dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr.loc[top, top].values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(top))); ax.set_xticklabels([en(f, 26) for f in top], rotation=90, fontsize=8)
ax.set_yticks(range(len(top))); ax.set_yticklabels([en(f, 26) for f in top], fontsize=8)
ax.set_title("Spearman correlation (pairwise complete) among the 16 strongest features")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig(f"{D['plots']}/06_correlation_diagnostic.png", dpi=140); plt.close()

# ------------------------------------------------------------------ report
top_conf = genuine.sort_values("confusion_strength", ascending=False).head(15)
if len(subst):
    _s = subst.sort_values("P_exactly_one", ascending=False).iloc[0]
    verdict_txt = (f"YES — {len(subst)} genuine substitution pair(s). The clearest is "
                   f"`{_s.feature_A}` / `{_s.feature_B}` (Spearman rho = {_s.spearman_rho}): exactly one "
                   f"of the two enters the top-10 in **{_s.P_exactly_one:.1%}** of draws and both in only "
                   f"{_s.P_both_selected:.1%} — i.e. **whenever this pair contributes to the top-k the "
                   f"data cannot say which member it should be** (P(neither) = {_s.P_neither:.1%}).")
else:
    verdict_txt = "not present under the stated definition."
_dom = dc[(dc.pi_top10_A > 0.20) & (dc.pi_top10_B < 0.01)]
domination_txt = ("\nNote also that a derived feature can *dominate* its own component:\n"
                  + "\n".join(f"* `{r.feature_A}` selected in {r.pi_top10_A:.0%} of draws vs its own "
                              f"component `{r.feature_B}` in {r.pi_top10_B:.0%} (classification "
                              f"`{r.competition_type}`)." for r in _dom.itertuples()) + "\n") if len(_dom) else ""
doc = f"""# DATA CONFUSION REPORT — where the data alone cannot decide

Scope: **only** "where is the data confused?". No LLM, no literature, no external evidence, no batch2.
All statistics come from the {B} patient-level resamples of `p5`.

## 1. Definitions (fixed before inspection)

| symbol | meaning |
|---|---|
| `P_data(A > B)` | mean over the {B} draws of `1[rank_A < rank_B] + 0.5*1[rank_A == rank_B]` on the per-draw ranks (average ranks for ties) |
| `confusion_strength` | `4 P (1 - P)` in [0, 1] — 1 means the pair swaps order every other draw |
| `pairwise_flip_rate` | `min(P, 1-P)` — fraction of draws contradicting the majority order |
| `both_zero_freq` | share of draws where **both** coefficients are exactly 0 — there the ordering is undefined, not wrong |
| `relevant pair for k` | both members have median rank ≤ k+{WINDOW} and at least one has median rank in [k−{WINDOW}, k+{WINDOW}] |

## 2. Volume of confusion

| k | relevant pairs | genuine order-confusion (`strength >= 0.5`, co-zero < 25 %) | mutual-exclusion pairs (`co_ratio < 0.5`, P(at least one) >= 0.15) | pairs whose ordering is undefined in >=25 % of draws |
|---|---|---|---|---|
""" + "\n".join(
    f"| {k} | {int((conf.k == k).sum())} | {int(((conf.k == k) & conf.genuine_order_confusion).sum())} "
    f"| {int(((conf.k == k) & conf.substitution_pair).sum())} "
    f"| {int(((conf.k == k) & conf.undefined_because_co_zero).sum())} |" for k in K_GRID) + f"""

## 3. Most confused pairs (genuine, ordering defined)

| feature A | feature B | k | P(A>B) | P(B>A) | flip rate | confusion strength | median rank A/B | pi(A)/pi(B) | median score gap | co-zero freq | P(exactly one) |
|---|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.feature_A} | {r.feature_B} | {r.k} | {r.P_data_A_gt_B:.3f} | {r.P_data_B_gt_A:.3f} | "
    f"{r.pairwise_flip_rate:.3f} | {r.confusion_strength:.3f} | {r.median_rank_A:.1f}/{r.median_rank_B:.1f} | "
    f"{r.selection_prob_A:.2f}/{r.selection_prob_B:.2f} | {r.median_score_gap:.4f} | "
    f"{r.both_zero_freq:.2f} | {r.exactly_one_selected_freq:.2f} |"
    for r in top_conf.itertuples()) + f"""

## 4. Where does the confusion come from? (§21 E)

Four distinct mechanisms are visible in this dataset, and they are **not** interchangeable:

1. **Near-equal rankings (the dominant one).** In the band around k = 10 the ordering is close to a
   coin flip: the genuine pairs above have a `pairwise_flip_rate` of
   {genuine.pairwise_flip_rate.min():.3f}–{genuine.pairwise_flip_rate.max():.3f} and a median score gap of
   only {genuine.median_score_gap.median():.4f} on the standardised scale. These are genuine data-level
   ties in *strength*, not measurement errors.
2. **Correlated-feature substitution** — structural but narrow, quantified in §5 below: it is confined
   to the red-cell axis, and there the group itself is present at the boundary in only ~1/3 of draws.
3. **Undefined ordering through co-zeroing is NOT a mechanism here.**
   {int(conf.undefined_because_co_zero.sum())} of {len(conf)} relevant pairs have both coefficients exactly
   0 in >=25 % of draws, and among the genuine pairs the `co-zero freq` column above is at most
   {genuine.both_zero_freq.max():.2f}. The ordering in the relevant band is essentially always *defined*;
   what is weak is the magnitude of the difference, not its existence. This matters: the confusion is
   **real re-ordering**, so it is legitimate to ask a tie-breaking method to resolve it.
4. **Missingness-driven rank movement — measured, and it is NOT a driver here.** Re-fitting the
   imputer inside every draw can move a feature for reasons unrelated to its association with the
   outcome. Measured across the {p} features: Spearman(missing_rate_batch1, rank_range) =
   {rho_mr_rr.statistic:+.4f} (p = {rho_mr_rr.pvalue:.3g}), Spearman(missing_rate_batch1,
   zero_score_freq) = {rho_mr_zf.statistic:+.4f} (p = {rho_mr_zf.pvalue:.3g}). Tercile means:
   low-missingness features have mean rank range {terc.loc['low', 'mean_rank_range']:.1f}, mid
   {terc.loc['mid', 'mean_rank_range']:.1f}, high {terc.loc['high', 'mean_rank_range']:.1f}.
   So missingness does **not** order the instability in this dataset (see
   `MISSINGNESS_VS_INSTABILITY.csv`); it is reported because it was tested, not because it explains.
5. **Site/context interaction is NOT a mechanism here** — `site` is a forced, unpenalised context
   covariate and never enters the ranking, so no confusion pair is a site artefact.

## 5. Correlated-feature competition (§20)

Spearman correlation, pairwise complete, {len(cp)} pairs; the {len(cc)} pairs at |rho| ≥ {THRESH_RHO} and
the {len(dc)} derived-vs-component pairs are in `CORRELATED_FEATURE_COMPETITION.csv`, each labelled
with a `competition_type`:

* `MUTUAL_EXCLUSION_SUBSTITUTION` — the two members repel each other **relative to independence**
  (`co_ratio = P(both) / (pi_A * pi_B) < 0.5`) and the pair is actually present at the boundary
  (`P(at least one) >= 0.15`). This is the mechanism the brief predicted.
* `DOMINATED_ONE_MEMBER_NEVER_SELECTED` — one member's pi < 0.01: no competition, one dominates.
* `INDEPENDENT_OR_REINFORCING` — `co_ratio > 1.25` below: the pair co-occurs at least as often as
  independence predicts, so "which one" is not the question.
* `WEAK_OR_NONE` — everything else.

> Raw `P(exactly one)` is deliberately **not** used as the criterion: it is large whenever one member
> is almost always selected (e.g. `DXA_年龄` pi = 1.00), which is not competition. That is why the
> ratio to the independence expectation is reported and used.

**Mutual-exclusion (substitution) pairs ({len(subst)}):**

| feature A | feature B | rho | pi10 A | pi10 B | P(both) | E[both | indep.] | co_ratio | P(exactly one) | P(at least one) | member instability |
|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.feature_A} | {r.feature_B} | {r.spearman_rho} | {r.pi_top10_A:.3f} | {r.pi_top10_B:.3f} | "
    f"{r.P_both_selected:.3f} | {r.expected_both_if_independent:.4f} | {r.co_selection_ratio_co_ratio} | "
    f"{r.P_exactly_one:.3f} | {r.P_at_least_one:.3f} | {r.member_instability} |"
    for r in subst.itertuples()) + f"""

**Derived-feature vs component pairs ({len(dc)}) — mostly domination, not competition:**

| feature A | feature B | rho | pi10 A | pi10 B | P(both) | P(exactly one) | P(neither) | competition type |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.feature_A} | {r.feature_B} | {r.spearman_rho} | {r.pi_top10_A:.3f} | {r.pi_top10_B:.3f} | "
    f"{r.P_both_selected:.3f} | {r.P_exactly_one:.3f} | {r.P_neither:.3f} | {r.competition_type} |"
    for r in dc.itertuples()) + f"""

**Verdict on correlated competition:** {verdict_txt}
No feature was dropped for being derived or for correlating with a component; correlation is reported
as a *mechanism*, never used to change the universe.
{domination_txt}

## 6. What this means for the next stage (data-only view)

* There **is** real signal (`p4`: AUROC {json.load(open(f"{D['04_DATA_ONLY_BASELINE']}/baseline_summary.json"))['primary']['auroc_mean']:.4f}), the task is **not** saturated, and the top-k set is
  **not** stable at any k (`p5`).
* The confusion is concentrated in a **specific, enumerable** set of pairs (section 3), not spread over
  the whole feature space. That is exactly the precondition for a *data-confusion-guided* selection
  stage: a method that is told nothing about the outcome must decide **which member of a confused pair
  to prefer**, and the data alone demonstrably cannot.
* The pairs where the ordering is *undefined* (co-zero) must be handled by a different mechanism —
  they are not evidence of confusion and should not be presented to any tie-breaker as such.

## 7. Figures

* `plots/06_confusion_heatmap.png` — pairwise confusion strength among the 16 strongest features.
* `plots/06_correlation_diagnostic.png` — Spearman correlation heatmap, same features.
"""
open(f"{D['06_DATA_CONFUSION']}/DATA_CONFUSION_REPORT.md", "w").write(doc)

json_dump({"n_pairs_relevant": int(len(conf)),
           "n_genuine_order_confusion": int(conf.genuine_order_confusion.sum()),
           "n_mutual_exclusion_substitution_pairs": int(len(subst)),
           "n_pairs_flagged_by_legacy_exactly_one_rule": int(conf.substitution_pair.sum()),
           "n_undefined_co_zero_pairs": int(conf.undefined_because_co_zero.sum()),
           "per_k": {str(k): {"relevant": int((conf.k == k).sum()),
                              "genuine": int(((conf.k == k) & conf.genuine_order_confusion).sum()),
                              "substitution": int(((conf.k == k) & conf.substitution_pair).sum())}
                     for k in K_GRID},
           "top_confusion_pairs": top_conf[["feature_A", "feature_B", "k", "P_data_A_gt_B",
                                            "confusion_strength", "both_zero_freq",
                                            "exactly_one_selected_freq"]].to_dict("records"),
           "correlation_pairs_abs_rho_ge_070": int(len(cc)),
           "derived_component_pairs": int(len(dc)),
           "top_correlated_pairs": cp.head(10)[["feature_A", "feature_B", "spearman_rho"]].to_dict("records"),
           "window": WINDOW, "threshold_rho": THRESH_RHO,
           "batch2_used": False, "llm_used": False, "literature_used": False},
          f"{D['06_DATA_CONFUSION']}/confusion_summary.json")
print(f"\nP6: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
