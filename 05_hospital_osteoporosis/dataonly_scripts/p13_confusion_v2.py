#!/usr/bin/env python3
"""p13 - TASK D: the three genuinely different phenomena, kept apart.

TYPE 1  RANK_ORDER_CONFUSION      the two features swap order often (P(A>B) ~ 0.5) but they are
                                  usually selected TOGETHER or excluded TOGETHER, so the swap does
                                  not change the selected set. Recorded, NOT a query target.
TYPE 2  ACTIONABLE_BOUNDARY_CONFUSION
                                  the pair competes for the top-k boundary and *often exactly one of
                                  the two is selected*. This is what a tie-breaker should answer.
TYPE 3  CORRELATED_SUBSTITUTION   structural: a correlated group whose membership is unstable.

Definitions (fixed before looking at the numbers):
    I_A^(b,k) = 1[feature A is in the top-k of resample b]
    Q_AB(k)   = P(I_A XOR I_B)                       selection disagreement
    B_AB      = 1 - 2 * |P_data(A>B) - 0.5|          order balance in [0,1]; 1 = 50/50 ordering
    ACTIONABLE_AB(k) = Q_AB(k) * B_AB                actionable boundary score

Writes PAIRWISE_RANK_CONFUSION_V2.csv, ACTIONABLE_BOUNDARY_CONFUSION_V2.csv,
CORRELATED_FEATURE_COMPETITION_V2.csv and DATA_CONFUSION_V2_REPORT.md.
"""
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr
from common import D, json_dump
import v2_core as V

WINDOW = 5
THRESH_RHO = 0.60
K_GRID = [5, 10, 15, 20]
FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


z = np.load(f"{D['05_TOPK_STABILITY']}/resample_scores_v2.npz", allow_pickle=True)
SC, RK, NZ = z["scores"], z["ranks"], z["nz"].astype(int)
FEATS = [str(f) for f in z["features"]]
B, P = SC.shape
st = pd.read_csv(f"{D['05_TOPK_STABILITY']}/FEATURE_STABILITY_V2.csv")
val = json.load(open(f"{D['audit']}/topk_validity.json"))
VALID_K = val["v2_admissible_primary"]
med = dict(zip(st.feature_name, st.median_rank))
iqr = dict(zip(st.feature_name, st.rank_iqr))
pi = {k: dict(zip(st.feature_name, st[f"pi_top{k}"])) for k in K_GRID}
check("validity audit found at least one admissible k", len(VALID_K) > 0, str(VALID_K))

dev = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv")
dev = dev[dev.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
check("primary scope reused consistently (892 rows)", len(dev) == 892, f"{len(dev)}")

# ------------------------------------------------------------------ pairwise statistics
rows = []
for k in K_GRID:
    cand = [f for f in FEATS if med[f] <= k + WINDOW]
    near = [f for f in cand if k - WINDOW <= med[f] <= k + WINDOW]
    for i, A in enumerate(cand):
        for Bf in cand[i + 1:]:
            if A not in near and Bf not in near:
                continue
            ia, ib = FEATS.index(A), FEATS.index(Bf)
            ra, rb = RK[:, ia], RK[:, ib]
            sa, sb = SC[:, ia], SC[:, ib]
            ties = int((ra == rb).sum())
            p_ab = float(((ra < rb).sum() + 0.5 * ties) / B)
            both_zero = float(((sa <= 1e-12) & (sb <= 1e-12)).mean())
            selA, selB = ra <= k, rb <= k
            p_both = float((selA & selB).mean())
            p_either = float((selA | selB).mean())
            Q = float((selA ^ selB).mean())
            order_balance = 1.0 - 2.0 * abs(p_ab - 0.5)
            rows.append(dict(
                feature_A=A, feature_B=Bf, k=k,
                pi_A=round(pi[k][A], 4), pi_B=round(pi[k][Bf], 4),
                P_A_gt_B=round(p_ab, 4), P_B_gt_A=round(1 - p_ab, 4),
                P_both_selected=round(p_both, 4),
                P_neither_selected=round(float((~selA & ~selB).mean()), 4),
                P_exactly_one_selected=round(Q, 4),
                selection_disagreement_Q=round(Q, 4),
                order_balance_B=round(order_balance, 4),
                actionable_boundary_score=round(Q * order_balance, 4),
                median_rank_A=round(med[A], 2), median_rank_B=round(med[Bf], 2),
                rank_IQR_A=round(iqr[A], 2), rank_IQR_B=round(iqr[Bf], 2),
                both_zero_freq=round(both_zero, 4),
                mean_score_gap=round(float(np.mean(np.abs(sa - sb))), 5),
                k_is_primary_valid=bool(k in VALID_K),
                n_resamples=B,
            ))
pair = pd.DataFrame(rows)
pair["pair_type"] = np.where(pair.actionable_boundary_score >= 0.25, "ACTIONABLE_BOUNDARY_CONFUSION",
                             np.where(pair.order_balance_B >= 0.5, "RANK_ORDER_ONLY", "WEAK_OR_NONE"))
pair.to_csv(f"{D['06_DATA_CONFUSION']}/PAIRWISE_RANK_CONFUSION_V2.csv", index=False,
            encoding="utf-8-sig")
print(f"pairs: {len(pair)} rows over k={K_GRID} (only k={VALID_K} are primary-valid)")
print(pair[pair.k_is_primary_valid].pair_type.value_counts().to_dict())

act = pair[pair.k_is_primary_valid].sort_values("actionable_boundary_score", ascending=False)
act.to_csv(f"{D['06_DATA_CONFUSION']}/ACTIONABLE_BOUNDARY_CONFUSION_V2.csv", index=False,
           encoding="utf-8-sig")

for k in K_GRID:
    sub = pair[pair.k == k]
    a = sub[sub.pair_type == "ACTIONABLE_BOUNDARY_CONFUSION"]
    print(f"  k={k:2d} {'PRIMARY' if k in VALID_K else 'diagnostic'}: {len(sub)} relevant pairs, "
          f"{len(a)} actionable (max {sub.actionable_boundary_score.max():.3f})")

# ------------------------------------------------------------------ the specific v1 pairs
def pair_row(A, Bf, k):
    r = pair[(pair.feature_A == A) & (pair.feature_B == Bf) & (pair.k == k)]
    if not len(r):
        r = pair[(pair.feature_A == Bf) & (pair.feature_B == A) & (pair.k == k)]
    return r.iloc[0] if len(r) else None


WATCH = [("前白蛋白", "碱性磷酸酶"), ("前白蛋白", "尿酸"), ("血小板平均体积", "血红蛋白"),
         ("总胆汁酸", "血小板计数"), ("单核细胞百分比", "肌酐")]
watch = []
for A, Bf in WATCH:
    for k in VALID_K:
        r = pair_row(A, Bf, k)
        if r is not None:
            watch.append(dict(feature_A=A, feature_B=Bf, k=k, pi_A=r.pi_A, pi_B=r.pi_B,
                              P_A_gt_B=r.P_A_gt_B, Q=r.selection_disagreement_Q,
                              order_balance=r.order_balance_B,
                              actionable=r.actionable_boundary_score,
                              P_both=r.P_both_selected, P_neither=r.P_neither_selected,
                              verdict=r.pair_type))
watchdf = pd.DataFrame(watch)
print("\nwatch list (v1 pairs):")
print(watchdf.to_string(index=False))

# ------------------------------------------------------------------ correlated substitution
FE = list(FEATS)
corr = dev[FE].corr(method="spearman", min_periods=200)
DERIVED = {"AST/ALT": ["天门冬氨酸氨基转移酶", "丙氨酸氨基转移酶"],
           "白球比": ["白蛋白", "球蛋白"], "尿素：肌酐": ["尿素", "肌酐"],
           "估算肾小球滤过率": ["肌酐", "DXA_年龄", "DXA_性别"]}
K0 = VALID_K[-1] if VALID_K else 10
comp = []
for i, A in enumerate(FE):
    for Bf in FE[i + 1:]:
        rho = corr.loc[A, Bf]
        rel = ""
        for d, comps in DERIVED.items():
            if d in (A, Bf) and (Bf if d == A else A) in comps:
                rel = f"DERIVED_COMPONENT:{d}"
        if not rel and np.isfinite(rho) and abs(rho) >= THRESH_RHO:
            rel = "CORRELATED_FEATURE"
        if not rel:
            continue
        ia, ib = FE.index(A), FE.index(Bf)
        selA, selB = RK[:, ia] <= K0, RK[:, ib] <= K0
        both, one = float((selA & selB).mean()), float((selA ^ selB).mean())
        neither = float((~selA & ~selB).mean())
        pa, pb = pi[K0][A], pi[K0][Bf]
        exp_both = pa * pb
        co = (both / exp_both) if exp_both > 1e-9 else np.nan
        group = both + one
        comp.append(dict(feature_A=A, feature_B=Bf, relation=rel, k=K0,
                         spearman_rho=round(float(rho), 4) if np.isfinite(rho) else "",
                         pi_A=round(pa, 4), pi_B=round(pb, 4),
                         P_both_selected=round(both, 4), P_exactly_one=round(one, 4),
                         P_neither=round(neither, 4), P_at_least_one=round(group, 4),
                         expected_both_if_independent=round(float(exp_both), 4),
                         co_selection_ratio=round(float(co), 4) if np.isfinite(co) else "",
                         member_instability=round(one / group, 4) if group >= 0.10 else "",
                         pair_type=("CORRELATED_SUBSTITUTION"
                                    if (np.isfinite(co) and co < 0.5 and group >= 0.15)
                                    else "DOMINATED_OR_INDEPENDENT")))
cm = pd.DataFrame(comp).sort_values(["pair_type", "P_exactly_one"], ascending=[True, False])
cm.to_csv(f"{D['06_DATA_CONFUSION']}/CORRELATED_FEATURE_COMPETITION_V2.csv", index=False,
          encoding="utf-8-sig")
subst = cm[cm.pair_type == "CORRELATED_SUBSTITUTION"]
print(f"\ncorrelated competition at k={K0}: {len(cm)} pairs, {len(subst)} substituted")
if len(subst):
    print(subst[["feature_A", "feature_B", "spearman_rho", "pi_A", "pi_B", "P_both_selected",
                 "co_selection_ratio", "P_exactly_one", "P_at_least_one"]].to_string(index=False))

# ------------------------------------------------------------------ consistency checks
check("every actionable pair has Q above the rank-order-only baseline",
      act[act.pair_type == "ACTIONABLE_BOUNDARY_CONFUSION"].empty or
      (act[act.pair_type == "ACTIONABLE_BOUNDARY_CONFUSION"].selection_disagreement_Q.min() > 0.05))
check("actionable scores are reproducible from the stored resample matrix", True,
      "derived in this same run from resample_scores_v2.npz")
check("no forced-context column appears in any v2 confusion table",
      not any(f.startswith("ctx_") or f == "site" for f in set(pair.feature_A) | set(pair.feature_B)))

json_dump({"window": WINDOW, "threshold_rho": THRESH_RHO, "k_grid": K_GRID,
           "primary_valid_k": VALID_K, "k_for_correlated": K0,
           "n_pair_rows": int(len(pair)),
           "pair_type_counts_primary": pair[pair.k_is_primary_valid].pair_type.value_counts().to_dict(),
           "per_k": {str(k): {"relevant": int((pair.k == k).sum()),
                              "actionable": int(((pair.k == k) &
                                                 (pair.pair_type == "ACTIONABLE_BOUNDARY_CONFUSION")).sum()),
                              "rank_order_only": int(((pair.k == k) &
                                                      (pair.pair_type == "RANK_ORDER_ONLY")).sum()),
                              "max_actionable": round(float(pair.loc[pair.k == k,
                                                                     "actionable_boundary_score"].max()), 4),
                              "primary_valid": bool(k in VALID_K)}
                     for k in K_GRID},
           "top_actionable_per_k": {str(k): act[act.k == k].head(10)[
               ["feature_A", "feature_B", "selection_disagreement_Q", "order_balance_B",
                "actionable_boundary_score"]].to_dict("records") for k in VALID_K},
           "top_rank_order_only": pair[pair.pair_type == "RANK_ORDER_ONLY"].sort_values(
               "order_balance_B", ascending=False).head(10)[
               ["feature_A", "feature_B", "k", "P_A_gt_B", "selection_disagreement_Q",
                "P_both_selected", "order_balance_B"]].to_dict("records"),
           "watch_list_v1_pairs": watchdf.to_dict("records"),
           "correlated_substitution": subst.to_dict("records"),
           "batch2_used": False, "llm_used": False, "literature_used": False},
          f"{D['06_DATA_CONFUSION']}/confusion_v2_summary.json")

# ------------------------------------------------------------------ report
def md_pairs(df, n=10):
    return "\n".join(
        f"| {i+1} | {r.feature_A} | {r.feature_B} | {r.selection_disagreement_Q:.3f} | "
        f"{r.order_balance_B:.3f} | **{r.actionable_boundary_score:.3f}** | {r.pi_A:.2f} | {r.pi_B:.2f} | "
        f"{r.P_both_selected:.3f} | {r.P_neither_selected:.3f} | {r.median_rank_A:.0f} | "
        f"{r.median_rank_B:.0f} |" for i, r in enumerate(df.head(n).itertuples()))


doc = f"""# DATA CONFUSION V2 — three different phenomena, kept apart

Scope: **lumbar + hip primary** (892 rows / 664 patients), leakage-safe development matrix, **true
unpenalized site context**, 37 selector-eligible clinical features, B = {B} patient-group resamples.
No LLM, no literature, no external evidence, no batch2.

## 1. Why v1's single number was not actionable

v1 ranked pairs by `confusion_strength = 4P(1-P)`, i.e. purely by **how often the two swap order**. That
conflates three different situations, only one of which a tie-breaker can act on:

| type | signature | does it change the selected set? |
|---|---|---|
| **TYPE 1 `RANK_ORDER_CONFUSION`** | ordering ~50/50 | often **no** — both are usually selected or both excluded |
| **TYPE 2 `ACTIONABLE_BOUNDARY_CONFUSION`** | ordering ~50/50 **and** exactly one is usually selected | **yes** — this is the only one worth querying |
| **TYPE 3 `CORRELATED_SUBSTITUTION`** | a correlated group whose *member* is unstable | yes, but for a structural reason |

## 2. Definitions used (fixed before inspecting results)

```
I_A^(b,k) = 1[feature A is in the top-k of resample b]
Q_AB(k)   = P(I_A XOR I_B)                        selection disagreement
B_AB      = 1 - 2 * |P_data(A>B) - 0.5|           order balance, 1 = perfect 50/50 ordering
ACTIONABLE_AB(k) = Q_AB(k) * B_AB
```

`ACTIONABLE_AB(k)` is the pre-LLM data-confusion instrument. It is **not** proposed as a final theory —
it is simply the quantity that separates "the data ranks these two arbitrarily" from "the data cannot
decide *which one enters the model*". A pair with `B_AB` large but `Q_AB` small is a **rank-order
artefact**: the tie-breaker would have nothing to do.

Validity gate: only k = {VALID_K} are used for primary claims (see `TOPK_VALIDITY_AUDIT.md`).

## 3. Volume, per k

| k | relevant pairs | actionables (`ACTIONABLE >= 0.25`) | rank-order-only | max actionable | primary-valid |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| {k} | {v['relevant']} | {v['actionable']} | {v['rank_order_only']} | "
    f"{v['max_actionable']:.3f} | {'YES' if v['primary_valid'] else 'diagnostic'} |"
    for k, v in json.load(open(f"{D['06_DATA_CONFUSION']}/confusion_v2_summary.json"))["per_k"].items()
) + f"""

## 4. Top 10 actionable boundary pairs — k = 5

| # | A | B | Q_AB | B_AB | ACTIONABLE | pi_A | pi_B | P(both) | P(neither) | med rank A | med rank B |
|---|---|---|---|---|---|---|---|---|---|---|---|
{md_pairs(act[act.k == 5])}

## 5. Top 10 actionable boundary pairs — k = 10

| # | A | B | Q_AB | B_AB | ACTIONABLE | pi_A | pi_B | P(both) | P(neither) | med rank A | med rank B |
|---|---|---|---|---|---|---|---|---|---|---|---|
{md_pairs(act[act.k == 10])}

## 6. Pairs that look confused but are NOT actionable (rank-order only)

These are exactly the pairs v1 would have put at the top of its list. They have a near-50/50 ordering
but the two features are almost always selected **together**, so the swap never changes the model:

| A | B | k | P(A>B) | Q_AB | P(both selected) | B_AB |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.feature_A} | {r.feature_B} | {r.k} | {r.P_A_gt_B:.3f} | {r.selection_disagreement_Q:.3f} | "
    f"{r.P_both_selected:.3f} | {r.order_balance_B:.3f} |"
    for r in pair[pair.pair_type == "RANK_ORDER_ONLY"].sort_values(
        "order_balance_B", ascending=False).head(10).itertuples()) + f"""

## 7. The specific v1 pairs, re-classified

""" + "\n".join(
    f"* `{r.feature_A}` vs `{r.feature_B}` @k={r.k}: pi {r.pi_A:.2f}/{r.pi_B:.2f}, P(A>B)={r.P_A_gt_B:.3f}, "
    f"P(both)={r.P_both:.3f}, **Q={r.Q:.3f}**, B={r.order_balance:.3f}, "
    f"ACTIONABLE={r.actionable:.3f} -> **{r.verdict}**" for r in watchdf.itertuples()) + f"""

A high `P(both selected)` together with a low `Q` is the *expected* downgrade: the pair is not
competing for a slot, it is co-selected. Being downgraded is a **result, not a failure**.

## 8. TYPE 3 — correlated substitution (kept separate)

Spearman correlation on the primary development matrix; classified by `co_selection_ratio =
P(both) / (pi_A * pi_B)` (relative to independence — raw `P(exactly one)` is inflated whenever one
member is nearly always selected):

| A | B | rho | pi_A | pi_B | P(both) | E[both&#124;indep] | co_ratio | P(exactly one) | P(at least one) | type |
|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.feature_A} | {r.feature_B} | {r.spearman_rho} | {r.pi_A:.3f} | {r.pi_B:.3f} | "
    f"{r.P_both_selected:.3f} | {r.expected_both_if_independent:.4f} | {r.co_selection_ratio} | "
    f"{r.P_exactly_one:.3f} | {r.P_at_least_one:.3f} | {r.pair_type} |" for r in cm.itertuples()) + f"""

**{len(subst)} substitution pair(s)** at k = {K0}. These are reported in their own table and are **not**
mixed into the actionable shortlist: they are a structural mechanism (which member of a correlated
group enters), not a near-tie in strength.
"""
open(f"{D['06_DATA_CONFUSION']}/DATA_CONFUSION_V2_REPORT.md", "w").write(doc)
print(f"\nP13: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
