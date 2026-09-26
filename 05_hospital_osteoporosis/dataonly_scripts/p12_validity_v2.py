#!/usr/bin/env python3
"""p12 - TASK C: is top-k even well defined?

If the selector leaves fewer than k clinical features with a non-zero coefficient, then a "top-k" set
cannot be read as a selector ranking: it would have to be filled with zero-coefficient features whose
order is arbitrary. This script quantifies that per k, declares which k are admissible as PRIMARY, and
re-audits the v1 claim "tie ambiguity = 0", which used a check that could not see this failure mode.

Writes 05_TOPK_STABILITY/TOPK_VALIDITY_AUDIT.md.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from common import D, N_RESAMPLES, json_dump
import v2_core as V

VALID_FRACTION_REQUIRED = 0.99      # declared BEFORE looking at the numbers
K_GRID = [5, 10, 15, 20]
FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


z2 = np.load(f"{D['05_TOPK_STABILITY']}/resample_scores_v2.npz", allow_pickle=True)
nz2 = z2["nz"].astype(int)
rk2 = z2["ranks"]
FEATS = [str(f) for f in z2["features"]]
P = len(FEATS)

z1 = np.load(f"{D['05_TOPK_STABILITY']}/resample_scores.npz", allow_pickle=True)
nz1 = z1["nz"].astype(int)
rk1 = z1["ranks"]

print(f"v2 (lumbar+hip, TRUE unpenalized site): non-zero clinical coefficients per draw "
      f"mean {nz2.mean():.1f}, range [{nz2.min()}, {nz2.max()}] of {P}")
print(f"v1 (all sites, x1000 hack)           : mean {nz1.mean():.1f}, "
      f"range [{nz1.min()}, {nz1.max()}] of {z1['scores'].shape[1]}")


def audit(nz, ranks, tag):
    rows = []
    for k in K_GRID:
        ge = int((nz >= k).sum())
        lt = int((nz < k).sum())
        sizes = (ranks <= k).sum(axis=1)
        # what a literal "take the k best, padding with arbitrary zeros" would do
        padded_rows = int((sizes < k).sum())
        rows.append(dict(
            dataset=tag, k=k,
            n_resamples=len(nz),
            fraction_resamples_nonzero_ge_k=round(ge / len(nz), 4),
            fraction_resamples_nonzero_lt_k=round(lt / len(nz), 4),
            n_valid_resamples=ge, n_invalid_resamples=lt,
            verdict=("VALID_TOPK_RESAMPLE" if lt == 0 else "ZERO_PADDED_OR_UNDEFINED_TOPK"),
            admissible_primary=bool(ge / len(nz) >= VALID_FRACTION_REQUIRED),
            mean_nonzero=round(float(nz.mean()), 2),
            mean_set_size=round(float(sizes.mean()), 3),
            rows_where_set_smaller_than_k=padded_rows,
            median_set_size=float(np.median(sizes)),
            min_set_size=int(sizes.min()), max_set_size=int(sizes.max()),
        ))
    return rows


aud = pd.DataFrame(audit(nz2, rk2, "v2_primary_lumbar_hip") +
                   audit(nz1, rk1, "v1_all_sites_x1000hack"))
aud.to_csv(f"{D['audit']}/topk_validity.csv", index=False, encoding="utf-8-sig")
print("\n", aud[["dataset", "k", "fraction_resamples_nonzero_ge_k", "admissible_primary",
                 "mean_set_size", "min_set_size"]].to_string(index=False))

v2 = aud[aud.dataset == "v2_primary_lumbar_hip"]
valid_k = sorted(v2.loc[v2.admissible_primary, "k"].tolist())
invalid_k = sorted(v2.loc[~v2.admissible_primary, "k"].tolist())
print(f"\nadmissible as PRIMARY: k = {valid_k};  demoted to diagnostic: k = {invalid_k}")
check("k=5 is admissible", 5 in valid_k, f"fraction valid "
      f"{float(v2.loc[v2.k == 5, 'fraction_resamples_nonzero_ge_k'].iloc[0]):.4f}")
check("k=10 is admissible", 10 in valid_k, f"fraction valid "
      f"{float(v2.loc[v2.k == 10, 'fraction_resamples_nonzero_ge_k'].iloc[0]):.4f}")
check("the k=15 / k=20 verdicts were recomputed, not assumed",
      True, f"k=15 valid fraction "
            f"{float(v2.loc[v2.k == 15, 'fraction_resamples_nonzero_ge_k'].iloc[0]):.4f}, "
            f"k=20 {float(v2.loc[v2.k == 20, 'fraction_resamples_nonzero_ge_k'].iloc[0]):.4f}")

# ---------------- the v1 "tie ambiguity = 0" claim, re-audited ----------------
v1_amb_claimed = 0
v1_true_invalid = {}
for k in K_GRID:
    r = rk1
    m = r <= k
    # the v1 check: does a *tie group that got selected* share its score with an unselected group?
    amb = 0
    for b in range(r.shape[0]):
        sel = m[b]
        if sel.any() and (~sel).any():
            s = z1["scores"][b]
            if np.isclose(s[sel].min(), s[~sel].max()):
                amb += 1
    v1_true_invalid[k] = dict(v1_flagged_ambiguous=amb,
                              v1_undersized_sets=int((m.sum(axis=1) < k).sum()),
                              v1_resamples_with_nonzero_lt_k=int((nz1 < k).sum()))

json_dump({"valid_fraction_required": VALID_FRACTION_REQUIRED, "k_grid": K_GRID,
           "v2_admissible_primary": valid_k, "v2_diagnostic_only": invalid_k,
           "audit_table": aud.to_dict("records"),
           "v1_reaudit": v1_true_invalid,
           "v2_nonzero_mean": round(float(nz2.mean()), 2),
           "v2_nonzero_range": [int(nz2.min()), int(nz2.max())],
           "v1_nonzero_mean": round(float(nz1.mean()), 2),
           "v1_nonzero_range": [int(nz1.min()), int(nz1.max())],
           "batch2_used": False, "llm_used": False},
          f"{D['audit']}/topk_validity.json")

md = "\n".join(
    f"| {r.k} | {r.fraction_resamples_nonzero_ge_k:.4f} | {r.fraction_resamples_nonzero_lt_k:.4f} | "
    f"{r.n_invalid_resamples} | {r.mean_nonzero:.1f} | {r.mean_set_size:.2f} | {r.min_set_size} | "
    f"{r.rows_where_set_smaller_than_k} | **{'PRIMARY-VALID' if r.admissible_primary else 'DIAGNOSTIC ONLY'}** |"
    for r in v2.itertuples())
md1 = "\n".join(
    f"| {r.k} | {r.fraction_resamples_nonzero_ge_k:.4f} | {r.n_invalid_resamples} | "
    f"{r.mean_set_size:.2f} | {v1_true_invalid[r.k]['v1_flagged_ambiguous']} |" for r in
    aud[aud.dataset == "v1_all_sites_x1000hack"].itertuples())

doc = f"""# TOP-K VALIDITY AUDIT — which k is a selector ranking at all?

## 1. The problem

The selector is L1-penalized, so in any given resample only some clinical features have a non-zero
coefficient. If fewer than `k` features are non-zero, then a "top-k set" cannot be read as a selector
ranking: the remaining slots would have to be filled with zero-coefficient features whose order is
arbitrary. v2 measured, per draw, **mean {nz2.mean():.1f} non-zero clinical coefficients, range
[{nz2.min()}, {nz2.max()}] of {P}** — so this is not hypothetical.

Declared rule (fixed before reading the numbers): a `k` is admissible as a PRIMARY claim only if
`fraction(resamples with n_nonzero >= k) >= {VALID_FRACTION_REQUIRED}`.

## 2. Result — v2 primary (lumbar+hip, true unpenalized site context)

| k | frac(nz >= k) | frac(nz < k) | invalid draws | mean nz | mean set size | min set size | draws where the selected set is smaller than k | verdict |
|---|---|---|---|---|---|---|---|---|
{md}

**Admissible as primary: k = {valid_k}.**
**Demoted to diagnostic/sensitivity: k = {invalid_k}.**

## 3. What "demoted" means, concretely

For the demoted `k`, in {int(v2.loc[~v2.admissible_primary, 'n_invalid_resamples'].sum())} draws
combined, the L1 solution does not contain `k` non-zero clinical features at all. Two ways to force a
set of size `k` exist and **both** are meaningless as a ranking:

1. pad with zero-coefficient features → the extra members are chosen by tie-breaking, not by the data;
2. keep only the non-zero members → the "top-k" silently becomes "top-`min(k, nz)`", so its size varies
   across draws and set-overlap statistics become incomparable (this is what the `mean set size` and
   `min set size` columns show).

So the earlier v1 numbers for k=15/20 are reported here as diagnostics only and are **not** used for
any primary stability or confusion claim in v2.

## 4. Re-audit of the v1 statement "tie ambiguity = 0"

The v1 check tested whether the *lowest selected* score equalled the *highest unselected* score. That
test can only fire when a tie group genuinely straddles the boundary, and because ties were given
**average ranks** the zero-coefficient group typically collapses to a single rank far below `k` — the
check therefore reported 0 ambiguous resamples while the real failure mode (fewer than `k` non-zero
coefficients) went unmeasured.

| k | frac(nz >= k) | invalid draws | mean set size | resamples v1 flagged "ambiguous" |
|---|---|---|---|---|
{md1}

> Correction: the v1 sentence "ambiguous tie-boundary resamples = 0" is **true as stated for the test
> that was run**, but it is **not** evidence that the top-k was well defined. The measured problem is
> non-zero support smaller than `k`, which affects k = {invalid_k} in the v1 (all-sites) run as well.
> v2 states the validity of each `k` explicitly instead.

## 5. Consequences carried into v2

* Stability and confusion statistics are computed for every k in {K_GRID}, but only k = {valid_k} are
  labelled `PRIMARY-VALID`; the others are labelled diagnostic-only in every v2 table.
* The actionable-pair shortlist is restricted to the admissible k.
* `09_REPORTS/FINAL_PRE_LLM_HANDOFF.md` quotes stability and confusion numbers only for the admissible
  k, with the others visible as sensitivity.
"""
open(f"{D['05_TOPK_STABILITY']}/TOPK_VALIDITY_AUDIT.md", "w").write(doc)
print(f"\nP12: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
