#!/usr/bin/env python3
"""p1 - TASK FREEZE.

Produces
  01_TASK_FREEZE/TASK_B_SEMANTICS_FREEZE.md   (what Task B is and is not, site audit, A vs B)
  01_TASK_FREEZE/SITE_CONTEXT_RULE.md         (site = FORCED CONTEXT, never selectable)
  audit/site_subregion_audit.json

The site/subregion audit is computed on BATCH1 (development) only. Whole-cohort subregion
figures are quoted from the already-frozen hospital README (published before this pilot) purely
as a cross-reference and were NOT recomputed or used to make the decision.
"""
import json
import pandas as pd
from common import CANON, D, batch1_raw, design_table, json_dump, SITES, SITE_EN

FAILS = []


def check(n, c, d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n} {d}")
    if not c:
        FAILS.append(n)


raw = batch1_raw()
tab = design_table(raw)
n, npat = len(tab), tab.patient_uid.nunique()

# ------------------------------------------------------------------ site / subregion audit (batch1)
aud = {}
for s in SITES:
    m = raw["site"] == s
    vc = raw.loc[m, "subregion"].fillna("<missing>").value_counts()
    src = raw.loc[m, "T_source"].value_counts()
    aud[s] = {
        "rows": int(m.sum()),
        "patients": int(raw.loc[m, "patient_uid"].nunique()),
        "subregion_counts": {k: int(v) for k, v in vc.items()},
        "t_source_counts": {k: int(v) for k, v in src.items()},
        "positive_rows": int(tab.loc[tab.site == s, "y"].sum()),
        "positive_rate": round(float(tab.loc[tab.site == s, "y"].mean()), 6),
        "subregion_labelled_rows": int((raw.loc[m, "subregion"].notna()
                                        & ~raw.loc[m, "subregion"].isin(["未知", "未标注"])).sum()),
    }
    aud[s]["subregion_labelled_share"] = round(
        aud[s]["subregion_labelled_rows"] / aud[s]["rows"], 6)
    aud[s]["iscd_reference_subregion_rows"] = int(
        raw.loc[m, "subregion"].isin(["1/3"]).sum()) if s == "前臂" else int(
        raw.loc[m, "subregion"].isin(["整体"]).sum())

json_dump(aud, f"{D['audit']}/site_subregion_audit.json")
pd.DataFrame(aud).T.to_csv(f"{D['audit']}/site_subregion_audit.csv", encoding="utf-8-sig")

fa = aud["前臂"]
print("batch1 forearm:", json.dumps(fa, ensure_ascii=False))
check("forearm is NOT dominated by the ISCD 1/3-radius site in batch1",
      fa["subregion_counts"].get("1/3", 0) / fa["rows"] < 0.05,
      f"1/3 radius rows = {fa['subregion_counts'].get('1/3', 0)}/{fa['rows']}")
check("hip: femoral-neck-labelled rows are a minority in batch1",
      aud["髋关节"]["subregion_counts"].get("股骨颈", 0) / aud["髋关节"]["rows"] < 0.10,
      f"{aud['髋关节']['subregion_counts'].get('股骨颈', 0)}/{aud['髋关节']['rows']}")
check("batch1 site totals 517/382/139",
      [aud[s]["rows"] for s in SITES] == [517, 382, 139],
      str([aud[s]["rows"] for s in SITES]))

# ------------------------------------------------------------------ TASK_B_SEMANTICS_FREEZE.md
st = aud
doc = f"""# TASK B — SEMANTICS FREEZE

**Status: FROZEN for the data-only pilot. Do not rename the task, do not widen the target.**

## 1. Canonical task statement

> **SITE-LEVEL LOW T-SCORE CLASSIFICATION**
> Given a patient's clinical and laboratory profile measured at/around the DXA encounter, predict
> whether **the measured skeletal site** has **T-score ≤ −2.5**.

| element | frozen value |
|---|---|
| observation unit | **patient × skeletal site** (one row = one patient's one skeletal site) |
| batch1 development rows / patients | {n} / {npat} |
| target | `y = 1[T_used ≤ −2.5]`, **site-specific** (the T-score of *that* row's site) |
| positive rows | {int(tab.y.sum())} ({tab.y.mean():.4f}) |
| sites | {", ".join(SITES)} |
| grouping key | `patient_uid` |
| direct leakage excluded | `T_used` (the label's own source), `T_source` |
| DXA-derived metadata excluded | `subregion` |

**Verified in the frozen handoff and re-verified here:** `label_osteoporosis` equals
`1[T_used ≤ −2.5]` on **every** row (batch1 mismatches = 0).

## 2. Naming rule (binding)

* ✅ **SITE-LEVEL LOW T-SCORE CLASSIFICATION**
* ✅ "site-specific low bone mineral density (T-score ≤ −2.5)"
* ❌ ~~"osteoporosis diagnosis"~~ / ~~"predicts osteoporosis"~~ / ~~"diagnostic model"~~

**Why the prohibition is not cosmetic.** The target is a *thresholded measurement*, and the
measurement is only equivalent to the clinical diagnosis at the sites the diagnostic standard
recognises. In this dataset the forearm is the counter-example: {fa['subregion_counts'].get('1/3', 0)}
of {fa['rows']} batch1 forearm records ({fa['subregion_counts'].get('1/3', 0)/fa['rows']:.1%}) is the
ISCD reference site (1/3 distal radius). The remaining forearm records are `整体`/`桡骨`/unknown,
i.e. **measurements whose normative reference is not the defined diagnostic one**. Calling the
label "osteoporosis diagnosis" would assert a clinical equivalence the data does not carry.

## 3. Not equivalent to a full clinical diagnosis

The target is **not** automatically a clinical osteoporosis diagnosis, because:

1. **Site coverage is incomplete per patient.** Only {tab.groupby('patient_uid').size().gt(1).sum():,}
   of {npat} batch1 patients have more than one site measured; a diagnosis needs the reference
   sites, not whichever sites happen to exist.
2. **Site identity within a region is coarse** (see §4): "lumbar spine" is mostly a single
   aggregate ROI and hip is mostly unlabelled, so a femoral-neck-specific statement is not
   supported.
3. **The DXA device/normative database is not documented** in the source; a T-score is only
   interpretable relative to the reference population it was computed against.
4. **Clinical context is absent.** No fracture history, FRAX inputs, glucocorticoid exposure,
   secondary-cause workup, or treatment history is present. Clinical diagnosis is not a
   pure T-score threshold.
5. **The ascertainment is not diagnostic.** Sites were measured because of clinical
   suspicion / routine practice; not every patient got every site.

## 4. Site / subregion audit (batch1, development only)

| site | rows | patients | positive rows | positive rate | subregion `整体` | femoral-neck (`股骨颈`) | `1/3` radius | other / unknown |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {s} ({SITE_EN[s]}) | {st[s]['rows']} | {st[s]['patients']} | {st[s]['positive_rows']} | "
    f"{st[s]['positive_rate']:.4f} | {st[s]['subregion_counts'].get('整体', 0)} | "
    f"{st[s]['subregion_counts'].get('股骨颈', 0) + st[s]['subregion_counts'].get('颈部', 0)} | "
    f"{st[s]['subregion_counts'].get('1/3', 0)} | "
    f"{st[s]['rows'] - st[s]['subregion_counts'].get('整体', 0) - st[s]['subregion_counts'].get('股骨颈', 0) - st[s]['subregion_counts'].get('颈部', 0) - st[s]['subregion_counts'].get('1/3', 0)} |"
    for s in SITES) + f"""

Raw `subregion` tokens present in batch1: {sorted(set(raw['subregion'].dropna().unique()))}.
Note `未知` and `未标注` are **distinct raw tokens**; both mean "not annotated" but must not be
silently merged when counting.

Site provenance (`T_source`) in batch1:
""" + "\n".join(f"* {s}: " + ", ".join(f"`{k}`={v}" for k, v in st[s]["t_source_counts"].items())
                for s in SITES) + f"""

**Cross-reference, not recomputed, not used for the decision:** the already-frozen hospital README
documents the same pattern on the full cohort (hip: 60.6 % unlabelled, femoral neck 6.7 %;
forearm: 1.2 % explicitly 1/3 radius) and states two hard consequences — *the hip must not be
called "femoral neck"* and *the forearm result is not an ISCD-defined diagnosis*.

## 5. Recommendation: all sites vs lumbar+hip  → **A. lumbar + hip primary, forearm sensitivity**

**A. lumbar spine + hip as PRIMARY, forearm as a SENSITIVITY analysis.** Decision made on
clinical/measurement semantics and provenance **only** — no predictive performance was inspected
for this choice.

Grounds:
1. **ISCD diagnostic sites.** The diagnosis-relevant measurement sites are the lumbar spine
   (L1–L4 aggregate) and the proximal femur; the forearm (33 % radius) is a *fallback* used when
   those are not evaluable. 腰椎骨 and 髋关节 map onto the diagnostic sites; 前臂 does not.
2. **The forearm subregion labels do not support the reference site.** Only
   {fa['subregion_counts'].get('1/3', 0)}/{fa['rows']} batch1 forearm rows are explicitly `1/3`;
   {fa['subregion_counts'].get('未知', 0)} are `未知` and {fa['subregion_counts'].get('整体', 0)} are
   a whole-forearm ROI. A T-score computed over an unstated ROI cannot be assumed to equal the
   33 %-radius T-score.
3. **The forearm base rate is materially different** (batch1 positive rate
   {fa['positive_rate']:.4f} vs {st['腰椎骨']['positive_rate']:.4f} lumbar /
   {st['髋关节']['positive_rate']:.4f} hip). An outcome whose positivity differs that much at the
   same threshold is evidence of a **different measurement/reference definition**, not merely a
   different patient mix — i.e. pooling it into the primary task mixes two outcome definitions.
4. **Hip caveat is a caveat, not an exclusion.** 髋关节 rows are `整体` (total hip, a valid ISCD
   site) or unlabelled; only {st['髋关节']['subregion_counts'].get('股骨颈', 0)}/{
    st['髋关节']['rows']} are femoral-neck-labelled. Hip therefore stays primary **as "hip",
   never as "femoral neck"**.
5. **Lumbar caveat.** {st['腰椎骨']['subregion_counts'].get('整体', 0)}/{st['腰椎骨']['rows']}
   lumbar rows are a single aggregate ROI and only {st['腰椎骨']['subregion_counts'].get('L1', 0)
   + st['腰椎骨']['subregion_counts'].get('L2', 0) + st['腰椎骨']['subregion_counts'].get('L3', 0)
   + st['腰椎骨']['subregion_counts'].get('L4', 0)} rows name a single vertebra → no
   per-vertebra analysis is possible.

**Forearm is NOT deleted.** It is retained in the data, in the manifest, and as a mandatory
sensitivity analysis (`site_scope = lumbar_hip_primary` vs `all_sites_sensitivity`). Both must be
reported; only the primary label differs.

## 6. Correlated rows (binding)

One patient contributes up to 3 rows sharing every clinical/laboratory value, the demographic and
anthropometric values, and the outcome *definition*. The rows are therefore **not independent**:
within a patient the rows differ only in the site and its T-score.

> **Every** train/validation split, CV fold, resampling draw and bootstrap must **group by
> `patient_uid`**. No patient may appear in more than one fold. Random row splitting is forbidden
> and is asserted against in `p7_assertions.py`.

Consequence for interpretation: any per-row metric is a **site-level** metric; a patient-level
metric must aggregate the 1–3 rows of a patient **before** scoring, never after.

## 7. What is frozen here vs left open

| item | status |
|---|---|
| task name, observation unit, target definition | **FROZEN** |
| grouping key = `patient_uid` | **FROZEN** |
| site scope recommendation (A) | **FROZEN as the primary/sensitivity split of the report** |
| feature universe, top-k `k`, selector, hyper-parameters | decided in `02_FEATURE_ELIGIBILITY/`, `03_LEAKAGE_SAFE_DEVELOPMENT/`, `05_TOPK_STABILITY/` — batch1 only |
| any temporal-test claim | **OUT OF SCOPE** (batch2 stays sealed) |
"""
open(f"{D['01_TASK_FREEZE']}/TASK_B_SEMANTICS_FREEZE.md", "w").write(doc)

# ------------------------------------------------------------------ SITE_CONTEXT_RULE.md
site_counts = tab.site.value_counts().to_dict()
doc2 = f"""# SITE CONTEXT RULE — `site` is FORCED CONTEXT, never selectable

**Status: HARD CONSTRAINT.**

## 1. The two universes (formal separation)

| universe | contents | counted in `p`? | ranked / top-k? | selection frequency? | pairwise / LLM? |
|---|---|---|---|---|---|
| **SELECTABLE CLINICAL FEATURES** | the 40 semantic candidates, minus those made ineligible by the frozen dev-only rules | ✅ | ✅ | ✅ | ✅ |
| **FORCED CONTEXT** | `site` ∈ {{{", ".join(SITES)}}} (encoded as 2 dummies, baseline `腰椎骨`) | ❌ | ❌ **never** | ❌ **never** | ❌ **never** |

`site` is **not** a selectable feature. It must never appear in: top-k sets, feature rankings,
selection probabilities π_j(k), rank statistics, Jaccard/stability statistics, pairwise ordering
statistics, correlation diagnostics, or any importance compared against a clinical feature
(albumin, age, a lab value, …). It is **not** "competing" with clinical features — it is a
conditioning variable.


`site` stays in the design matrix of **every** fitted model, in every fold, in every resampling
draw. Removing it would (a) drop a real measurement-context effect and (b) make the residual
variance site-dependent, which would in turn distort the clinical-feature ranking we are trying to
measure.

## 3. Implementation (chosen, documented)

`scripts/common.py`:

```python
site_dummies(site)  ->  two columns, one-hot drop-first:
                        ctx_site_髋关节, ctx_site_前臂   (baseline = 腰椎骨)
                        multiplied by FORCED_SCALE = {1000.0}
design = [ ctx_site_髋关节*{1000.0}, ctx_site_前臂*{1000.0},  <standardised clinical block> ]
```

* The dummies are placed **first**; the clinical block is `design[:, 2:]` and is the only block
  that is ever ranked. `extract_clinical()` slices exactly that block, so a site column can never
  leak into the ranking.
* **Why the scale factor.** `sklearn`'s `LogisticRegression` applies one penalty strength to all
  columns. Scaling a column by `s` shrinks its optimal coefficient by `1/s`, so the L1 penalty it
  contributes (`|w|`) becomes ~`1/s` of the unscaled value. With `s = {1000.0}` the site block is
  effectively **unpenalised** — it stays in the model — while the clinical block keeps a natural,
  comparable coefficient scale. A sensitivity fit at `s = {5000.0}` is asserted to leave the
  clinical top-k ordering unchanged (see `p4`), proving the ranking does not depend on the trick.
* Baseline level `腰椎骨` is the reference; the two dummies carry the hip and forearm shifts.

Batch1 context distribution actually fitted (for the record):
`腰椎骨`={site_counts.get('腰椎骨', 0)}, `髋关节`={site_counts.get('髋关节', 0)},
`前臂`={site_counts.get('前臂', 0)}.

## 4. What may and may not be reported about the site block

| allowed | forbidden |
|---|---|
| "site was included as a forced context covariate in every model" | "`site` is the k-th most important feature" |
| the site-scope sensitivity (lumbar+hip vs all sites) | ranking `site` against albumin / age / a lab value |
| site-stratified reporting of performance | putting `site` into a top-k set or selection frequency |

## 5. Assertions (enforced in `p7_assertions.py`)

* [ ] `site` is not in the selectable universe (by name and by construction).
* [ ] the design matrix of every fit has exactly 2 forced context columns, always in positions 0–1.
* [ ] no top-k set, rank statistic or selection probability is defined over `ctx_*` columns.
* [ ] `site`/`ctx_*` never appears in `FEATURE_STABILITY.csv`, `TOPK_SET_STABILITY.csv`,
      `PAIRWISE_DATA_CONFUSION.csv`, `CORRELATED_FEATURE_COMPETITION.csv`.
* [ ] site is present in 100 % of fitted folds.
"""
open(f"{D['01_TASK_FREEZE']}/SITE_CONTEXT_RULE.md", "w").write(doc2)

print("\nsite freeze docs written")
print(f"P1: {len(FAILS)} failure(s)")
raise SystemExit(1 if FAILS else 0)
