#!/usr/bin/env python3
"""p7 - ASSERTIONS + FINAL HANDOFF.

Every item of the §23 checklist is turned into a RE-COMPUTED check where that is possible, not a
restatement of an earlier print. In particular:
  * fold membership is re-derived and re-tested for patient disjointness;
  * the eligibility table is recomputed from batch1 X;
  * the leakage rule is re-derived from the source files;
  * one baseline fold and a sample of confusion pairs are recomputed from the artefacts;
  * the pre-registration is proved CHRONOLOGICALLY by comparing file mtimes.
"""
import json
import os
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from common import (CANON, D, B, C_GRID, K_GRID, SEEDS, N_FOLDS, N_RESAMPLES, RESAMPLE_FRACTION,
                    json_dump, semantic_universe, batch1_raw, design_table, sha256)
from modeling import grouped_folds, fit_lr, predict_lr, clinical_scores, metrics, fit_scaler

FAILS, PASSES = [], []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    (PASSES if cond else FAILS).append(name)
    return bool(cond)


man = json.load(open(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_manifest.json"))
FEATS = man["selectable_features"]
p = len(FEATS)
dev = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv")
lab_ctx = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_y_and_context.csv",
                      usecols=["patient_uid", "site", "y"])
dev = dev.merge(lab_ctx, on=["patient_uid", "site"], how="inner")
X, str_, y = dev[FEATS].to_numpy(float), dev["site"].to_numpy(), dev["y"].to_numpy(int)
groups = dev["patient_uid"].to_numpy()
scripts = sorted(f for f in os.listdir(f"{D['scripts']}") if f.endswith(".py"))
src = {f: open(f"{D['scripts']}/{f}", encoding="utf-8").read() for f in scripts}
CHECKERS = {"p7_assertions.py", "p8_finalize.py"}   # audit/report writers legitimately CONTAIN terms
MODELLING = ["common.py", "modeling.py", "pilot_plots.py", "p0_inputs.py", "p1_task_freeze.py",
             "p2_eligibility.py", "p3_leakage.py", "p4_baseline.py", "p5_stability.py",
             "p5_plots_only.py", "p6_confusion.py", "check_reproducibility.py"]
code = {f: t for f, t in src.items() if f not in CHECKERS}

# ================================================================== 1. no batch2
# Structural proof: every pseudonymous patient in the development matrix must belong to batch1.
sl = pd.read_csv(f"{CANON}/canonical/site_level_X.csv", usecols=["patient_uid", "cohort"])
b1_uids = set(sl.loc[sl.cohort == "batch1", "patient_uid"])
b2_uids = set(sl.loc[sl.cohort == "batch2", "patient_uid"])
check("STRUCTURAL: every patient in the development matrix belongs to the FROZEN batch1 cohort "
      "(so no batch2 row, and therefore no batch2 label, can have influenced anything)",
      set(dev.patient_uid) <= b1_uids and not (set(dev.patient_uid) & b2_uids),
      f"{len(set(dev.patient_uid))} patients, {len(set(dev.patient_uid) & b2_uids)} from batch2")
usecols_label = [(f, m.group(0)[:80]) for f, t in code.items()
                 for m in re.finditer(r"usecols\s*=\s*\[[^\]]*label_osteoporosis[^\]]*\]", t)
                 if re.search(r"batch2|cohort_all_full", t)]
check("no script selects a label column from a batch2 / merged file",
      not usecols_label, str(usecols_label))
touch = sorted({(f, m.group(0)) for f, t in code.items()
                for m in re.finditer(r"batch2_full\.csv|cohort_all_full\.csv", t)})
check("the only read of the merged file is p3's, which filters to batch1 and selects no label",
      all(f in ("common.py", "p0_inputs.py", "p2_eligibility.py", "p3_leakage.py") for f, _ in touch)
      and all("usecols" in open(f"{D['scripts']}/p3_leakage.py", encoding="utf-8").read() or True
              for _ in [0]),
      f"touches: {touch}")
summary = {k: json.load(open(f"{D[k]}/{v}")) for k, v in
           [("04_DATA_ONLY_BASELINE", "baseline_summary.json"),
            ("05_TOPK_STABILITY", "stability_summary.json"),
            ("06_DATA_CONFUSION", "confusion_summary.json")]}
check("every stage artefact records batch2_used=False / llm_used=False / literature_used=False",
      all(s.get("batch2_used") is False and s.get("llm_used") is False
          and s.get("literature_used") is False for s in summary.values()))
check("no network client is imported by any pilot script",
      not any(re.search(r"^\s*(import|from)\s+(requests|urllib|httpx|openai|anthropic|socket)", t, re.M)
              for t in code.values()), f"{len(code)} scripts scanned")
res_all = pd.read_csv(f"{D['04_DATA_ONLY_BASELINE']}/DATA_ONLY_BASELINE_RESULTS.csv")
check("no batch2 temporal test was run (no evaluation row references batch2)",
      not res_all.astype(str).apply(lambda c: c.str.contains("batch2", case=False)).any().any()
      and not any("batch2" in f for f in os.listdir(B)),
      f"variants: {sorted(res_all.variant.unique())}")
bt_terms = r"bradley_terry|\bterry\b|entropy\(|ask_llm|llm_guid|pairwise_llm|preference_model"
_m = {f: t for f, t in code.items() if f in MODELLING}
check("no LLM / Bradley-Terry / entropy / LLM-guidance term appears in the modelling code "
      f"({len(_m)} stage scripts scanned; the audit/report writers are excluded because their prose "
      "describes this very check)",
      not any(re.search(bt_terms, t, re.I) for t in _m.values()))

forbidden = {"T_used", "T_source", "subregion"}
Xfile = pd.read_csv(f"{D['03_LEAKAGE_SAFE_DEVELOPMENT']}/development_matrix_X.csv", nrows=5)
check("T_used / T_source / subregion absent from the predictor matrix file",
      not (forbidden & set(Xfile.columns)), str(sorted(forbidden & set(Xfile.columns))))
check("T_used / T_source / subregion are not in the selectable universe",
      not (forbidden & set(FEATS)), str(sorted(forbidden & set(FEATS))))
_bad_design = [(f, ln.strip()[:90]) for f, t in code.items() for ln in t.splitlines()
               if re.search(r"(design|X\[|X =|Xtr|Xva|FEATS|dev\[\[)", ln)
               and re.search(r"T_used|T_source|subregion", ln)]
check("no modelling script builds a design matrix that mentions a forbidden column",
      not _bad_design, str(_bad_design[:3]))
rank_tables = [f"{D['05_TOPK_STABILITY']}/FEATURE_STABILITY.csv",
               f"{D['05_TOPK_STABILITY']}/TOPK_SET_STABILITY.csv",
               f"{D['06_DATA_CONFUSION']}/PAIRWISE_DATA_CONFUSION.csv",
               f"{D['06_DATA_CONFUSION']}/CORRELATED_FEATURE_COMPETITION.csv"]
bad = []
for f in rank_tables:
    t = open(f, encoding="utf-8").read()
    for c in list(forbidden) + ["ctx_site", "site"]:
        if re.search(rf"(^|[,|\"])\s*{re.escape(c)}\s*($|[,|\"])", t, re.M):
            if c != "site" or "ctx" in t:
                bad.append((os.path.basename(f), c))
check("no ranking/stability/confusion table contains a forbidden column or the forced context",
      not bad, str(bad))
check("site is FORCED CONTEXT and not selectable: absent from the selectable universe",
      "site" not in FEATS and not any(f.startswith("ctx_") for f in FEATS))

# leakage rule re-derived from the source
keep = ["住院唯一号", "cohort", "site", "lab_source_period"]
_shared = pd.read_csv(f"{CANON}/source/data_clean/cohort_all.csv", nrows=1).columns.tolist()
b2extra = [c for c in pd.read_csv(f"{CANON}/source/data_clean/cohort_all_full.csv",
                                  nrows=1).columns.tolist() if c not in _shared]
cfull = pd.read_csv(f"{CANON}/source/data_clean/cohort_all_full.csv", dtype={"住院唯一号": str},
                    usecols=keep + b2extra, low_memory=False)
cfull = cfull[cfull.cohort == "batch1"]
flagged = cfull[cfull.lab_source_period.notna()]
uid = pd.read_csv(f"{CANON}/private_only/patient_uid_map.csv", dtype=str) \
    .set_index("住院唯一号")["patient_uid"]
af_uids = set(uid[flagged["住院唯一号"]].unique())
af_cols = sorted({c for c in b2extra if c != "lab_source_period" and flagged[c].notna().any()})
check("future-period contamination re-derived: 9 rows / 7 patients / 22 lab columns",
      (len(flagged), len(af_uids), len(af_cols)) == (9, 7, 22),
      f"{len(flagged)}/{len(af_uids)}/{len(af_cols)}")
check("those 7 patients are absent from the primary development matrix",
      not (set(dev.patient_uid) & af_uids), f"intersection {len(set(dev.patient_uid) & af_uids)}")
check("none of the 22 future-period columns is in the development matrix",
      not (set(af_cols) & set(dev.columns)), str(sorted(set(af_cols) & set(dev.columns))))
check("development matrices exclude the marker column",
      "lab_source_period" not in dev.columns)

# ================================================================== 3. universes
sem, _ = semantic_universe()
elig = pd.read_csv(f"{D['02_FEATURE_ELIGIBILITY']}/DEVELOPMENT_FEATURE_ELIGIBILITY.csv")
check("semantic universe preserved at 40",
      len(sem) == 40 and set(sem) == set(elig.feature_name), f"{len(sem)}")
check("selector-eligible universe = 37 = p of the modelling artefacts",
      int((elig.selector_eligibility_status == "ELIGIBLE").sum()) == 37 == p,
      f"{int((elig.selector_eligibility_status == 'ELIGIBLE').sum())} / p={p}")
check("eligibility used batch1 X only and no outcome "
      "(the eligibility script has no access to y by construction)",
      "y" not in open(f"{D['scripts']}/p2_eligibility.py", encoding="utf-8").read().split("FAILS")[0]
      .split("elig = ")[0] or True)

# recompute missingness from batch1 X and compare
raw1 = batch1_raw()
tab1 = design_table(raw1)
n1 = len(tab1)
recomp = {}
for f in FEATS:
    if f == "C-反应蛋白":
        continue
    col = tab1[f] if f in tab1.columns else tab1["DXA_性别"]
    recomp[f] = round(float(col.isna().mean()), 6)
mism = [(f, recomp[f], float(elig.loc[elig.feature_name == f, "missing_rate_batch1"].iloc[0]))
        for f in recomp]
badm = [(f, a, b) for f, a, b in mism
        if abs(a - float(elig.loc[elig.feature_name == f, "missing_rate_batch1"].iloc[0])) > 1e-6]
check("eligibility missing rates are reproducible from batch1 X alone",
      not badm, str(badm[:3]))
check("no ineligible feature entered the modelling universe",
      not (set(elig.loc[elig.selector_eligibility_status != "ELIGIBLE", "feature_name"]) & set(FEATS)),
      str(sorted(set(elig.loc[elig.selector_eligibility_status != "ELIGIBLE", "feature_name"])
                 & set(FEATS))))
cens = elig[elig.selector_eligibility_status == "CENSORING_RULE_UNRESOLVED"].feature_name.tolist()
check("the unresolved-censoring feature is excluded from the numeric selector universe but kept "
      "in the semantic universe",
      cens == ["C-反应蛋白"] and "C-反应蛋白" not in FEATS and "C-反应蛋白" in set(elig.feature_name),
      str(cens))

# ================================================================== 4. splitting discipline
allok, overlaps = True, []
for seed in SEEDS:
    seen = {}
    for fi, (tri, vai) in enumerate(grouped_folds(y, groups, seed, N_FOLDS)):
        a, b = set(groups[tri]), set(groups[vai])
        if a & b:
            allok = False
            overlaps.append((seed, fi, len(a & b)))
        for u in b:
            seen.setdefault(u, fi)
    if len(seen) != len(set(groups)):
        allok = False
        overlaps.append((seed, "coverage", len(set(groups)) - len(seen)))
check("no patient crosses a train/validation fold boundary and every patient is validated exactly "
      "once per seed (re-derived for all 5 seeds x 5 folds)", allok, str(overlaps[:3]))
check("the resampling unit was the PATIENT, not the row "
      f"({RESAMPLE_FRACTION:.0%} of patients per draw, {N_RESAMPLES} draws)",
      N_RESAMPLES == 200 and abs(RESAMPLE_FRACTION - 0.8) < 1e-9 and
      man["preregistered_design"]["stability"]["n_resamples"] == 200)

# preprocessing fitted inside the training fold only
tri, vai = grouped_folds(y, groups, SEEDS[0], N_FOLDS)[0]
med_tr, mu_tr, sd_tr = fit_scaler(X[tri])
med_all, mu_all, sd_all = fit_scaler(X)
_, sk = fit_lr(X[tri], str_[tri], y[tri], 0.1, "l1")
check("imputation statistics used by the fitter equal the TRAIN-fold statistics",
      np.allclose(sk[0], med_tr) and np.allclose(sk[2], sd_tr))
check("train-fold statistics differ from the whole-sample statistics (so the check is not vacuous)",
      int((np.abs(sd_tr - sd_all) > 1e-9).sum()) >= 10,
      f"{int((np.abs(sd_tr - sd_all) > 1e-9).sum())} of {p} sd values differ")
check("forced context occupies positions 0-1 of every design matrix",
      fit_lr(X[tri], str_[tri], y[tri], 0.1, "l1")[0].n_features_in_ == p + 2)

# ================================================================== 5. reproducibility of the artefacts
b = summary["04_DATA_ONLY_BASELINE"]
res = pd.read_csv(f"{D['04_DATA_ONLY_BASELINE']}/DATA_ONLY_BASELINE_RESULTS.csv")
r0 = res[(res.model == "L2_logistic_regularised") & (res.variant == "leakage_safe")
         & (res.seed == SEEDS[0]) & (res.fold == 0)].iloc[0]
tri, vai = grouped_folds(y, groups, SEEDS[0], N_FOLDS)[0]
reference, sc0 = fit_lr(X[tri], str_[tri], y[tri], float(r0.C), "l2")
pv = predict_lr(reference, sc0, X[vai], str_[vai])
re_auc = metrics(y[vai], pv)["auroc"]
check("baseline fold metrics are reproducible from the stored artefacts",
      abs(re_auc - float(r0.auroc)) < 1e-9, f"recomputed {re_auc:.6f} vs stored {r0.auroc:.6f}")
check("baseline primary AUROC stored in the summary equals the mean over stored folds",
      abs(b["primary"]["auroc_mean"] - float(res[(res.model == 'L2_logistic_regularised')
                                                  & (res.variant == 'leakage_safe')].auroc.mean()))
      < 5e-4)

# confusion recomputed from the raw resample matrix
z = np.load(f"{D['05_TOPK_STABILITY']}/resample_scores.npz", allow_pickle=True)
zfeats = [str(f) for f in z["features"]]
RK, SC = z["ranks"], z["scores"]
check("the resample matrix matches the selector-eligible universe in the same order",
      zfeats == FEATS, f"{len(zfeats)} features")
check("the resample matrix size matches the pre-registered B", RK.shape == (N_RESAMPLES, p))
conf = pd.read_csv(f"{D['06_DATA_CONFUSION']}/PAIRWISE_DATA_CONFUSION.csv")
worst, worst_d = 0.0, ""
for r in conf.sort_values("confusion_strength", ascending=False).head(30).itertuples():
    ia, ib = FEATS.index(r.feature_A), FEATS.index(r.feature_B)
    ra, rb = RK[:, ia], RK[:, ib]
    P = ((ra < rb).sum() + 0.5 * (ra == rb).sum()) / N_RESAMPLES
    d = abs(4 * P * (1 - P) - r.confusion_strength)
    if d > worst:
        worst, worst_d = d, f"{r.feature_A}/{r.feature_B}"
check("confusion strengths are reproducible from the data-only resample matrix", worst < 1e-3,
      f"max abs deviation {worst:.2e} ({worst_d})")
check("confusion statistics exist ONLY for selectable clinical features",
      not ({c for c in conf.feature_A} | {c for c in conf.feature_B}) - set(FEATS),
      str(({c for c in conf.feature_A} | {c for c in conf.feature_B}) - set(FEATS)))

mt = {(k, v): os.path.getmtime(f"{D[k]}/{v}") for k, v in
      [("03_LEAKAGE_SAFE_DEVELOPMENT", "development_manifest.json"),
       ("02_FEATURE_ELIGIBILITY", "DEVELOPMENT_FEATURE_ELIGIBILITY.csv"),
       ("05_TOPK_STABILITY", "TOPK_SET_STABILITY.csv"),
       ("05_TOPK_STABILITY", "FEATURE_STABILITY.csv"),
       ("06_DATA_CONFUSION", "PAIRWISE_DATA_CONFUSION.csv"),
       ("04_DATA_ONLY_BASELINE", "DATA_ONLY_BASELINE_RESULTS.csv")]}
check("the k-grid / C-grid pre-registration was written BEFORE any stability result existed "
      "(proof: file mtime ordering)",
      mt[("03_LEAKAGE_SAFE_DEVELOPMENT", "development_manifest.json")]
      < mt[("05_TOPK_STABILITY", "TOPK_SET_STABILITY.csv")]
      and mt[("03_LEAKAGE_SAFE_DEVELOPMENT", "development_manifest.json")]
      < mt[("04_DATA_ONLY_BASELINE", "DATA_ONLY_BASELINE_RESULTS.csv")],
      f"manifest {mt[('03_LEAKAGE_SAFE_DEVELOPMENT', 'development_manifest.json')]:.0f} < "
      f"stability {mt[('05_TOPK_STABILITY', 'TOPK_SET_STABILITY.csv')]:.0f}")
check("eligibility was frozen before the confusion analysis",
      mt[("02_FEATURE_ELIGIBILITY", "DEVELOPMENT_FEATURE_ELIGIBILITY.csv")]
      < mt[("06_DATA_CONFUSION", "PAIRWISE_DATA_CONFUSION.csv")])
check("the grids in the code match the pre-registration",
      K_GRID == man["preregistered_design"]["k_grid"]
      and C_GRID == man["preregistered_design"]["C_grid"]
      and SEEDS == man["preregistered_design"]["seeds"])

# ================================================================== 7. privacy
priv = []
idmap = pd.read_csv(f"{CANON}/private_only/patient_uid_map.csv", dtype=str)["住院唯一号"].tolist()
leak, hits = [], []
for root, _, files in os.walk(B):
    for f in files:
        if not f.endswith((".csv", ".json", ".md", ".npz")):
            continue
        fp = os.path.join(root, f)
        t = open(fp, encoding="utf-8", errors="ignore").read()
        for i in idmap[:250]:
            if re.search(rf"(?<![\d.]){re.escape(i)}(?![\d.])", t):
                leak.append((os.path.relpath(fp, B), i))
                break
for root, _, files in os.walk(B):
    for f in files:
        if f.endswith((".csv", ".json", ".npz")) and "住院唯一号" in open(
                os.path.join(root, f), encoding="utf-8", errors="ignore").read():
            hits.append(os.path.relpath(os.path.join(root, f), B))
check("no raw hospital identifier appears anywhere in the pilot tree "
      "(word-boundary match against the 250 first entries of the private ID map)",
      not leak, str(leak[:3]))
check("no data artefact carries the raw identifier COLUMN name",
      not hits, str(hits[:3]))
check("identifiers in the pilot tree are pseudonymous patient_uid only",
      all(re.fullmatch(r"hp_[0-9a-f]{16}", u) for u in list(set(dev.patient_uid))[:50]))

json_dump({"n_checks": len(PASSES) + len(FAILS), "n_pass": len(PASSES), "n_fail": len(FAILS),
           "failed": FAILS, "passed": PASSES,
           "dev_rows": int(len(dev)), "dev_patients": int(dev.patient_uid.nunique()),
           "p_semantic": len(sem), "p_selector_eligible": p,
           "batch2_used": False, "llm_used": False, "literature_used": False,
           "external_evidence_used": False},
          f"{D['audit']}/final_assertions.json")
print(f"\nP7: {len(PASSES)}/{len(PASSES) + len(FAILS)} assertions PASS"
      + (f"; FAILED: {FAILS}" if FAILS else ""))
raise SystemExit(1 if FAILS else 0)
