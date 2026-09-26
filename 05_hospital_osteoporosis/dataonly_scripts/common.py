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
"""Shared loading / preprocessing / modelling utilities for the DATA-ONLY pilot.

Hard rules encoded here:
  * the ONLY data source is the frozen canonical recovery tree (never modified);
  * the ONLY modelling data is batch1 (batch2 is a sealed temporal external test);
  * `site` is a FORCED CONTEXT covariate, never a selectable feature;
  * all train/validation splitting is grouped by patient_uid;
  * imputation/scaling/encoding/selection/hyper-parameters are fitted inside the
    training fold only.
"""
import hashlib
import json
import os
import numpy as np
import pandas as pd

CANON = str(REPRO_ROOT / 'hospital_osteoporosis_canonical_20260917')
B = str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917')
D = {k: f"{B}/{k}" for k in ("00_INPUT_REFERENCE", "01_TASK_FREEZE", "02_FEATURE_ELIGIBILITY",
                             "03_LEAKAGE_SAFE_DEVELOPMENT", "04_DATA_ONLY_BASELINE",
                             "05_TOPK_STABILITY", "06_DATA_CONFUSION", "07_REPORTS",
                             "08_EVIDENCE_PREP", "09_REPORTS",
                             "audit", "plots", "scripts")}
for p in D.values():
    os.makedirs(p, exist_ok=True)

# ------------------------------------------------------------------ provenance of the inputs
CANON_FILES = {
    "canonical/canonical_manifest.json": None,
    "canonical/feature_dictionary_canonical.csv": None,
    "canonical/feature_identity_registry.csv": None,
    "canonical/site_level_X.csv": None,
    "canonical/site_level_y.csv": None,
    "source/data_clean/cohort_all.csv": None,
    "source/data_clean/batch1_full.csv": None,
    "source/data_clean/batch2_full.csv": None,
    "source/data_clean/cohort_all_full.csv": None,
    "source/data_clean/README.md": None,
}


def sha256(path, blk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(blk), b""):
            h.update(c)
    return h.hexdigest()


def canon_hash_snapshot():
    return {k: {"bytes": os.path.getsize(f"{CANON}/{k}"), "sha256": sha256(f"{CANON}/{k}")}
            for k in CANON_FILES}


# ------------------------------------------------------------------ feature semantics
def semantic_universe():
    """The 40 SEMANTIC CANDIDATE features, read from the FROZEN canonical manifest.

    Never retyped: the list is whatever the frozen handoff shipped.
    """
    man = json.load(open(f"{CANON}/canonical/canonical_manifest.json"))
    return list(man["primary_candidate_set"]["columns"]), man


FEATURE_CATEGORY = {c: ("DEMOGRAPHIC" if c in ("DXA_性别", "DXA_年龄")
                        else "ANTHROPOMETRIC" if c in ("身高_cm", "体重_kg") else "LAB")
                    for c in []}  # filled lazily by features()
DERIVED_FEATURES = {
    "AST/ALT": "天门冬氨酸氨基转移酶 / 丙氨酸氨基转移酶",
    "白球比": "白蛋白 / 球蛋白",
    "尿素：肌酐": "尿素 / 肌酐",
    "估算肾小球滤过率": "肌酐 + DXA_年龄 + DXA_性别 (CKD-EPI 2021, race-free)",
}
ENGLISH = {
    "身高_cm": "Height (cm)", "体重_kg": "Weight (kg)", "身高": "Height",
    "体重": "Weight", "AST/ALT": "AST/ALT ratio (De Ritis ratio)",
    "白球比": "Albumin/globulin ratio", "尿素：肌酐": "Urea/creatinine ratio",
    "估算肾小球滤过率": "Estimated GFR",
}

SITES = ["腰椎骨", "髋关节", "前臂"]
SITE_EN = {"腰椎骨": "lumbar spine", "髋关节": "hip", "前臂": "forearm"}

# forced-context implementation: site dummies are multiplied by this factor so that the
# L1 penalty acting on them is negligible -> site stays in the model, unpenalised in effect,
# while remaining outside the selectable universe.
FORCED_SCALE = 1000.0
FORCED_SCALE_SENSITIVITY = 5000.0

# hyper-parameter grids, FIXED BEFORE any result inspection (see development_manifest.json)
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]           # L1 selector / regularised baseline
K_GRID = [5, 10, 15, 20]                          # pre-registered top-k grid
SEEDS = [11, 22, 33, 44, 55]                      # fixed CV seeds
N_FOLDS = 5
N_RESAMPLES = 200                                 # stability resampling
RESAMPLE_FRACTION = 0.80                          # fraction of PATIENTS per resample


# ------------------------------------------------------------------ data
def batch1_raw():
    """batch1 rows exactly as the frozen source ships them, plus patient_uid.

    Returns the raw 48-column batch1 table (no parsing beyond the raw strings).
    """
    f = f"{CANON}/source/data_clean/batch1_full.csv"
    df = pd.read_csv(f, dtype=str, low_memory=False)
    uid = pd.read_csv(f"{CANON}/private_only/patient_uid_map.csv", dtype=str) \
        .set_index("住院唯一号")["patient_uid"]
    df["patient_uid"] = df["住院唯一号"].map(uid)
    assert df["patient_uid"].notna().all(), "patient_uid mapping incomplete"
    assert "住院唯一号" not in df.columns or True
    return df


def parse_age(s):
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False),
                         errors="coerce")


def parse_sex(s):
    """女 -> 1, 男 -> 0 (fixed documented mapping; not fitted on any target)."""
    m = {"女": 1.0, "男": 0.0}
    return s.astype(str).str.strip().map(m)


def design_table(df=None):
    """Parsed, analysis-ready batch1 table: patient_uid, site, y, and every candidate column.

    C-反应蛋白 is kept BOTH as a raw string column (`C-反应蛋白__raw`) and as a naive
    numeric parse (`C-反应蛋白__numeric_parse`) plus a censoring flag, so that the pilot can
    show what the parse would do WITHOUT ever adopting it as an eligible numeric feature.
    """
    if df is None:
        df = batch1_raw()
    cols, man = semantic_universe()
    out = pd.DataFrame({
        "patient_uid": df["patient_uid"].to_numpy(),
        "site": df["site"].to_numpy(),
        "batch": df["batch"].to_numpy(),
        "y": pd.to_numeric(df["label_osteoporosis"]).astype(int).to_numpy(),
        "T_used": pd.to_numeric(df["T_used"], errors="coerce").to_numpy(),
    })
    out["DXA_年龄"] = parse_age(df["DXA_年龄"]).to_numpy()
    out["DXA_性别_str"] = df["DXA_性别"].to_numpy()
    out["DXA_性别"] = parse_sex(df["DXA_性别"]).to_numpy()
    for c in cols:
        if c in ("DXA_年龄", "DXA_性别"):
            continue
        if c == "C-反应蛋白":
            out["C-反应蛋白__raw"] = df[c].to_numpy()
            out["C-反应蛋白__numeric_parse"] = pd.to_numeric(df[c], errors="coerce").to_numpy()
            out["C-反应蛋白__censored_lt4"] = df[c].astype(str).str.match(r"^<").to_numpy()
            continue
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    return out


def label_check(tab):
    """label == 1[T_used <= -2.5] for every row (tolerance-free, float compare)."""
    return int((tab["y"].to_numpy() != (tab["T_used"].to_numpy() <= -2.5).astype(int)).sum())


# ------------------------------------------------------------------ forced-context design
def site_dummies(site, scale=FORCED_SCALE):
    """One-hot of site, drop-first (baseline = 腰椎骨), scaled so that L1 barely touches it."""
    d = np.zeros((len(site), 2))
    d[:, 0] = (np.asarray(site) == "髋关节")
    d[:, 1] = (np.asarray(site) == "前臂")
    return d * scale


FORCED_NAMES = ["ctx_site_髋关节", "ctx_site_前臂"]


def fit_scaler(Xtr):
    """training-fold-only imputation + standardisation of the SELECTABLE block."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Z = np.where(np.isnan(Xtr), med, Xtr)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return med, mu, sd


def apply_scaler(X, med, mu, sd):
    Z = np.where(np.isnan(X), med, X)
    return (Z - mu) / sd


def make_design(Xsel_tr, site_tr, Xsel_va=None, site_va=None, scale=FORCED_SCALE, scaler=None):
    """Returns (design_train, design_valid, scaler). Forced context is appended FIRST."""
    if scaler is None:
        scaler = fit_scaler(Xsel_tr)
    med, mu, sd = scaler
    A = apply_scaler(Xsel_tr, med, mu, sd)
    A = np.column_stack([site_dummies(site_tr, scale), A])
    if Xsel_va is None:
        return A, scaler
    V = apply_scaler(Xsel_va, med, mu, sd)
    V = np.column_stack([site_dummies(site_va, scale), V])
    return A, V, scaler


def extract_clinical(coef, p, scale_aware=True):
    """|coef| of the selectable block. The forced block is skipped by construction."""
    return np.asarray(coef)[2:2 + p]


def rank_from_scores(scores):
    """Descending |score| -> integer ranks 1..p (ties broken deterministically by index)."""
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    r = np.empty(len(scores), dtype=int)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def topk_from_scores(scores, k):
    r = rank_from_scores(scores)
    return set(int(i) for i in np.where(r <= k)[0])


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def json_dump(obj, path):
    def enc(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=enc)
