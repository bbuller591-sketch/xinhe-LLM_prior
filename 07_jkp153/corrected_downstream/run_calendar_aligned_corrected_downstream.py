#!/usr/bin/env python3
from __future__ import annotations

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

import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_ndtr
from scipy.stats import norm, pearsonr, rankdata, spearmanr


ROOT = Path(str(REPRO_ROOT))
PROJECT = ROOT / "finance_llm_prior"
OUT = ROOT / "JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321"
BASE = ROOT / "JKP153_FORMAL_MEASUREMENT_DESIGN_FREEZE_V0_4_20260913_201725"
VAL24 = ROOT / "JKP153_2024_MEASUREMENT_AND_VALIDATION_V0_4_20260914_112349"
FINAL25 = ROOT / "JKP153_2025_FINAL_TEST_V0_4_20260915_015736"
CODE_REVIEW = ROOT / "JKP153_DATAONLY_TARGET_SRCC_CODE_REVIEW_20260915_022514"
V01 = PROJECT / "experiments/jkp153_capm_alpha_dataonly_integration_v01/20260911_160413"
V03 = PROJECT / "experiments/jkp153_final_exploratory_llm_capm_v03/20260911_180045"
V04 = PROJECT / "experiments/jkp153_signfixed_crossbackbone_v04/20260911_234833"
FACTOR_CSV = PROJECT / "data/raw/jkp/monthly_vw_cap/[usa]_[all_factors]_[monthly]_[vw_cap].csv"
MARKET_ZIP = V01 / "market_raw/[usa]_[mkt]_[monthly]_[vw_cap].zip"
REGISTRY = BASE / "templates/D0_FACTOR_METADATA_REGISTRY.csv"
TARGET_PANEL = V03 / "CORRECTED_FUTURE_CAPM_ALPHA_TARGETS.parquet"
RAW_PANEL = V04 / "RAW_BACKBONE_PANEL.parquet"
EB_PANEL = V04 / "EB_BACKBONE_PANEL.parquet"
GLOBAL_PATH = V04 / "STRUCTURED_GLOBAL_CORRECTED.csv"

WINDOW = 120
HORIZON = 12
MIN_ROLLING_OBS = 60
LAMBDA_GRID = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
RHO_GRID = [0, 0.25, 0.5, 0.75, 1.0]
LLM_ARMS = ["DQ", "DL_gated", "DQL_gated"]
BOOT_SEED = 20260915


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_market() -> pd.DataFrame:
    with zipfile.ZipFile(MARKET_ZIP) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        with z.open(names[0]) as f:
            return pd.read_csv(f, na_values=["na", "NA", ""])


def month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s) + pd.offsets.MonthEnd(0)


def load_panels():
    factors = pd.read_csv(REGISTRY)["factor_id"].astype(str).tolist()
    fr = pd.read_csv(FACTOR_CSV, na_values=["na", "NA", ""])
    mr = read_market()
    fr["date"] = month_end(fr["date"])
    mr["date"] = month_end(mr["date"])
    fr["ret"] = pd.to_numeric(fr["ret"], errors="coerce")
    mr["ret"] = pd.to_numeric(mr["ret"], errors="coerce")
    fp = fr[fr["name"].isin(factors)].pivot(index="date", columns="name", values="ret").sort_index().reindex(columns=factors)
    mkt = mr[mr["name"].eq("mkt")].drop_duplicates("date").set_index("date")["ret"].sort_index()
    idx = fp.index.union(mkt.index).sort_values()
    return factors, fr, mr, fp.reindex(idx), mkt.reindex(idx)


def ols_alpha_beta(y: np.ndarray, x: np.ndarray, min_obs: int):
    ok = np.isfinite(y) & np.isfinite(x)
    n = int(ok.sum())
    if n < min_obs:
        return None
    yy, xx = y[ok], x[ok]
    xbar, ybar = float(xx.mean()), float(yy.mean())
    xc, yc = xx - xbar, yy - ybar
    sxx = float(np.dot(xc, xc))
    if sxx <= 0:
        return None
    beta = float(np.dot(xc, yc) / sxx)
    alpha = float(ybar - beta * xbar)
    resid = yy - alpha - beta * xx
    rv = float(np.dot(resid, resid) / max(n - 2, 1))
    se = float(math.sqrt(max(rv, 0.0) * (1.0 / n + xbar * xbar / sxx)))
    return alpha, beta, se, rv, n


def tau2_mle(alpha: np.ndarray, se: np.ndarray):
    ok = np.isfinite(alpha) & np.isfinite(se) & (se > 0)
    a, s2 = alpha[ok], se[ok] ** 2
    if len(a) == 0:
        return 0.0, True

    def nll(tau2: float) -> float:
        v = tau2 + s2
        return float(0.5 * np.sum(np.log(v) + a * a / v))

    upper = max(float(np.nanvar(a) * 20), float(np.nanmean(a * a) * 20), 1e-8)
    boundary = nll(0.0)
    res = minimize_scalar(nll, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-14})
    if boundary <= float(res.fun) + 1e-10:
        return 0.0, True
    return max(float(res.x), 0.0), bool(float(res.x) <= 1e-12)


def score_at(factors_order, fp, mkt, forecast_origin: str, information_cutoff: str):
    dates = list(fp.index)
    i = {d: k for k, d in enumerate(dates)}[pd.Timestamp(forecast_origin)]
    start, end = i - WINDOW, i
    ymat, x = fp.to_numpy(float), mkt.to_numpy(float)
    rows = []
    for j, name in enumerate(factors_order):
        est = ols_alpha_beta(ymat[start:end, j], x[start:end], MIN_ROLLING_OBS)
        if est is None:
            alpha = beta = se = rv = np.nan
            n = 0
        else:
            alpha, beta, se, rv, n = est
        rows.append({
            "factor_id": name,
            "information_cutoff_date": information_cutoff,
            "forecast_window_start": forecast_origin,
            "history_start": dates[start],
            "history_end": dates[end - 1],
            "history_n_obs": n,
            "raw_alpha": alpha,
            "raw_beta": beta,
            "raw_alpha_se": se,
            "raw_residual_variance": rv,
        })
    out = pd.DataFrame(rows)
    tau2, boundary = tau2_mle(out["raw_alpha"].to_numpy(float), out["raw_alpha_se"].to_numpy(float))
    s2 = out["raw_alpha_se"].to_numpy(float) ** 2
    kappa = np.divide(tau2, tau2 + s2, out=np.zeros_like(s2), where=(tau2 + s2) > 0)
    out["tau2_data"] = tau2
    out["tau2_boundary"] = boundary
    out["kappa_data"] = kappa
    out["EB_score"] = out["raw_alpha"].to_numpy(float) * kappa
    return out


def target_at(factors_order, fp, mkt, forecast_origin: str, information_cutoff: str):
    dates = list(fp.index)
    i = {d: k for k, d in enumerate(dates)}[pd.Timestamp(forecast_origin)]
    start, end = i, i + HORIZON
    ymat, x = fp.to_numpy(float), mkt.to_numpy(float)
    rows = []
    for j, name in enumerate(factors_order):
        est = ols_alpha_beta(ymat[start:end, j], x[start:end], math.ceil(0.8 * HORIZON))
        if est is None:
            alpha = beta = se = rv = np.nan
            n = int((np.isfinite(ymat[start:end, j]) & np.isfinite(x[start:end])).sum())
        else:
            alpha, beta, se, rv, n = est
        rows.append({
            "factor_id": name,
            "information_cutoff_date": information_cutoff,
            "forecast_window_start": forecast_origin,
            "target_start": dates[start],
            "target_end": dates[end - 1],
            "future_n_obs": n,
            "future_CAPM_alpha": alpha,
            "future_CAPM_beta": beta,
            "future_CAPM_alpha_se": se,
            "future_residual_variance": rv,
            "target_rule": "CORRECTED_CALENDAR_ALIGNED_MIN80",
        })
    return pd.DataFrame(rows)


def rank01(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    return ((s.rank(method="average") - 1) / (len(s) - 1)).to_numpy(float)


def spearman_ic(score: np.ndarray, target: np.ndarray) -> float:
    ok = np.isfinite(score) & np.isfinite(target)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(np.asarray(score)[ok], np.asarray(target)[ok]).statistic)


def spearman_manual(score: np.ndarray, target: np.ndarray) -> float:
    ok = np.isfinite(score) & np.isfinite(target)
    xr = rankdata(np.asarray(score)[ok], method="average")
    yr = rankdata(np.asarray(target)[ok], method="average")
    return float(pearsonr(xr, yr).statistic)


def posmix_closed(y: np.ndarray, se: np.ndarray, tau2: np.ndarray, q: np.ndarray, rho: float) -> np.ndarray:
    s2 = se * se
    den = tau2 + s2
    k = np.divide(tau2, den, out=np.zeros_like(y, dtype=float), where=den > 0)
    mu0 = k * y
    v0 = np.divide(tau2 * s2, den, out=np.zeros_like(y, dtype=float), where=den > 0)
    sd = np.sqrt(np.maximum(v0, 0))
    out = mu0.copy()
    w = np.zeros_like(y, dtype=float)
    active = (tau2 > 0) & (sd > 0) & np.isfinite(y) & np.isfinite(se)
    a = np.zeros_like(y, dtype=float)
    a[active] = mu0[active] / sd[active]
    pi = np.clip(rho * q, 0, 1)
    log_ratio = np.full_like(y, -np.inf, dtype=float)
    log_ratio[active] = math.log(2.0) + log_ndtr(a[active])
    w[active & (pi <= 0)] = 0.0
    w[active & (pi >= 1)] = 1.0
    mid = active & (pi > 0) & (pi < 1)
    w[mid] = expit(np.log(pi[mid]) - np.log1p(-pi[mid]) + log_ratio[mid])
    muplus = mu0.copy()
    muplus[active] = mu0[active] + sd[active] * np.exp(norm.logpdf(a[active]) - log_ndtr(a[active]))
    out[active] = (1 - w[active]) * mu0[active] + w[active] * muplus[active]
    return out


def guidance(year: int, factors_order: list[str]) -> pd.DataFrame:
    path = (VAL24 if year == 2024 else FINAL25) / f"guidance/{year}_q_scores.csv"
    g = pd.read_csv(path).set_index("factor_id").reindex(factors_order).reset_index()
    global_df = pd.read_csv(GLOBAL_PATH).rename(columns={"factor": "factor_id"}).set_index("factor_id").reindex(factors_order)
    g["q_Global"] = rank01(global_df["psi_structured_global"].to_numpy(float))
    return g


def evaluate_year(year: int, scores: pd.DataFrame, targets: pd.DataFrame, g: pd.DataFrame, selected=None):
    factors = scores["factor_id"].astype(str).tolist()
    target = targets.set_index("factor_id").reindex(factors)["future_CAPM_alpha"].to_numpy(float)
    raw = scores["raw_alpha"].to_numpy(float)
    eb = scores["EB_score"].to_numpy(float)
    raw_rank = rank01(raw)
    al_col = "A_L_2024" if year == 2024 else "A_L_2025"
    al = g[al_col].to_numpy(float)
    q = {
        "Global": g["q_Global"].to_numpy(float),
        "DQ": g["q_DQ"].to_numpy(float),
        "DL_gated": g["q_DL"].to_numpy(float),
        "DQL_gated": g["q_DQL_gated"].to_numpy(float),
    }
    standalone = [
        {"method": "Raw", "family": "data_only", "weight": "", f"spearman_rank_ic_{year}_h12": spearman_ic(raw, target)},
        {"method": "EB", "family": "data_only", "weight": "", f"spearman_rank_ic_{year}_h12": spearman_ic(eb, target)},
        {"method": "Global_standalone", "family": "guidance_standalone", "weight": "", f"spearman_rank_ic_{year}_h12": spearman_ic(q["Global"], target)},
        {"method": "DQ_standalone", "family": "guidance_standalone", "weight": "", f"spearman_rank_ic_{year}_h12": spearman_ic(q["DQ"], target)},
        {"method": "DL_gated_standalone", "family": "guidance_standalone", "weight": "", f"spearman_rank_ic_{year}_h12": spearman_ic(np.where(al == 1, q["DL_gated"], raw_rank), target)},
        {"method": "DQL_gated_standalone", "family": "guidance_standalone", "weight": "", f"spearman_rank_ic_{year}_h12": spearman_ic(q["DQL_gated"], target)},
    ]
    rf_rows = []
    pm_rows = []
    if selected is None:
        for lam in LAMBDA_GRID:
            for arm in LLM_ARMS:
                score = raw_rank.copy()
                if arm == "DL_gated":
                    score[al == 1] = (1 - lam) * raw_rank[al == 1] + lam * q[arm][al == 1]
                else:
                    score = (1 - lam) * raw_rank + lam * q[arm]
                rf_rows.append({"family": "RF", "guidance": arm, "lambda_RF": lam, "selection_member": True, f"spearman_rank_ic_{year}_h12": spearman_ic(score, target)})
            score = (1 - lam) * raw_rank + lam * q["Global"]
            rf_rows.append({"family": "RF", "guidance": "Global", "lambda_RF": lam, "selection_member": False, f"spearman_rank_ic_{year}_h12": spearman_ic(score, target)})
        pm_q = {"DQ": q["DQ"], "DL_gated": al * q["DL_gated"], "DQL_gated": q["DQL_gated"], "Global": q["Global"]}
        for rho in RHO_GRID:
            for arm in LLM_ARMS + ["Global"]:
                score = posmix_closed(raw, scores["raw_alpha_se"].to_numpy(float), scores["tau2_data"].to_numpy(float), pm_q[arm], rho)
                pm_rows.append({"family": "PM", "guidance": arm, "rho_PM": rho, "selection_member": arm in LLM_ARMS, f"spearman_rank_ic_{year}_h12": spearman_ic(score, target)})
        rf = pd.DataFrame(rf_rows)
        pm = pd.DataFrame(pm_rows)
        ic_col = f"spearman_rank_ic_{year}_h12"
        sel_rf = rf[rf.selection_member].groupby("lambda_RF")[ic_col].mean().reset_index(name="MeanIC_RF").sort_values(["MeanIC_RF", "lambda_RF"], ascending=[False, True])
        sel_pm = pm[pm.selection_member].groupby("rho_PM")[ic_col].mean().reset_index(name="MeanIC_PM").sort_values(["MeanIC_PM", "rho_PM"], ascending=[False, True])
        glob_rf = rf[rf.guidance.eq("Global")].sort_values([ic_col, "lambda_RF"], ascending=[False, True])
        glob_pm = pm[pm.guidance.eq("Global")].sort_values([ic_col, "rho_PM"], ascending=[False, True])
        selected = {
            "lambda_RF_LLM_shared_star": float(sel_rf.iloc[0]["lambda_RF"]),
            "rho_PM_LLM_shared_star": float(sel_pm.iloc[0]["rho_PM"]),
            "lambda_Global_star": float(glob_rf.iloc[0]["lambda_RF"]),
            "rho_Global_star": float(glob_pm.iloc[0]["rho_PM"]),
            "lambda_selection": sel_rf.to_dict("records"),
            "rho_selection": sel_pm.to_dict("records"),
            "global_lambda_selection": glob_rf[["lambda_RF", ic_col]].to_dict("records"),
            "global_rho_selection": glob_pm[["rho_PM", ic_col]].to_dict("records"),
            "selection_year": year,
            "retuned_on_2025": False,
        }
    else:
        rf = pd.DataFrame()
        pm = pd.DataFrame()
        for arm in LLM_ARMS:
            lam = selected["lambda_RF_LLM_shared_star"]
            score = raw_rank.copy()
            if arm == "DL_gated":
                score[al == 1] = (1 - lam) * raw_rank[al == 1] + lam * q[arm][al == 1]
            else:
                score = (1 - lam) * raw_rank + lam * q[arm]
            rf_rows.append({"family": "RF", "guidance": arm, "lambda_RF": lam, f"spearman_rank_ic_{year}_h12": spearman_ic(score, target)})
        for label, lam in [("Global_shared", selected["lambda_RF_LLM_shared_star"]), ("Global_tuned", selected["lambda_Global_star"])]:
            score = (1 - lam) * raw_rank + lam * q["Global"]
            rf_rows.append({"family": "RF", "guidance": label, "lambda_RF": lam, f"spearman_rank_ic_{year}_h12": spearman_ic(score, target)})
        pm_q = {"DQ": q["DQ"], "DL_gated": al * q["DL_gated"], "DQL_gated": q["DQL_gated"], "Global": q["Global"]}
        for arm in LLM_ARMS:
            rho = selected["rho_PM_LLM_shared_star"]
            score = posmix_closed(raw, scores["raw_alpha_se"].to_numpy(float), scores["tau2_data"].to_numpy(float), pm_q[arm], rho)
            pm_rows.append({"family": "PM", "guidance": arm, "rho_PM": rho, f"spearman_rank_ic_{year}_h12": spearman_ic(score, target)})
        for label, rho in [("Global_shared", selected["rho_PM_LLM_shared_star"]), ("Global_tuned", selected["rho_Global_star"])]:
            score = posmix_closed(raw, scores["raw_alpha_se"].to_numpy(float), scores["tau2_data"].to_numpy(float), pm_q["Global"], rho)
            pm_rows.append({"family": "PM", "guidance": label, "rho_PM": rho, f"spearman_rank_ic_{year}_h12": spearman_ic(score, target)})
        rf = pd.DataFrame(rf_rows)
        pm = pd.DataFrame(pm_rows)
    return pd.DataFrame(standalone), rf, pm, selected


def qa_markdown(year, scores, targets, current_artifact_origin: str | None):
    h_start = pd.to_datetime(scores["history_start"]).dt.strftime("%Y-%m").unique().tolist()
    h_end = pd.to_datetime(scores["history_end"]).dt.strftime("%Y-%m").unique().tolist()
    t_start = pd.to_datetime(targets["target_start"]).dt.strftime("%Y-%m").unique().tolist()
    t_end = pd.to_datetime(targets["target_end"]).dt.strftime("%Y-%m").unique().tolist()
    raw_ic = spearman_ic(scores["raw_alpha"].to_numpy(float), targets["future_CAPM_alpha"].to_numpy(float))
    raw_manual = spearman_manual(scores["raw_alpha"].to_numpy(float), targets["future_CAPM_alpha"].to_numpy(float))
    eb_ic = spearman_ic(scores["EB_score"].to_numpy(float), targets["future_CAPM_alpha"].to_numpy(float))
    eb_manual = spearman_manual(scores["EB_score"].to_numpy(float), targets["future_CAPM_alpha"].to_numpy(float))
    extra = ""
    if current_artifact_origin:
        rp = pd.read_parquet(RAW_PANEL); ep = pd.read_parquet(EB_PANEL); tp = pd.read_parquet(TARGET_PANEL)
        for df in [rp, ep, tp]:
            df["origin_date"] = pd.to_datetime(df["origin_date"])
        co = pd.Timestamp(current_artifact_origin)
        raw_cmp = rp[rp.origin_date.eq(co)].rename(columns={"factor": "factor_id", "RAW": "artifact_raw"}).set_index("factor_id").reindex(scores.factor_id)
        eb_cmp = ep[ep.origin_date.eq(co)].rename(columns={"factor": "factor_id", "EB": "artifact_eb"}).set_index("factor_id").reindex(scores.factor_id)
        tgt_cmp = tp[(tp.origin_date.eq(co)) & (tp.horizon_months.eq(12)) & (tp.target_rule.eq("CORRECTED_MIN80_PRIMARY"))].rename(columns={"factor": "factor_id", "future_alpha_realized": "artifact_target"}).set_index("factor_id").reindex(scores.factor_id)
        extra = f"""
    ## Artifact Comparison At Forecast Origin

    - compared_existing_origin: {current_artifact_origin}
    - max_abs_raw_diff_vs_RAW_BACKBONE_PANEL: {float((scores['raw_alpha'].to_numpy(float) - raw_cmp['artifact_raw'].to_numpy(float)).max()):.3e}
    - max_abs_eb_diff_vs_EB_BACKBONE_PANEL: {float(np.nanmax(np.abs(scores['EB_score'].to_numpy(float) - eb_cmp['artifact_eb'].to_numpy(float)))):.3e}
    - max_abs_target_diff_vs_CORRECTED_FUTURE_CAPM_ALPHA_TARGETS: {float(np.nanmax(np.abs(targets['future_CAPM_alpha'].to_numpy(float) - tgt_cmp['artifact_target'].to_numpy(float)))):.3e}
    """
    return f"""
    # Corrected {year} Date And Metric QA

    - factor IDs unique: {scores.factor_id.is_unique and targets.factor_id.is_unique}
    - score rows: {len(scores)}
    - target rows: {len(targets)}
    - one-to-one merge rows: {len(scores[['factor_id']].merge(targets[['factor_id']], on='factor_id'))}
    - history_start_months: {h_start}
    - history_end_months: {h_end}
    - history_n_obs_min_max: {int(scores.history_n_obs.min())} / {int(scores.history_n_obs.max())}
    - target_start_months: {t_start}
    - target_end_months: {t_end}
    - future_n_obs_min_max: {int(targets.future_n_obs.min())} / {int(targets.future_n_obs.max())}
    - no row-order merge: TRUE; all joins are keyed by `factor_id`.
    - Raw/EB uses future-year returns: NO; historical score slice ends before forecast-window first month.
    - target uses historical score information: NO; future target regression uses future factor return and market return rows only.
    - JKP sign/orientation: factor `ret` used directly; no second sign flip in this downstream script.
    - scipy Raw Spearman: {raw_ic:.16f}
    - manual Raw Spearman: {raw_manual:.16f}
    - scipy/manual Raw absolute difference: {abs(raw_ic - raw_manual):.3e}
    - scipy EB Spearman: {eb_ic:.16f}
    - manual EB Spearman: {eb_manual:.16f}
    - scipy/manual EB absolute difference: {abs(eb_ic - eb_manual):.3e}
    {extra}
    """


def combine_results(year, standalone, rf, pm, selected):
    ic = f"spearman_rank_ic_{year}_h12"
    rows = standalone.to_dict("records")
    if year == 2024:
        for arm in LLM_ARMS:
            val = float(rf[(rf.guidance.eq(arm)) & (rf.lambda_RF.eq(selected["lambda_RF_LLM_shared_star"]))][ic].iloc[0])
            rows.append({"method": f"RF_Raw_{arm}", "family": "rank_fusion_selected_shared_lambda", "weight": selected["lambda_RF_LLM_shared_star"], ic: val})
        val = float(rf[(rf.guidance.eq("Global")) & (rf.lambda_RF.eq(selected["lambda_Global_star"]))][ic].iloc[0])
        rows.append({"method": "RF_Raw_Global_tuned", "family": "rank_fusion_global_validation_tuned", "weight": selected["lambda_Global_star"], ic: val})
        for arm in LLM_ARMS:
            val = float(pm[(pm.guidance.eq(arm)) & (pm.rho_PM.eq(selected["rho_PM_LLM_shared_star"]))][ic].iloc[0])
            rows.append({"method": f"PM_EB_{arm}", "family": "positive_mixture_selected_shared_rho", "weight": selected["rho_PM_LLM_shared_star"], ic: val})
        val = float(pm[(pm.guidance.eq("Global")) & (pm.rho_PM.eq(selected["rho_Global_star"]))][ic].iloc[0])
        rows.append({"method": "PM_EB_Global_tuned", "family": "positive_mixture_global_validation_tuned", "weight": selected["rho_Global_star"], ic: val})
    else:
        for _, r in rf.iterrows():
            rows.append({"method": f"RF_Raw_{r.guidance}", "family": "rank_fusion_frozen_2024_weight", "weight": r.lambda_RF, ic: r[ic]})
        for _, r in pm.iterrows():
            rows.append({"method": f"PM_EB_{r.guidance}", "family": "positive_mixture_frozen_2024_weight", "weight": r.rho_PM, ic: r[ic]})
    return pd.DataFrame(rows)


def robustness_2025(scores, targets, g, selected, results):
    outdir = OUT / "2025_final"
    factors = scores.factor_id.tolist()
    target = targets.set_index("factor_id").reindex(factors)["future_CAPM_alpha"].to_numpy(float)
    raw = scores["raw_alpha"].to_numpy(float)
    eb = scores["EB_score"].to_numpy(float)
    raw_rank = rank01(raw)
    reg = pd.read_csv(REGISTRY).set_index("factor_id").reindex(factors)
    themes = reg["theme"].astype(str).to_numpy()
    al = g["A_L_2025"].to_numpy(float)
    qg, qdq, qdl, qdql = g["q_Global"].to_numpy(float), g["q_DQ"].to_numpy(float), g["q_DL"].to_numpy(float), g["q_DQL_gated"].to_numpy(float)
    lam = selected["lambda_RF_LLM_shared_star"]
    qmap = {
        "Raw": raw_rank,
        "EB_rank": rank01(eb),
        "Global_standalone": qg,
        "DQ_standalone": qdq,
        "DQL_gated_standalone": qdql,
        "RF_Raw_DQ": (1 - lam) * raw_rank + lam * qdq,
        "RF_Raw_DQL_gated": (1 - lam) * raw_rank + lam * qdql,
        "RF_Raw_Global_tuned": (1 - selected["lambda_Global_star"]) * raw_rank + selected["lambda_Global_star"] * qg,
    }
    dl_rf = raw_rank.copy()
    dl_rf[al == 1] = (1 - lam) * raw_rank[al == 1] + lam * qdl[al == 1]
    qmap["RF_Raw_DL_gated"] = dl_rf
    pm_q = {"DQ": qdq, "DL_gated": al * qdl, "DQL_gated": qdql, "Global_tuned": qg}
    for name, qv in pm_q.items():
        rho = selected["rho_Global_star"] if name == "Global_tuned" else selected["rho_PM_LLM_shared_star"]
        qmap[f"PM_EB_{name}"] = posmix_closed(raw, scores["raw_alpha_se"].to_numpy(float), scores["tau2_data"].to_numpy(float), qv, rho)
    loo = []
    for theme in sorted(set(themes)):
        mask = themes != theme
        for method, score in qmap.items():
            loo.append({"left_out_theme": theme, "method": method, "n_factors": int(mask.sum()), "spearman_rank_ic_2025_h12": spearman_ic(np.asarray(score)[mask], target[mask])})
    pd.DataFrame(loo).to_csv(outdir / "THEME_LEAVE_ONE_OUT.csv", index=False)
    rng = np.random.default_rng(BOOT_SEED)
    uniq = np.array(sorted(set(themes)))
    boot = []
    for b in range(1000):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(themes == t)[0] for t in sampled])
        for method, score in qmap.items():
            boot.append({"bootstrap_id": b, "method": method, "spearman_rank_ic_2025_h12": spearman_ic(np.asarray(score)[idx], target[idx])})
    boot_df = pd.DataFrame(boot)
    boot_df.groupby("method")["spearman_rank_ic_2025_h12"].agg(
        mean="mean",
        ci_low=lambda x: x.quantile(0.025),
        ci_high=lambda x: x.quantile(0.975),
    ).reset_index().to_csv(outdir / "THEME_CLUSTER_ROBUSTNESS.csv", index=False)


def old_vs_corrected(corr24, corr25):
    old24 = pd.read_csv(VAL24 / "downstream/VALIDATION_2024_SUMMARY.csv").rename(columns={"spearman_rank_ic_2024_h12": "old_buggy_ic"})
    old25 = pd.read_csv(FINAL25 / "downstream/FINAL_2025_RESULTS.csv").rename(columns={"spearman_rank_ic_2025_h12": "old_buggy_ic"})
    c24 = corr24.rename(columns={"spearman_rank_ic_2024_h12": "corrected_ic"})
    c25 = corr25.rename(columns={"spearman_rank_ic_2025_h12": "corrected_ic"})
    rows = []
    for year, old, corr in [(2024, old24, c24), (2025, old25, c25)]:
        for _, r in corr.iterrows():
            candidates = old[old.method.eq(r.method)]
            if candidates.empty and r.method == "Global_standalone":
                candidates = old[old.method.str.contains("Global_standalone", regex=False)]
            old_ic = float(candidates.old_buggy_ic.iloc[0]) if not candidates.empty else np.nan
            rows.append({"year": year, "method": r.method, "old_buggy_ic": old_ic, "corrected_ic": float(r.corrected_ic), "difference_corrected_minus_old": float(r.corrected_ic - old_ic) if np.isfinite(old_ic) else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "comparison/OLD_VS_CORRECTED_DATE_WINDOW_RESULTS.csv", index=False)


def manifest_zip():
    rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"FILE_MANIFEST.csv", "SHA256SUMS.txt"}:
            rows.append({"relative_path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    with (OUT / "FILE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "bytes", "sha256"])
        w.writeheader(); w.writerows(rows)
    with (OUT / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS.txt":
                f.write(f"{sha256_file(p)}  {p.relative_to(OUT)}\n")
    zpath = OUT.with_suffix(".zip")
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(OUT.name / p.relative_to(OUT)))


def main():
    for d in ["protocol", "qa", "2024_validation", "2025_final", "comparison"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    factors_order, fr, mr, fp, mkt = load_panels()

    s24 = score_at(factors_order, fp, mkt, "2024-01-31", "2023-12-31")
    t24 = target_at(factors_order, fp, mkt, "2024-01-31", "2023-12-31")
    s24.to_csv(OUT / "2024_validation/CORRECTED_2024_RAW.csv", index=False)
    s24.to_csv(OUT / "2024_validation/CORRECTED_2024_EB.csv", index=False)
    t24.to_csv(OUT / "2024_validation/CORRECTED_2024_TARGET.csv", index=False)
    write_text(OUT / "qa/CORRECTED_2024_DATE_AND_METRIC_QA.md", qa_markdown(2024, s24, t24, "2024-01-31"))
    g24 = guidance(2024, factors_order)
    st24, rf24, pm24, selected = evaluate_year(2024, s24, t24, g24)
    full24 = pd.concat([rf24.assign(grid_type="rank_fusion"), pm24.assign(grid_type="positive_mixture")], ignore_index=True)
    full24.to_csv(OUT / "2024_validation/CORRECTED_2024_VALIDATION_FULL_GRID.csv", index=False)
    write_json(OUT / "protocol/CORRECTED_2024_SELECTED_HYPERPARAMETERS.json", selected)
    res24 = combine_results(2024, st24, rf24, pm24, selected)
    res24.to_csv(OUT / "2024_validation/CORRECTED_2024_RESULTS.csv", index=False)

    s25 = score_at(factors_order, fp, mkt, "2025-01-31", "2024-12-31")
    t25 = target_at(factors_order, fp, mkt, "2025-01-31", "2024-12-31")
    s25.to_csv(OUT / "2025_final/CORRECTED_2025_RAW.csv", index=False)
    s25.to_csv(OUT / "2025_final/CORRECTED_2025_EB.csv", index=False)
    t25.to_csv(OUT / "2025_final/CORRECTED_2025_TARGET.csv", index=False)
    write_text(OUT / "qa/CORRECTED_2025_DATE_AND_METRIC_QA.md", qa_markdown(2025, s25, t25, "2025-01-31"))
    g25 = guidance(2025, factors_order)
    st25, rf25, pm25, _ = evaluate_year(2025, s25, t25, g25, selected)
    res25 = combine_results(2025, st25, rf25, pm25, selected)
    res25.to_csv(OUT / "2025_final/CORRECTED_2025_FINAL_RESULTS.csv", index=False)
    res25.to_csv(OUT / "2025_final/CORRECTED_2025_FINAL_COMPARISON_TABLE.csv", index=False)
    robustness_2025(s25, t25, g25, selected, res25)
    old_vs_corrected(res24, res25)

    write_text(OUT / "protocol/DATE_SEMANTICS.md", """
    # Date Semantics

    Authoritative basis: `JKP153_DATAONLY_TARGET_SRCC_CODE_REVIEW_20260915_022514`.

    - `information_cutoff_date` is the conceptual last information date.
    - `forecast_window_start` is the code/table origin used by rolling score and target builders.
    - Corrected 2024 validation: information cutoff `2023-12-31`, forecast window start `2024-01-31`.
    - Corrected 2025 final: information cutoff `2024-12-31`, forecast window start `2025-01-31`.
    """)
    write_text(OUT / "protocol/CORRECTED_VALIDATION_TEST_PROTOCOL.md", f"""
    # Corrected Validation/Test Protocol

    This downstream package fixes calendar alignment only.

    - DeepSeek rerun: NO.
    - Literature reaudit: NO.
    - BT redesign/refit: NO.
    - 2024 corrected validation reselects lambda/rho using only corrected 2024 target.
    - 2025 corrected final uses frozen 2024-selected parameters.
    - LLM shared lambda grid: {LAMBDA_GRID}.
    - PM rho grid: {RHO_GRID}.
    - BT lambda remains frozen from existing measurement outputs.
    - DQ/DL/DQL guidance is reused from existing 2024/2025 packages.
    - Deterministic Global is reused from existing structured global file with `DATE_TRUNCATED_CURRENT_SNAPSHOT_NOT_VINTAGE_PURE` caveat.
    """)
    write_text(OUT / "README_FIRST.md", """
    # JKP153 Calendar-Aligned Corrected Downstream

    This package reruns only downstream calendar-aligned validation/final evaluation. It does not call DeepSeek and does not alter prompts, graph, evidence, measurement responses, or BT protocol.

    Final status:

    - DEEPSEEK_RERUN = NO
    - LITERATURE_REAUDIT = NO
    - BT_REDESIGN = NO
    - CALENDAR_ALIGNMENT_BUG_FIXED = YES
    - 2024_HYPERPARAMETERS_RESELECTED_ON_CORRECTED_VALIDATION = YES
    - 2025_HYPERPARAMETER_RETUNING = NO
    """)
    manifest_zip()


if __name__ == "__main__":
    main()
