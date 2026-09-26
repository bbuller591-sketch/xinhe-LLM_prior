#!/usr/bin/env python3
"""v2_core - exact masked-penalty logistic regression, plus the v2 data plumbing.

Why this module exists
----------------------
The v1 pilot kept the site block out of the L1 penalty by scaling its dummies by 1000, which only
makes the penalty *small*. That is an exploratory hack, not an unpenalized block. Here the objective
is optimised directly:

    F(w) = sum_i logloss_i(w)  +  lam * sum_j mask_j * |w_j|        (kind="l1")
    F(w) = sum_i logloss_i(w)  +  lam * sum_j mask_j *  w_j^2       (kind="l2")

with a per-coordinate penalty mask:
    mask = 0  ->  intercept and the `site` dummies  (NEVER penalized)
    mask = 1  ->  the selectable clinical features  (penalized)

Solved by FISTA (proximal gradient) on the smooth log-loss with a data-derived step size
(1/L, L = 0.25 * lambda_max(X'X) via a seeded power iteration), so the run is bit-reproducible.
The solution is verified against the KKT conditions of the *actual* objective, and the solver is
cross-validated against sklearn's liblinear on a problem where the two objectives coincide.

The clinical block is the only block that is ever ranked: `clinical_scores()` slices exactly the
coordinates after the forced context, so a site coefficient can never enter a ranking.
"""
import numpy as np

from common import D, C_GRID, SEEDS, N_FOLDS  # re-used unchanged (v1 artefacts keep their exact code)

# Pre-registered for the v2 stage. `lam = 1 / C` keeps the SAME regularisation strength as the v1
# liblinear convention (liblinear minimizes ||w||_1 + C * sum logloss), so v1 and v2 are comparable.
LAM_GRID = [1.0 / c for c in C_GRID]                      # [333.3, 100, 33.3, 10, 3.33]
LAM_GRID_EXTENDED = LAM_GRID + [1.0, 1.0 / 3.0]
SITE_PRIMARY = ["腰椎骨", "髋关节"]
SITE_SENSITIVITY_ONLY = ["前臂"]
FORCED_CONTEXT_NAMES = ["intercept", "ctx_site_髋关节"]     # primary scope has 2 levels


# ------------------------------------------------------------------ solver
def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
                    np.exp(np.clip(z, -500, 500)) / (1.0 + np.exp(np.clip(z, -500, 500))))


def logloss_sum(Xd, y, w):
    z = np.clip(Xd @ w, -500, 500)
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


def grad_logloss(Xd, y, w):
    return Xd.T @ (sigmoid(Xd @ w) - y)


def lambda_max_power(A, n_iter=300, seed=20260918):
    """Deterministic largest eigenvalue of A'A (seeded power iteration)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(n_iter):
        u = A.T @ (A @ v)
        nrm = np.linalg.norm(u)
        if nrm == 0:
            return 0.0
        v = u / nrm
        lam = nrm
    return float(lam)


def prox_apply(v, step, pen, kind):
    if kind == "l1":
        return np.sign(v) * np.maximum(np.abs(v) - step * pen, 0.0)
    return v / (1.0 + 2.0 * step * pen)


def objective(Xd, y, w, mask, lam, kind):
    pen = np.abs(w) if kind == "l1" else w ** 2
    return logloss_sum(Xd, y, w) + float(lam) * float(np.sum(np.asarray(mask) * pen))


def _polish(Xd, y, mask, lam, kind, w0, rounds=200, tol=1e-12):
    """Active-set damped Newton on the FREE coordinates (unpenalized + non-zero penalized).

    FISTA alone stops at a KKT residual of ~1e-6 for a sensible iteration budget; this takes the
    *same* optimum to machine precision in a few 39x39 solves, which is what makes "the site penalty
    is exactly zero" a measurable statement rather than an approximate one.

    Two details matter for convergence:
      * the step is capped ONLY where the cap ratio is genuinely positive (a coordinate that would
        cross zero). Capping otherwise would replace full Newton steps with 99%-damped ones and stall
        the residual at the FISTA level;
      * a coordinate driven to ~0 by a capped step is SNAPPED to exactly 0 and dropped from the free
        set, otherwise the next round recomputes the same cap and the polish spins in place.
    """
    w = w0.copy()
    d = Xd.shape[1]
    for _ in range(rounds):
        free = ((mask == 0) | (np.abs(w) > 1e-14)) if kind == "l1" else np.ones(d, bool)
        if not free.any():
            break
        fi = np.flatnonzero(free)
        pen_pos = np.flatnonzero(mask[fi] > 0)
        sign = np.sign(w[fi][pen_pos])
        sign[sign == 0] = 1.0
        for _inner in range(200):
            Xf, wf = Xd[:, fi], w[fi]
            p = sigmoid(Xf @ wf)
            g = Xf.T @ (p - y)
            H = (Xf * (p * (1.0 - p))[:, None]).T @ Xf
            g = g.copy()
            H = H.copy()
            if kind == "l1":
                g[pen_pos] += lam * sign
            else:
                g[pen_pos] += 2.0 * lam * wf[pen_pos]
                H[pen_pos, pen_pos] += 2.0 * lam
            if float(np.max(np.abs(g))) < tol:
                break
            H = H + 1e-12 * np.eye(len(fi))
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break
            alpha = 1.0
            if kind == "l1" and len(pen_pos):
                num, den = wf[pen_pos], -step[pen_pos]
                ok = (num * den) > 0                      # only these can cross zero
                ratio = np.where(ok, num / np.where(ok, den, 1.0), np.inf)
                cap = float(np.min(ratio))
                if cap < 1.0:
                    alpha = 0.99 * cap
            f_old = objective(Xd, y, w, mask, lam, kind)
            for _ls in range(60):                          # backtracking safeguard
                cand = w.copy()
                cand[fi] = wf + alpha * step
                if objective(Xd, y, cand, mask, lam, kind) <= f_old + 1e-15:
                    break
                alpha *= 0.5
            w = cand
            if kind == "l1":
                snap = np.flatnonzero(np.abs(w) < 1e-13)
                if len(snap):
                    w[snap] = 0.0
            if alpha < 1.0:
                break                                      # active set changed -> rebuild
            if float(np.max(np.abs(step))) < tol:
                break
    return w


def fit_masked(Xd, y, mask, lam, kind="l1", max_iter=20000, tol=1e-9, polish=True):
    """Exact minimiser of the masked-penalty logistic objective. Deterministic.

    The input is forced into a canonical C-contiguous float64 layout first: with different memory
    layouts BLAS takes different summation paths and the result can move by ~1e-15, i.e. the solver
    would not be a function of its *values* alone.
    """
    Xd = np.ascontiguousarray(Xd, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    n, d = Xd.shape
    L = 0.25 * lambda_max_power(Xd)
    step = 1.0 / L if L > 0 else 1.0
    mask = np.ascontiguousarray(mask, dtype=float)
    pen = float(lam) * mask
    w = np.zeros(d)
    z = w.copy()
    t = 1.0
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        g = grad_logloss(Xd, y, z)
        w_new = prox_apply(z - step * g, step, pen, kind)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        delta = float(np.max(np.abs(w_new - w)))
        w, t = w_new, t_new
        if delta < tol:
            break
    if polish:
        w_p = _polish(Xd, y, mask, lam, kind, w)
        if objective(Xd, y, w_p, mask, lam, kind) <= objective(Xd, y, w, mask, lam, kind) + 1e-12:
            w = w_p
    g = grad_logloss(Xd, y, w)
    pen_terms = np.abs(w) if kind == "l1" else w ** 2
    obj = logloss_sum(Xd, y, w) + float(lam) * float(np.sum(mask * pen_terms))
    info = {"kind": kind, "lam": float(lam), "objective": obj, "n_iter": n_iter,
            "n_nonzero_clinical": int((np.abs(w[2:]) > 1e-12).sum()),
            "n_nonzero": int((np.abs(w) > 1e-12).sum()),
            "step": step, "lipschitz": L, "polished": bool(polish),
            "grad_scale": float(np.max(np.abs(g)))}
    if kind == "l1":
        unpen = mask == 0
        info["kkt_unpenalized_max_grad"] = float(np.max(np.abs(g[unpen]))) if unpen.any() else 0.0
        nz = (np.abs(w) > 1e-12) & (mask > 0)
        zz = (np.abs(w) <= 1e-12) & (mask > 0)
        info["kkt_l1_nonzero_max_dev"] = float(np.max(np.abs(np.abs(g[nz]) - lam))) if nz.any() else 0.0
        info["kkt_l1_zero_max_excess"] = float(np.max(np.abs(g[zz]) - lam)) if zz.any() else 0.0
        info["penalty_at_solution"] = float(lam * np.sum(mask * np.abs(w)))
    else:
        unpen = mask == 0
        info["kkt_unpenalized_max_grad"] = float(np.max(np.abs(g[unpen]))) if unpen.any() else 0.0
        info["kkt_l2_max_dev"] = float(np.max(np.abs(g[mask > 0] + 2.0 * lam * w[mask > 0]))) \
            if (mask > 0).any() else 0.0
        info["penalty_at_solution"] = float(lam * np.sum(mask * w ** 2))
    return w, info


# ------------------------------------------------------------------ design / plumbing
def design(Xsel, site, include_site=True):
    """[intercept | site dummies (unscaled) | standardized clinical block]. Mask is returned too."""
    n = len(Xsel)
    parts = [np.ones((n, 1))]
    mask = [0.0]
    if include_site:
        hip = (np.asarray(site) == "髋关节").astype(float).reshape(-1, 1)
        parts.append(hip)
        mask.append(0.0)
    p = Xsel.shape[1]
    parts.append(np.asarray(Xsel, dtype=float))
    mask += [1.0] * p
    return np.column_stack(parts), np.asarray(mask)


def forced_context_cols(site, include_site=True):
    return 2 if include_site else 1


def clinical_scores(w, n_ctx):
    """|coefficient| of the SELECTABLE clinical block only (forced context is sliced away)."""
    return np.abs(np.asarray(w).ravel()[n_ctx:])


def fit_scaler_v2(Xtr):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Z = np.where(np.isnan(Xtr), med, Xtr)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return med, mu, sd


def apply_scaler_v2(X, sc):
    med, mu, sd = sc
    return (np.where(np.isnan(X), med, X) - mu) / sd


def fit_full(Xraw, site, y, lam, kind="l1", include_site=True, mask_kind="clinical"):
    """Standardize on the rows given (training rows only!), then solve the masked problem."""
    sc = fit_scaler_v2(Xraw)
    Xs = apply_scaler_v2(Xraw, sc)
    Xd, mask = design(Xs, site, include_site=include_site)
    if mask_kind == "all":            # only for the solver cross-validation against liblinear
        mask = np.ones_like(mask)
    w, info = fit_masked(Xd, y, mask, lam, kind=kind)
    n_ctx = forced_context_cols(site, include_site)
    return w, info, sc, n_ctx


def predict_full(w, sc, Xraw, site, include_site=True):
    Xs = apply_scaler_v2(Xraw, sc)
    Xd, _ = design(Xs, site, include_site=include_site)
    return sigmoid(Xd @ w)
