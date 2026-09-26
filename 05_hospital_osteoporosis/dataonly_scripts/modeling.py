#!/usr/bin/env python3
"""Modelling helpers for the pilot.

Every function here respects the pilot's hard rules:
  * `site` enters as FORCED CONTEXT (scaled dummies, positions 0-1) and is never returned as a score;
  * the returned "scores" are the |coefficient| of the SELECTABLE clinical block only;
  * preprocessing is fitted on the training rows passed in, never on the whole cohort.
"""
import inspect
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from common import FORCED_SCALE, fit_scaler, apply_scaler, site_dummies

_LR_PARAMS = inspect.signature(LogisticRegression.__init__).parameters
_USE_L1_RATIO = "l1_ratio" in _LR_PARAMS


def _lr(C, penalty, max_iter=5000, tol=1e-5, random_state=0):
    """`random_state` MUST be set: the liblinear solver (used for L1) SHUFFLES the data using it,
    and with the default `None` every run draws a different seed. That made the L1-based results
    (the selector ranking and the permuted-label null) differ between otherwise identical runs.
    lbfgs (L2) is deterministic either way, but the seed is passed anyway for explicitness."""
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    if _USE_L1_RATIO:
        return LogisticRegression(l1_ratio=(1.0 if penalty == "l1" else 0.0), C=C,
                                  solver=solver, max_iter=max_iter, tol=tol,
                                  random_state=random_state)
    return LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, tol=tol,
                              random_state=random_state)


def fit_plain_l2(A, y, C=1.0):
    """Forced-context-only reference model (no selectable block)."""
    m = _lr(C, "l2", max_iter=1000)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(A, y)
    return m


def grouped_folds(y, groups, seed, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def _design(Xtr, str_tr, Xva=None, str_va=None, scale=FORCED_SCALE, scaler=None):
    """Builds EITHER the training block OR the validation block (never both, so that the training
    frame can be omitted safely when only scoring)."""
    if scaler is None:
        scaler = fit_scaler(Xtr)
    med, mu, sd = scaler
    if Xva is None:
        A = np.column_stack([site_dummies(str_tr, scale), apply_scaler(Xtr, med, mu, sd)])
        return A, scaler, None
    V = np.column_stack([site_dummies(str_va, scale), apply_scaler(Xva, med, mu, sd)])
    return None, scaler, V


def fit_lr(Xtr, str_tr, ytr, C, penalty="l1", scale=FORCED_SCALE, scaler=None):
    A, scaler, _ = _design(Xtr, str_tr, None, None, scale=scale, scaler=scaler)
    m = _lr(C, penalty)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(A, ytr)
    return m, scaler


def predict_lr(m, scaler, Xva, str_va, scale=FORCED_SCALE):
    _, _, V = _design(None, None, Xva, str_va, scale=scale, scaler=scaler)
    return m.predict_proba(V)[:, 1]


def clinical_scores(m, p):
    """|coef| of the selectable block (positions 2..2+p). Forced context is skipped by construction."""
    return np.abs(np.asarray(m.coef_).ravel()[2:2 + p])


def select_C(Xtr, str_tr, ytr, gtr, C_grid, penalty="l1", n_splits=4, seed=0):
    """Inner group-aware CV inside the TRAINING rows only -> picks C by mean AUROC."""
    best, best_C, detail = -np.inf, C_grid[0], []
    for C in C_grid:
        aucs = []
        for tri, vai in grouped_folds(ytr, gtr, seed + int(round(C * 1e4)), n_splits):
            if len(np.unique(ytr[tri])) < 2 or len(np.unique(ytr[vai])) < 2:
                continue
            m, sc = fit_lr(Xtr[tri], str_tr[tri], ytr[tri], C, penalty)
            pv = predict_lr(m, sc, Xtr[vai], str_va=str_tr[vai])
            aucs.append(roc_auc_score(ytr[vai], pv))
        mu = float(np.mean(aucs)) if aucs else -np.inf
        detail.append((C, mu))
        if mu > best:
            best, best_C = mu, C
    return best_C, float(best), detail


def metrics(y, p):
    return dict(auroc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, (p >= 0.5).astype(int))))


def nested_cv(X, str_, y, groups, C_grid, penalty="l1", n_splits=5, seeds=(11,), scale=FORCED_SCALE,
              inner_splits=4, return_oof=False):
    """Grouped nested CV. Returns per-(seed,fold) metric rows plus the chosen C per fold."""
    rows = []
    oof = {}
    for seed in seeds:
        for fi, (tri, vai) in enumerate(grouped_folds(y, groups, seed, n_splits)):
            C, inner_auc, _ = select_C(X[tri], str_[tri], y[tri], groups[tri], C_grid, penalty,
                                       inner_splits, seed + fi)
            m, sc = fit_lr(X[tri], str_[tri], y[tri], C, penalty, scale=scale)
            pv = predict_lr(m, sc, X[vai], str_[vai], scale=scale)
            pt = predict_lr(m, sc, X[tri], str_[tri], scale=scale)
            rows.append(dict(seed=seed, fold=fi, C=C, inner_auc=inner_auc,
                             n_train=len(tri), n_valid=len(vai),
                             n_patients_valid=int(len(np.unique(groups[vai]))),
                             **metrics(y[vai], pv),
                             auroc_train=float(roc_auc_score(y[tri], pt))))
            oof[(seed, fi)] = (vai, pv)
    if return_oof:
        return rows, oof
    return rows


def site_only_cv(str_, y, groups, n_splits=5, seeds=(11,)):
    """Reference: forced context only, no clinical features."""
    rows = []
    for seed in seeds:
        for fi, (tri, vai) in enumerate(grouped_folds(y, groups, seed, n_splits)):
            m = fit_plain_l2(site_dummies(str_[tri]), y[tri], C=1.0)
            pv = m.predict_proba(site_dummies(str_[vai]))[:, 1]
            rows.append(dict(seed=seed, fold=fi, **metrics(y[vai], pv)))
    return rows


def permuted_label_cv(X, str_, y, groups, C_grid, penalty="l1", n_splits=5, seeds=(11,),
                      n_repeats=3, rng_seed=7):
    """Null calibration.

    The labels are permuted **within each site stratum**, so the marginal prevalence and the
    site-conditional prevalence are both preserved exactly while any feature->label association is
    destroyed. A correctly working pipeline must return AUROC ~ 0.5 on this null.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for rep in range(n_repeats):
        yp = y.copy()
        for s in np.unique(str_):
            idx = np.where(str_ == s)[0]
            yp[idx] = rng.permutation(yp[idx])
        for seed in seeds:
            for fi, (tri, vai) in enumerate(grouped_folds(yp, groups, seed, n_splits)):
                if len(np.unique(yp[tri])) < 2 or len(np.unique(yp[vai])) < 2:
                    continue
                C, _, _ = select_C(X[tri], str_[tri], yp[tri], groups[tri], C_grid, penalty, 3, seed)
                m, sc = fit_lr(X[tri], str_[tri], yp[tri], C, penalty)
                pv = predict_lr(m, sc, X[vai], str_[vai])
                rows.append(dict(repeat=rep, seed=seed, fold=fi, C=C, **metrics(yp[vai], pv)))
    return rows
