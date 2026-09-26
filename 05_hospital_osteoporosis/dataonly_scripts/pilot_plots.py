#!/usr/bin/env python3
"""Figure helpers for the data-only pilot.

Plots label features by their FROZEN canonical English name (read from
canonical/feature_identity_registry.csv) so that the figures are legible on any system without CJK
fonts; the CSV/markdown tables keep the original Chinese names.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CANON, D, K_GRID

_reg = pd.read_csv(f"{CANON}/canonical/feature_identity_registry.csv")
EN = dict(zip(_reg.original_chinese_name, _reg.canonical_english_name))
# The frozen registry keeps the hospital's own column names for these two (its "English name" is the
# same Chinese string); add ASCII display names so the figures render on any system.
EN.update({"DXA_性别": "Sex (DXA record)", "DXA_年龄": "Age (DXA record)",
           "身高_cm": "Height", "体重_kg": "Weight"})


def en(name, maxlen=30):
    s = EN.get(name, name)
    return s if len(s) <= maxlen else s[:maxlen - 1] + "…"


def fig_selection_probability(st, path):
    order = st.sort_values("median_rank").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    cols = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    y = np.arange(len(order))
    h = 0.20
    for i, (k, col) in enumerate(zip(K_GRID, cols)):
        ax.barh(y + (i - 1.5) * h, order[f"pi_top{k}"], height=h, label=f"k={k}", color=col, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels([en(f) for f in order.feature_name], fontsize=8)
    ax.axvline(0.5, ls=":", c="grey"); ax.axvline(0.9, ls="--", c="grey")
    ax.invert_yaxis()
    ax.set_xlabel("selection probability $\\pi_j(k)$ over B resamples")
    ax.set_title("Data-only selection probability (clinical features only)")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def fig_rank_variability(st, ranks, path):
    order = st.sort_values("median_rank").reset_index(drop=True)
    feats = list(order.feature_name)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.boxplot([ranks[:, i] for i in range(ranks.shape[1])],
               tick_labels=[en(f) for f in feats], vert=False, showfliers=False)
    for k in K_GRID:
        ax.axvline(k, ls=":", c="grey")
        ax.text(k, 0.3, f"k={k}", fontsize=7, color="grey")
    ax.invert_yaxis(); ax.set_xlabel("rank (1 = strongest |coefficient|)")
    ax.set_title("Data-only rank variability across resamples")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def fig_topk_jaccard(ranks, path):
    from collections import Counter
    B = ranks.shape[0]
    fig, ax = plt.subplots(1, len(K_GRID), figsize=(4 * len(K_GRID), 3.4), sharey=True)
    for i, k in enumerate(K_GRID):
        sets = [frozenset(np.where(ranks[b] <= k)[0]) for b in range(B)]
        js = [len(sets[a] & sets[c]) / max(len(sets[a] | sets[c]), 1)
              for a in range(B) for c in range(a + 1, B)]
        ax[i].hist(js, bins=30, color="#4C72B0")
        ax[i].set_title(f"k={k}\nmean J={np.mean(js):.3f}, distinct={len(Counter(sets))}")
        ax[i].set_xlabel("pairwise Jaccard")
    ax[0].set_ylabel("pairs")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
