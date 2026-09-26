#!/usr/bin/env python3
"""Regenerate the top-k stability figures from the saved artefacts (no recomputation)."""
import numpy as np
import pandas as pd
from common import D
from pilot_plots import fig_selection_probability, fig_rank_variability, fig_topk_jaccard

z = np.load(f"{D['05_TOPK_STABILITY']}/resample_scores.npz", allow_pickle=True)
st = pd.read_csv(f"{D['05_TOPK_STABILITY']}/FEATURE_STABILITY.csv")
fig_selection_probability(st, f"{D['plots']}/05_selection_probability.png")
fig_rank_variability(st, z["ranks"], f"{D['plots']}/05_rank_variability.png")
fig_topk_jaccard(z["ranks"], f"{D['plots']}/05_topk_jaccard.png")
print("figures regenerated from saved artefacts")
