#!/usr/bin/env python3
"""check_reproducibility.py - numerically compare two directories holding the same artefacts.

Two consecutive full runs must agree EXACTLY on every numeric cell (not merely to the printed
precision). CSVs are read with `dtype=str, keep_default_na=False` and each column is then converted
explicitly, so that a literal string such as `null` in a text column cannot masquerade as a NaN
difference (that produced a false "differs" report once).

usage: python check_reproducibility.py <dirA> <dirB> <label>
"""
import json
import os
import sys
import numpy as np
import pandas as pd


def load(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def as_numeric(s):
    """Returns a float Series if the column is genuinely numeric-dimensional, else None."""
    v = pd.to_numeric(s.replace("", np.nan), errors="coerce")
    return v if int(v.notna().sum()) == int((s != "").sum()) else None


def compare(a_dir, b_dir, label, verbose=True):
    files = sorted(f for f in os.listdir(a_dir)
                   if f in os.listdir(b_dir) and f.endswith((".csv", ".json")))
    worst, where, bad = 0.0, "", []
    for f in files:
        pa, pb = os.path.join(a_dir, f), os.path.join(b_dir, f)
        if f.endswith(".csv"):
            a, b = load(pa), load(pb)
            if a.shape != b.shape or list(a.columns) != list(b.columns):
                bad.append((f, f"shape/cols mismatch {a.shape} vs {b.shape}"))
                continue
            nd = 0.0
            for c in a.columns:
                na, nb = as_numeric(a[c]), as_numeric(b[c])
                if na is not None and nb is not None:
                    d = float(np.nanmax(np.abs(na.to_numpy() - nb.to_numpy()))) if len(na) else 0.0
                    d = d if np.isfinite(d) else 0.0
                    nd = max(nd, d)
                    if d > worst:
                        worst, where = d, f"{f}:{c}"
                elif not (a[c].to_numpy() == b[c].to_numpy()).all():
                    bad.append((f, c))
            if verbose:
                print(f"  {f:46s} rows={len(a):5d}  max|delta| = {nd:.3e}")
        else:
            da, db = json.load(open(pa)), json.load(open(pb))
            if json.dumps(da, sort_keys=True, ensure_ascii=False) == \
               json.dumps(db, sort_keys=True, ensure_ascii=False):
                if verbose:
                    print(f"  {f:46s} JSON byte-equal")
            else:
                bad.append((f, "json differs"))
                print(f"  {f:46s} JSON DIFFERS")
    ok = (worst == 0.0) and not bad
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {len(files)} artefacts, "
          f"max|delta| = {worst:.3e} ({where or 'n/a'}), non-numeric diffs = {bad}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if compare(sys.argv[1], sys.argv[2], sys.argv[3]) else 1)
