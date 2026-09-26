

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
import sys,os,pathlib,importlib.util,numpy as np,pandas as pd
base=pathlib.Path(str(REPRO_ROOT / '02_gse272769_sepsis/gse272769/selective_shuffle'))
spec=importlib.util.spec_from_file_location('m3base',base/'run_nested272_shufflebase.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
orig=m.MEAS.copy()
def shuffled(rep):
    rng=np.random.default_rng(2026091901+rep); z=orig.copy()
    # canonicalize direction, then permute complete semantic/confidence bundle within arm
    canon=np.where(z.gene_i.astype(str)<z.gene_j.astype(str),z.hard_y_i_over_j.astype(float),1-z.hard_y_i_over_j.astype(float))
    z['_canon_y']=canon
    bundle=['_canon_y','certainty_1_minus_H','either_U','both_U']
    for arm,idx in z.groupby('arm').groups.items():
        idx=np.asarray(list(idx)); donor=rng.permutation(idx)
        for c in bundle:z.loc[idx,c]=orig.assign(_canon_y=canon).loc[donor,c].to_numpy()
    z['hard_y_i_over_j']=np.where(z.gene_i.astype(str)<z.gene_j.astype(str),z._canon_y,1-z._canon_y)
    return z.drop(columns=['_canon_y'])
start=int(sys.argv[1]);end=int(sys.argv[2])
for rep in range(start,end+1):
    out=base/f'rep_{rep:03d}'; done=out/'selective_NESTED_AGGREGATE.csv'
    if done.exists(): continue
    out.mkdir(parents=True,exist_ok=True); os.environ['selective_SHUFFLE_OUT']=str(out); m.MEAS=shuffled(rep); m.run_task('SEPSIS_GSE272769')
