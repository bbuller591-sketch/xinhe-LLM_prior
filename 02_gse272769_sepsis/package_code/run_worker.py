

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
import sys,os,pathlib,importlib.util,numpy as np
base=pathlib.Path(str(REPRO_ROOT / '02_gse272769_sepsis/gse272769/tmp_k40_90_shuffle'))
spec=importlib.util.spec_from_file_location('m3base',base/'run_nested_shufflebase.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
orig=m.MEAS.copy()
canon0=np.where(orig.gene_i.astype(str)<orig.gene_j.astype(str),orig.hard_y_i_over_j.astype(float),1-orig.hard_y_i_over_j.astype(float))
orig2=orig.copy(); orig2['_canon_y']=canon0
def shuffled(rep):
    rng=np.random.default_rng(2026091901+rep); z=orig2.copy()
    bundle=['_canon_y','certainty_1_minus_H','either_U','both_U']
    for arm,idx0 in z.groupby('arm').groups.items():
        idx=np.asarray(list(idx0)); donor=rng.permutation(idx)
        for c in bundle: z.loc[idx,c]=orig2.loc[donor,c].to_numpy()
    z['hard_y_i_over_j']=np.where(z.gene_i.astype(str)<z.gene_j.astype(str),z._canon_y,1-z._canon_y)
    return z.drop(columns=['_canon_y'])
start,end=map(int,sys.argv[1:3])
for rep in range(start,end+1):
    out=base/f'rep_{rep:03d}'; done=out/'selective_NESTED_AGGREGATE.csv'
    if done.exists(): continue
    out.mkdir(parents=True,exist_ok=True); os.environ['selective_SHUFFLE_OUT']=str(out)
    m.MEAS=shuffled(rep); m.run_task('SEPSIS_GSE272769')
