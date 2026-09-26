import sys,os,pathlib,importlib.util,numpy as np
base=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('m3base',base/'run_top30_top50_nested.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
orig30=m.MEAS30.copy(); orig50=m.MEAS50.copy()
def shuffled(orig,rep):
    rng=np.random.default_rng(2026091901+rep); z=orig.copy()
    canon=np.where(z.gene_i.astype(str)<z.gene_j.astype(str),z.hard_y_i_over_j.astype(float),1-z.hard_y_i_over_j.astype(float))
    z['_canon_y']=canon; orig2=orig.copy(); orig2['_canon_y']=canon
    bundle=['_canon_y','certainty_1_minus_H','either_U','both_U']
    for arm,idx0 in z.groupby('arm').groups.items():
        idx=np.asarray(list(idx0)); donor=rng.permutation(idx)
        for c in bundle: z.loc[idx,c]=orig2.loc[donor,c].to_numpy()
    z['hard_y_i_over_j']=np.where(z.gene_i.astype(str)<z.gene_j.astype(str),z._canon_y,1-z._canon_y)
    return z.drop(columns=['_canon_y'])
start,end=map(int,sys.argv[1:3])
for rep in range(start,end+1):
    out=base.parent/'09_REPRODUCED_SHUFFLE'/f'rep_{rep:03d}'
    done=out/'selective_NESTED_AGGREGATE.csv'
    if done.exists(): continue
    out.mkdir(parents=True,exist_ok=True); os.environ['selective_REPRO_OUT']=str(out)
    m.MEAS30=shuffled(orig30,rep); m.MEAS50=shuffled(orig50,rep)
    m.run_task('SEPSIS_GSE272769')
