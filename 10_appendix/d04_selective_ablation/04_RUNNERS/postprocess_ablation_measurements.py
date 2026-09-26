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
from pathlib import Path
import json, math, hashlib
import pandas as pd
import numpy as np
from scipy.special import expit

ROOT=Path(str(REPRO_ROOT / '10_appendix/d04_selective_ablation'))
PKG=Path(str(REPRO_ROOT / 'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919'))
TASK='BREAST_GSE25055_GSE25065'
OUT=ROOT/'03_MEASUREMENTS'
old_meas=pd.read_csv(PKG/'MEASUREMENTS/selective/selective_PAIR_SOURCE_MEASUREMENTS.csv')
q=pd.read_csv(ROOT/'02_QUERY_AUDIT/MISSING_selective_ABLATION_QUERIES_PREAUTH.csv',dtype=str,keep_default_na=False)
c=pd.read_parquet(OUT/'MISSING_selective_CALLS_COMPLETE.parquet')
assert len(q)==2028 and len(c)==2028
assert (c.parse_status=='PASS').all()
z=q[['query_id','unordered_pair_id','arm','order','gene_A','gene_B','prompt_sha256','evidence_packet_sha256']].merge(
    c,on=['query_id','unordered_pair_id','arm','order','gene_A','gene_B','prompt_sha256','evidence_packet_sha256'],validate='one_to_one')
rows=[]
for (arm,pair),g in z.groupby(['arm','unordered_pair_id']):
    assert len(g)==2 and set(g.order)=={'AB','BA'}
    ab=g[g.order=='AB'].iloc[0]
    ba=g[g.order=='BA'].iloc[0]
    assert ab.gene_A==ba.gene_B and ab.gene_B==ba.gene_A
    lab=float(ab.logp_A)-float(ab.logp_B)
    lba=float(ba.logp_A)-float(ba.logp_B)
    ell=(lab-lba)/2.0
    p=float(expit(ell))
    H=-(p*math.log(p,2)+(1-p)*math.log(1-p,2)) if 0<p<1 else 0.0
    rows.append({'task':TASK,'arm':arm,'unordered_pair_id':pair,'gene_i':ab.gene_A,'gene_j':ab.gene_B,
                 'symmetrized_logit_i_vs_j':ell,'p_i_over_j':p,
                 'hard_y_i_over_j':1.0 if p>0.5 else (0.0 if p<0.5 else 0.5),
                 'H_AB_bits':H,'certainty_1_minus_H':1-H,'either_U':bool((g.response_token=='U').any()),
                 'measurement_source':'new_ablation_completion'})
new_meas=pd.DataFrame(rows)
old=old_meas.copy()
old['measurement_source']='old_frozen_selective'
combined=pd.concat([old,new_meas],ignore_index=True,sort=False)
combined=combined.sort_values(['unordered_pair_id','arm']).reset_index(drop=True)
# audit full candidate universe coverage
union=pd.read_csv(ROOT/'01_MANIFEST/BREAST_CANDIDATE_PAIR_UNION_1834.csv')
expected_pairs=set(union.unordered_pair_id.astype(str))
got_pairs=set(combined.unordered_pair_id.astype(str).unique())
status={
 'status':'PASS' if expected_pairs==got_pairs and len(combined)==1834*2 else 'FAIL',
 'old_pair_source_rows':int(len(old)),'new_pair_source_rows':int(len(new_meas)),'combined_pair_source_rows':int(len(combined)),
 'combined_unique_pairs':int(combined.unordered_pair_id.nunique()),
 'missing_pairs_after_combine':sorted(expected_pairs-got_pairs)[:20],
 'extra_pairs_after_combine':sorted(got_pairs-expected_pairs)[:20],
 'provider_status':json.loads((OUT/'MISSING_selective_CALL_STATUS.json').read_text()),
}
combined.to_csv(OUT/'selective_ABLATION_PAIR_SOURCE_MEASUREMENTS_1834.csv',index=False)
status['combined_sha256']=hashlib.sha256((OUT/'selective_ABLATION_PAIR_SOURCE_MEASUREMENTS_1834.csv').read_bytes()).hexdigest()
(OUT/'MEASUREMENT_COMBINE_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
if status['status']!='PASS': raise SystemExit(2)
