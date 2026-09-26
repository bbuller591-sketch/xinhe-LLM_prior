#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def q(s,p): return float(s.dropna().quantile(p)) if s.notna().any() else float('nan')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--year',type=int,required=True);ap.add_argument('--edge-probs',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
    e=pd.read_csv(a.edge_probs);rows=[]
    subsets={
      'BROAD':e.in_broad.astype(bool),
      'SELECTIVE_H085':e.in_selective_union_H085.astype(bool),
      'SELECTIVE_H090':e.selective_primary_H090.astype(bool),
      'SELECTIVE_H095':e.selective_strict_H095.astype(bool)
    }
    for name,m in subsets.items():
      z=e[m]
      for arm in ['DQ','DQL']:
        valid=z[f'{arm}_valid'].fillna(False).astype(bool)
        zz=z[valid]
        rows.append({
          'year':a.year,'graph':name,'arm':arm,'edges_total':len(z),'edges_valid':int(valid.sum()),'valid_rate':float(valid.mean()) if len(z) else np.nan,
          'p_q05':q(zz[f'{arm}_p_i_gt_j'],.05),'p_q25':q(zz[f'{arm}_p_i_gt_j'],.25),'p_median':q(zz[f'{arm}_p_i_gt_j'],.5),'p_q75':q(zz[f'{arm}_p_i_gt_j'],.75),'p_q95':q(zz[f'{arm}_p_i_gt_j'],.95),
          'HL_q25':q(zz[f'{arm}_HL_bits'],.25),'HL_median':q(zz[f'{arm}_HL_bits'],.5),'HL_q75':q(zz[f'{arm}_HL_bits'],.75),
          'certainty_mean':float(zz[f'{arm}_certainty'].mean()) if len(zz) else np.nan,
          'order_gap_median':q(zz[f'{arm}_order_gap_mean'],.5),'repeat_logit_sd_median':q(zz[f'{arm}_repeat_logit_sd'],.5),
          'hard_T_rate_mean':float(z[f'{arm}_hard_T_rate'].mean()),'hard_U_rate_mean':float(z[f'{arm}_hard_U_rate'].mean()),
          'mean_valid_repeat_pairs':float(z[f'{arm}_valid_repeat_pairs'].mean())
        })
    out=pd.DataFrame(rows);a.out_dir.mkdir(parents=True,exist_ok=True);out.to_csv(a.out_dir/f'MEASUREMENT_DIAGNOSTICS_{a.year}.csv',index=False)
    # source contrast on common-valid edges
    cv=e[e.DQ_valid.fillna(False).astype(bool)&e.DQL_valid.fillna(False).astype(bool)].copy()
    cv['delta_logit_DQL_minus_DQ']=np.log(cv.DQL_p_i_gt_j.clip(1e-6,1-1e-6)/(1-cv.DQL_p_i_gt_j.clip(1e-6,1-1e-6)))-np.log(cv.DQ_p_i_gt_j.clip(1e-6,1-1e-6)/(1-cv.DQ_p_i_gt_j.clip(1e-6,1-1e-6)))
    sc={'year':a.year,'common_valid_edges':len(cv),'mean_delta_logit':float(cv.delta_logit_DQL_minus_DQ.mean()),'median_abs_delta_logit':float(cv.delta_logit_DQL_minus_DQ.abs().median()),
        'fraction_sign_flip':float(((cv.DQ_p_i_gt_j-.5)*(cv.DQL_p_i_gt_j-.5)<0).mean())}
    (a.out_dir/f'SOURCE_CONTRAST_DIAGNOSTIC_{a.year}.json').write_text(json.dumps(sc,indent=2)+'\n')
    print(out.to_string(index=False));print(json.dumps(sc,indent=2))
if __name__=='__main__':main()
