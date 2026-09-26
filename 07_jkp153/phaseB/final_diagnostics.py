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
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
ROOT=Path(str(REPRO_ROOT))
P=ROOT/'finance_llm_prior/experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
COR=ROOT/'JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321'
META=pd.read_csv(ROOT/'JKP153_FORMAL_MEASUREMENT_DESIGN_FREEZE_V0_4_20260913_201725/templates/D0_FACTOR_METADATA_REGISTRY.csv')[['factor_id','factor_name','theme']]
B=json.loads((P/'results/2024/B_FREEZE_2024.json').read_text())
def r01(x):
    s=pd.Series(x); return ((s.rank(method='average')-1)/(len(s)-1)).to_numpy(float)
def sp(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    return float(spearmanr(x[ok],y[ok]).statistic)
def load_data(year):
    if year==2024:
        s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv'); t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
    else:
        s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv'); t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
    t=t.set_index('factor_id').reindex(s.factor_id)
    return s,t
def hvec(fids,e,arm,thr=.90):
    pos={f:i for i,f in enumerate(fids)}; h=np.zeros(len(fids))
    z=e[e.HD_bits>=thr]
    for x in z.to_dict('records'):
        if not bool(x[f'{arm}_valid']): continue
        p=float(x[f'{arm}_p_i_gt_j']); c=float(x[f'{arm}_certainty'])
        if not np.isfinite(p) or not np.isfinite(c) or p==.5: continue
        sign=1 if p>.5 else -1; w=float(x['HD_bits'])*c
        h[pos[x['factor_i']]] += sign*w/6
        h[pos[x['factor_j']]] -= sign*w/6
    return h
for year in [2024,2025]:
    s,t=load_data(year); fids=s.factor_id.astype(str).tolist(); raw=r01(s.raw_alpha); y=t.future_CAPM_alpha.to_numpy(float)
    e=pd.read_csv(P/f'results/{year}/EDGE_PROBABILITIES_{year}.csv')
    b=float(B['selective_H090'])
    out=s[['factor_id','raw_alpha']].copy().merge(META,on='factor_id',how='left')
    out['target_future_capm_alpha']=y; out['raw_rank01']=raw
    for arm in ['DQ','DQL']:
        h=hvec(fids,e,arm,.90); out[f'selective_{arm}_h']=h; out[f'selective_{arm}_score']=raw+(b/152.0)*h
        out[f'selective_{arm}_rank01']=r01(out[f'selective_{arm}_score'])
    out.to_csv(P/f'results/{year}/selective_PRIMARY_FACTOR_SCORES_{year}.csv',index=False)
    themes=sorted(out.theme.dropna().unique())
    rows=[]
    for drop in ['NONE']+themes:
        keep=np.ones(len(out),bool) if drop=='NONE' else out.theme.ne(drop).to_numpy()
        rr=sp(out.loc[keep,'raw_rank01'],out.loc[keep,'target_future_capm_alpha'])
        dq=sp(out.loc[keep,'selective_DQ_score'],out.loc[keep,'target_future_capm_alpha'])
        dql=sp(out.loc[keep,'selective_DQL_score'],out.loc[keep,'target_future_capm_alpha'])
        rows.append({'year':year,'dropped_theme':drop,'n':int(keep.sum()),'Raw_ic':rr,'selective_DQ_ic':dq,'selective_DQL_ic':dql,
                     'DQ_minus_Raw':dq-rr,'DQL_minus_Raw':dql-rr,'DQL_minus_DQ':dql-dq})
    pd.DataFrame(rows).to_csv(P/f'results/{year}/selective_PRIMARY_THEME_ROBUSTNESS_{year}.csv',index=False)

# Cross-year graph and measurement reproducibility, target-free
a=pd.read_csv(P/'results/2024/EDGE_PROBABILITIES_2024.csv')
b=pd.read_csv(P/'results/2025/EDGE_PROBABILITIES_2025.csv')
def kdf(df):
    x=df.copy(); x['pair_key']=x.apply(lambda r:'||'.join(sorted([str(r.factor_i),str(r.factor_j)])),axis=1); return x
a,b=kdf(a),kdf(b)
rows=[]
for graphcol,name in [('in_broad','BROAD'),('in_selective_union_H085','SELECTIVE_H085'),('selective_primary_H090','SELECTIVE_H090'),('selective_strict_H095','SELECTIVE_H095')]:
    sa=set(a.loc[a[graphcol].astype(bool),'pair_key']); sb=set(b.loc[b[graphcol].astype(bool),'pair_key']); inter=sa&sb; union=sa|sb
    rows.append({'graph':name,'n2024':len(sa),'n2025':len(sb),'overlap':len(inter),'jaccard':len(inter)/len(union)})
pd.DataFrame(rows).to_csv(P/'results/CROSS_YEAR_GRAPH_REPRODUCIBILITY.csv',index=False)
m=a.merge(b,on='pair_key',suffixes=('_2024','_2025'))
meas=[]
for graphcol,name in [('in_broad','BROAD'),('selective_primary_H090','SELECTIVE_H090')]:
    z=m[m[f'{graphcol}_2024'].astype(bool)&m[f'{graphcol}_2025'].astype(bool)]
    for arm in ['DQ','DQL']:
        ok=z[f'{arm}_valid_2024'].fillna(False).astype(bool)&z[f'{arm}_valid_2025'].fillna(False).astype(bool)
        q=z[ok]
        p24=q[f'{arm}_p_i_gt_j_2024'].to_numpy(float); p25=q[f'{arm}_p_i_gt_j_2025'].to_numpy(float)
        meas.append({'graph':name,'arm':arm,'common_valid_edges':len(q),
          'prob_spearman':float(spearmanr(p24,p25).statistic) if len(q)>2 else np.nan,
          'direction_agreement':float(((p24-.5)*(p25-.5)>0).mean()) if len(q) else np.nan,
          'mean_abs_prob_change':float(np.mean(np.abs(p24-p25))) if len(q) else np.nan})
pd.DataFrame(meas).to_csv(P/'results/CROSS_YEAR_MEASUREMENT_REPRODUCIBILITY.csv',index=False)

# Summary
r24=pd.read_csv(P/'results/2024/selective_PRIMARY_THEME_ROBUSTNESS_2024.csv')
r25=pd.read_csv(P/'results/2025/selective_PRIMARY_THEME_ROBUSTNESS_2025.csv')
summary={
 '2024_leave_one_theme_out':{
   'DQ_minus_Raw_positive':int((r24[r24.dropped_theme!='NONE'].DQ_minus_Raw>0).sum()),
   'DQL_minus_Raw_positive':int((r24[r24.dropped_theme!='NONE'].DQL_minus_Raw>0).sum()),
   'DQL_minus_DQ_positive':int((r24[r24.dropped_theme!='NONE'].DQL_minus_DQ>0).sum()),
   'themes':int((r24.dropped_theme!='NONE').sum())},
 '2025_leave_one_theme_out':{
   'DQ_minus_Raw_positive':int((r25[r25.dropped_theme!='NONE'].DQ_minus_Raw>0).sum()),
   'DQL_minus_Raw_positive':int((r25[r25.dropped_theme!='NONE'].DQL_minus_Raw>0).sum()),
   'DQL_minus_DQ_positive':int((r25[r25.dropped_theme!='NONE'].DQL_minus_DQ>0).sum()),
   'themes':int((r25.dropped_theme!='NONE').sum())}
}
(P/'results/FINAL_DIAGNOSTIC_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print('\nGraph reproducibility'); print(pd.read_csv(P/'results/CROSS_YEAR_GRAPH_REPRODUCIBILITY.csv').to_string(index=False))
print('\nMeasurement reproducibility'); print(pd.read_csv(P/'results/CROSS_YEAR_MEASUREMENT_REPRODUCIBILITY.csv').to_string(index=False))
