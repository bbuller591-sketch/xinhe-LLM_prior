#!/usr/bin/env python3
from __future__ import annotations

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
import json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(str(REPRO_ROOT))
PROJECT=ROOT/'finance_llm_prior'
OUT=PROJECT/'experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
FREEZE=ROOT/'JKP153_FORMAL_MEASUREMENT_DESIGN_FREEZE_V0_4_20260913_201725'
PHASEA=PROJECT/'experiments/jkp153_data_confusion_phaseA_20260919_025045'
FACTOR_CSV=PROJECT/'data/raw/jkp/monthly_vw_cap/[usa]_[all_factors]_[monthly]_[vw_cap].csv'
MARKET_ZIP=PROJECT/'experiments/jkp153_capm_alpha_dataonly_integration_v01/20260911_160413/market_raw/[usa]_[mkt]_[monthly]_[vw_cap].zip'
SEED_BASE=20260919
B=1000
BLOCK=12
RANK_GAP_MAX=3
H_UNION=0.85
H_PRIMARY=0.90
H_STRICT=0.95
MAX_EDGES=459
R=3

def month_end(s):
    return pd.to_datetime(s)+pd.offsets.MonthEnd(0)

def read_market():
    with zipfile.ZipFile(MARKET_ZIP) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(names)!=1: raise RuntimeError(names)
        with z.open(names[0]) as f:
            return pd.read_csv(f,na_values=['na','NA',''])

def load_panels():
    fids=pd.read_csv(FREEZE/'templates/D0_FACTOR_METADATA_REGISTRY.csv')['factor_id'].astype(str).tolist()
    fr=pd.read_csv(FACTOR_CSV,na_values=['na','NA',''])
    mr=read_market()
    fr['date']=month_end(fr['date']); mr['date']=month_end(mr['date'])
    fr['ret']=pd.to_numeric(fr.ret,errors='coerce'); mr['ret']=pd.to_numeric(mr.ret,errors='coerce')
    fp=fr[fr.name.isin(fids)].pivot(index='date',columns='name',values='ret').sort_index().reindex(columns=fids)
    mkt=mr[mr.name.eq('mkt')].drop_duplicates('date').set_index('date').ret.sort_index()
    return fids,fp,mkt

def raw_alpha(Y,x):
    xb=float(x.mean()); yb=Y.mean(0); xc=x-xb; yc=Y-yb
    den=float(xc@xc)
    beta=(xc[:,None]*yc).sum(0)/den
    return yb-beta*xb

def rankhi(x):
    return rankdata(-np.asarray(x,float),method='average')

def mbi(n,L,rng):
    starts=rng.integers(0,n-L+1,size=math.ceil(n/L))
    return np.concatenate([np.arange(s,s+L) for s in starts])[:n]

def entropy_bits(p):
    q=min(max(float(p),1e-12),1-1e-12)
    return -(q*math.log2(q)+(1-q)*math.log2(1-q))

def packet_eligibility(year):
    path=FREEZE/f'inputs/FACTOR_EVIDENCE_PACKETS_{year}.jsonl'
    out={}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        o=json.loads(line)
        fid=o['factor']['factor_id']
        out[fid]=o.get('llm_measurement_status')!='ABSTAIN_NO_APPROVED_ELIGIBLE_EVIDENCE'
    return out

def build_year(year,fids,fp,mkt):
    if year==2024:
        start,end='2014-01-31','2023-12-31'
    else:
        start,end='2015-01-31','2024-12-31'
    idx=pd.date_range(start,end,freq='ME')
    Y=fp.reindex(idx).to_numpy(float); x=mkt.reindex(idx).to_numpy(float)
    if Y.shape!=(120,len(fids)) or not np.isfinite(Y).all() or not np.isfinite(x).all():
        raise RuntimeError(f'{year}: incomplete historical panel {Y.shape}')
    r0=raw_alpha(Y,x); ranks=rankhi(r0)
    seed=SEED_BASE+(year-2024)*100+BLOCK
    rng=np.random.default_rng(seed)
    boot=np.empty((B,len(fids)),float)
    for b in range(B):
        ix=mbi(len(x),BLOCK,rng)
        boot[b]=raw_alpha(Y[ix],x[ix])
    elig=packet_eligibility(year)
    rows=[]
    for a in range(len(fids)):
        for b in range(a+1,len(fids)):
            gap=abs(float(ranks[a]-ranks[b]))
            if gap>RANK_GAP_MAX: continue
            # canonical endpoint labels are lexical, not data-rank ordered
            xfid,yfid=sorted((fids[a],fids[b]))
            xi=fids.index(xfid); yi=fids.index(yfid)
            gt=int((boot[:,xi]>boot[:,yi]).sum()); eq=int((boot[:,xi]==boot[:,yi]).sum())
            p=(gt+0.5*eq)/B; H=entropy_bits(p)
            if H < H_UNION: continue
            rows.append({
                'factor_i':xfid,'factor_j':yfid,
                'raw_reference_rank_i':float(ranks[xi]),'raw_reference_rank_j':float(ranks[yi]),
                'raw_reference_rank_gap':gap,
                'pD_i_gt_j':p,'HD_bits':H,
                'primary_H_ge_0p90':H>=H_PRIMARY,
                'strict_sensitivity_H_ge_0p95':H>=H_STRICT,
                'literature_eligible_i':bool(elig[xfid]),
                'literature_eligible_j':bool(elig[yfid]),
                'DQL_call_required':bool(elig[xfid] or elig[yfid]),
            })
    df=pd.DataFrame(rows)
    df=df.sort_values(['HD_bits','raw_reference_rank_gap','factor_i','factor_j'],
                      ascending=[False,True,True,True]).reset_index(drop=True)
    if len(df)>MAX_EDGES:
        df=df.head(MAX_EDGES).copy()
        cap=True
    else: cap=False
    df.insert(0,'edge_id',[f'SC{year}_{i:04d}' for i in range(1,len(df)+1)])
    df['origin_year']=year
    df['graph_seed']=seed
    df.to_csv(OUT/f'QUERY_GRAPH_UNION_H085_{year}.csv',index=False)
    df[df.primary_H_ge_0p90].to_csv(OUT/f'QUERY_GRAPH_PRIMARY_H090_{year}.csv',index=False)
    df[df.strict_sensitivity_H_ge_0p95].to_csv(OUT/f'QUERY_GRAPH_STRICT_H095_{year}.csv',index=False)
    summary={
      'year':year,'history_start':start,'history_end':end,'B':B,'block_len':BLOCK,'seed':seed,
      'rank_gap_max':RANK_GAP_MAX,'union_entropy_min':H_UNION,'primary_entropy_min':H_PRIMARY,
      'strict_entropy_min':H_STRICT,'max_edges':MAX_EDGES,'cap_active':cap,
      'union_edges':int(len(df)),'primary_edges':int(df.primary_H_ge_0p90.sum()),
      'strict_edges':int(df.strict_sensitivity_H_ge_0p95.sum()),
      'union_nodes':int(len(set(df.factor_i)|set(df.factor_j))),
      'primary_nodes':int(len(set(df.loc[df.primary_H_ge_0p90,'factor_i'])|set(df.loc[df.primary_H_ge_0p90,'factor_j']))),
      'DQL_call_required_union_edges':int(df.DQL_call_required.sum()),
      'DQ_calls_R3_ABBA':int(len(df)*2*R),
      'DQL_calls_R3_ABBA':int(df.DQL_call_required.sum()*2*R),
      'total_planned_calls_union':int((len(df)+df.DQL_call_required.sum())*2*R),
      'future_target_read':False,'LLM_output_read':False,
    }
    (OUT/f'QUERY_GRAPH_SUMMARY_{year}.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

def main():
    fids,fp,mkt=load_panels()
    s24=build_year(2024,fids,fp,mkt)
    s25=build_year(2025,fids,fp,mkt)
    allsum={'protocol':'JKP153_FULLRANK_DATACONFUSION_PHASEB_GRAPH_V1','summaries':[s24,s25]}
    (OUT/'QUERY_GRAPH_SUMMARY_ALL.json').write_text(json.dumps(allsum,indent=2)+'\n')
    print(json.dumps(allsum,indent=2))

if __name__=='__main__':
    main()
