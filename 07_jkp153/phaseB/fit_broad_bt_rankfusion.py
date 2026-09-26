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
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

ROOT=Path(str(REPRO_ROOT))
COR=ROOT/'JKP153_CALENDAR_ALIGNED_CORRECTED_DOWNSTREAM_20260915_023321'
LAMBDA_BT=1.0
RFGRID=[0,.1,.2,.3,.5,.7,1.0]

def rank01(x):
    s=pd.Series(x);return ((s.rank(method='average')-1)/(len(s)-1)).to_numpy(float)
def sp(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);return float(spearmanr(x[ok],y[ok]).statistic)
def bt_fit(fids,games):
    pos={f:i for i,f in enumerate(fids)}
    ii=np.array([pos[a] for a,b,w in games],int);jj=np.array([pos[b] for a,b,w in games],int);yy=np.array([w for a,b,w in games],float)
    n=len(fids)
    def fg(th):
        z=th[ii]-th[jj];p=expit(z)
        loss=float(np.sum(np.logaddexp(0,z)-yy*z)+0.5*LAMBDA_BT*np.dot(th,th))
        grad=np.zeros(n);d=p-yy
        np.add.at(grad,ii,d);np.add.at(grad,jj,-d);grad+=LAMBDA_BT*th
        return loss,grad
    res=minimize(lambda t:fg(t)[0],np.zeros(n),jac=lambda t:fg(t)[1],method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12})
    th=res.x-res.x.mean()
    return th,{'success':bool(res.success),'message':str(res.message),'n_games':len(games),'fun':float(res.fun)}
def load_data(year):
    if year==2024:
        s=pd.read_csv(COR/'2024_validation/CORRECTED_2024_RAW.csv');t=pd.read_csv(COR/'2024_validation/CORRECTED_2024_TARGET.csv')
    else:
        s=pd.read_csv(COR/'2025_final/CORRECTED_2025_RAW.csv');t=pd.read_csv(COR/'2025_final/CORRECTED_2025_TARGET.csv')
    t=t.set_index('factor_id').reindex(s.factor_id);return s,t
def build_games(calls,graph,arm):
    broad=graph[graph.in_broad.astype(bool)]
    pairs={(r.measurement_edge_id):(r.factor_i,r.factor_j,bool(r.DQL_call_required)) for r in broad.itertuples()}
    games=[]
    for eid,(fi,fj,dqlreq) in pairs.items():
        usearm=arm if (arm!='DQL' or dqlreq) else 'DQ'
        c=calls[(calls.edge_id==eid)&(calls.arm==usearm)&(calls.final_status=='SUCCESS')]
        for r in c.itertuples():
            if r.hard_token not in ('A','B'):continue
            if r.order=='AB': winner_i=(r.hard_token=='A')
            else: winner_i=(r.hard_token=='B')
            games.append((fi,fj,1.0 if winner_i else 0.0))
    return games
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--year',type=int,required=True);ap.add_argument('--log',type=Path,required=True);ap.add_argument('--graph',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--lambda-freeze',type=Path);a=ap.parse_args()
    calls=pd.DataFrame([json.loads(x) for x in a.log.read_text().splitlines() if x.strip()]).drop_duplicates('measurement_call_id',keep='last')
    graph=pd.read_csv(a.graph);s,t=load_data(a.year);fids=s.factor_id.astype(str).tolist();raw=rank01(s.raw_alpha.to_numpy(float));y=t.future_CAPM_alpha.to_numpy(float)
    qs={};meta={}
    for arm in ['DQ','DQL']:
        games=build_games(calls,graph,arm);th,m=bt_fit(fids,games);qs[arm]=rank01(th);meta[arm]=m
        pd.DataFrame({'factor_id':fids,'theta':th,'q':qs[arm]}).to_csv(a.out_dir/f'BROAD_BT_{arm}_{a.year}.csv',index=False)
    rows=[]
    for lam in RFGRID:
        for arm in ['DQ','DQL']:
            score=(1-lam)*raw+lam*qs[arm]
            rows.append({'year':a.year,'lambda_RF':lam,'arm':arm,'rank_ic':sp(score,y)})
    grid=pd.DataFrame(rows);a.out_dir.mkdir(parents=True,exist_ok=True);grid.to_csv(a.out_dir/f'BROAD_BT_RF_GRID_{a.year}.csv',index=False)
    if a.year==2024:
        sel=grid.groupby('lambda_RF').rank_ic.mean().reset_index(name='mean_DQ_DQL_IC').sort_values(['mean_DQ_DQL_IC','lambda_RF'],ascending=[False,True])
        lstar=float(sel.iloc[0].lambda_RF);(a.out_dir/'BROAD_BT_RF_LAMBDA_FREEZE_2024.json').write_text(json.dumps({'lambda_RF_star':lstar},indent=2)+'\n')
    else:
        if not a.lambda_freeze:raise SystemExit('--lambda-freeze required')
        lstar=float(json.loads(a.lambda_freeze.read_text())['lambda_RF_star'])
    chosen=grid[grid.lambda_RF.eq(lstar)];m=dict(zip(chosen.arm,chosen.rank_ic))
    out={'year':a.year,'lambda_RF':lstar,'Raw_ic':sp(raw,y),'DQ_ic':m['DQ'],'DQL_ic':m['DQL'],'DQL_minus_DQ':m['DQL']-m['DQ'],'bt_meta':meta,
         'label':'2024_VALIDATION' if a.year==2024 else 'POST_FINAL_EXPLORATORY_REUSE'}
    (a.out_dir/f'BROAD_BT_RF_SUMMARY_{a.year}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
