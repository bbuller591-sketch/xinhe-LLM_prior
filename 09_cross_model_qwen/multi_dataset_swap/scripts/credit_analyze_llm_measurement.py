

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
import json, math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
M=WS/'04_METHOD_FREEZE'
R=Path(str(REPRO_ROOT / '09_cross_model_qwen/multi_dataset_swap/05_CREDIT_G/downstream'))
FEATURES=["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker"]
idx={f:i for i,f in enumerate(FEATURES)}
EPS=1e-6

def clip(p): return min(max(float(p),1e-4),1-1e-4)
def logit(p):
    p=clip(p); return math.log(p/(1-p))
def sigmoid(x):
    if x>=0: return 1/(1+math.exp(-x))
    ex=math.exp(x); return ex/(1+ex)
def entropy_conf(p):
    p=clip(p)
    H=-(p*math.log(p)+(1-p)*math.log(1-p))
    return 1-H/math.log(2)

def primary_pair_table(df):
    d=df[df.primary_measurement.astype(bool)].copy()
    rows=[]
    for pair_id,q in d.groupby('pair_id'):
        if set(q.presentation_order)!=set(['AB','BA']):
            raise RuntimeError((pair_id,q.presentation_order.tolist()))
        ab=q[q.presentation_order=='AB'].iloc[0]
        ba=q[q.presentation_order=='BA'].iloc[0]
        ca=ab.canonical_feature_a; cb=ab.canonical_feature_b
        abstain=(ab.first_token=='U') or (ba.first_token=='U')
        p_ab_can=1-float('nan')
        p_ba_can=1-float('nan')
        pbar=np.nan; gap=np.nan; c=np.nan
        if pd.notna(ab.p_presented_A_cond_AB):
            p_ab_can=clip(ab.p_presented_A_cond_AB)
        else:
            p_ab_can=np.nan
        if pd.notna(ba.p_presented_A_cond_AB):
            p_ba_can=clip(1.0-float(ba.p_presented_A_cond_AB))
        else:
            p_ba_can=np.nan
        if not abstain and np.isfinite(p_ab_can) and np.isfinite(p_ba_can):
            la,lb=logit(p_ab_can),logit(p_ba_can)
            gap=abs(la-lb)
            pbar=sigmoid((la+lb)/2)
            c=entropy_conf(pbar)
        rows.append({
            'pair_id':pair_id,'feature_a':ca,'feature_b':cb,
            'ab_first_token':ab.first_token,'ba_first_token':ba.first_token,
            'p_canonical_a_from_AB':p_ab_can,
            'p_canonical_a_from_BA':p_ba_can,
            'order_gap_logit':gap,'p_canonical_a_order_neutral':pbar,
            'c_edge_entropy':c,'primary_abstain':bool(abstain),
        })
    return pd.DataFrame(rows)

def fit_bt(pair):
    use=pair[~pair.primary_abstain & pair.p_canonical_a_order_neutral.notna()].copy()
    def unpack(x):
        g=np.zeros(len(FEATURES))
        g[:-1]=x
        g[-1]=-np.sum(x)
        return g
    def obj(x):
        g=unpack(x); val=0.0
        for _,r in use.iterrows():
            i,j=idx[r.feature_a],idx[r.feature_b]
            z=g[i]-g[j]
            p=float(r.p_canonical_a_order_neutral)
            # stable Bernoulli cross entropy with soft target
            val += np.logaddexp(0,z) - p*z
        return float(val)
    res=minimize(obj,np.zeros(len(FEATURES)-1),method='BFGS',options={'gtol':1e-10,'maxiter':10000})
    if not res.success:
        # optimizer precision-loss is acceptable only if gradient/nll are finite; retry L-BFGS-B
        res2=minimize(obj,res.x,method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-12})
        if res2.fun<=res.fun or res2.success: res=res2
    g=unpack(res.x)
    med=float(np.median(g)); mad=float(np.median(np.abs(g-med))); sg=max(mad,1e-6)
    h=np.tanh(g/sg)
    return g,h,sg,res,use

def feature_guidance(pair,g,h):
    rows=[]
    for f in FEATURES:
        inc=pair[(pair.feature_a==f)|(pair.feature_b==f)]
        good=inc[~inc.primary_abstain & inc.c_edge_entropy.notna()]
        cov=len(good)/6.0
        cj=float(good.c_edge_entropy.mean()) if len(good) else 0.0
        if len(good)<3: cj=0.0
        rows.append({
            'feature':f,'bt_score_g':float(g[idx[f]]),'bt_guidance_h':float(h[idx[f]]),
            'entropy_trust_cj':cj,'nonabstaining_incident_edges':int(len(good)),
            'design_degree':6,'coverage':cov,
        })
    return pd.DataFrame(rows)

def canonical_choice(row):
    tok=row.first_token
    if tok=='U': return 'U'
    if row.presentation_order=='AB':
        return 'A_CANON' if tok=='A' else 'B_CANON'
    return 'B_CANON' if tok=='A' else 'A_CANON'

def repeat_diag(df):
    sent=df[df.repeat_instability_sentinel.astype(bool)].copy()
    sent['canonical_choice']=sent.apply(canonical_choice,axis=1)
    sent['p_canonical_a']=sent.apply(lambda r: float(r.p_presented_A_cond_AB) if r.presentation_order=='AB' else 1-float(r.p_presented_A_cond_AB),axis=1)
    rows=[]
    for (pair_id,order),q in sent.groupby(['pair_id','presentation_order']):
        qs=q.sort_values('repeat_index')
        choices=qs.canonical_choice.tolist()
        p=qs.p_canonical_a.to_numpy(float)
        rows.append({
            'pair_id':pair_id,'presentation_order':order,'n_repeats':len(qs),
            'choices':'|'.join(choices),'choice_unanimous':len(set(choices))==1,
            'p_canonical_a_mean':float(np.mean(p)),'p_canonical_a_sd':float(np.std(p,ddof=1)) if len(p)>1 else 0.0,
            'p_canonical_a_range':float(np.max(p)-np.min(p)),
        })
    return pd.DataFrame(rows)

m12=pd.read_csv(R/'global_DEEPSEEK_MEASUREMENT.csv')
pair12=primary_pair_table(m12)
pair12.to_csv(R/'global_PAIR_MEASUREMENT.csv',index=False)
g,h,sg,res,use=fit_bt(pair12)
fg=feature_guidance(pair12,g,h)
fg.to_csv(R/'global_FEATURE_GUIDANCE.csv',index=False)
rep12=repeat_diag(m12)
rep12.to_csv(R/'global_REPEAT_DIAGNOSTICS.csv',index=False)

selective=pd.read_csv(R/'selective_DEEPSEEK_MEASUREMENT.csv')
pair3=primary_pair_table(selective)
pair3.to_csv(R/'selective_PAIR_MEASUREMENT.csv',index=False)
rep3=repeat_diag(selective)
rep3.to_csv(R/'selective_REPEAT_DIAGNOSTICS.csv',index=False)

m3graph=pd.read_csv(M/'selective_SELECTIVE_GRAPH_FREEZE.csv')
pairmap=pair3.set_index('pair_id')
selrows=[]
for s in ['L1','GBM_PERM','ELASTIC_NET']:
    key=s.lower()
    num={f:0.0 for f in FEATURES}; den={f:0.0 for f in FEATURES}
    for _,r in m3graph.iterrows():
        if not bool(r.formal_selective_measurement_eligible): continue
        if not bool(r[f'requested_{key}']): continue
        pm=pairmap.loc[r.pair_id]
        if bool(pm.primary_abstain) or pd.isna(pm.p_canonical_a_order_neutral): continue
        p=float(pm.p_canonical_a_order_neutral); c=float(pm.c_edge_entropy)
        u=float(r[f'u_data_{key}'])
        w=u*c
        sig=2*p-1
        num[r.feature_a]+=w*sig; den[r.feature_a]+=w
        num[r.feature_b]-=w*sig; den[r.feature_b]+=w
    for f in FEATURES:
        val=num[f]/den[f] if den[f]>0 else 0.0
        selrows.append({'selector':s,'feature':f,'h_selective':float(val),'eligible_weight_sum':float(den[f])})
pd.DataFrame(selrows).to_csv(R/'selective_SELECTOR_GUIDANCE.csv',index=False)

summary={
 'm12_primary_pairs':int(len(pair12)),
 'm12_primary_abstain_pairs':int(pair12.primary_abstain.sum()),
 'm12_bt_pairs_used':int(len(use)),
 'm12_bt_optimizer_success':bool(res.success),
 'm12_bt_optimizer_message':str(res.message),
 'm12_bt_scale_mad':sg,
 'm12_mean_order_gap_logit':float(pair12.order_gap_logit.mean()),
 'm12_median_entropy_trust_edge':float(pair12.c_edge_entropy.median()),
 'm12_repeat_order_cells':int(len(rep12)),
 'm12_repeat_nonunanimous_cells':int((~rep12.choice_unanimous).sum()),
 'selective_primary_pairs':int(len(pair3)),
 'selective_primary_abstain_pairs':int(pair3.primary_abstain.sum()),
 'selective_repeat_order_cells':int(len(rep3)),
 'selective_repeat_nonunanimous_cells':int((~rep3.choice_unanimous).sum()),
}
(R/'LLM_MEASUREMENT_AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print('\nM1/global_certainty feature guidance:')
print(fg.sort_values('bt_guidance_h',ascending=False).to_string(index=False))
print('\nM3 pairs:')
print(pair3.to_string(index=False))
print('\nM3 nonzero guidance:')
x=pd.DataFrame(selrows)
print(x[x.eligible_weight_sum>0].to_string(index=False))
