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
import json, math, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtri
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
DATA=Path(str(REPRO_ROOT / 'hospital_osteoporosis_dataonly_pilot_20260917'))
CANON=Path(str(REPRO_ROOT / 'hospital_osteoporosis_canonical_20260917'))
BASE=ROOT/'13_SELECTOR_ROBUSTNESS_EXPLORATORY_V1_0'
OUT=ROOT/'14_SELECTOR_selective_PORTABILITY_EXPLORATORY_V1_0'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(DATA/'scripts'))
import v2_core as V

NRES=200; FRAC=.80; WINDOW=5; THR=.25; LAMS={5:10.0,10:0.03}; KGRID=[5,10]
SEED=2026091804

def enet_fit_full(Xraw,site,y,lam,alpha,max_iter=30000,tol=1e-9):
    sc=V.fit_scaler_v2(Xraw); Xs=V.apply_scaler_v2(Xraw,sc); Xd,mask=V.design(Xs,site,include_site=True)
    Xd=np.ascontiguousarray(Xd,float); y=np.asarray(y,float); mask=np.asarray(mask,float)
    L=.25*V.lambda_max_power(Xd)+2*lam*(1-alpha); step=1.0/L
    w=np.zeros(Xd.shape[1]); z=w.copy(); t=1.
    l1=lam*alpha*mask; l2=lam*(1-alpha)*mask
    for _ in range(max_iter):
        g=V.grad_logloss(Xd,y,z)+2*l2*z
        v=z-step*g
        wn=np.sign(v)*np.maximum(np.abs(v)-step*l1,0)
        wn[mask==0]=v[mask==0]
        tn=(1+math.sqrt(1+4*t*t))/2
        zn=wn+((t-1)/tn)*(wn-w)
        if np.max(np.abs(wn-w))<tol: w=wn; break
        w,z,t=wn,zn,tn
    return w,sc,V.forced_context_cols(site,True)

def confusion_from_ranks(ranks,features,k):
    med=np.median(ranks,axis=0); B=ranks.shape[0]
    cand=[i for i in range(len(features)) if med[i]<=k+WINDOW]
    near=set(i for i in cand if k-WINDOW<=med[i]<=k+WINDOW)
    rows=[]
    for aa,ia in enumerate(cand):
        for ib in cand[aa+1:]:
            if ia not in near and ib not in near: continue
            ra,rb=ranks[:,ia],ranks[:,ib]
            ties=(ra==rb).sum()
            p_ab=((ra<rb).sum()+.5*ties)/B
            sa,sb=ra<=k,rb<=k
            Q=float((sa^sb).mean()); bal=1-2*abs(float(p_ab)-.5); act=Q*bal
            rows.append({'feature_A':features[ia],'feature_B':features[ib],'k':k,
                         'P_A_gt_B':float(p_ab),'selection_disagreement_Q':Q,'order_balance_B':bal,
                         'actionable_boundary_score':act,'P_both':float((sa&sb).mean()),
                         'P_neither':float((~sa&~sb).mean()),'median_rank_A':float(med[ia]),'median_rank_B':float(med[ib]),
                         'pair_type':'ACTIONABLE_BOUNDARY_CONFUSION' if act>=THR else ('RANK_ORDER_ONLY' if bal>=.5 else 'WEAK_OR_NONE')})
    return pd.DataFrame(rows)

def selective_anchor(D):
    r=rankdata(-np.asarray(D,float),method='average'); u=1-(r-.5)/len(D)
    return ndtri(np.clip(u,1e-6,1-1e-6))

def selective_solve(D,features,pairs,lam):
    x0=selective_anchor(D); idx={f:i for i,f in enumerate(features)}
    z=pairs[pairs.primary_weight>0]
    if z.empty: return x0,True
    obs=[(idx[r.feature_A],idx[r.feature_B],float(r.primary_weight),float(r.hard_target_A)) for r in z.itertuples()]
    wsum=sum(o[2] for o in obs)+1e-12; p=len(features)
    def fg(x):
        loss=.5/p*np.sum((x-x0)**2); grad=(x-x0)/p
        for i,j,w,y in obs:
            d=x[i]-x[j]; loss+=lam*w*(np.logaddexp(0,d)-y*d)/wsum
            rr=lam*w*(expit(d)-y)/wsum; grad[i]+=rr;grad[j]-=rr
        return float(loss),grad
    res=minimize(lambda x:fg(x)[0],x0,jac=lambda x:fg(x)[1],method='BFGS',options={'maxiter':3000,'gtol':1e-8})
    return res.x,bool(res.success)

def stable_topk(score,secondary,features,k):
    d=pd.DataFrame({'feature':features,'score':score,'secondary':secondary,'idx':np.arange(len(features))})
    d=d.sort_values(['score','secondary','idx'],ascending=[False,False,True],kind='mergesort')
    return list(d.feature.head(k))

def choose_pred_l2(X,site,y,g):
    spl=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=91019).split(np.zeros(len(y)),y,g))
    best=None
    for lam in V.LAM_GRID:
        auc=[]
        for tr,va in spl:
            w,info,sc,nctx=V.fit_full(X[tr],site[tr],y[tr],lam,kind='l2')
            auc.append(roc_auc_score(y[va],V.predict_full(w,sc,X[va],site[va])))
        m=float(np.mean(auc))
        if best is None or (m,float(lam))>best[0]: best=((m,float(lam)),float(lam))
    return best[1]

def metric(y,p):
    return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,(p>=.5).astype(int)))}

# data
man=json.load(open(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_manifest.json')); features=list(man['selectable_features'])
dev=pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_X.csv').merge(
    pd.read_csv(DATA/'03_LEAKAGE_SAFE_DEVELOPMENT/development_matrix_y_and_context.csv',usecols=['patient_uid','site','y']),on=['patient_uid','site'])
dev=dev[dev.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
X=dev[features].to_numpy(float); site=dev.site.to_numpy(); y=dev.y.to_numpy(int); groups=dev.patient_uid.to_numpy(); uids=np.unique(groups)
selsets=pd.read_csv(BASE/'BATCH1_FROZEN_SELECTED_SETS.csv')
scores=pd.read_csv(BASE/'SELECTOR_FEATURE_SCORES.csv')

# frozen params
status=json.load(open(BASE/'SELECTOR_ROBUSTNESS_STATUS.json'))
alpha=float(status['elasticnet_alpha']); enlam=float(status['elasticnet_lambda']); rlam=float(status['ridge_lambda'])

# full scores
enD=scores[scores.selector=='S1_ELASTICNET'].set_index('feature').loc[features].score.to_numpy(float)
rD=scores[scores.selector=='S2_RIDGERANK'].set_index('feature').loc[features].score.to_numpy(float)
sub=np.load(BASE/'SUBSAMPLETOPK_RESAMPLES.npz',allow_pickle=True)
subr=sub['ranks']; assert list(map(str,sub['features']))==features

# resampling Elastic/Ridge
ranks_by={}
for name in ['S1_ELASTICNET','S2_RIDGERANK']:
    rr=np.zeros((NRES,len(features)))
    for b in range(NRES):
        rng=np.random.default_rng(SEED+b)
        take=rng.choice(len(uids),size=int(round(FRAC*len(uids))),replace=False)
        mask=np.isin(groups,uids[take]); idx=np.where(mask)[0]
        if name=='S1_ELASTICNET':
            w,sc,nctx=enet_fit_full(X[idx],site[idx],y[idx],enlam,alpha)
        else:
            w,info,sc,nctx=V.fit_full(X[idx],site[idx],y[idx],rlam,kind='l2')
        d=V.clinical_scores(w,nctx); rr[b]=rankdata(-d,method='average')
        if (b+1)%50==0: print(name,'resample',b+1,flush=True)
    ranks_by[name]=rr
    np.savez_compressed(OUT/f'{name}_RESAMPLE_RANKS.npz',ranks=rr,features=np.array(features))
ranks_by['S3_SUBSAMPLETOPK']=subr

# broad LLM pair lookup
pm=pd.read_csv(ROOT/'08_LLM_MEASUREMENT/BT_ANALYSIS_V1_8_1/PAIR_MEASUREMENTS_ORDER_NEUTRAL_V1_8_1.csv')
def ukey(a,b): return '||'.join(sorted([str(a),str(b)]))
pm['ukey']=[ukey(a,b) for a,b in zip(pm.feature_A,pm.feature_B)]
lookup=pm.set_index('ukey')[['feature_A','feature_B','p_pair_semantic_A','H_pair','c_pair']].to_dict('index')
eligible=set(pd.read_csv(ROOT/'07_BROAD37_EVIDENCE/V0_8_1_AUDITED_CORPUS_AND_PACKETS/BROAD37_FEATURE_EVIDENCE_STATE_V0_8_1.csv').query("bt_graph_eligible == True").feature.astype(str))

m3sets=[]; confsum=[]; allpairs=[]
for selector in ['S1_ELASTICNET','S2_RIDGERANK','S3_SUBSAMPLETOPK']:
    rr=ranks_by[selector]
    for k in KGRID:
        cf=confusion_from_ranks(rr,features,k)
        cf['selector']=selector
        act=cf[cf.actionable_boundary_score>=THR].copy()
        rows=[]
        for r in act.itertuples():
            key=ukey(r.feature_A,r.feature_B)
            gate=int(r.feature_A in eligible and r.feature_B in eligible and key in lookup)
            if gate:
                m=lookup[key]
                pA=float(m['p_pair_semantic_A']) if r.feature_A==m['feature_A'] else 1-float(m['p_pair_semantic_A'])
                c=float(m['c_pair'])
            else:
                pA=.5;c=0.
            rows.append({**r._asdict(),'evidence_gate':gate,'p_selective_A':pA,'c_pair':c,
                         'hard_target_A':float(pA>.5),'primary_weight':float(r.actionable_boundary_score)*c*gate})
        pa=pd.DataFrame(rows)
        if len(pa)==0: pa=pd.DataFrame(columns=['feature_A','feature_B','k','evidence_gate','p_selective_A','c_pair','hard_target_A','primary_weight'])
        pa.to_csv(OUT/f'{selector}_K{k}_ACTIONABLE_WITH_LLM.csv',index=False)
        allpairs.append(pa)
        confsum.append({'selector':selector,'k':k,'n_relevant_pairs':len(cf),'n_actionable':len(act),
                        'n_evidence_eligible_actionable':int(pa.evidence_gate.sum()) if len(pa) else 0,
                        'sum_primary_weight':float(pa.primary_weight.sum()) if len(pa) else 0,
                        'mean_primary_weight':float(pa.primary_weight.mean()) if len(pa) else 0})
        if selector=='S1_ELASTICNET': D=enD
        elif selector=='S2_RIDGERANK': D=rD
        else:
            D=(rr<=k).mean(axis=0)
        latent,ok=selective_solve(D,features,pa,LAMS[k])
        ss=stable_topk(latent,D,features,k)
        base=selsets[(selsets.selector==selector)&(selsets.k==k)].iloc[0].selected_set
        m3sets.append({'selector':selector,'k':k,'lam':LAMS[k],'optimizer_success':ok,
                       'base_selected_set':base,'selective_selected_set':'|'.join(ss),
                       'added_vs_base':'|'.join(sorted(set(ss)-set(base.split('|')))),
                       'removed_vs_base':'|'.join(sorted(set(base.split('|'))-set(ss)))})
pd.DataFrame(confsum).to_csv(OUT/'CONFUSION_ROUTING_SUMMARY.csv',index=False)
sets=pd.DataFrame(m3sets); sets.to_csv(OUT/'selective_PORTABILITY_SELECTED_SETS.csv',index=False)

# fit predictors full Batch1
preds_meta=[]; cache={}
for r in sets.itertuples():
    fs=set(str(r.selective_selected_set).split('|')); ft=tuple(f for f in features if f in fs)
    if ft not in cache:
        inds=[features.index(f) for f in ft]; XX=X[:,inds]; lam=choose_pred_l2(XX,site,y,groups)
        w,info,sc,nctx=V.fit_full(XX,site,y,lam,kind='l2')
        mid='PM3_%02d'%(len(cache)+1); np.savez_compressed(OUT/f'{mid}_MODEL.npz',w=w,med=sc[0],mu=sc[1],sd=sc[2])
        cache[ft]=(mid,lam,w,sc)
    mid,lam,w,sc=cache[ft]
    preds_meta.append({'selector':r.selector,'k':int(r.k),'lam':float(r.lam),'selected_set':'|'.join(ft),'model_id':mid,'predictor_l2_lambda':lam})
pidx=pd.DataFrame(preds_meta);pidx.to_csv(OUT/'selective_PORTABILITY_PREDICTOR_INDEX.csv',index=False)

# Batch2
cx=pd.read_csv(CANON/'canonical/site_level_X.csv'); cy=pd.read_csv(CANON/'canonical/site_level_y.csv')
cx=cx.copy();cx['y']=pd.to_numeric(cy.iloc[:,0]).astype(int).to_numpy()
b2=cx[(cx.cohort.astype(str)=='batch2')&cx.site.isin(V.SITE_PRIMARY)].reset_index(drop=True)
yb=b2.y.to_numpy(int); sb=b2.site.to_numpy(); uid=b2.patient_uid.astype(str).to_numpy()

def matrix(feat):
    z=b2[list(feat)].copy()
    for c in feat:
        if c=='DXA_性别': z[c]=z[c].astype(str).str.strip().map({'女':1.,'男':0.})
        else: z[c]=pd.to_numeric(z[c],errors='coerce')
    return z.to_numpy(float)

preds={}
for ft,(mid,lam,w,sc) in cache.items(): preds[mid]=V.predict_full(w,sc,matrix(ft),sb)

base_res=pd.read_csv(BASE/'BATCH2_EXPLORATORY_DATAONLY_RESULTS.csv')
rows=[]
for r in pidx.itertuples():
    met=metric(yb,preds[r.model_id]); br=base_res[(base_res.selector==r.selector)&(base_res.k==r.k)].iloc[0]
    rows.append({'selector':r.selector,'k':int(r.k),'lam':r.lam,'selective_selected_set':r.selected_set,'model_id':r.model_id,**met,
                 'delta_auroc_vs_own_dataonly':met['auroc']-float(br.auroc),
                 'delta_auprc_vs_own_dataonly':met['auprc']-float(br.auprc),
                 'delta_balacc_vs_own_dataonly':met['balanced_accuracy']-float(br.balanced_accuracy)})
res=pd.DataFrame(rows)

# bootstrap vs own data-only prediction: reconstruct baseline models from BASE
base_idx=pd.read_csv(BASE/'FROZEN_PREDICTOR_INDEX.csv')
base_preds={}
for rr in base_idx.itertuples():
    if rr.selector not in set(sets.selector): continue
    feat=tuple(str(rr.selected_set).split('|'))
    dat=np.load(BASE/f'{rr.predictor_id}_MODEL.npz')
    base_preds[(rr.selector,int(rr.k))]=V.predict_full(dat['w'],(dat['med'],dat['mu'],dat['sd']),matrix(feat),sb)

rng=np.random.default_rng(2026091805); patients=np.unique(uid); boot=[]
for b in range(2000):
    samp=rng.choice(patients,size=len(patients),replace=True)
    inds=np.concatenate([np.where(uid==u)[0] for u in samp]); yy=yb[inds]
    if len(np.unique(yy))<2: continue
    for r in pidx.itertuples():
        p=preds[r.model_id][inds]; p0=base_preds[(r.selector,int(r.k))][inds]
        boot.append({'b':b,'selector':r.selector,'k':int(r.k),
                     'd_auroc':roc_auc_score(yy,p)-roc_auc_score(yy,p0),
                     'd_auprc':average_precision_score(yy,p)-average_precision_score(yy,p0)})
bd=pd.DataFrame(boot); cis=[]
for (s,k),g in bd.groupby(['selector','k']):
    cis.append({'selector':s,'k':int(k),'d_auroc_lo':float(g.d_auroc.quantile(.025)),'d_auroc_hi':float(g.d_auroc.quantile(.975)),
                'p_d_auroc_gt0':float((g.d_auroc>0).mean()),'d_auprc_lo':float(g.d_auprc.quantile(.025)),
                'd_auprc_hi':float(g.d_auprc.quantile(.975)),'p_d_auprc_gt0':float((g.d_auprc>0).mean())})
final=res.merge(pd.DataFrame(cis),on=['selector','k'],how='left')
final.to_csv(OUT/'selective_PORTABILITY_BATCH2_EXPLORATORY_RESULTS.csv',index=False)

status={'status':'PASS_POSTHOC_EXPLORATORY','new_llm_calls':0,'actionable_threshold':THR,'window':WINDOW,'n_resamples':NRES,
        'etas_reused_from_main':LAMS,'batch2_already_seen_before_extension':True,
        'interpretation':'selector-specific selective portability sensitivity only'}
(OUT/'selective_PORTABILITY_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(status,ensure_ascii=False,indent=2))
print('\nROUTING\n',pd.DataFrame(confsum).to_string(index=False))
print('\nSETS\n',sets.to_string(index=False))
print('\nBATCH2\n',final.to_string(index=False))
