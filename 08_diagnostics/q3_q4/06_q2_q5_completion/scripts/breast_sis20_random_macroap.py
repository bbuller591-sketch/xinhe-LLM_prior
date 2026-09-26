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
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(str(REPRO_ROOT / 'GSE25055_GSE25065_FORMAL_REPRODUCIBILITY_PACKAGE'))
OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion/artifacts/BREAST_Q3_RANDOM_MACROAP_SIS20'))
OUT.mkdir(parents=True,exist_ok=True)
TASK='BREAST_GSE25055_GSE25065'; NULL_SEED=2026092207; NREP=1000
LAMS=np.array([0,.03,.1,.3,1,3,10.],float); SEL='SIS'; K=20
SRC=ROOT/'DATA/FROZEN'/TASK; reference=ROOT/'METHOD_ARTIFACTS/reference'; ROUTING=ROOT/'METHOD_ARTIFACTS/selective_ROUTING'; FINAL=ROOT/'METHOD_ARTIFACTS/selective_FINAL'
X=np.load(SRC/'X_development.npy').astype(float); y=np.load(SRC/'y_development.npy').astype(int)
Xte=np.load(SRC/'X_sealed_validation.npy').astype(float); yte=np.load(SRC/'y_sealed_validation.npy').astype(int)
feat=pd.read_csv(SRC/'features_p2000.csv'); splits=pd.read_csv(reference/'OUTER_SPLITS.csv')
finalpars=pd.read_csv(FINAL/'FINAL_SELECTOR_PARAMS.csv'); actual_lam=pd.read_csv(FINAL/'FINAL_SELECTED_ETA.csv')
sealed_ref=pd.read_csv(ROOT/'RESULTS/SEALED_VALIDATION_PRIMARY_RESULTS.csv')
sealed_ref=sealed_ref[(sealed_ref.task==TASK)&(sealed_ref.analysis=='PRIMARY')]
reference_ref=sealed_ref[sealed_ref.method=='reference'].set_index(['selector','k']); selective_ref=sealed_ref[sealed_ref.method=='selective'].set_index(['selector','k'])

def data_anchor(D):
    r=rankdata(-D,method='average'); u=1-(r-.5)/len(D); return norm.ppf(np.clip(u,1e-6,1-1e-6))
def sis_score(A,b):
    bc=b-b.mean(); ac=A-A.mean(0); den=np.sqrt((ac*ac).sum(0)*(bc*bc).sum())
    return np.nan_to_num(np.abs((ac*bc[:,None]).sum(0)/np.where(den==0,np.nan,den)),nan=0.)
def stable_topk(z,D,k):
    return np.lexsort((np.arange(len(z)),-D,-z))[:k]

outer={}
for fold in range(1,6):
    tr=splits[(splits.fold==fold)&(splits.role=='outer_train')].sample_index.to_numpy(int)
    va=splits[(splits.fold==fold)&(splits.role=='outer_validation')].sample_index.to_numpy(int)
    D=sis_score(X[tr],y[tr]); outer[fold]={'tr':tr,'va':va,'D':D,'sD':data_anchor(D)}
Dfull=sis_score(X,y); full={'D':Dfull,'sD':data_anchor(Dfull)}

meas=pd.read_csv(ROOT/'MEASUREMENTS/selective/selective_PAIR_SOURCE_MEASUREMENTS.csv'); meas=meas[meas.task==TASK].copy()
pairs=sorted(meas.unordered_pair_id.astype(str).unique()); arms=sorted(meas.arm.astype(str).unique())
pair_to_pos={p:i for i,p in enumerate(pairs)}; arm_to_pos={a:i for i,a in enumerate(arms)}
name_to_idx={str(g):int(i) for i,g in zip(feat.feature_index,feat.gene_symbol)}
route={}
for fold in range(1,6):
    cf=pd.read_csv(ROUTING/f'fold{fold}_{SEL}'/f'K{K}_PAIR_CONFUSION.csv')
    if len(cf) and 'pair_type' in cf.columns: cf=cf[cf.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION']
    rr=[]
    for r in cf.itertuples():
        a,b=sorted([str(r.feature_A),str(r.feature_B)]); pair=a+'||'+b
        rr.append((pair,name_to_idx[a],name_to_idx[b],float(r.actionable_boundary_score)))
    route[fold]=rr
fc=pd.read_csv(FINAL/'FINAL_CROSSFITTED_selective_CONSTRAINTS.csv')
g=fc[(fc.selector==SEL)&(fc.k==K)]
q=g.groupby(['unordered_pair_id','gene_i','gene_j'],as_index=False).A_CF.first()
final_geom=[]
for r in q.itertuples():
    a,b=str(r.unordered_pair_id).split('||',1)
    final_geom.append((str(r.unordered_pair_id),name_to_idx[a],name_to_idx[b],float(r.A_CF)))

def random_constraints(geom,rand_y,rand_c):
    out=[]
    for pair,i,j,A in geom:
        pi=pair_to_pos[pair]
        for arm in arms:
            ai=arm_to_pos[arm]
            out.append((i,j,float(rand_y[pi,ai]),A*float(rand_c[pi,ai])/2.))
    return out
def optimize_active(sD,cc,lam):
    if lam==0 or not cc: return sD.copy()
    ii=np.array([r[0] for r in cc],int); jj=np.array([r[1] for r in cc],int)
    yy=np.array([r[2] for r in cc]); w=np.array([r[3] for r in cc]); W=w.sum()
    if W<=0:return sD.copy()
    active=np.unique(np.r_[ii,jj]); am={v:q for q,v in enumerate(active)}
    li=np.array([am[v] for v in ii]); lj=np.array([am[v] for v in jj]); base=sD[active].copy(); P=len(sD)
    def fg(v):
        d=v[li]-v[lj]; ce=np.logaddexp(0,d)-yy*d; diff=v-base
        f=.5*np.dot(diff,diff)/P+lam*np.dot(w,ce)/W
        gg=diff/P; rr=lam*(w/W)*(expit(d)-yy); np.add.at(gg,li,rr); np.add.at(gg,lj,-rr)
        return float(f),gg
    r=minimize(lambda v:fg(v)[0],base,jac=lambda v:fg(v)[1],method='L-BFGS-B',
               options={'maxiter':1000,'maxls':50,'ftol':1e-12,'gtol':1e-8})
    if not r.success:
        x0=r.x if np.all(np.isfinite(r.x)) else base
        r=minimize(lambda v:fg(v)[0],x0,jac=lambda v:fg(v)[1],method='L-BFGS-B',
                   options={'maxiter':3000,'maxls':300,'ftol':1e-13,'gtol':1e-7})
    z=sD.copy(); z[active]=r.x; return z

def metrics(Atr,btr,Ava,bva,cols):
    sc=StandardScaler().fit(Atr[:,cols])
    mod=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',class_weight='balanced',
                           max_iter=5000,random_state=2026091901)
    mod.fit(sc.transform(Atr[:,cols]),btr)
    pp=mod.predict_proba(sc.transform(Ava[:,cols]))[:,1]
    auc=float(roc_auc_score(bva,pp))
    ap_pos=float(average_precision_score(bva,pp))
    ap_neg=float(average_precision_score(1-bva,1-pp))
    return auc,0.5*(ap_pos+ap_neg)

# development reference at lam=0
dev_ref=[]
for fold in range(1,6):
    os=outer[fold]; cols=stable_topk(os['sD'],os['D'],K)
    dev_ref.append(metrics(X[os['tr']],y[os['tr']],X[os['va']],y[os['va']],cols)[0])
dev_ref=float(np.mean(dev_ref))

def run_chunk(reps):
    rows=[]
    for rep in reps:
        rng=np.random.default_rng(np.random.SeedSequence([NULL_SEED,rep]))
        p=rng.uniform(0,1,(len(pairs),len(arms)))
        H=-(p*np.log2(np.clip(p,1e-12,1))+(1-p)*np.log2(np.clip(1-p,1e-12,1)))
        c=1-H; yhard=(p>.5).astype(float)
        token=hashlib.sha256(p.tobytes()).hexdigest()
        vals=[]
        for lam in LAMS:
            aa=[]
            for fold in range(1,6):
                os=outer[fold]; cc=random_constraints(route[fold],yhard,c)
                z=optimize_active(os['sD'],cc,float(lam)); cols=stable_topk(z,os['D'],K)
                aa.append(metrics(X[os['tr']],y[os['tr']],X[os['va']],y[os['va']],cols)[0])
            vals.append((float(lam),float(np.mean(aa))))
        best=max(v for _,v in vals); chosen=min(e for e,v in vals if np.isclose(v,best,atol=1e-12,rtol=0))
        cc=random_constraints(final_geom,yhard,c); z=optimize_active(full['sD'],cc,chosen); cols=stable_topk(z,full['D'],K)
        auc,mapv=metrics(X,y,Xte,yte,cols)
        rows.append(dict(replicate=rep,draw_sha256=token,chosen_lam=chosen,best_dev_auc=best,
                         sealed_auc=auc,sealed_macro_ap=mapv,selected_indices='|'.join(map(str,cols.tolist()))))
    return rows

chunks=[list(range(i,NREP,8)) for i in range(8)]; rows=[]; t=time.time()
with ProcessPoolExecutor(max_workers=8) as ex:
    fs=[ex.submit(run_chunk,ch) for ch in chunks]
    for n,f in enumerate(as_completed(fs),1):
        rows.extend(f.result()); print('chunk',n,'rows',len(rows),'sec',round(time.time()-t,1),flush=True)
R=pd.DataFrame(rows).sort_values('replicate'); R.to_csv(OUT/'RANDOM_SIS20_WITH_MACROAP_1000.csv',index=False)
# identity against existing saved AUROC null
old=pd.read_csv(str(REPRO_ROOT / '08_diagnostics/q3_q4/02_q3_random_null/BREAST_GSE25055_GSE25065/RANDOM_PROBABILITY_NULL_1000.csv'))
old=old[(old.selector==SEL)&(old.k==K)].sort_values('replicate')
assert np.all(old.replicate.to_numpy()==R.replicate.to_numpy())
maxdiff=float(np.max(np.abs(old.sealed_auc_frozen_support.to_numpy()-R.sealed_auc.to_numpy())))
obs_map=float(selective_ref.loc[(SEL,K),'macro_ap']); ref_map=float(reference_ref.loc[(SEL,K),'macro_ap']); v=R.sealed_macro_ap
summary=dict(n=NREP,auroc_identity_max_abs_diff=maxdiff,reference_macro_ap=ref_map,observed_macro_ap=obs_map,
             observed_delta_macro_ap=obs_map-ref_map,random_null_mean_macro_ap=float(v.mean()),
             random_null_mean_delta_macro_ap=float(v.mean()-ref_map),
             random_macro_ap_p=float((1+int((v>=obs_map-1e-12).sum()))/(NREP+1)),
             random_macro_ap_q025=float(v.quantile(.025)),random_macro_ap_q975=float(v.quantile(.975)))
pd.DataFrame([summary]).to_csv(OUT/'SUMMARY.csv',index=False)
print(summary)
