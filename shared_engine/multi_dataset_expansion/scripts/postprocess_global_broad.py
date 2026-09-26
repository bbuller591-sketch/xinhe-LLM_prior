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
import json,hashlib,math
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
B=ROOT/'14_global_BROAD_FREEZE'
MEASROOT=ROOT/'15_global_BROAD_MEASUREMENT'
OUT=ROOT/'16_global_POSTPROCESS'
OUT.mkdir(parents=True,exist_ok=True)
FP='aeb56401ca74e127821c4f9126dcb669'
LAM=1e-3
ARMS=[
 ('BREAST_GSE25055_GSE25065','ARM_GSE41998'),
 ('BREAST_GSE25055_GSE25065','ARM_GSE32646'),
 ('SEPSIS_GSE65682','ARM_EMTAB4451'),
 ('SEPSIS_GSE65682','ARM_EMTAB7581')]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def full_calls():
    newp=MEASROOT/'02_FULL_NEW/summaries/CALLS_COMPLETE.csv'
    if not newp.exists(): raise RuntimeError('FULL_NEW_CALLS_NOT_COMPLETE')
    new=pd.read_csv(newp,dtype=str,keep_default_na=False)
    if len(new)!=156320 or new.query_id.nunique()!=156320: raise RuntimeError(f'NEW_CALL_COUNT {len(new)}')
    allowed_status={'PASS','PASS_LOGPROB_ONLY_CONTENT_NONCONFORMANT'}
    if not new.parse_status.isin(allowed_status).all(): raise RuntimeError('NEW_PARSE_FAIL')
    if int((new.parse_status=='PASS_LOGPROB_ONLY_CONTENT_NONCONFORMANT').sum())!=7:
        raise RuntimeError('UNEXPECTED_CONTENT_NONCONFORMANT_COUNT')
    if set(new.system_fingerprint)!={FP}: raise RuntimeError('NEW_FP_FAIL')

    manifests=[]
    for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
        q=pd.read_parquet(B/'QUERIES'/f'{task}_global_BROAD_D20_QUERIES.parquet')
        manifests.append(q[['query_id','task','unordered_pair_id','arm','order','gene_A','gene_B',
                            'prompt_sha256','evidence_packet_sha256','reusable_from_selective']])
    q=pd.concat(manifests,ignore_index=True)
    assert len(q)==156400 and q.query_id.nunique()==156400
    reuseq=q[q.reusable_from_selective.astype(bool)].copy()
    assert len(reuseq)==80
    selective=pd.read_csv(ROOT/'10_LLM_MEASUREMENT/02_FULL_selective/summaries/CALLS_COMPLETE.csv',dtype=str,keep_default_na=False)
    reuse=reuseq.merge(selective,on='query_id',suffixes=('_q',''),validate='one_to_one')
    if len(reuse)!=80: raise RuntimeError('selective_REUSE_COUNT')
    if not (reuse.prompt_sha256_q==reuse.prompt_sha256).all(): raise RuntimeError('REUSE_PROMPT_HASH')
    if not (reuse.evidence_packet_sha256_q==reuse.evidence_packet_sha256).all(): raise RuntimeError('REUSE_EVIDENCE_HASH')
    if not (reuse.parse_status=='PASS').all() or set(reuse.system_fingerprint)!={FP}: raise RuntimeError('REUSE_MEAS_FAIL')
    keep=['timestamp_utc','query_id','unordered_pair_id','arm','order','gene_A','gene_B',
          'prompt_sha256','evidence_packet_sha256','provider_model','system_fingerprint','parse_status',
          'response_token','semantic_choice_gene','logp_A','logp_B','logp_U','pA_vs_B','entropy_AB_bits',
          'pU_threeway_topset','latency_ms','attempts','prompt_tokens','completion_tokens','total_tokens',
          'prompt_cache_hit_tokens','prompt_cache_miss_tokens']
    reuse2=reuse[keep].copy()
    taskmap=q.set_index('query_id').task.to_dict()
    new2=new.copy()
    if 'task' not in new2.columns: new2['task']=new2.query_id.map(taskmap)
    reuse2['task']=reuse2.query_id.map(taskmap)
    allc=pd.concat([new2,reuse2],ignore_index=True,sort=False)
    allc['measurement_origin']=np.where(allc.query_id.isin(set(reuseq.query_id)),'REUSED_selective_IDENTICAL','NEW_BROAD')
    if len(allc)!=156400 or allc.query_id.nunique()!=156400: raise RuntimeError('FULL_LOGICAL_CALL_COUNT')
    qq=q.set_index('query_id')
    chk=allc.set_index('query_id')
    if not (chk.loc[qq.index,'prompt_sha256'].astype(str).values==qq.prompt_sha256.astype(str).values).all(): raise RuntimeError('FULL_PROMPT_HASH')
    if not (chk.loc[qq.index,'evidence_packet_sha256'].astype(str).values==qq.evidence_packet_sha256.astype(str).values).all(): raise RuntimeError('FULL_EVIDENCE_HASH')
    allc.to_parquet(OUT/'BROAD_D20_ALL_CALLS.parquet',index=False,compression='zstd')
    return allc

def neutralize(calls):
    rows=[]
    for (task,arm,pair),g in calls.groupby(['task','arm','unordered_pair_id'],sort=False):
        if len(g)!=2 or set(g.order)!={'AB','BA'}: raise RuntimeError(f'ABBA {task} {arm} {pair}')
        ab=g[g.order=='AB'].iloc[0]; ba=g[g.order=='BA'].iloc[0]
        if str(ab.gene_A)!=str(ba.gene_B) or str(ab.gene_B)!=str(ba.gene_A): raise RuntimeError('ORIENTATION')
        lab=float(ab.logp_A)-float(ab.logp_B)
        lba=float(ba.logp_A)-float(ba.logp_B)
        ell=(lab-lba)/2.0; ordercomp=(lab+lba)/2.0; p=float(expit(ell))
        H=-(p*math.log(p,2)+(1-p)*math.log(1-p,2)) if 0<p<1 else 0.0
        rows.append({'task':task,'arm':arm,'unordered_pair_id':pair,
                     'gene_i':str(ab.gene_A),'gene_j':str(ab.gene_B),
                     'symmetrized_logit_i_vs_j':ell,'order_component':ordercomp,
                     'p_i_over_j':p,'H_AB_bits':H,'certainty_1_minus_H':1-H,
                     'hard_y_i_over_j':1.0 if p>0.5 else (0.0 if p<0.5 else np.nan),
                     'response_token_AB':ab.response_token,'response_token_BA':ba.response_token,
                     'content_contract_ok_AB':ab.parse_status=='PASS',
                     'content_contract_ok_BA':ba.parse_status=='PASS',
                     'pair_content_contract_ok':ab.parse_status=='PASS' and ba.parse_status=='PASS',
                     'either_U':ab.response_token=='U' or ba.response_token=='U',
                     'both_U':ab.response_token=='U' and ba.response_token=='U',
                     'origin_AB':ab.measurement_origin,'origin_BA':ba.measurement_origin})
    P=pd.DataFrame(rows)
    if len(P)!=78200: raise RuntimeError(f'PAIR_COUNT {len(P)}')
    P.to_csv(OUT/'BROAD_D20_PAIR_NEUTRALIZED.csv',index=False)
    return P

def fit_bt(task,arm,P,degree):
    E=pd.read_csv(B/task/arm/f'EDGES_D{degree}.csv',dtype=str)
    eset=set(E.unordered_pair_id)
    Q=P[(P.task==task)&(P.arm==arm)&P.unordered_pair_id.isin(eset)].copy()
    if len(Q)!=len(E): raise RuntimeError(f'EDGE_MEAS_COUNT {task} {arm} d{degree} {len(Q)} {len(E)}')
    nodes=pd.read_csv(B/task/arm/'NODES.csv',dtype={'GeneID':str})
    genes=nodes.gene_symbol.astype(str).tolist(); idx={g:i for i,g in enumerate(genes)}
    ii=Q.gene_i.map(idx).to_numpy(); jj=Q.gene_j.map(idx).to_numpy()
    if np.isnan(ii).any() or np.isnan(jj).any(): raise RuntimeError('NODE_MAP')
    ii=ii.astype(int); jj=jj.astype(int)
    y=Q.hard_y_i_over_j.to_numpy(float); mask=np.isfinite(y)
    ii2=ii[mask]; jj2=jj[mask]; y2=y[mask]
    cert=Q.loc[mask,'certainty_1_minus_H'].to_numpy(float)
    def fit(w):
        def fg(x):
            z=x[ii2]-x[jj2]; pr=expit(z)
            ce=np.logaddexp(0,z)-y2*z
            f=float(np.dot(w,ce)+.5*LAM*np.dot(x,x))
            rr=w*(pr-y2); grad=np.zeros_like(x)
            np.add.at(grad,ii2,rr); np.add.at(grad,jj2,-rr); grad+=LAM*x
            return f,grad
        res=minimize(lambda x:fg(x),np.zeros(len(genes)),jac=True,method='L-BFGS-B',
                     options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
        g=res.x-res.x.mean()
        r=rankdata(-g,method='average')
        psi=2*(1-(r-1)/(len(g)-1))-1
        return res,g,psi
    r1,g1,p1=fit(np.ones(mask.sum()))
    r2,g2,p2=fit(cert)
    if not r1.success or not r2.success: raise RuntimeError(f'BT_OPT_FAIL {task} {arm} d{degree}')
    S=pd.DataFrame({'task':task,'arm':arm,'degree':degree,'feature_index':nodes.feature_index,
                    'gene_symbol':genes,'GeneID':nodes.GeneID,'g_global':g1,'psi_global':p1,'g_global_certainty':g2,'psi_global_certainty':p2})
    ad=OUT/task/arm; ad.mkdir(parents=True,exist_ok=True)
    S.to_csv(ad/f'BT_global_D{degree}_SCORES.csv',index=False)
    st={'task':task,'arm':arm,'degree':degree,'n_nodes':len(genes),'n_edges':len(Q),
        'n_hard_ties_zero_weight':int((~mask).sum()),'mean_entropy':float(Q.H_AB_bits.mean()),
        'median_entropy':float(Q.H_AB_bits.median()),'mean_abs_order_component':float(Q.order_component.abs().mean()),
        'either_U_rate':float(Q.either_U.mean()),'both_U_rate':float(Q.both_U.mean()),
        'global_success':bool(r1.success),'global_nit':int(r1.nit),'global_fun':float(r1.fun),
        'global_certainty_success':bool(r2.success),'global_certainty_nit':int(r2.nit),'global_certainty_fun':float(r2.fun),
        'corr_psi_global':float(np.corrcoef(p1,p2)[0,1])}
    (ad/f'BT_D{degree}_STATUS.json').write_text(json.dumps(st,indent=2),encoding='utf-8')
    return S,st

def meta(task,degree,scores):
    feat=pd.read_csv(ROOT/'03_FROZEN_DATA'/task/'features_p2000.csv',dtype={'GeneID':str})
    base=feat[['feature_index','gene_symbol','GeneID']].copy()
    arms=[a for t,a in ARMS if t==task]
    for method in ['global','global_certainty']:
        cols=[]
        for arm in arms:
            s=scores[(task,arm,degree)][['gene_symbol',f'psi_{method}']].rename(columns={f'psi_{method}':f'{arm}_{method}'})
            base=base.merge(s,on='gene_symbol',how='left')
            cols.append(f'{arm}_{method}')
        arr=base[cols].to_numpy(float)
        avail=np.isfinite(arr)
        n=avail.sum(axis=1)
        h=np.divide(np.nansum(arr,axis=1),n,out=np.zeros(len(base)),where=n>0)
        base[f'h_{method}']=h
        base[f'n_sources_{method}']=n
        if len(cols)>=2:
            base[f'source_range_{method}']=np.where(n>=2,np.nanmax(arr,axis=1)-np.nanmin(arr,axis=1),np.nan)
            signs=np.sign(arr)
            same=np.array([len(set(row[np.isfinite(row)]))==1 if np.isfinite(row).sum()>=2 else np.nan for row in signs],object)
            base[f'source_sign_agree_{method}']=same
    od=OUT/task; od.mkdir(parents=True,exist_ok=True)
    base.to_csv(od/f'global_META_GUIDANCE_D{degree}.csv',index=False)
    return base

calls=full_calls()
P=neutralize(calls)
scores={}; stats=[]
for task,arm in ARMS:
    for degree in [20,10]:
        s,st=fit_bt(task,arm,P,degree); scores[(task,arm,degree)]=s; stats.append(st)
for task in ['BREAST_GSE25055_GSE25065','SEPSIS_GSE65682']:
    for d in [20,10]: meta(task,d,scores)
status={'status':'PASS_global_SOURCE_PRESERVING_BT','n_logical_calls':156400,'n_pair_source':78200,
        'degrees':[20,10],'lambda_BT':LAM,'provider_fingerprint':FP,
        'source_status':stats,'uses_sealed_validation':False,
        'all_calls_sha256':sha(OUT/'BROAD_D20_ALL_CALLS.parquet'),
        'pair_neutralized_sha256':sha(OUT/'BROAD_D20_PAIR_NEUTRALIZED.csv')}
(OUT/'global_POSTPROCESS_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
