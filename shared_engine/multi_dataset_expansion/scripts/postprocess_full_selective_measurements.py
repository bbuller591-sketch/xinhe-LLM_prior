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
from scipy.special import expit

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
Q=ROOT/'10_LLM_MEASUREMENT/02_FULL_selective/FULL_selective_QUERY_TABLE.csv'
C=ROOT/'10_LLM_MEASUREMENT/02_FULL_selective/summaries/CALLS_COMPLETE.csv'
OUT=ROOT/'11_selective_POSTPROCESS'
OUT.mkdir(parents=True,exist_ok=True)

q=pd.read_csv(Q,dtype=str,keep_default_na=False)
c=pd.read_csv(C,dtype=str,keep_default_na=False)
assert len(q)==8862 and len(c)==8862 and q.query_id.nunique()==8862 and c.query_id.nunique()==8862
meas_cols=['query_id','parse_status','provider_model','system_fingerprint','response_token','semantic_choice_gene',
           'logp_A','logp_B','logp_U','pA_vs_B','entropy_AB_bits','pU_threeway_topset','latency_ms',
           'prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens',
           'prompt_sha256','evidence_packet_sha256','call_key','attempts']
z=q.merge(c[meas_cols],on='query_id',suffixes=('_q','_c'),validate='one_to_one')
assert (z.prompt_sha256_q==z.prompt_sha256_c).all()
assert (z.evidence_packet_sha256_q==z.evidence_packet_sha256_c).all()
assert (z.parse_status=='PASS').all()
assert z.provider_model.nunique()==1 and z.provider_model.iloc[0]=='deepseek-flash'
assert z.system_fingerprint.nunique()==1 and z.system_fingerprint.iloc[0]=='aeb56401ca74e127821c4f9126dcb669'

# verify every retained row has a valid unique cache object
cache=ROOT/'10_LLM_MEASUREMENT/CACHE'
missing=[k for k in z.call_key.astype(str) if not (cache/f'{k}.json').exists()]
assert not missing
assert z.call_key.nunique()==8862

rows=[]
for (arm,pair),g in z.groupby(['arm','unordered_pair_id']):
    assert len(g)==2 and set(g.order)=={'AB','BA'}
    ab=g[g.order=='AB'].iloc[0]; ba=g[g.order=='BA'].iloc[0]
    assert ab.gene_A==ba.gene_B and ab.gene_B==ba.gene_A
    lab=float(ab.logp_A)-float(ab.logp_B)
    lba=float(ba.logp_A)-float(ba.logp_B)
    ell=(lab-lba)/2.0
    p=float(expit(ell))
    H=-(p*math.log(p,2)+(1-p)*math.log(1-p,2)) if 0<p<1 else 0.0
    pab=float(expit(lab)); pba_can=float(expit(-lba))
    task='BREAST_GSE25055_GSE25065' if str(ab.query_id).startswith('BREAST_') else 'SEPSIS_GSE65682'
    rows.append({
      'task':task,'arm':arm,'unordered_pair_id':pair,
      'gene_i':ab.gene_A,'gene_j':ab.gene_B,
      'logit_AB':lab,'logit_BA_presented':lba,'symmetrized_logit_i_vs_j':ell,
      'p_i_over_j':p,'hard_y_i_over_j':1.0 if p>0.5 else (0.0 if p<0.5 else 0.5),
      'H_AB_bits':H,'certainty_1_minus_H':1-H,
      'p_i_AB_order':pab,'p_i_BA_order':pba_can,
      'abs_order_probability_gap':abs(pab-pba_can),
      'response_token_AB':ab.response_token,'response_token_BA':ba.response_token,
      'U_AB':ab.response_token=='U','U_BA':ba.response_token=='U',
      'either_U':ab.response_token=='U' or ba.response_token=='U',
      'both_U':ab.response_token=='U' and ba.response_token=='U',
      'semantic_choice_AB':ab.semantic_choice_gene,'semantic_choice_BA':ba.semantic_choice_gene,
      'hard_semantic_choice_agree':ab.semantic_choice_gene==ba.semantic_choice_gene,
      'mean_call_entropy_bits':0.5*(float(ab.entropy_AB_bits)+float(ba.entropy_AB_bits)),
      'evidence_packet_sha256':ab.evidence_packet_sha256_q
    })
M=pd.DataFrame(rows).sort_values(['task','arm','unordered_pair_id'],kind='mergesort').reset_index(drop=True)
assert len(M)==4431
M.to_csv(OUT/'selective_PAIR_SOURCE_MEASUREMENTS.csv',index=False)

src=[]
for (task,pair),g in M.groupby(['task','unordered_pair_id']):
    if len(g)>=2:
        ys=g.hard_y_i_over_j.to_numpy(float); ps=g.p_i_over_j.to_numpy(float)
        src.append({'task':task,'unordered_pair_id':pair,'n_sources':len(g),
                    'hard_direction_agreement':bool(np.all(ys==ys[0])),
                    'max_pairwise_p_gap':float(ps.max()-ps.min()),
                    'mean_certainty':float(g.certainty_1_minus_H.mean()),
                    'any_U':bool(g.either_U.any())})
S=pd.DataFrame(src)
S.to_csv(OUT/'selective_SOURCE_DISAGREEMENT.csv',index=False)

for col in ['prompt_tokens','completion_tokens','latency_ms','entropy_AB_bits','attempts',
            'prompt_cache_hit_tokens','prompt_cache_miss_tokens']:
    c[col]=pd.to_numeric(c[col],errors='coerce')
arm_audit=(c.groupby('arm',as_index=False)
    .agg(n_calls=('query_id','size'),U_calls=('response_token',lambda x:int((x=='U').sum())),
         mean_entropy=('entropy_AB_bits','mean'),mean_prompt_tokens=('prompt_tokens','mean'),
         mean_latency_ms=('latency_ms','mean'),max_attempts=('attempts','max')))
arm_audit['U_rate']=arm_audit.U_calls/arm_audit.n_calls
arm_audit.to_csv(OUT/'selective_CALL_AUDIT_BY_ARM.csv',index=False)

pair_arm=(M.groupby(['task','arm'],as_index=False)
    .agg(n_pair_source=('unordered_pair_id','size'),mean_certainty=('certainty_1_minus_H','mean'),
         median_certainty=('certainty_1_minus_H','median'),mean_order_gap=('abs_order_probability_gap','mean'),
         p95_order_gap=('abs_order_probability_gap',lambda x:float(np.quantile(x,.95))),
         either_U_rate=('either_U','mean'),both_U_rate=('both_U','mean'),
         hard_order_choice_agreement=('hard_semantic_choice_agree','mean')))
pair_arm.to_csv(OUT/'selective_PAIR_SOURCE_AUDIT_BY_ARM.csv',index=False)

status={
 'status':'PASS_FULL_MEASUREMENT_POSTPROCESS',
 'n_calls':8862,'n_pair_source_measurements':4431,
 'provider_model':'deepseek-flash','system_fingerprint':'aeb56401ca74e127821c4f9126dcb669',
 'all_parse_pass':True,'all_call_keys_unique':True,'all_cache_objects_present':True,
 'response_token_counts':c.response_token.value_counts().to_dict(),
 'U_call_rate':float((c.response_token=='U').mean()),
 'mean_prompt_tokens':float(c.prompt_tokens.mean()),'total_prompt_tokens':int(c.prompt_tokens.sum()),
 'total_completion_tokens':int(c.completion_tokens.sum()),
 'mean_latency_ms':float(c.latency_ms.mean()),'median_latency_ms':float(c.latency_ms.median()),
 'pair_source_mean_certainty':float(M.certainty_1_minus_H.mean()),
 'pair_source_median_certainty':float(M.certainty_1_minus_H.median()),
 'pair_source_either_U_rate':float(M.either_U.mean()),
 'pair_source_both_U_rate':float(M.both_U.mean()),
 'pair_source_mean_order_gap':float(M.abs_order_probability_gap.mean()),
 'pair_source_p95_order_gap':float(np.quantile(M.abs_order_probability_gap,.95)),
 'pair_source_hard_order_agreement':float(M.hard_semantic_choice_agree.mean()),
 'multi_source_pair_count':int(len(S)),
 'multi_source_hard_direction_agreement':float(S.hard_direction_agreement.mean()) if len(S) else None,
 'multi_source_mean_max_p_gap':float(S.max_pairwise_p_gap.mean()) if len(S) else None,
 'calls_complete_sha256':hashlib.sha256(C.read_bytes()).hexdigest(),
 'pair_measurements_sha256':hashlib.sha256((OUT/'selective_PAIR_SOURCE_MEASUREMENTS.csv').read_bytes()).hexdigest(),
 'execution_incident_note':(
   'A foreground tool timeout caused transient duplicate runner processes. Duplicate runners were terminated; '
   'the retained scientific dataset contains exactly one validated cache object per frozen query_id. '
   'Logical retained-result cost is known; exact provider-side duplicate-request billing cannot be reconstructed from local cache alone.')
}
(OUT/'selective_POSTPROCESS_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
print('\nCALL AUDIT BY ARM\n',arm_audit.to_string(index=False))
print('\nPAIR-SOURCE AUDIT BY ARM\n',pair_arm.to_string(index=False))
print('\nSOURCE DISAGREEMENT\n', S.groupby('task').agg(n=('unordered_pair_id','size'),direction_agreement=('hard_direction_agreement','mean'),mean_max_p_gap=('max_pairwise_p_gap','mean'),any_U=('any_U','mean')).to_string() if len(S) else 'none')
