#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.special import expit

PKG=Path(__file__).resolve().parents[1]
OUT=PKG/'REPRODUCE/OUTPUT'
OUT.mkdir(parents=True,exist_ok=True)

q=pd.read_csv(PKG/'MEASUREMENTS/selective/selective_QUERIES_PREAUTH.csv',dtype=str,keep_default_na=False)
c=pd.read_parquet(PKG/'MEASUREMENTS/selective/selective_CALLS_COMPLETE.parquet')
ref=pd.read_csv(PKG/'MEASUREMENTS/selective/selective_PAIR_SOURCE_MEASUREMENTS.csv')

need=['query_id','unordered_pair_id','arm','order','gene_A','gene_B','logp_A','logp_B',
      'response_token','semantic_choice_gene','entropy_AB_bits','prompt_sha256','evidence_packet_sha256']
z=q[['query_id','unordered_pair_id','arm','order','gene_A','gene_B','prompt_sha256','evidence_packet_sha256']].merge(
    c[[x for x in need if x in c.columns]],on='query_id',suffixes=('_q','_c'),validate='one_to_one')
assert len(z)==5308 and z.query_id.nunique()==5308
assert (z.prompt_sha256_q==z.prompt_sha256_c).all()
assert (z.evidence_packet_sha256_q==z.evidence_packet_sha256_c).all()

rows=[]
for (arm,pair),g in z.groupby(['arm_q','unordered_pair_id_q']):
    assert len(g)==2 and set(g.order_q)=={'AB','BA'}
    ab=g[g.order_q=='AB'].iloc[0]
    ba=g[g.order_q=='BA'].iloc[0]
    assert ab.gene_A_q==ba.gene_B_q and ab.gene_B_q==ba.gene_A_q
    lab=float(ab.logp_A)-float(ab.logp_B)
    lba=float(ba.logp_A)-float(ba.logp_B)
    ell=(lab-lba)/2.0
    p=float(expit(ell))
    H=-(p*math.log(p,2)+(1-p)*math.log(1-p,2)) if 0<p<1 else 0.0
    rows.append({
      'arm':arm,'unordered_pair_id':pair,
      'gene_i':ab.gene_A_q,'gene_j':ab.gene_B_q,
      'symmetrized_logit_i_vs_j':ell,'p_i_over_j':p,
      'hard_y_i_over_j':1.0 if p>0.5 else (0.0 if p<0.5 else 0.5),
      'H_AB_bits':H,'certainty_1_minus_H':1-H
    })
got=pd.DataFrame(rows)
cmp=got.merge(ref[['arm','unordered_pair_id','gene_i','gene_j','symmetrized_logit_i_vs_j',
                   'p_i_over_j','hard_y_i_over_j','H_AB_bits','certainty_1_minus_H']],
              on=['arm','unordered_pair_id','gene_i','gene_j'],suffixes=('_repro','_ref'),validate='one_to_one')
metrics=['symmetrized_logit_i_vs_j','p_i_over_j','hard_y_i_over_j','H_AB_bits','certainty_1_minus_H']
diffs={m:float(np.max(np.abs(cmp[m+'_repro']-cmp[m+'_ref']))) for m in metrics}
ok=bool(len(cmp)==2654 and max(diffs.values())<1e-12)
got.to_csv(OUT/'REPRODUCED_selective_PAIR_SOURCE_MEASUREMENTS.csv',index=False)
status={'status':'PASS' if ok else 'FAIL','n_pair_source_cells':len(cmp),'max_abs_differences':diffs}
(OUT/'selective_MEASUREMENT_REPRODUCTION_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
if not ok: raise SystemExit(2)
