import csv,glob,math
rows=[]
for f in glob.glob('gse272769/tmp_k40_90_llm/measurement/summaries/CALLS_COMPLETE_*.csv'):
    with open(f) as h: rows.extend(csv.DictReader(h))
assert len(rows)==19676
by={}
for r in rows: by.setdefault((r['arm'],r['unordered_pair_id']),{})[r['order']]=r
fields=['arm','unordered_pair_id','gene_i','gene_j','symmetrized_logit_i_vs_j','p_i_over_j','hard_y_i_over_j','H_AB_bits','certainty_1_minus_H','p_i_AB_order','p_i_BA_order','abs_order_probability_gap','either_U','both_U','hard_semantic_choice_agree']
new=[]
for (arm,pair),d in sorted(by.items()):
    assert set(d)=={'AB','BA'}
    ab,ba=d['AB'],d['BA']; gi,gj=pair.split('||',1)
    lab=float(ab['logp_A'])-float(ab['logp_B']); lba=float(ba['logp_A'])-float(ba['logp_B']); z=(lab-lba)/2
    p=1/(1+math.exp(-z)) if abs(z)<700 else float(z>0)
    H=0 if p in (0,1) else -(p*math.log(p,2)+(1-p)*math.log(1-p,2))
    pab=float(ab['pA_vs_B']); pba=1-float(ba['pA_vs_B'])
    new.append(dict(arm=arm,unordered_pair_id=pair,gene_i=gi,gene_j=gj,symmetrized_logit_i_vs_j=z,p_i_over_j=p,hard_y_i_over_j=int(p>=.5),H_AB_bits=H,certainty_1_minus_H=1-H,p_i_AB_order=pab,p_i_BA_order=pba,abs_order_probability_gap=abs(pab-pba),either_U=(ab['response_token']=='U' or ba['response_token']=='U'),both_U=(ab['response_token']=='U' and ba['response_token']=='U'),hard_semantic_choice_agree=(ab['semantic_choice_gene']==ba['semantic_choice_gene'])))
out='gse272769/tmp_k40_90_llm/measurement/selective_PAIR_SOURCE_MEASUREMENTS_NEW.csv'
with open(out,'w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(new)
merged={}
for f in ['gse272769/measurement/postprocess/selective_PAIR_SOURCE_MEASUREMENTS.csv',out]:
    with open(f) as h:
        for r in csv.DictReader(h): merged.setdefault((r['arm'],r['unordered_pair_id']),r)
mout='gse272769/tmp_k40_90_llm/measurement/selective_PAIR_SOURCE_MEASUREMENTS_MERGED.csv'
with open(mout,'w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(merged.values())
print(len(rows),len(new),len(merged))
