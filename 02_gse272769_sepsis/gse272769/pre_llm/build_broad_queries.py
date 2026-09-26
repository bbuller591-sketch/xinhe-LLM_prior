import pandas as pd,json,hashlib
from pathlib import Path
B=Path('gse272769'); G=B/'pre_llm/broad_graphs'; EV=B/'evidence'; OUT=B/'pre_llm';
S={
'ARM_MARS_GSE65682':('GSE65682_MARS_28D_GENE_EVIDENCE.csv',{'comparator':'28-day nonsurvivor vs survivor','modality':'whole-blood leukocyte transcriptomic expression; Affymetrix HTA 2.0','design':'independent MARS ICU sepsis cohort; n=479 (114 nonsurvivors,365 survivors), within first 24h ICU','limitation':'28-day rather than 30-day; already-completed project dataset used only as evidence'}),
'ARM_VANISH_EMTAB7581':('EMTAB7581_28D_GENE_EVIDENCE.csv',{'comparator':'day-28 dead vs alive','modality':'blood transcriptomic expression; Illumina HumanHT-12 v4','design':'independent VANISH septic-shock RCT; n=176 (48 dead,128 alive), enrollment blood','limitation':'narrower septic-shock population; randomized vasopressor/steroid context'}),
'ARM_GSE95233_D01':('GSE95233_D01_28D_GENE_EVIDENCE.csv',{'comparator':'day-28 nonsurvivor vs survivor','modality':'whole-blood transcriptomic expression; Affymetrix GPL570','design':'independent French septic-shock cohort; admission-only n=51 (17 nonsurvivors,34 survivors)','limitation':'small cohort; narrower septic-shock population; repeats excluded'})}
T={a:(pd.read_csv(EV/f).set_index('gene_symbol'),m) for a,(f,m) in S.items()}; TASK="You are comparing two candidate genes for a fixed variable-selection task in adult human sepsis. The target is 30-day mortality versus survival using whole-blood transcriptomic measurements obtained within 24 hours of ICU admission. Choose the gene with stronger task-aligned evidence of differential association with short-term (28/30-day) mortality, regardless of direction. Do not reward generic sepsis importance, severity, pathway popularity, or evidence from a materially different timepoint/modality. Use only the supplied source evidence. If insufficient or effectively tied, abstain. Return exactly one token: A, B, or U.";SYSTEM="Follow the user's measurement instruction exactly. Do not call tools, browse the web, or retrieve external information during this judgment. Output only the requested semantic token."
def f(x):
 try:return f'{float(x):.6g}'
 except:return 'NA'
def rec(r,a,m):return f"Source arm: {a}\nComparator: {m['comparator']}\nModality/design: {m['modality']}; {m['design']}\nHigher in nonsurvivors: {r.higher_in_positive}\nStandardized mean difference: {f(r.standardized_mean_difference)}\nUnivariate AUC for nonsurvival: {f(r.univariate_auc_positive)}\nAbsolute AUC distance from 0.5: {f(r.abs_auc_distance_from_0_5)}\nWelch p-value: {f(r.welch_t_pvalue)}; BH q-value: {f(r.welch_bh_qvalue_candidate_universe)}\nSource counts: n={int(r.n)}, nonsurvivor={int(r.n_positive)}, survivor={int(r.n_negative)}\nKnown limitation: {m['limitation']}"
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
rows=[]
for arm,(d,m) in T.items():
 e=pd.read_csv(G/arm/'EDGES_D20.csv')
 for rr in e.itertuples():
  a0,b0=rr.gene_A,rr.gene_B; packet=json.dumps({'arm':arm,'pair':rr.unordered_pair_id,'A':d.loc[a0].to_dict(),'B':d.loc[b0].to_dict(),'meta':m},sort_keys=True,default=str);ph=sha(packet)
  for order,(a,b) in [('AB',(a0,b0)),('BA',(b0,a0))]:
   p=TASK+f"\n\nGene A: {a}\nEvidence for Gene A:\n{rec(d.loc[a],arm,m)}\n\nGene B: {b}\nEvidence for Gene B:\n{rec(d.loc[b],arm,m)}\n\nAnswer with exactly one token: A, B, or U."
   rows.append({'query_id':f'SEPSIS_GSE272769__BROAD_D20__{arm}__{rr.unordered_pair_id}__{order}','arm':arm,'unordered_pair_id':rr.unordered_pair_id,'order':order,'gene_A':a,'gene_B':b,'prompt_text':p,'prompt_sha256':sha(p),'evidence_packet_sha256':ph,'allowed_tokens':'A|B|U','query_role':'global_BROAD_D20'})
q=pd.DataFrame(rows);fp=OUT/'SEPSIS_GSE272769_global_BROAD_D20_PREAUTH.csv';q.to_csv(fp,index=False);man={'task':'SEPSIS_GSE272769','scope':'global_PRIMARY_BROAD_D20','execution_authorized':False,'n_queries':len(q),'n_edges':len(q)//2,'by_arm':q.groupby('arm').size().astype(int).to_dict(),'system_prompt':SYSTEM,'system_prompt_sha256':sha(SYSTEM),'query_csv_sha256':hashlib.sha256(fp.read_bytes()).hexdigest(),'d10_zero_extra_calls':True};json.dump(man,open(OUT/'SEPSIS_GSE272769_global_BROAD_MANIFEST.json','w'),indent=2);print(json.dumps(man,indent=2))
