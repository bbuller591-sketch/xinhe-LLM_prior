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
import pandas as pd, numpy as np, json, hashlib, math

PKG=Path(str(REPRO_ROOT / 'GSE272769_M3_TOP30_TOP50_REPRODUCIBLE_20260919'))
WS=Path(str(REPRO_ROOT / '02_gse272769_sepsis/gse272769'))
OUT=Path(str(REPRO_ROOT / '08_diagnostics/q3_q4/06_q2_q5_completion/artifacts/GSE272769_Q2_MATCHED'))
OUT.mkdir(parents=True,exist_ok=True)
M=pd.read_csv(PKG/'03_MEASUREMENT/selective_PAIR_SOURCE_MEASUREMENTS_K50.csv')
feat=pd.read_csv(PKG/'01_FROZEN_DATA/features_p1500.csv')
genes=feat.gene_symbol.astype(str).tolist()

# Reconstruct exact Selective pair-source occurrence slots for ElasticNet k=50.
occ=[]
for fold in range(1,6):
    q=pd.read_csv(PKG/f'04_ROUTING/fold{fold}_ELASTICNET/K50_PAIR_CONFUSION.csv')
    q=q[q.pair_type=='ACTIONABLE_BOUNDARY_CONFUSION'].copy()
    for rr in q.itertuples():
        pair='||'.join(sorted([str(rr.feature_A),str(rr.feature_B)]))
        z=M[M.unordered_pair_id.eq(pair)]
        m=len(z)
        if m==0: continue
        for arm in z.arm.astype(str):
            occ.append({'selective_pair':pair,'arm':arm,'fold':fold,
                        'Adiv':float(rr.actionable_boundary_score)/m})
OCC=pd.DataFrame(occ)
slots=(OCC.groupby(['selective_pair','arm'],as_index=False)
       .agg(n_fold_occurrences=('fold','size'),
            folds=('fold',lambda x:'|'.join(map(str,sorted(x)))),
            Adivs=('Adiv',lambda x:'|'.join(f'{v:.17g}' for v in x))))
assert len(slots)==402 and len(OCC)==409

# Full evidence tables and formal prompt metadata.
SRC={
 'ARM_GAINS_VALIDATION':('EMTAB4451_28D_GENE_EVIDENCE.csv',
    '28-day non-survivor versus survivor',
    'blood/leukocyte transcriptomic expression; Illumina HumanHT-12 v4; independent GAinS validation CAP-sepsis cohort; n=106 (52 non-survivors, 54 survivors)',
    'CAP-sepsis population; E-MTAB-4421 is same study family and is not counted separately'),
 'ARM_VANISH':('EMTAB7581_28D_GENE_EVIDENCE.csv',
    'day-28 dead versus alive',
    'blood transcriptomic expression; Illumina HumanHT-12 v4; independent VANISH septic-shock randomized-trial cohort; n=176 (48 dead, 128 alive)',
    'narrower septic-shock population; randomized vasopressor/steroid treatment context'),
 'ARM_GSE95233_D01':('GSE95233_D01_28D_GENE_EVIDENCE.csv',
    'day-28 non-survivor versus survivor',
    'whole-blood transcriptomic expression; Affymetrix GPL570; independent French septic-shock cohort; admission D01 only, n=51 (17 non-survivors, 34 survivors)',
    'small cohort and narrower septic-shock population; D2/D3 repeats excluded')
}
EV={a:pd.read_csv(WS/'evidence'/f).set_index('gene_symbol') for a,(f,_,_,_) in SRC.items()}
TASK=("You are comparing two candidate genes for a fixed variable-selection task in adult human sepsis. "
"The target is 30-day mortality versus survival using whole-blood transcriptomic measurements obtained within 24 hours of ICU admission. "
"Choose the gene with stronger task-aligned evidence of differential association with short-term 28/30-day mortality, regardless of whether expression is higher in non-survivors or survivors. "
"Do not reward generic sepsis importance, severity, pathway popularity, prognosis at an unspecified horizon, or evidence from a materially different timepoint, population, or biological modality unless represented in the supplied source evidence. "
"Use only the supplied source evidence. If the supplied evidence is insufficient or effectively tied, abstain. Return exactly one token: A, B, or U.")
def fmt(x):
    try:return f'{float(x):.6g}'
    except:return 'NA'
def block(g,arm):
    r=EV[arm].loc[g]; _,comp,design,lim=SRC[arm]
    return (f"Source arm: {arm}\nComparator: {comp}\nModality/design: {design}\n"
            f"Identity mapping: stable NCBI GeneID where available\n"
            f"Outcome-positive direction: {r.higher_in_positive}\n"
            f"Standardized mean difference (positive minus negative): {fmt(r.standardized_mean_difference)}\n"
            f"Univariate AUC for positive outcome: {fmt(r.univariate_auc_positive)}\n"
            f"Absolute AUC distance from 0.5: {fmt(r.abs_auc_distance_from_0_5)}\n"
            f"Welch p-value: {fmt(r.welch_t_pvalue)}\n"
            f"Source sample counts: n={int(r.n)}, positive={int(r.n_positive)}, negative={int(r.n_negative)}\n"
            f"Known limitation: {lim}")
def sha(x): return hashlib.sha256(x.encode()).hexdigest()

# Exclude any pair-source identity already used by the target Selective arm.
banned=set(zip(M.unordered_pair_id.astype(str),M.arm.astype(str)))
sampled={}
for arm,g in slots.groupby('arm'):
    n=len(g)
    avail=[x for x in genes if x in EV[arm].index and bool(EV[arm].loc[x,'available'])]
    rng=np.random.default_rng(np.random.SeedSequence([2026092211, sum(map(ord,arm))]))
    chosen=[];seen=set();tries=0
    while len(chosen)<n:
        a,b=rng.choice(avail,size=2,replace=False)
        p='||'.join(sorted([str(a),str(b)])); tries+=1
        if p in seen or (p,arm) in banned: continue
        seen.add(p); chosen.append(p)
        if tries>1000000: raise RuntimeError('sampling failed')
    sampled[arm]=chosen

maps=[]
for arm,g in slots.groupby('arm',sort=True):
    gg=g.sort_values(['selective_pair']).reset_index(drop=True)
    for rr,pair in zip(gg.itertuples(),sampled[arm]):
        a,b=pair.split('||')
        maps.append({'arm':arm,'selective_pair_slot':rr.selective_pair,'global_pair':pair,
                     'gene_i':a,'gene_j':b,'folds':rr.folds,'Adivs':rr.Adivs,
                     'n_fold_occurrences':int(rr.n_fold_occurrences)})
MAP=pd.DataFrame(maps)
assert len(MAP)==402 and MAP.global_pair.nunique()>=228
MAP.to_csv(OUT/'GLOBAL_SLOT_MAPPING.csv',index=False)

qs=[]
for rr in MAP.itertuples():
    arm=rr.arm; a0,b0=rr.gene_i,rr.gene_j
    packet={'arm':arm,'pair':rr.global_pair,'A':EV[arm].loc[a0].to_dict(),
            'B':EV[arm].loc[b0].to_dict(),'formal_meta':SRC[arm][1:]}
    ph=sha(json.dumps(packet,sort_keys=True,default=str))
    for order,(a,b) in [('AB',(a0,b0)),('BA',(b0,a0))]:
        prompt=(TASK+f"\n\nGene A: {a}\nEvidence for Gene A:\n{block(a,arm)}"
                f"\n\nGene B: {b}\nEvidence for Gene B:\n{block(b,arm)}"
                "\n\nAnswer with exactly one token: A, B, or U.")
        qs.append({'query_id':f'SEPSIS_GSE272769__Q2_GLOBAL_MATCHED__{arm}__{rr.global_pair}__{order}',
                   'unordered_pair_id':rr.global_pair,'arm':arm,'order':order,
                   'gene_A':a,'gene_B':b,'prompt_text':prompt,'prompt_sha256':sha(prompt),
                   'evidence_packet_sha256':ph,'allowed_tokens':'A|B|U','query_role':'Q2_MATCHED_GLOBAL'})
Q=pd.DataFrame(qs); assert len(Q)==804
Q.to_csv(OUT/'Q2_GLOBAL_MATCHED_QUERIES.csv',index=False)

# Use the prior empirical token calibration for a transparent pre-run estimate.
cal=json.load(open(WS/'pre_llm/TOKEN_COST_ESTIMATE.json'))['calibration']
est=cal['linear_intercept']+cal['linear_slope']*Q.prompt_text.str.len()
est=np.maximum(est,1)
base=json.load(open(WS/'pre_llm/TOKEN_COST_ESTIMATE.json'))
peak_per_prompt=(base['cost_peak_all_cache_miss_usd']-0)/base['conservative_prompt_tokens_2rmse_total']
# This is deliberately conservative: scale the old all-miss estimate by token ratio.
old=json.load(open(WS/'pre_llm/TOKEN_COST_ESTIMATE.json'))
cost_peak=old['cost_peak_all_cache_miss_usd']*(float(est.sum())/old['conservative_prompt_tokens_2rmse_total'])
cost_off=old['cost_off_peak_all_cache_miss_usd']*(float(est.sum())/old['conservative_prompt_tokens_2rmse_total'])
summary={'status':'READY_FOR_Q2_MATCHED_GLOBAL_MEASUREMENT','dataset':'GSE272769','selector':'ElasticNet','k':50,
         'selective_unique_pair_source_budget':len(slots),'selective_pair_source_fold_occurrences':len(OCC),
         'new_queries_ab_ba':len(Q),'by_arm_pair_source':MAP.arm.value_counts().astype(int).to_dict(),
         'by_arm_calls':Q.arm.value_counts().astype(int).to_dict(),
         'estimated_prompt_tokens':int(est.sum()),'estimated_cost_off_peak_usd':float(cost_off),
         'estimated_cost_peak_usd':float(cost_peak),
         'pair_selection':'data-confusion-independent deterministic RNG over all source-eligible frozen candidate pairs; excludes target Selective pair-source identities',
         'slot_matching':'preserves source arm, unique measurement count, fold-occurrence pattern, and per-fold actionable-weight slots; semantic pair identities replaced globally',
         'new_llm_calls_required':True}
(OUT/'PRE_RUN_COST_AND_DESIGN.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
