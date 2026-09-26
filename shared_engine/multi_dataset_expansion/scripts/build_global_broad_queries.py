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
import hashlib,json
import pandas as pd
import numpy as np

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
B=ROOT/'14_global_BROAD_FREEZE'
EV=ROOT/'05_EXTERNAL_EVIDENCE/gene_level_sources'
M3Q=ROOT/'08_PRE_LLM_QUERIES'
OUT=B/'QUERIES'
OUT.mkdir(parents=True,exist_ok=True)

SYSTEM_PROMPT="Follow the user's measurement instruction exactly. Do not call tools, browse the web, or retrieve external information during this judgment. Output only the requested semantic token."

TASKS={
'BREAST_GSE25055_GSE25065':{
 'task_text':(
  "You are comparing two candidate genes for a fixed variable-selection task in HER2-negative human breast cancer. "
  "The target is pathological complete response (pCR) versus residual/non-complete pathological response after neoadjuvant chemotherapy. "
  "Choose the gene with stronger task-aligned evidence of differential association with this response target, regardless of whether expression is higher in pCR or non-pCR. "
  "Do not reward generic breast-cancer importance, prognosis, subtype popularity, pathway popularity, or treatment-response evidence from a materially different setting unless it is represented in the supplied source evidence. "
  "Use only the supplied source evidence. If the supplied evidence is insufficient or effectively tied, abstain. "
  "Return exactly one token: A, B, or U."
 ),
 'sources':{
  'ARM_GSE41998':('BREAST_GSE41998_HER2NEG_PCR',{
    'comparator':'pCR versus non-pCR in strict HER2-negative pretreatment tumors',
    'modality':'Affymetrix GPL571 U133A 2.0 transcriptomic expression',
    'design':'independent randomized phase-II pretreatment biopsy cohort; n=229 (63 pCR, 166 non-pCR)',
    'limitation':'AC followed by ixabepilone or paclitaxel; treatment-arm heterogeneity differs from the development cohort'
  }),
  'ARM_GSE32646':('BREAST_GSE32646_HER2NEG_PCR',{
    'comparator':'pCR versus nCR in HER2-negative pretreatment tumors',
    'modality':'Affymetrix GPL570 U133 Plus 2.0 transcriptomic expression',
    'design':'independent Osaka pretreatment biopsy cohort; n=81 (15 pCR, 66 nCR)',
    'limitation':'smaller cohort; paclitaxel followed by FEC'
  })
 }
},
'SEPSIS_GSE65682':{
 'task_text':(
  "You are comparing two candidate genes for a fixed variable-selection task in human sepsis. "
  "The target is 28-day mortality versus survival using blood transcriptomic measurements obtained at admission/early presentation. "
  "Choose the gene with stronger task-aligned evidence of differential association with 28-day mortality, regardless of whether expression is higher in non-survivors or survivors. "
  "Do not reward generic sepsis importance, severity, pathway popularity, or mortality evidence from a materially different timepoint or biological modality unless it is represented in the supplied source evidence. "
  "Use only the supplied source evidence. If the supplied evidence is insufficient or effectively tied, abstain. "
  "Return exactly one token: A, B, or U."
 ),
 'sources':{
  'ARM_EMTAB4451':('SEPSIS_EMTAB4451_28D_MORTALITY',{
    'comparator':'28-day non-survivor versus survivor',
    'modality':'blood/leukocyte transcriptomic expression; Illumina HumanHT-12 v4 / A-MEXP-2210',
    'design':'independent GAinS CAP-sepsis cohort; expression subset n=106 (52 non-survivors, 54 survivors)',
    'limitation':'CAP-sepsis population; overlapping GAinS derivative cohorts are not treated as separate evidence arms'
  }),
  'ARM_EMTAB7581':('SEPSIS_EMTAB7581_VANISH_28D_MORTALITY',{
    'comparator':'day-28 dead versus alive',
    'modality':'blood transcriptomic expression; Illumina HumanHT-12 v4 / A-MEXP-2210',
    'design':'independent VANISH septic-shock randomized-trial cohort; n=176 (48 dead, 128 alive)',
    'limitation':'narrower septic-shock population; randomized vasopressor/steroid treatments'
  })
 }
}
}

def fnum(x):
    if pd.isna(x): return 'NA'
    try: return f'{float(x):.6g}'
    except: return str(x)

def evidence_record(r,arm,meta):
    return (
       f"Source arm: {arm}\n"
       f"Comparator: {meta['comparator']}\n"
       f"Modality/design: {meta['modality']}; {meta['design']}\n"
       f"Identity mapping: current NCBI protein-coding GeneID mapped to this platform\n"
       f"Outcome-positive direction: {r.higher_in_positive if not pd.isna(r.higher_in_positive) else 'NA'}\n"
       f"Standardized mean difference (positive minus negative): {fnum(r.standardized_mean_difference)}\n"
       f"Univariate AUC for positive outcome: {fnum(r.univariate_auc_positive)}\n"
       f"Absolute AUC distance from 0.5: {fnum(r.abs_auc_distance_from_0_5)}\n"
       f"Welch p-value: {fnum(r.welch_t_pvalue)}; BH q-value within candidate universe: {fnum(r.welch_bh_qvalue_candidate_universe)}\n"
       f"Source sample counts: n={int(r.n)}, positive={int(r.n_positive)}, negative={int(r.n_negative)}\n"
       f"Known limitation: {meta['limitation']}"
    )

def sha(x): return hashlib.sha256(x.encode()).hexdigest()
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

all_manifest=[]
for task,cfg in TASKS.items():
    source_tables={}
    for arm,(prefix,meta) in cfg['sources'].items():
        g=pd.read_csv(EV/f'{prefix}_GENE_EVIDENCE.csv')
        g=g[g.available.fillna(False).astype(bool)].set_index('gene_symbol')
        source_tables[arm]=(g,meta)
    selective=pd.read_csv(M3Q/f'{task}_selective_QUERIES_PREAUTH.csv',dtype=str,keep_default_na=False)
    m3ab=selective[selective.order=='AB'][['arm','unordered_pair_id','gene_A','gene_B']].drop_duplicates()
    m3orient={(str(x.arm),str(x.unordered_pair_id)):(str(x.gene_A),str(x.gene_B)) for x in m3ab.itertuples()}
    rows=[]
    for arm in sorted(cfg['sources']):
        E=pd.read_csv(B/task/arm/'EDGES_D20.csv',dtype=str)
        g,meta=source_tables[arm]
        for rr in E.sort_values('unordered_pair_id',kind='mergesort').itertuples():
            a0,b0=str(rr.gene_A),str(rr.gene_B)
            # If this pair-source cell already exists in the frozen selective cache,
            # inherit its AB orientation exactly so prompt/evidence hashes match.
            if (arm,str(rr.unordered_pair_id)) in m3orient:
                a0,b0=m3orient[(arm,str(rr.unordered_pair_id))]
            if a0 not in g.index or b0 not in g.index: raise RuntimeError('GRAPH_INELIGIBLE_GENE')
            evidence_packet=json.dumps({
              'task':task,'arm':arm,'unordered_pair':rr.unordered_pair_id,
              'gene_1':a0,'gene_1_evidence':g.loc[a0].to_dict(),
              'gene_2':b0,'gene_2_evidence':g.loc[b0].to_dict(),
              'source_meta':meta},sort_keys=True,default=str,separators=(',',':'))
            eph=sha(evidence_packet)
            for order,(a,b) in [('AB',(a0,b0)),('BA',(b0,a0))]:
                prompt=(cfg['task_text']+
                    f"\n\nGene A: {a}\nEvidence for Gene A:\n{evidence_record(g.loc[a],arm,meta)}"
                    f"\n\nGene B: {b}\nEvidence for Gene B:\n{evidence_record(g.loc[b],arm,meta)}"
                    "\n\nAnswer with exactly one token: A, B, or U.")
                rows.append({
                  'query_id':f"{task}__{arm}__{rr.unordered_pair_id}__{order}",
                  'task':task,'unordered_pair_id':rr.unordered_pair_id,'arm':arm,'order':order,
                  'gene_A':a,'gene_B':b,'prompt_text':prompt,'prompt_chars':len(prompt),
                  'prompt_sha256':sha(prompt),'evidence_packet_sha256':eph,
                  'allowed_tokens':'A|B|U','query_role':'global_BROAD_D20'
                })
    d=pd.DataFrame(rows)
    expected=80000 if task.startswith('BREAST') else 76400
    assert len(d)==expected and d.query_id.nunique()==expected
    # Validate AB/BA.
    for (arm,pair),z in d.groupby(['arm','unordered_pair_id']):
        assert len(z)==2 and set(z.order)=={'AB','BA'}
    # Exact overlap scientific-equivalence audit against frozen selective query table.
    common=set(d.query_id)&set(selective.query_id)
    z=d[d.query_id.isin(common)][['query_id','prompt_sha256','evidence_packet_sha256']].merge(
        selective[selective.query_id.isin(common)][['query_id','prompt_sha256','evidence_packet_sha256']],
        on='query_id',suffixes=('_broad','_selective'),validate='one_to_one')
    assert (z.prompt_sha256_broad==z.prompt_sha256_selective).all()
    assert (z.evidence_packet_sha256_broad==z.evidence_packet_sha256_selective).all()
    d['reusable_from_selective']=d.query_id.isin(common)
    # Full prompt table compressed; lightweight hash table CSV.
    pq=OUT/f'{task}_global_BROAD_D20_QUERIES.parquet'
    d.to_parquet(pq,index=False,compression='zstd')
    light=d.drop(columns=['prompt_text'])
    csvp=OUT/f'{task}_global_BROAD_D20_QUERY_MANIFEST.csv'
    light.to_csv(csvp,index=False)
    status={
      'task':task,'n_queries':len(d),'n_pair_source_cells':len(d)//2,
      'n_reusable_queries_from_selective':len(common),'n_new_queries':len(d)-len(common),
      'all_overlap_prompt_hashes_identical':True,'all_overlap_evidence_hashes_identical':True,
      'query_parquet_sha256':sha_file(pq),'query_manifest_csv_sha256':sha_file(csvp),
      'system_prompt_sha256':sha(SYSTEM_PROMPT),'execution_authorized':False
    }
    (OUT/f'{task}_global_BROAD_D20_QUERY_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    all_manifest.append(status)
    print(json.dumps(status,indent=2))

(OUT/'ALL_BROAD_QUERY_STATUS.json').write_text(json.dumps(all_manifest,indent=2),encoding='utf-8')
