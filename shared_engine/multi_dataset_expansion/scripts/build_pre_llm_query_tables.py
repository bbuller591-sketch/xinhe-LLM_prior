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

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
BUD=ROOT/'07_PRE_LLM_BUDGET'
EV=ROOT/'05_EXTERNAL_EVIDENCE/gene_level_sources'
OUT=ROOT/'08_PRE_LLM_QUERIES'
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

def build(task):
    elig=BUD/f'{task}_PAIR_SOURCE_ELIGIBILITY.csv'
    if not elig.exists():
        print(task,'eligibility missing'); return
    e=pd.read_csv(elig)
    cfg=TASKS[task]
    source_tables={}
    for arm,(prefix,meta) in cfg['sources'].items():
        g=pd.read_csv(EV/f'{prefix}_GENE_EVIDENCE.csv')
        g=g[g.available.fillna(False).astype(bool)].set_index('gene_symbol')
        source_tables[arm]=(g,meta)
    rows=[]
    unique=e[e.pair_source_eligible.fillna(False).astype(bool)].copy()
    unique=unique.sort_values(['arm','unordered_pair'],kind='mergesort')
    for rr in unique.itertuples():
        a0,b0=str(rr.feature_A),str(rr.feature_B)
        g,meta=source_tables[rr.arm]
        assert a0 in g.index and b0 in g.index
        # canonical pair-specific evidence text is hashed independent of AB/BA order
        evidence_packet=json.dumps({
          'task':task,'arm':rr.arm,'unordered_pair':rr.unordered_pair,
          'gene_1':a0,'gene_1_evidence':g.loc[a0].to_dict(),
          'gene_2':b0,'gene_2_evidence':g.loc[b0].to_dict(),
          'source_meta':meta},sort_keys=True,default=str,separators=(',',':'))
        eph=sha(evidence_packet)
        for order,(a,b) in [('AB',(a0,b0)),('BA',(b0,a0))]:
            ea=evidence_record(g.loc[a],rr.arm,meta)
            eb=evidence_record(g.loc[b],rr.arm,meta)
            prompt=(cfg['task_text']+
                f"\n\nGene A: {a}\nEvidence for Gene A:\n{ea}"
                f"\n\nGene B: {b}\nEvidence for Gene B:\n{eb}"
                "\n\nAnswer with exactly one token: A, B, or U.")
            rows.append({
              'query_id':f"{task}__{rr.arm}__{rr.unordered_pair}__{order}",
              'unordered_pair_id':rr.unordered_pair,'arm':rr.arm,'order':order,
              'gene_A':a,'gene_B':b,'prompt_text':prompt,'prompt_sha256':sha(prompt),
              'evidence_packet_sha256':eph,'allowed_tokens':'A|B|U',
              'query_role':'selective_ACTIONABLE_BOUNDARY_CONFUSION',
              'n_outer_folds':int(rr.n_outer_folds),
              'n_fold_selector_k_routes':int(rr.n_fold_selector_k_routes),
              'max_actionable':float(rr.max_actionable)
            })
    out=pd.DataFrame(rows)
    assert out.query_id.nunique()==len(out)
    # exact AB/BA pair-arm symmetry
    assert len(out)%2==0
    for (pair,arm),z in out.groupby(['unordered_pair_id','arm']):
        assert set(z.order)=={'AB','BA'} and len(z)==2
        x=z.set_index('order')
        assert x.loc['AB','gene_A']==x.loc['BA','gene_B']
        assert x.loc['AB','gene_B']==x.loc['BA','gene_A']
        assert x.evidence_packet_sha256.nunique()==1
    out.to_csv(OUT/f'{task}_selective_QUERIES_PREAUTH.csv',index=False)
    with open(OUT/f'{task}_selective_QUERIES_PREAUTH.jsonl','w',encoding='utf-8') as f:
      for d in out.to_dict('records'): f.write(json.dumps(d,ensure_ascii=False)+'\n')
    summary={
      'task':task,'n_queries':len(out),'n_pair_source_cells':len(out)//2,
      'n_unique_pairs':out.unordered_pair_id.nunique(),
      'n_source_arms':out.arm.nunique(),
      'system_prompt':SYSTEM_PROMPT,'system_prompt_sha256':sha(SYSTEM_PROMPT),
      'query_csv_sha256':hashlib.sha256((OUT/f'{task}_selective_QUERIES_PREAUTH.csv').read_bytes()).hexdigest(),
      'execution_authorized':False
    }
    (OUT/f'{task}_QUERY_MANIFEST.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

for t in TASKS: build(t)
