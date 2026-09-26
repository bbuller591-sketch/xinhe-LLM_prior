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
import json, hashlib
import pandas as pd

ROOT=Path(str(REPRO_ROOT / '10_appendix/d04_selective_ablation'))
PKG=Path(str(REPRO_ROOT / 'BREAST_reference_global_selective_COMPLETE_REPRODUCIBILITY_20260919'))
TASK='BREAST_GSE25055_GSE25065'
OUT=ROOT/'02_QUERY_AUDIT'
OUT.mkdir(parents=True,exist_ok=True)

system_prompt='Follow the user\'s measurement instruction exactly. Do not call tools, browse the web, or retrieve external information during this judgment. Output only the requested semantic token.'
oldq=pd.read_csv(PKG/'MEASUREMENTS/selective/selective_QUERIES_PREAUTH.csv',dtype=str,keep_default_na=False)
old_pairs=set(oldq.unordered_pair_id.unique())
union=pd.read_csv(ROOT/'01_MANIFEST/BREAST_CANDIDATE_PAIR_UNION_1834.csv',dtype=str)
missing=union[~union.unordered_pair_id.isin(old_pairs)].copy()

arm_specs={
 'ARM_GSE32646':{
   'evidence_csv':PKG/'DATA/EXTERNAL_EVIDENCE_PROCESSED/BREAST_GSE32646_HER2NEG_PCR_GENE_EVIDENCE.csv',
   'comparator':'pCR versus nCR in HER2-negative pretreatment tumors',
   'design':'Affymetrix GPL570 U133 Plus 2.0 transcriptomic expression; independent Osaka pretreatment biopsy cohort; n=81 (15 pCR, 66 nCR)',
   'limitation':'smaller cohort; paclitaxel followed by FEC'
 },
 'ARM_GSE41998':{
   'evidence_csv':PKG/'DATA/EXTERNAL_EVIDENCE_PROCESSED/BREAST_GSE41998_HER2NEG_PCR_GENE_EVIDENCE.csv',
   'comparator':'pCR versus non-pCR in strict HER2-negative pretreatment tumors',
   'design':'Affymetrix GPL571 U133A 2.0 transcriptomic expression; independent randomized phase-II pretreatment biopsy cohort; n=229 (63 pCR, 166 non-pCR)',
   'limitation':'AC followed by ixabepilone or paclitaxel; treatment-arm heterogeneity differs from the development cohort'
 }
}

evidence={}
for arm,spec in arm_specs.items():
    df=pd.read_csv(spec['evidence_csv'])
    evidence[arm]={str(r.gene_symbol):r for r in df.itertuples(index=False)}

header=("You are comparing two candidate genes for a fixed variable-selection task in HER2-negative human breast cancer. "
"The target is pathological complete response (pCR) versus residual/non-complete pathological response after neoadjuvant chemotherapy. "
"Choose the gene with stronger task-aligned evidence of differential association with this response target, regardless of whether expression is higher in pCR or non-pCR. "
"Do not reward generic breast-cancer importance, prognosis, subtype popularity, pathway popularity, or treatment-response evidence from a materially different setting unless it is represented in the supplied source evidence. "
"Use only the supplied source evidence. If the supplied evidence is insufficient or effectively tied, abstain. Return exactly one token: A, B, or U.")

def fmt_num(x):
    return f'{float(x):.6g}'

def evidence_block(label,gene,arm):
    r=evidence[arm][gene]
    spec=arm_specs[arm]
    return (f"Gene {label}: {gene}\n"
            f"Evidence for Gene {label}:\n"
            f"Source arm: {arm}\n"
            f"Comparator: {spec['comparator']}\n"
            f"Modality/design: {spec['design']}\n"
            f"Identity mapping: current NCBI protein-coding GeneID mapped to this platform\n"
            f"Outcome-positive direction: {str(r.higher_in_positive)}\n"
            f"Standardized mean difference (positive minus negative): {fmt_num(r.standardized_mean_difference)}\n"
            f"Univariate AUC for positive outcome: {fmt_num(r.univariate_auc_positive)}\n"
            f"Absolute AUC distance from 0.5: {fmt_num(r.abs_auc_distance_from_0_5)}\n"
            f"Welch p-value: {fmt_num(r.welch_t_pvalue)}; BH q-value within candidate universe: {fmt_num(r.welch_bh_qvalue_candidate_universe)}\n"
            f"Source sample counts: n={int(r.n)}, positive={int(r.n_positive)}, negative={int(r.n_negative)}\n"
            f"Known limitation: {spec['limitation']}")

def prompt(geneA,geneB,arm):
    return header+'\n\n'+evidence_block('A',geneA,arm)+'\n\n'+evidence_block('B',geneB,arm)+'\n\nAnswer with exactly one token: A, B, or U.'

def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

# Spot-check prompt reconstruction on historical queries.
checks=[]
for idx,row in oldq.groupby('arm').head(10).iterrows():
    p=prompt(str(row.gene_A),str(row.gene_B),str(row.arm))
    checks.append({'query_id':row.query_id,'arm':row.arm,'order':row.order,'match_text':p==row.prompt_text,'match_sha':sha(p)==row.prompt_sha256,'got_sha':sha(p),'ref_sha':row.prompt_sha256})
check_df=pd.DataFrame(checks)
check_df.to_csv(OUT/'PROMPT_RECONSTRUCTION_SPOTCHECK.csv',index=False)
if not (check_df.match_text.all() and check_df.match_sha.all()):
    bad=check_df[~(check_df.match_text & check_df.match_sha)]
    raise SystemExit('PROMPT_RECONSTRUCTION_FAILED '+bad.to_string())

rows=[]
for pair in sorted(missing.unordered_pair_id):
    a,b=pair.split('||',1)
    for arm in ['ARM_GSE32646','ARM_GSE41998']:
        for order,gA,gB in [('AB',a,b),('BA',b,a)]:
            text=prompt(gA,gB,arm)
            ev_payload=json.dumps({'arm':arm,'gene_A':gA,'gene_B':gB,'evidence_A':evidence_block('A',gA,arm),'evidence_B':evidence_block('B',gB,arm)},sort_keys=True,ensure_ascii=False)
            rows.append({
                'query_id':f'{TASK}__{arm}__{pair}__{order}',
                'unordered_pair_id':pair,'arm':arm,'order':order,'gene_A':gA,'gene_B':gB,
                'prompt_text':text,'prompt_sha256':sha(text),'evidence_packet_sha256':sha(ev_payload),
                'allowed_tokens':'A|B|U','query_role':'selective_ABLATION_CANDIDATE_UNIVERSE_COMPLETION',
                'n_outer_folds':'','n_fold_selector_k_routes':'','max_actionable':''
            })
q=pd.DataFrame(rows)
q.to_csv(OUT/'MISSING_selective_ABLATION_QUERIES_PREAUTH.csv',index=False)
manifest={
 'task':TASK,
 'system_prompt':system_prompt,
 'system_prompt_sha256':sha(system_prompt),
 'old_unique_pairs':len(old_pairs),
 'candidate_union_pairs':int(union.unordered_pair_id.nunique()),
 'missing_unique_pairs':int(missing.unordered_pair_id.nunique()),
 'new_queries':int(len(q)),
 'arms':['ARM_GSE32646','ARM_GSE41998'],
 'orders':['AB','BA'],
 'prompt_spotcheck_status':'PASS',
 'query_csv_sha256':hashlib.sha256((OUT/'MISSING_selective_ABLATION_QUERIES_PREAUTH.csv').read_bytes()).hexdigest()
}
(OUT/'MISSING_QUERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
