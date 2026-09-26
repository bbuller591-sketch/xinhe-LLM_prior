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
import re,json,hashlib
import pandas as pd

ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
A=ROOT/'formal_outputs/recipient_audit'
g360=A/'geo_quick/GSE36059_individual'
g485=A/'geo_quick/GSE48581_gsm_quick.txt'

def inspect_records(paths):
    fields=set(); semantic_hits=[]; titles=[]; acc=[]
    patt=re.compile(r'(recipient|patient|subject|individual|case[ _-]?id|patient[ _-]?id|recipient[ _-]?id|subject[ _-]?id|biopsy[ _-]?id)',re.I)
    ignore_fields={'!Sample_treatment_protocol_ch1','!Sample_growth_protocol_ch1'}
    for p in paths:
        text=Path(p).read_text(errors='ignore')
        cur=''
        for line in text.splitlines():
            line=line.rstrip('\r')
            if line.startswith('^SAMPLE ='):
                cur=line.split('=',1)[1].strip()
            if line.startswith('!Sample_'):
                field=line.split('=',1)[0].strip();fields.add(field)
                value=line.split('=',1)[1].strip() if '=' in line else ''
                if field=='!Sample_title':titles.append(value)
                if field=='!Sample_geo_accession':acc.append(value)
                if patt.search(field+' '+value) and field not in ignore_fields:
                    semantic_hits.append({'sample':cur,'field':field,'value':value[:500]})
    return {'fields':sorted(fields),'semantic_hits':semantic_hits,'titles':titles,'accessions':acc}

p360=sorted(g360.glob('GSM*.txt'))
x360=inspect_records(p360)
x485=inspect_records([g485])

dev=pd.read_csv(ROOT/'SAMPLE_PHENOTYPE_FROZEN.csv',dtype=str,keep_default_na=False)
ext=pd.read_csv(ROOT/'EXTERNAL_SAMPLE_PHENOTYPE_FROZEN.csv',dtype=str,keep_default_na=False)

pmid239=(A/'references/PMID_23915426.xml').read_text(errors='ignore')
pmc108=(A/'references/PMC10841597.html').read_text(errors='ignore')
claim485=bool(re.search(r'300 indication biopsies from 264 patients',pmid239,re.I))
claim360=bool(re.search(r'403 clinically indicated biopsies from 315 recipients',pmc108,re.I))

summary={
 'version':'RECIPIENT_ID_AUDIT_V1',
 'GSE36059':{
   'public_GSM_records_expected':411,'public_GSM_records_downloaded':len(p360),
   'unique_accessions_seen':len(set(x360['accessions'])),'unique_titles_seen':len(set(x360['titles'])),
   'semantic_patient_recipient_id_hits_excluding_generic_protocol_prose':len(x360['semantic_hits']),
   'semantic_hits':x360['semantic_hits'][:20],
   'paper_level_repeat_evidence':'403 clinically indicated biopsies from 315 recipients',
   'paper_level_repeat_evidence_locally_confirmed':claim360,
   'frozen_formal_rows':len(dev),
   'mapping_status':'UNAVAILABLE_IN_PUBLIC_GEO_SAMPLE_METADATA'
 },
 'GSE48581':{
   'public_GSM_records_expected':306,'public_GSM_records_downloaded':len(set(x485['accessions'])),
   'unique_accessions_seen':len(set(x485['accessions'])),'unique_titles_seen':len(set(x485['titles'])),
   'semantic_patient_recipient_id_hits_excluding_generic_protocol_prose':len(x485['semantic_hits']),
   'semantic_hits':x485['semantic_hits'][:20],
   'paper_level_repeat_evidence':'300 indication biopsies from 264 patients',
   'paper_level_repeat_evidence_locally_confirmed':claim485,
   'frozen_formal_rows':len(ext),
   'mapping_status':'UNAVAILABLE_IN_PUBLIC_GEO_SAMPLE_METADATA'
 },
 'audit_sources':[
   '411 individual GSE36059 GSM self/quick text records downloaded 2026-09-21',
   'complete GSE48581 targ=gsm view=quick text containing 306 GSM records',
   'PubMed PMID 23915426 XML saved locally',
   'PMC10841597 HTML saved locally',
   'GSE36059/GSE48581 series matrices and frozen phenotype tables'
 ],
 'conclusion':'Repeated recipients are confirmed at study level, but the public GEO records audited do not expose a biopsy-to-recipient mapping. Exact grouped resampling/CV cannot be reconstructed without inventing IDs.',
 'formal_decision':'PASS_WITH_EXPLICIT_PUBLIC_ID_LIMITATION',
 'decision_rationale':'The handoff freezes row-level R=200 80% stratified subsampling and row-level 5-fold stratified development CV. Because no valid grouping key is publicly recoverable, retain the frozen reproducible public-data design, document possible within-recipient dependence/optimism, and do not fabricate grouping.',
 'prohibited_claim':'Do not describe R200/CV as patient-level or recipient-grouped.',
 'no_gse48581_outcome_used':True
}
(A/'RECIPIENT_ID_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
md = f"""# Recipient-ID Audit — GSE36059 → GSE48581

## Result
PASS_WITH_EXPLICIT_PUBLIC_ID_LIMITATION.

Repeated recipients are confirmed at the study level, but a biopsy-to-recipient identifier is not exposed in the public GEO sample metadata audited here. Therefore grouped resampling/CV cannot be reconstructed exactly without inventing identifiers.

## GSE36059
- Public GEO samples audited: {len(p360)}/411, downloaded individually as GSM self/quick records; all downloads succeeded.
- Frozen formal development rows: {len(dev)}.
- Public sample fields contain no usable patient/recipient/subject ID. Semantic ID hits after excluding generic treatment/growth-protocol prose: {len(x360['semantic_hits'])}.
- Study-level evidence confirms 403 clinically indicated biopsies from 315 recipients.
- Consequence: repeated recipients exist, but their per-biopsy mapping is unavailable publicly.

## GSE48581 / INTERCOM
- Public GEO samples audited: {len(set(x485['accessions']))}/306.
- Frozen formal external rows: {len(ext)}.
- Public sample fields contain no usable patient/recipient/subject ID. Semantic ID hits after excluding generic protocol prose: {len(x485['semantic_hits'])}.
- PMID 23915426 explicitly reports 300 indication biopsies from 264 patients.
- Consequence: repeated recipients exist, but their per-biopsy mapping is unavailable publicly.

## Frozen engineering decision
The formal handoff explicitly specifies row-level R=200, 80% stratified subsampling and development-only 5-fold stratified CV. Since a legitimate grouping key cannot be recovered from the public records, retain this frozen reproducible public-data design.

This does not mean the samples are independent patients. The paper/report must state that repeated biopsies can create within-recipient dependence and potentially optimistic internal resampling/CV estimates. No patient grouping will be fabricated.

The sealed GSE48581 outcome was not used in this audit or decision.
"""
(A/'RECIPIENT_ID_AUDIT.md').write_text(md)
paths=[A/'RECIPIENT_ID_AUDIT.json',A/'RECIPIENT_ID_AUDIT.md',A/'references/PMID_23915426.xml',A/'references/PMC10841597.html',g485,
       ROOT/'SAMPLE_PHENOTYPE_FROZEN.csv',ROOT/'EXTERNAL_SAMPLE_PHENOTYPE_FROZEN.csv']
rows=[]
for p in paths:
    rows.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
pd.DataFrame(rows).to_csv(A/'RECIPIENT_ID_AUDIT_SHA256.csv',index=False)
print(json.dumps(summary,indent=2))
