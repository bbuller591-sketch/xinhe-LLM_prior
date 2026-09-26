

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
import pandas as pd, json, hashlib

WS=Path(str(REPRO_ROOT / 'credit_g_m0m3_20260918'))
M=WS/"04_METHOD_FREEZE"; E=WS/"03_EVIDENCE"
SYSTEM="You are a measurement instrument for pairwise variable relevance in consumer-credit repayment risk. Follow the user instruction exactly. Do not call tools, browse the web, or retrieve external information. Do not try to identify or name a benchmark dataset. Your entire response must be exactly one token: A, B, or U."

construct={
"checking_status":"status of the applicant's checking account with the bank",
"duration":"credit duration in months",
"credit_history":"history of compliance with previous or concurrent credit contracts",
"purpose":"purpose for which the credit is needed",
"credit_amount":"credit amount",
"savings_status":"the applicant's savings",
"employment":"duration of employment with the current employer",
"installment_commitment":"credit installments as a percentage of disposable income",
"personal_status":"combined sex and marital-status category",
"other_parties":"whether there is another debtor or guarantor for the credit",
"residence_since":"length of time the applicant has lived at the present residence",
"property_magnitude":"the applicant's most valuable property category",
"age":"age in years",
"other_payment_plans":"installment plans from providers other than the credit-giving bank",
"housing":"type of housing in which the applicant lives",
"existing_credits":"number of credits including the current one at the same bank",
"job":"job category / employment qualification",
"num_dependents":"number of people financially dependent on the applicant",
"own_telephone":"whether a telephone landline is registered in the applicant's name",
"foreign_worker":"foreign-worker status",
}

def hashtext(s): return hashlib.sha256(s.encode()).hexdigest()

def render_noev(a,b):
    return f"""Task:
A lender observes applicant information at or before loan origination.
The outcome is whether the borrower later shows bad repayment performance / non-compliance with the credit contract.

Compare the two applicant variables below using your general pretrained knowledge about consumer-credit risk.
Do not infer a specific named dataset and do not use benchmark-specific feature rankings.

A: {construct[a]}
B: {construct[b]}

Choose:
A = A is more likely to have stronger predictive relevance to bad repayment risk.
B = B is more likely to have stronger predictive relevance to bad repayment risk.
U = the comparison is not directionally meaningful enough to choose A or B.

Output exactly one token: A, B, or U."""

packets={}
for f in ["purpose","installment_commitment","housing"]:
    packets[f]=json.loads((E/f"selective_PACKET_{f}.json").read_text())

def packet_text(f):
    p=packets[f]
    return f"Source {p['source_id']} ({p['evidence_state']}): {p['summary']} Caveat: {p['scope_caveat']}"

def render_selective(a,b):
    return f"""Task:
A lender observes applicant information at or before loan origination.
The outcome is whether the borrower later shows bad repayment performance / non-compliance with the credit contract.

Compare the two applicant variables using ONLY the supplied audited evidence summaries plus ordinary interpretation of the variable definitions.
Do not infer a named benchmark and do not import remembered benchmark-specific feature rankings.

A: {construct[a]}
Evidence for A:
{packet_text(a)}

B: {construct[b]}
Evidence for B:
{packet_text(b)}

Choose:
A = the supplied evidence supports A as having stronger predictive relevance.
B = the supplied evidence supports B as having stronger predictive relevance.
U = the supplied evidence is insufficient, non-comparable, or too conflicted to support a directional comparison.

Output exactly one token: A, B, or U."""

broad=pd.read_csv(M/"global_BROAD_GRAPH_FREEZE.csv")
# deterministic sentinel = six smallest SHA256(pair_id|20260918)
broad["sentinel_hash"]=broad.pair_id.map(lambda x:hashtext(f"{x}|20260918"))
sent=set(broad.sort_values("sentinel_hash").head(6).pair_id)
rows=[]
for _,r in broad.iterrows():
    ca,cb=r.feature_a,r.feature_b
    for order in ["AB","BA"]:
        left,right=(ca,cb) if order=="AB" else (cb,ca)
        reps=[0,1,2] if r.pair_id in sent else [0]
        for rep in reps:
            user=render_noev(left,right)
            rows.append({
                "query_id":f"M12_{r.pair_id}_{order}_R{rep}",
                "measurement_family":"global_SHARED",
                "arm":"NO_RETRIEVED_EVIDENCE",
                "pair_id":r.pair_id,
                "canonical_feature_a":ca,
                "canonical_feature_b":cb,
                "presentation_order":order,
                "presented_A_feature":left,
                "presented_B_feature":right,
                "repeat_index":rep,
                "primary_measurement":rep==0,
                "repeat_instability_sentinel":r.pair_id in sent,
                "system_prompt_sha256":hashtext(SYSTEM),
                "user_prompt_sha256":hashtext(user),
                "user_prompt":user,
            })
m12=pd.DataFrame(rows)
assert len(m12)==144
assert m12[m12.primary_measurement].shape[0]==120
m12.to_csv(M/"global_QUERY_MANIFEST.csv",index=False)

m3g=pd.read_csv(M/"selective_SELECTIVE_GRAPH_FREEZE.csv")
m3g=m3g[m3g.formal_selective_measurement_eligible.astype(bool)].copy()
rows=[]
for _,r in m3g.iterrows():
    ca,cb=r.feature_a,r.feature_b
    for order in ["AB","BA"]:
        left,right=(ca,cb) if order=="AB" else (cb,ca)
        for rep in [0,1,2]:
            user=render_selective(left,right)
            rows.append({
                "query_id":f"selective_{r.pair_id}_{order}_R{rep}",
                "measurement_family":"selective_AUDITED_EVIDENCE",
                "arm":"AUDITED_EXTERNAL_EVIDENCE",
                "pair_id":r.pair_id,
                "canonical_feature_a":ca,
                "canonical_feature_b":cb,
                "presentation_order":order,
                "presented_A_feature":left,
                "presented_B_feature":right,
                "repeat_index":rep,
                "primary_measurement":rep==0,
                "repeat_instability_sentinel":True,
                "system_prompt_sha256":hashtext(SYSTEM),
                "user_prompt_sha256":hashtext(user),
                "user_prompt":user,
            })
selective=pd.DataFrame(rows)
assert len(selective)==18 and selective[selective.primary_measurement].shape[0]==6
selective.to_csv(M/"selective_QUERY_MANIFEST.csv",index=False)

for name,df in [("global",m12),("selective",selective)]:
    joined="\n".join(df.query_id.astype(str)+"|"+df.user_prompt_sha256.astype(str))
    (M/f"{name}_QUERY_MANIFEST.sha256").write_text(hashtext(joined)+"\n")

forbidden=["German Credit","Statlog","OpenML","UCI","South German"]
for name,df in [("global",m12),("selective",selective)]:
    bad=[]
    for i,p in enumerate(df.user_prompt):
        for term in forbidden:
            if term.lower() in p.lower(): bad.append((i,term))
    if bad: raise RuntimeError((name,bad[:10]))

audit={
 "system_prompt_sha256":hashtext(SYSTEM),
 "global_total_calls":len(m12),
 "global_primary_calls":int(m12.primary_measurement.sum()),
 "global_repeat_calls":int((~m12.primary_measurement).sum()),
 "global_sentinel_pairs":sorted(sent),
 "selective_total_calls":len(selective),
 "selective_primary_calls":int(selective.primary_measurement.sum()),
 "selective_repeat_calls":int((~selective.primary_measurement).sum()),
 "selective_pairs":m3g.pair_id.tolist(),
 "forbidden_dataset_identity_terms_absent":True,
}
(M/"QUERY_MANIFEST_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps(audit,indent=2))
