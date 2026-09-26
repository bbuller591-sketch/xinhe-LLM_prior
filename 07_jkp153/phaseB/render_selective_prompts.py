#!/usr/bin/env python3
from __future__ import annotations

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
import csv, hashlib, json
from pathlib import Path
import pandas as pd

ROOT=Path(str(REPRO_ROOT))
OUT=ROOT/'finance_llm_prior/experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
FREEZE=ROOT/'JKP153_FORMAL_MEASUREMENT_DESIGN_FREEZE_V0_4_20260913_201725'
ARMS=('DQ','DQL'); ORDERS=('AB','BA'); REPEATS=(1,2,3)
QUESTION=("Compare the two displayed factors. Which factor has stronger evidence that, after the already-applied "
"published-direction orientation, its long-run CAPM alpha is persistently positive, rather than being mainly "
"attributable to sampling noise, data mining, fragile implementation, or lack of external transportability?")
SYSTEM=("You are a careful finance research measurement assistant. Do not use tools or web access. "
"Your entire response must be exactly one token: A, B, T, or U.")
COMMON=("Use only the evidence displayed below together with general methodological principles. "
"Do not use remembered factor-specific returns, t-statistics, replication outcomes, rankings, sign conventions, "
"publication outcomes, or other factor-specific facts that are not displayed. Do not browse the web or call external tools. "
"Missing evidence is not negative evidence. Prefer directly applicable evidence, independent support, robustness to "
"implementation choices, and external transportability.\n\n"
"When quantitative return/alpha/t-statistic fields are displayed, they have already been oriented to the factor's "
"published direction. Positive displayed values support persistent positive published-direction CAPM alpha; negative "
"displayed values count against that target. Do not flip or reinterpret the displayed sign from memory.\n\n"
"When audited literature evidence is displayed, preserve its stated scope and guards. HXZ and Chen-Zimmermann are one "
"R dimension; McLean-Pontiff's 58% post-publication decline is study-level context; GHZ multivariate nonselection is not "
"zero univariate predictability; absence of eligible literature evidence is not negative evidence.\n\n"
"Output A if Factor A has stronger evidence, B if Factor B has stronger evidence, T if the displayed evidence supports "
"approximately a tie/no meaningful directional distinction, and U if the displayed evidence is insufficient or not "
"applicable enough to make the comparison.")

def htext(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

def load_csv(path,key):
    with open(path,newline='',encoding='utf-8-sig') as f:
        return {r[key].strip():r for r in csv.DictReader(f)}

def load_packets(path):
    out={}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            o=json.loads(line); out[o['factor']['factor_id']]=o
    return out

def factor_base(m):
    d=m.get('canonical_definition','').strip()
    if not d or m.get('definition_status','').startswith('MISSING'):
        raise ValueError(f"missing frozen canonical_definition for {m['factor_id']}")
    return f"Factor ID: {m['factor_id']}\nFactor name: {m['factor_name']}\nTheme: {m['theme']}\nConstruction/definition: {d}"

def qblock(q):
    if q.get('dq_status','').strip().startswith('BLOCKED'):
        raise ValueError(f"DQ blocked for {q['factor_id']}")
    fields=['alpha_point_estimate_full','ols_t_full','hac12_t_full']
    if any(q.get(x,'').strip()=='' for x in fields):
        raise ValueError(f"DQ full fields missing for {q['factor_id']}")
    return ("World-ex-US quantitative evidence (observation-date truncated to the prediction origin):\n"
      f"- full-sample oriented external non-US alpha: {q['alpha_point_estimate_full']}\n"
      f"- full-sample oriented OLS t-stat: {q['ols_t_full']}\n"
      f"- full-sample oriented HAC12 t-stat: {q['hac12_t_full']}\n"
      f"- post-original-sample oriented alpha: {q['alpha_point_estimate_post_original'] or 'NA'}\n"
      f"- post-original-sample oriented OLS t-stat: {q['ols_t_post_original'] or 'NA'}\n"
      f"- post-original-sample oriented HAC12 t-stat: {q['hac12_t_post_original'] or 'NA'}\n"
      f"- post-original stats available: {q['post_original_stats_available']}\n"
      f"- quantitative information cutoff: {q['observation_date_cutoff']}\n"
      f"- data-vintage status: {q['vintage_status']}")

def lblock(p):
    if p['llm_measurement_status']=='ABSTAIN_NO_APPROVED_ELIGIBLE_EVIDENCE':
        return ("Audited literature evidence:\n"
                "- No approved eligible literature evidence under the frozen protocol for this origin.\n"
                "- Absence of evidence is not negative evidence.")
    lines=['Audited literature evidence:']
    for e in p['evidence']:
        lines += [
          f"[Dimension {e['dimension']} | {e['evidence_unit_id']}]",
          f"Construction mapping: {e.get('construction_mapping','')}",
          f"Evidence summary: {e.get('evidence_summary','')}",
          f"Audited interpretation: {e.get('audited_interpretation','')}",
          f"Dimension guard: {e.get('dimension_guard','')}",
        ]
        for s in e.get('sources',[]):
            lines.append("Source provenance: "+" | ".join(str(s.get(k,'')) for k in ('source_ids','titles','pages','version_id','first_public_date')))
    return '\n'.join(lines)

def build_factor(fid,arm,meta,dq,pack):
    parts=[factor_base(meta[fid]),qblock(dq[fid])]
    if arm=='DQL': parts.append(lblock(pack[fid]))
    return '\n'.join(parts)

def render_year(year):
    graph=pd.read_csv(OUT/f'QUERY_GRAPH_UNION_H085_{year}.csv')
    meta=load_csv(FREEZE/'templates/D0_FACTOR_METADATA_REGISTRY.csv','factor_id')
    dq=load_csv(FREEZE/f'outputs/{year}/DQ_INPUT_{year}.csv','factor_id')
    pack=load_packets(FREEZE/f'inputs/FACTOR_EVIDENCE_PACKETS_{year}.jsonl')
    pdir=OUT/f'prompts/{year}'; pdir.mkdir(parents=True,exist_ok=True)
    base_rows=[]
    for e in graph.to_dict('records'):
        fi,fj=e['factor_i'],e['factor_j']
        for arm in ARMS:
            if arm=='DQL' and not bool(e['DQL_call_required']): continue
            for order in ORDERS:
                A,B=(fi,fj) if order=='AB' else (fj,fi)
                fa=build_factor(A,arm,meta,dq,pack); fb=build_factor(B,arm,meta,dq,pack)
                user=(f"{COMMON}\n\nSCIENTIFIC QUESTION\n{QUESTION}\n\nFACTOR A\n{fa}\n\nFACTOR B\n{fb}\n\n"
                      "Return exactly one token: A, B, T, or U.")
                full=SYSTEM+'\n\n'+user
                prompt_id=f"{e['edge_id']}_{arm}_{order}"
                fn=pdir/f'{prompt_id}.txt'; fn.write_text(full+'\n',encoding='utf-8')
                base_rows.append({
                  'origin_year':year,'edge_id':e['edge_id'],'factor_i':fi,'factor_j':fj,
                  'arm':arm,'order':order,'display_A':A,'display_B':B,
                  'HD_bits':e['HD_bits'],'primary_H_ge_0p90':e['primary_H_ge_0p90'],
                  'strict_sensitivity_H_ge_0p95':e['strict_sensitivity_H_ge_0p95'],
                  'DQL_call_required':e['DQL_call_required'],
                  'prompt_file':str(fn.relative_to(OUT)),'prompt_sha256':htext(full),
                })
    base=pd.DataFrame(base_rows)
    base.to_csv(OUT/f'BASE_PROMPT_SCHEDULE_{year}.csv',index=False)
    calls=[]
    for row in base.to_dict('records'):
        for rep in REPEATS:
            call=dict(row)
            call['measurement_repeat']=rep
            call['measurement_call_id']=f"{row['edge_id']}_{row['arm']}_{row['order']}_R{rep}"
            calls.append(call)
    calls=pd.DataFrame(calls)
    # deterministic randomized execution order, independent of target/model output
    seed=f'JKP153_PHASEB_EXEC_{year}_20260919'
    calls['_key']=calls.measurement_call_id.map(lambda x:htext(seed+'|'+x))
    calls=calls.sort_values('_key').drop(columns='_key').reset_index(drop=True)
    calls.insert(0,'execution_index',range(1,len(calls)+1))
    calls.to_csv(OUT/f'EXECUTION_SCHEDULE_{year}.csv',index=False)
    return {
      'year':year,'union_edges':len(graph),'base_prompts':len(base),'calls_R3':len(calls),
      'DQ_calls':int((calls.arm=='DQ').sum()),'DQL_calls':int((calls.arm=='DQL').sum()),
      'primary_calls':int(calls.primary_H_ge_0p90.astype(bool).sum()),
      'target_read':False,'model_called':False,
    }

def main():
    s=[render_year(2024),render_year(2025)]
    (OUT/'PROMPT_RENDER_SUMMARY.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(s,indent=2))

if __name__=='__main__': main()
