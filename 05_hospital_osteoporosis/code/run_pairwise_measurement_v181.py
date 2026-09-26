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
import argparse, json, hashlib, math, os, re, time, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd

ROOT=Path(str(REPRO_ROOT / 'HOSPITAL_OSTEOPOROSIS_EXTERNAL_EVIDENCE_V1_2_20260918'))
PACKDIR=ROOT/'07_BROAD37_EVIDENCE/V0_8_1_AUDITED_CORPUS_AND_PACKETS'
GRAPH=PACKDIR/'global_COMPLETE_GRAPH_V0_8_1.csv'
PROMPT_SPEC=ROOT/'02_PROTOCOL/PAIRWISE_PROMPT_SPEC_V1_8_1.json'
GATE=ROOT/'08_LLM_MEASUREMENT/PRE_LLM_FREEZE_GATE_V1_8_1.json'
ENV=ROOT/'LOCAL_SECRETS/deepseek_api.env'
OUT=ROOT/'08_LLM_MEASUREMENT/ARM_E1_V1_8_1'
CACHE=OUT/'cache'; CACHE.mkdir(parents=True,exist_ok=True)
SUMMARY=OUT/'summaries'; SUMMARY.mkdir(parents=True,exist_ok=True)

def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')

def load_env():
    vals={}
    for line in ENV.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1); vals[k.strip()]=v.strip()
    if not vals.get('DEEPSEEK_API_KEY'): raise RuntimeError('API_KEY_MISSING')
    return vals

def packet_path(feature):
    safe=re.sub(r'[^A-Za-z0-9._-]+','_',feature)
    return PACKDIR/f'{safe}_PACKET_V0_8_1.json'

def load_packet(feature):
    return json.loads(packet_path(feature).read_text())

def render_cards(packet,prefix):
    lines=[]
    for c in packet['cards']:
        t=c['card_text']
        # Rewrite leading E##: to A.E##:/B.E##: only.
        t=re.sub(r'^E(\d\d):',prefix+r'.E\1:',t)
        lines.append(t)
    return '\n'.join(lines) if lines else 'NONE'

def first_semantic_token(lpcontent):
    for pos in lpcontent or []:
        tok=str(pos.get('token',''))
        if tok.strip():
            return pos
    return None

def logsumexp(vals):
    m=max(vals); return m+math.log(sum(math.exp(x-m) for x in vals))

def parse_response(resp,valid_A,valid_B):
    ch=resp['choices'][0]; content=ch['message'].get('content','') or ''
    lpcontent=(ch.get('logprobs') or {}).get('content') or []
    pos=first_semantic_token(lpcontent)
    if pos is None: return {'parse_status':'FAIL_NO_LOGPROB_POSITION','content':content}
    alts=pos.get('top_logprobs') or []
    la=[float(x['logprob']) for x in alts if str(x.get('token','')).strip()=='A']
    lb=[float(x['logprob']) for x in alts if str(x.get('token','')).strip()=='B']
    first=re.search(r'\S+',content)
    first_txt=first.group(0).strip() if first else ''
    if first_txt not in ('A','B'): return {'parse_status':'FAIL_FIRST_TOKEN','content':content,'first_text_token':first_txt}
    if not la or not lb: return {'parse_status':'FAIL_AB_NOT_BOTH_TOP20','content':content,'first_text_token':first_txt,
                                 'A_in_top20':bool(la),'B_in_top20':bool(lb)}
    L_A=logsumexp(la); L_B=logsumexp(lb); z=logsumexp([L_A,L_B]); pA=math.exp(L_A-z)
    lines=[x.strip() for x in content.strip().splitlines()]
    ca=next((x.split('=',1)[1].strip() for x in lines if x.startswith('CITES_A=')),None)
    cb=next((x.split('=',1)[1].strip() for x in lines if x.startswith('CITES_B=')),None)
    rat=next((x.split('=',1)[1].strip() for x in lines if x.startswith('RATIONALE=')),None)
    if ca is None or cb is None or rat is None:
        return {'parse_status':'FAIL_FORMAT','content':content,'p_display_A':pA,'first_text_token':first_txt}
    def cites(x):
        if x=='NONE': return []
        return [z.strip() for z in x.split(',') if z.strip()]
    cia,cib=cites(ca),cites(cb)
    badA=[x for x in cia if x not in valid_A]; badB=[x for x in cib if x not in valid_B]
    if badA or badB:
        return {'parse_status':'FAIL_CITATION_ID','content':content,'p_display_A':pA,'first_text_token':first_txt,
                'bad_cites_A':badA,'bad_cites_B':badB}
    return {'parse_status':'PASS','content':content,'p_display_A':pA,'first_text_token':first_txt,
            'hard_winner_display':first_txt,'cites_A':cia,'cites_B':cib,'rationale':rat,
            'rationale_words':len(rat.split()),'A_in_top20':True,'B_in_top20':True,
            'first_logprob_token':pos.get('token')}

def call_api(vals,payload,retries=6):
    body=json.dumps(payload,ensure_ascii=False).encode()
    err=None
    for a in range(retries):
        try:
            req=urllib.request.Request(vals['DEEPSEEK_BASE_URL'].rstrip('/')+'/chat/completions',data=body,
                headers={'Authorization':'Bearer '+vals['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read().decode())
        except Exception as e:
            err=repr(e); time.sleep(min(30,2**a))
    raise RuntimeError('API_FAILED '+str(err))

def make_key(pair_id,orientation,pa,pb,prompt_hash,model,repeat):
    obj={'pair_id':pair_id,'orientation':orientation,'packet_A_sha256':pa['packet_sha256'],'packet_B_sha256':pb['packet_sha256'],
         'prompt_spec_sha256':prompt_hash,'model':model,'thinking':'disabled','temperature':1.0,'max_tokens':220,
         'logprobs':True,'top_logprobs':20,'repeat':repeat,'arm':'ARM_E1'}
    raw=json.dumps(obj,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(raw.encode()).hexdigest(),obj

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-pairs',type=int,default=None); ap.add_argument('--diagnostic-repeats',action='store_true')
    args=ap.parse_args()
    gate=json.loads(GATE.read_text())
    if gate.get('status')!='PASS_AUTHORIZE_EXPERIMENTAL_LLM_MEASUREMENT_V1_8_1': raise RuntimeError('PRE_LLM_GATE_NOT_PASS')
    vals=load_env(); model=vals.get('DEEPSEEK_MODEL')
    if model!='deepseek-flash': raise RuntimeError('MODEL_NOT_FROZEN_DEEPSEEK_FLASH')
    ps=json.loads(PROMPT_SPEC.read_text()); SYS=ps['system_template']; USER=ps['user_template']; ph=ps['prompt_spec_sha256']
    graph=pd.read_csv(GRAPH,dtype=str).sort_values('broad_pair_id')
    if args.max_pairs is not None: graph=graph.head(args.max_pairs)
    diag_ids=set()
    if args.diagnostic_repeats:
        allg=pd.read_csv(GRAPH,dtype=str)
        rank=sorted([(hashlib.sha256((p+'|20260918').encode()).hexdigest(),p) for p in allg.broad_pair_id])
        diag_ids={p for _,p in rank[:int(len(allg)*0.10)]}
    rows=[]; failures=[]
    for pi,r in graph.iterrows():
        pair_id=r.broad_pair_id; fa=r.feature_A; fb=r.feature_B
        repeats=[0] + ([1,2] if args.diagnostic_repeats and pair_id in diag_ids else [])
        for repeat in repeats:
            for orientation in ['AB','BA']:
                if orientation=='AB': da,db=fa,fb
                else: da,db=fb,fa
                pa,pb=load_packet(da),load_packet(db)
                key,keyobj=make_key(pair_id,orientation,pa,pb,ph,model,repeat)
                cpath=CACHE/f'{key}.json'
                if cpath.exists():
                    rec=json.loads(cpath.read_text())
                    rows.append(rec['summary']); continue
                A_cards=render_cards(pa,'A'); B_cards=render_cards(pb,'B')
                user=USER.format(FEATURE_A_NAME=pa['canonical_feature_name'],PACKET_A_CARDS_WITH_IDS_PREFIXED_A=A_cards,
                                 FEATURE_B_NAME=pb['canonical_feature_name'],PACKET_B_CARDS_WITH_IDS_PREFIXED_B=B_cards)
                payload={'model':model,'messages':[{'role':'system','content':SYS},{'role':'user','content':user}],
                         'thinking':{'type':'disabled'},'temperature':1.0,'max_tokens':220,'logprobs':True,'top_logprobs':20,'stream':False}
                t0=now(); resp=call_api(vals,payload)
                validA={f'A.{c["evidence_id"]}' for c in pa['cards']}; validB={f'B.{c["evidence_id"]}' for c in pb['cards']}
                parsed=parse_response(resp,validA,validB)
                provider_model=resp.get('model'); fingerprint=resp.get('system_fingerprint')
                if provider_model!='deepseek-flash' or not fingerprint:
                    parsed={'parse_status':'FAIL_PROVIDER_ROUTE_OR_FINGERPRINT',**parsed}
                # Resolve to semantic feature_A probability/winner.
                pdisp=parsed.get('p_display_A')
                if pdisp is not None:
                    p_sem_fa=pdisp if orientation=='AB' else 1-pdisp
                else: p_sem_fa=None
                hw=parsed.get('hard_winner_display')
                if hw in ('A','B'):
                    winner_sem=(da if hw=='A' else db)
                else: winner_sem=None
                summ={'timestamp_utc':t0,'call_key':key,'pair_id':pair_id,'semantic_feature_A':fa,'semantic_feature_B':fb,
                      'orientation':orientation,'display_feature_A':da,'display_feature_B':db,'repeat_index':repeat,
                      'packet_A_sha256':pa['packet_sha256'],'packet_B_sha256':pb['packet_sha256'],'prompt_spec_sha256':ph,
                      'requested_model':model,'provider_model':provider_model,'system_fingerprint':fingerprint,
                      'parse_status':parsed.get('parse_status'),'p_display_A':pdisp,'p_semantic_feature_A':p_sem_fa,
                      'hard_winner_display':hw,'hard_winner_semantic':winner_sem,'cites_A':'|'.join(parsed.get('cites_A',[])),
                      'cites_B':'|'.join(parsed.get('cites_B',[])),'rationale':parsed.get('rationale'),
                      'rationale_words':parsed.get('rationale_words'),'finish_reason':resp['choices'][0].get('finish_reason'),
                      'prompt_tokens':resp.get('usage',{}).get('prompt_tokens'),'completion_tokens':resp.get('usage',{}).get('completion_tokens'),
                      'total_tokens':resp.get('usage',{}).get('total_tokens'),'arm':'ARM_E1'}
                rec={'key_object':keyobj,'summary':summ,'parsed':parsed,'raw_response':resp}
                cpath.write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
                rows.append(summ)
                if summ['parse_status']!='PASS':
                    failures.append(summ)
                    pd.DataFrame(rows).to_csv(SUMMARY/'MEASUREMENT_CALLS_PARTIAL.csv',index=False)
                    raise RuntimeError('HARD_MEASUREMENT_FAILURE '+json.dumps(summ,ensure_ascii=False))
                if len(rows)%20==0: print(f'calls_completed_or_cached={len(rows)} last={pair_id} {orientation} r{repeat}',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(SUMMARY/'MEASUREMENT_CALLS_LATEST.csv',index=False)
    status={'status':'PASS','n_rows':len(df),'n_unique_call_keys':df.call_key.nunique(),'n_pairs':df.pair_id.nunique(),
            'repeat_indices':sorted(map(int,df.repeat_index.unique())),'n_failures':int((df.parse_status!='PASS').sum()),
            'provider_models':df.provider_model.value_counts().to_dict(),'fingerprints':df.system_fingerprint.value_counts().to_dict(),
            'prompt_tokens':int(df.prompt_tokens.fillna(0).sum()),'completion_tokens':int(df.completion_tokens.fillna(0).sum()),
            'timestamp_utc':now()}
    (SUMMARY/'MEASUREMENT_RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
