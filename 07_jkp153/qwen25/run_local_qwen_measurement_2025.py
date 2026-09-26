

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
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT=Path(str(REPRO_ROOT / 'finance_llm_prior'))
PHASEB=ROOT/'experiments/jkp153_fullrank_dataconfusion_phaseB_20260919_030238'
HERE=ROOT/'experiments/jkp153_qwen25_local_formal_20260919'
MODEL=ROOT/'models/Qwen2.5-14B-Instruct-cf98f3b'
REV='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'

ap=argparse.ArgumentParser()
ap.add_argument('--year',type=int,default=2025)
ap.add_argument('--shard-index',type=int,required=True)
ap.add_argument('--num-shards',type=int,required=True)
ap.add_argument('--batch-size',type=int,default=4)
a=ap.parse_args()
if a.year!=2025: raise SystemExit('2025 only in this frozen runner')
sched=pd.read_csv(HERE/'QWEN_EXECUTION_SCHEDULE_2025.csv')
jobs=sched[(sched.qwen_execution_index-1)%a.num_shards==a.shard_index].copy().reset_index(drop=True)
outp=HERE/'runs/2025'/f'calls.part{a.shard_index:02d}.jsonl'
print('shard',a.shard_index,'jobs',len(jobs),'out',outp,flush=True)
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
tok.padding_side='left'
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
label_ids={}
for x in ['A','B','T','U']:
    z=tok.encode(x,add_special_tokens=False)
    if len(z)!=1: raise RuntimeError(f'{x} tokenization {z}')
    label_ids[x]=z[0]
model=AutoModelForCausalLM.from_pretrained(
    MODEL,local_files_only=True,dtype=torch.bfloat16,
    device_map='auto',max_memory={0:'22GiB',1:'22GiB','cpu':'64GiB'},
    low_cpu_mem_usage=True)
model.eval()
first_dev=next(model.parameters()).device

done=set()
if outp.exists():
    for line in outp.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try: done.add(json.loads(line)['measurement_call_id'])
            except Exception: pass
jobs=jobs[~jobs.measurement_call_id.isin(done)].reset_index(drop=True)
print('remaining',len(jobs),'resume_done',len(done),flush=True)
with outp.open('a',encoding='utf-8') as fout:
  for start in range(0,len(jobs),a.batch_size):
    chunk=jobs.iloc[start:start+a.batch_size]
    texts=[]
    for _,r in chunk.iterrows():
        prompt=(PHASEB/str(r.prompt_file)).read_text(encoding='utf-8')
        texts.append(tok.apply_chat_template(
            [{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True))
    enc=tok(texts,return_tensors='pt',padding=True)
    enc={k:v.to(first_dev) for k,v in enc.items()}
    t0=time.time()
    with torch.inference_mode():
        res=model(**enc,use_cache=False,logits_to_keep=1)
    logits=res.logits[:,-1,:].float().cpu()
    del res,enc
    torch.cuda.empty_cache()
    elapsed=time.time()-t0
    for bi,(_,r) in enumerate(chunk.iterrows()):
        lg=logits[bi]
        vals={x:float(lg[i]) for x,i in label_ids.items()}
        v4=np.array([vals[x] for x in ['A','B','T','U']],dtype=float)
        v4=v4-v4.max(); p4=np.exp(v4); p4=p4/p4.sum()
        p4d=dict(zip(['A','B','T','U'],map(float,p4)))
        ab=np.array([vals['A'],vals['B']],dtype=float)
        ab=ab-ab.max(); pab=np.exp(ab); pab=pab/pab.sum()
        hard=max(p4d,key=p4d.get)
        rec={
          'measurement_call_id':str(r.measurement_call_id),
          'edge_id':str(r.measurement_edge_id),
          'arm':str(r.arm),'order':str(r.order),'measurement_repeat':1,
          'factor_i':str(r.factor_i),'factor_j':str(r.factor_j),
          'display_A':str(r.display_A),'display_B':str(r.display_B),
          'prompt_file':str(r.prompt_file),'prompt_sha256':str(r.prompt_sha256),
          'model_requested':'Qwen2.5-14B-Instruct',
          'model_revision':REV,'runtime':'LOCAL_HF_BF16',
          'hard_token':hard,
          'pA_cond_AB':float(pab[0]),'pB_cond_AB':float(pab[1]),
          'logit_A':vals['A'],'logit_B':vals['B'],'logit_T':vals['T'],'logit_U':vals['U'],
          'p4_A':p4d['A'],'p4_B':p4d['B'],'p4_T':p4d['T'],'p4_U':p4d['U'],
          'final_status':'SUCCESS','batch_forward_seconds':elapsed,
          'qwen_execution_index':int(r.qwen_execution_index)
        }
        fout.write(json.dumps(rec,ensure_ascii=False)+'\n')
        fout.flush()
    print(f'shard {a.shard_index} processed {min(start+a.batch_size,len(jobs))}/{len(jobs)}',flush=True)
print('DONE shard',a.shard_index,flush=True)
