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
import gzip,csv,requests,time,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(str(REPRO_ROOT / '01_renal_tcmr/formal_handoff/01_GSE36059_to_GSE48581_TCMR'))
matrix=Path(str(REPRO_ROOT / 'dataset_screening_20260921/data/GSE36059/GSE36059_series_matrix.txt.gz'))
out=ROOT/'formal_outputs/recipient_audit/geo_quick/GSE36059_individual'
out.mkdir(parents=True,exist_ok=True)
# get accessions without touching outcomes
acc=[]
with gzip.open(matrix,'rt',errors='replace') as h:
    for line in h:
        if line.startswith('!Sample_geo_accession'):
            z=next(csv.reader([line.rstrip()],delimiter='\t'));acc=[x.strip('"') for x in z[1:]];break
assert len(acc)==411
proxies={'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}
def one(a):
    p=out/(a+'.txt')
    if p.exists():
        s=p.read_text(errors='ignore')
        if ('^SAMPLE = '+a) in s and '!Sample_title =' in s:return (a,'CACHED',len(s))
    last=''
    for k in range(1,6):
        try:
            r=requests.get('https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi',
                params={'acc':a,'targ':'self','form':'text','view':'quick'},
                proxies=proxies,timeout=60,headers={'User-Agent':'Mozilla/5.0 research-audit/1.0'})
            if r.status_code==200 and ('^SAMPLE = '+a) in r.text and '!Sample_title =' in r.text:
                p.write_text(r.text,encoding='utf-8');return (a,'OK',len(r.text))
            last=f'HTTP{r.status_code} bytes={len(r.content)}'
        except Exception as e:last=repr(e)
        time.sleep(min(2*k,10))
    return (a,'FAIL',last)
rows=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    fut={ex.submit(one,a):a for a in acc}
    for i,f in enumerate(as_completed(fut),1):
        z=f.result();rows.append(z)
        if i==1 or i%40==0 or i==len(acc):print(i,len(acc),z,flush=True)
ok={a for a,s,*_ in rows if s in ('OK','CACHED')}
summary={'requested':len(acc),'ok':len(ok),'fail':len(acc)-len(ok),'failed':sorted(set(acc)-ok)}
(ROOT/'formal_outputs/recipient_audit/geo_quick/GSE36059_INDIVIDUAL_DOWNLOAD_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
