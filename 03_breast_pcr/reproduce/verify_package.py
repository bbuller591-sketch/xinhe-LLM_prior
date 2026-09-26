#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, sys
PKG=Path(__file__).resolve().parents[1]
SUMS=PKG/'PACKAGE_SHA256SUMS.txt'
if not SUMS.exists():
    raise SystemExit('PACKAGE_SHA256SUMS.txt is missing')
bad=[]; checked=0
for line in SUMS.read_text(encoding='utf-8').splitlines():
    line=line.strip()
    if not line: continue
    want,rel=line.split('  ',1)
    p=PKG/rel
    if not p.exists():
        bad.append({'file':rel,'reason':'missing'}); continue
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    got=h.hexdigest(); checked+=1
    if got!=want: bad.append({'file':rel,'reason':'hash_mismatch','want':want,'got':got})
status={'status':'PASS' if not bad else 'FAIL','checked_files':checked,'failures':bad}
print(json.dumps(status,indent=2))
if bad: raise SystemExit(2)
