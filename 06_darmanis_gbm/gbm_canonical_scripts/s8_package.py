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
"""GBM s8 - refresh manifest hashes, write SHA256SUMS.txt + FILE_MANIFEST.csv,
build the tarball and verify the round trip.
"""
import hashlib, json, os, shutil, subprocess, tarfile, tempfile, time

B = str(REPRO_ROOT / 'gbm_canonical_recovery_20260917')
CAN = f"{B}/canonical"
OUT = str(REPRO_ROOT)
TARBALL = f"{OUT}/GBM_CANONICAL_HANDOFF_20260917.tar.gz"
TAR_ROOT = "gbm_canonical_recovery_20260917"
BIG = {"raw/GSE84465_GBM_All_data.csv.gz",           # 20 MB, re-downloadable (URL in the audit)
       "processed/darmanis_counts_genes_by_cells.npy"}  # 337 MB, reproducible via s2
SELF = {"SHA256SUMS.txt", "FILE_MANIFEST.csv"}
t0 = time.time()

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# ---------------------------------------------------------------- inventory
files = []
for root, dirs, fs in os.walk(B):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for f in fs:
        p = os.path.join(root, f)
        r = os.path.relpath(p, B)
        if r.startswith("scripts/_") or r.endswith(".pyc"):
            continue
        files.append(r)
files.sort()
print(f"{len(files)} files in the tree")

# ---------------------------------------------------------------- manifest refresh
man = json.load(open(f"{CAN}/candidate_manifest.json"))
man["files"] = {f: {"bytes": os.path.getsize(f"{CAN}/{f}"), "sha256": sha(f"{CAN}/{f}")}
                for f in sorted(os.listdir(CAN))
                if os.path.isfile(f"{CAN}/{f}") and f != "candidate_manifest.json"}
man["self_hash_note"] = ("candidate_manifest.json is excluded from this map (a file cannot carry "
                         "its own sha256); its hash is listed in the tree-level SHA256SUMS.txt "
                         "and FILE_MANIFEST.csv")
man["packaging"] = {"tarball": TARBALL, "excluded_big_files": sorted(BIG),
                    "excluded_reason": "re-downloadable / reproducible; hashes recorded in FILE_MANIFEST.csv"}
json.dump(man, open(f"{CAN}/candidate_manifest.json", "w"), indent=2)
print("manifest hashes refreshed")

# ---------------------------------------------------------------- checksums
rows, sums = [], []
for r in files:
    p = f"{B}/{r}"
    b, h = os.path.getsize(p), sha(p)
    in_tar = r not in BIG
    rows.append((r, b, h, in_tar))
    if in_tar and r not in SELF:
        sums.append(f"{h}  {r}")
with open(f"{B}/SHA256SUMS.txt", "w") as f:
    f.write("\n".join(sums) + "\n")
with open(f"{B}/FILE_MANIFEST.csv", "w") as f:
    f.write("path,bytes,sha256,in_tarball\n")
    for r, b, h, in_tar in rows:
        f.write(f"{r},{b},{h},{int(in_tar)}\n")
print(f"SHA256SUMS.txt: {len(sums)} entries   FILE_MANIFEST.csv: {len(rows)} rows")

# ---------------------------------------------------------------- tarball
tarmembers = [(r, b, h, in_tar) for r, b, h, in_tar in rows if in_tar] + \
             [(f, os.path.getsize(f"{B}/{f}"), sha(f"{B}/{f}"), True) for f in sorted(SELF)]
if os.path.exists(TARBALL):
    os.remove(TARBALL)
with tarfile.open(TARBALL, "w:gz", compresslevel=6) as tf:
    for r, b, h, in_tar in tarmembers:
        tf.add(f"{B}/{r}", arcname=f"{TAR_ROOT}/{r}")
print(f"tarball {TARBALL}: {os.path.getsize(TARBALL):,} bytes, "
      f"{len(tarmembers)} members  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- round trip
tmp = tempfile.mkdtemp(prefix="gbm_handoff_verify_")
try:
    with tarfile.open(TARBALL) as tf:
        names = tf.getnames()
        tf.extractall(tmp)
    print(f"extracted {len(names)} members to {tmp}")
    res = subprocess.run(["sha256sum", "-c", "SHA256SUMS.txt"], cwd=f"{tmp}/{TAR_ROOT}",
                         capture_output=True, text=True)
    ok = res.stdout.count(": OK")
    bad = res.stdout.count("FAILED") + res.stderr.count("FAILED")
    print(f"sha256sum -c -> {ok} OK, {bad} FAILED")
    key = ["canonical/X_candidate_aligned.npy", "canonical/candidate_features_canonical.csv",
           "canonical/candidate_manifest.json", "FINAL_HANDOFF.md"]
    for k in key:
        print(f"   {k}: {'present' if os.path.exists(f'{tmp}/{TAR_ROOT}/{k}') else 'MISSING'}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print(f"tarball sha256 {sha(TARBALL)}")
print(f"done {time.time()-t0:.0f}s")
