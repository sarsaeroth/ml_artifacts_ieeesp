#!/usr/bin/env python3
"""DEX reuse per provider for corpus v3 (ML-guided paper, corpus table): share of a provider's
classified applications that share an identical top-level classes*.dex (SHA-256) with another
application of the same provider, plus DEX-file occurrence and distinct counts.
Reads APK zips on disk (no execution). Output: results/corpus_v3/dex_reuse_v3.json
"""
import hashlib
import json
import os
import re
import zipfile
from collections import defaultdict
from multiprocessing import get_context
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMS = ["cashraven", "ipidea", "ipninja", "luminati", "monkeysocks", "oxylabs", "popa", "traffmonetizer"]

def dex_hashes(r):
    try:
        with zipfile.ZipFile(os.path.join(ROOT, r["path"])) as z:
            # SHA-256 of each top-level classes*.dex
            hashes = []
            for n in z.namelist():
                if re.match(r"classes\d*\.dex$", n):
                    hashes.append(hashlib.sha256(z.read(n)).hexdigest())
            return r["sha256"], hashes
    except Exception:
        return r["sha256"], None

def main():
    man = []
    for l in open(os.path.join(ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
        man.append(json.loads(l))
    keep = []
    for r in man:
        if r["status"] == "kept" and r["family"] in FAMS:
            keep.append(r)
    dex = {}
    with get_context("fork").Pool(6) as p:
        for sha, hashes in p.imap_unordered(dex_hashes, keep, chunksize=8):
            dex[sha] = hashes
    fam = {}
    for r in keep:
        fam[r["sha256"]] = r["family"]
    unreadable = 0
    for v in dex.values():
        if v is None:
            unreadable += 1
    out = {"apps": len(keep), "unreadable": unreadable, "per_family": {}}
    occ = []
    for v in dex.values():
        if v:
            for h in v:
                occ.append(h)
    out["dex_occurrences"] = len(occ)
    out["dex_distinct"] = len(set(occ))
    for f in FAMS:
        by = defaultdict(set)
        apps = []
        for s in fam:
            if fam[s] == f:
                apps.append(s)
        # DEX hash -> apps of this provider containing it
        for s in apps:
            if dex.get(s):
                app_hashes = dex.get(s)
            else:
                app_hashes = []
            for h in app_hashes:
                by[h].add(s)
        # apps sharing at least one DEX file with another app of the provider
        shared = set()
        for s in apps:
            if dex.get(s):
                app_hashes = dex.get(s)
            else:
                app_hashes = []
            for h in app_hashes:
                if len(by[h]) > 1:
                    shared.add(s)
                    break
        out["per_family"][f] = {"apps": len(apps), "sharing": len(shared), "share": len(shared) / len(apps)}
    os.makedirs(os.path.join(ROOT, "results", "corpus_v3"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "corpus_v3", "dex_reuse_v3.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
