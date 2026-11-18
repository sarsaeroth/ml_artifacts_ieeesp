#!/usr/bin/env python3
"""Manifest-declared vs programmatic SDK integration on corpus v3 (the #1756 appendix table).

For every corpus-v3 class application (status kept, split class), parse AndroidManifest.xml with androguard
(parsing only) and record whether at least one declared component (activity, service, receiver or provider)
lies in the provider's SDK namespaces (exp_label_audit.SDK, converted to dotted package form). Reported per
provider: applications parsed, applications with an SDK component, rate, and parse failures.
Run with python3 -P (androguard is in the user site-packages). Output: results/datasets/proxy_v3/manifest_v3.json
(resumable)
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_label_audit as A

ROOT = A.ROOT
OUT = os.path.join(ROOT, "results", "datasets", "proxy_v3", "manifest_v3.json")


def main():
    from androguard.core.apk import APK
    from loguru import logger
    logger.remove()
    # kept class-split apps of corpus v3
    man = []
    for l in open(os.path.join(ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
        man.append(json.loads(l))
    apps = []
    for m in man:
        if m["status"] == "kept" and m.get("split") == "class":
            apps.append(m)
    # resume from the previous output if present
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {"apps": {}}
    for i, m in enumerate(apps, 1):
        sha = m["sha256"]
        if sha in res["apps"]:
            continue
        # SDK prefixes as dotted package names (Lcom/foo/ -> com.foo.)
        pre_list = []
        for p in A.SDK[m["family"]]:
            pre_list.append(p.lstrip("L").replace("/", "."))
        pre = tuple(pre_list)
        rec = {"family": m["family"]}
        try:
            a = APK(os.path.join(ROOT, m["path"]))
            comps = list(a.get_activities()) + list(a.get_services()) + list(a.get_receivers()) + list(a.get_providers())
            n_sdk = 0
            for c in comps:
                if c and c.startswith(pre):
                    n_sdk += 1
            rec["sdk_components"] = n_sdk
        except Exception as e:
            rec["error"] = type(e).__name__
        res["apps"][sha] = rec
        if i % 200 == 0:
            json.dump(res, open(OUT, "w"))
            print(f"  {i}/{len(apps)}", flush=True)
    # per-provider summary
    summ = defaultdict(lambda: {"parsed": 0, "declared": 0, "failed": 0})
    for r in res["apps"].values():
        s = summ[r["family"]]
        if "error" in r:
            s["failed"] += 1
        else:
            s["parsed"] += 1
            if r["sdk_components"] > 0:
                s["declared"] += 1
    for s in summ.values():
        if s["parsed"]:
            s["rate"] = s["declared"] / s["parsed"]
        else:
            s["rate"] = None
    res["summary"] = dict(summ)
    json.dump(res, open(OUT, "w"), indent=1)
    for f, s in sorted(summ.items()):
        print(f, s, flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
