#!/usr/bin/env python3
"""What does the corpus-v3 classifier key on: the proxy SDK, co-bundled ad libraries, or neither?

ad_confound_check.py on corpus v3 and its eight classes, extended with the SDK itself. From one parse of
each FCG, identifier WL vectors (d=1024, the deployed coloring) are built over five method subsets and
cross-validated under identical DEX-grouped folds (exp_harness.groups_for, StratifiedGroupKFold, in-fold
upsampling, linear SGD):
  full        all methods (the deployed setting)
  ad_masked   ad/analytics/tracking methods removed (ad_confound_check.AD_PREFIXES)
  ads_only    only ad/analytics methods
  sdk_masked  every provider's SDK namespaces removed (exp_label_audit.SDK, all eight classes)
  sdk_only    only SDK-namespace methods
sdk_masked high means the classes are separable without the SDK (template, bundled libraries, host code);
sdk_only high means the SDK alone suffices. Up to 200 apps per class (all of the smaller classes),
seed 42. Also reports, per class, the median share of non-library methods in ad and SDK namespaces.
Output: results/datasets/proxy_v3/confound_v3.json
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import ad_confound_check as AC
import exp_harness as H
import exp_label_audit as A
import gen_wl_vectors as G

OUT = os.path.join(H.ROOT, "results", "datasets", "proxy_v3", "confound_v3.json")
DIM = 1024
PER_FAM = 200
SEED = 42
REPEATS = 3
# every provider's SDK namespace prefixes, in FAMS_V3 order
_sdk_all = []
for _f in A.FAMS_V3:
    for _p in A.SDK[_f]:
        _sdk_all.append(_p)
SDK_ALL = tuple(_sdk_all)
CONDS = {"full": lambda l: True,
         "ad_masked": lambda l: not l.startswith(AC.AD_PREFIXES),
         "ads_only": lambda l: l.startswith(AC.AD_PREFIXES),
         "sdk_masked": lambda l: not l.startswith(SDK_ALL),
         "sdk_only": lambda l: l.startswith(SDK_ALL)}


def main():
    assert H.V3, "run with DATASET=proxy_v3"
    rng = np.random.default_rng(SEED)
    man = []
    for l in open(os.path.join(H.ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
        man.append(json.loads(l))
    by = defaultdict(list)
    for m in sorted(man, key=lambda m: m["sha256"]):
        if m["status"] == "kept" and m.get("split") == "class":
            by[m["family"]].append(m["sha256"])
    rows = {}
    for c in CONDS:
        rows[c] = []
    ids = []
    fams = []
    share = defaultdict(lambda: {"ad": [], "sdk": []})
    for f in A.FAMS_V3:
        lst = by[f][:]
        rng.shuffle(lst)
        used = 0
        for sha in lst:
            g = H.load_graph(f, sha, "fcg")
            if g is None or not g[0]:
                continue
            nl, su, pr = g
            canon = {}
            for n, r in nl.items():
                canon[n] = G.canonical_label(r, "fcg")
            # one WL vector per method subset; an empty subset gives the zero vector
            for c, keep in CONDS.items():
                sub = {}
                for n, raw in nl.items():
                    if keep(canon[n]):
                        sub[n] = raw
                if not sub:
                    vec = np.zeros(DIM, dtype=np.float32)
                else:
                    vec = G.wl_hashed_vector(sub, su, pr, kind="fcg", dim=DIM).astype(np.float32)
                rows[c].append(vec)
                del sub
            # share of non-library methods in ad and in SDK namespaces
            dev = []
            for l in canon.values():
                if not l.startswith(G.LIB_PREFIXES):
                    dev.append(l)
            n_ad = 0
            n_sdk = 0
            for l in dev:
                if l.startswith(AC.AD_PREFIXES):
                    n_ad += 1
                if l.startswith(SDK_ALL):
                    n_sdk += 1
            share[f]["ad"].append(n_ad / max(1, len(dev)))
            share[f]["sdk"].append(n_sdk / max(1, len(dev)))
            ids.append(sha)
            fams.append(f)
            used += 1
            del g, nl, su, pr, canon
            if used >= PER_FAM:
                break
        print(f"[{f}] {used} apps", flush=True)
    ids = np.array(ids)
    fams = np.array(fams)
    y, le = H.labels(fams)
    groups = H.groups_for(ids)
    folds = H.folds(y, groups, REPEATS)
    share_median = {}
    for f, s in share.items():
        share_median[f] = {}
        for k, v in s.items():
            share_median[f][k] = float(np.median(v))
    res = {"n": int(len(ids)), "counts": dict(Counter(fams.tolist())), "dim": DIM,
           "share_median": share_median,
           "conditions": {}}
    for c in CONDS:
        X = np.vstack(rows[c])
        f1 = []
        per = []
        for r, k, tr, te in folds:
            up = H.upsample(tr, y, 1000 * r + k)
            m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
            m.fit(X[up], y[up])
            p = m.predict(X[te])
            f1.append(f1_score(y[te], p, average="macro"))
            per.append(f1_score(y[te], p, average=None, labels=range(len(le.classes_))))
        per_family_f1 = {}
        mean_per = np.mean(per, 0)
        for cls_name, v in zip(le.classes_, mean_per):
            per_family_f1[cls_name] = float(v)
        res["conditions"][c] = {"macro_f1": float(np.mean(f1)), "sd": float(np.std(f1)),
                                "per_family_f1": per_family_f1}
        parts = []
        for k, v in res["conditions"][c]["per_family_f1"].items():
            parts.append(f"{k}={v:.2f}")
        print(f"{c:11} macro-F1 {np.mean(f1):.3f} +/- {np.std(f1):.3f}  " + " ".join(parts), flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
