#!/usr/bin/env python3
"""Search-space reduction on corpus v3 (the #1756 counts: ranked methods, method-APK entry points, methods in
transitive slices), the old pipeline's selection rule applied to the eight providers.

Ranking: exp_label_audit's provenance ranking (linear SGD on the v3 identifier FCG vectors, d=1024; exact
linear SHAP split of each bucket over its contributing methods, provenance_localization.split_bucket_credit),
aggregated over the same PER_FAM=25 sampled applications per provider (seed 42), credit toward the provider's
own class. Candidates are the FCG methods left after the vectorizer's library filter.
  ranked methods  the top TOP_FAM=10 methods of each provider plus the TOP_GLOBAL=20 methods with the largest
                  credit summed over all providers' rankings; reported before and after deduplication
  entry points    each ranked method is instantiated in the sampled application (of the provider whose
                  ranking selected it) where it received the most credit; a global-list method in every
                  provider whose sampled applications contain it with positive credit, at that provider's
                  best application. Distinct (method, APK) pairs.
  slices          the methods reachable from each entry point along call edges in that application's FCG
                  (forward transitive closure, the entry included); per application the union over its entry
                  points, summed over applications. Reported for all methods and for methods outside the
                  vectorizer's library prefixes.
Output: results/datasets/proxy_v3/search_space_v3.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H
import exp_label_audit as A
import gen_wl_vectors as G
import provenance_localization as P

OUT = os.path.join(H.ROOT, "results", "datasets", "proxy_v3", "search_space_v3.json")
TOP_FAM = 10
TOP_GLOBAL = 20


def main():
    assert H.V3, "run with DATASET=proxy_v3"
    fams = A.FAMS_V3
    ids, fam, X = A.labels("v3", fams)
    m = A.train(fam, X)
    Ex = X.mean(0)
    classes = list(m.classes_)
    del X
    rng = np.random.default_rng(A.SEED)
    agg = {}
    best = {}            # method -> (credit, sha) of the app where it got the most credit
    for f in fams:
        agg[f] = defaultdict(float)
        best[f] = {}
    sampled = {}
    for f in fams:
        w = m.coef_[classes.index(f)].astype(np.float64)
        lst = sorted(ids[fam == f].tolist())
        rng.shuffle(lst)
        used = []
        for sha in lst:
            g = H.load_graph(f, sha, "fcg")
            if g is None or len(g[0]) == 0:
                continue
            x, prov, nrm = P.provenance_vector(g[0], g[1], g[2])  # (nodes, successors, predecessors)
            del g
            if x is None:
                continue
            for meth, c in P.split_bucket_credit(prov, w, Ex, nrm, w * (x - Ex)).items():
                agg[f][meth] += c
                if meth in best[f]:
                    prev = best[f][meth][0]
                else:
                    prev = -np.inf
                if c > prev:
                    best[f][meth] = (c, sha)
            used.append(sha)
            if len(used) >= A.PER_FAM:
                break
        sampled[f] = used
        print(f"[{f}] {len(used)} applications, {len(agg[f])} candidate methods", flush=True)

    # ranked methods: per-provider top and global top
    per_fam = {}
    for f in fams:
        per_fam[f] = sorted(agg[f], key=lambda k: -agg[f][k])[:TOP_FAM]
    tot = defaultdict(float)
    for f in fams:
        for k, v in agg[f].items():
            tot[k] += v
    glob = sorted(tot, key=lambda k: -tot[k])[:TOP_GLOBAL]
    ranked_raw = 0
    for v in per_fam.values():
        ranked_raw += len(v)
    ranked_raw += len(glob)
    ranked = set(glob)
    for v in per_fam.values():
        ranked.update(v)
    entries = set()                          # (family, sha, method)
    for f in fams:
        for k in per_fam[f]:
            entries.add((f, best[f][k][1], k))
    for k in glob:
        for f in fams:
            if k in best[f] and best[f][k][0] > 0:
                entries.add((f, best[f][k][1], k))
    pairs = set()
    for _, sha, k in entries:
        pairs.add((sha, k))
    print(f"ranked {ranked_raw} -> {len(ranked)} after deduplication; {len(pairs)} method-APK entry points", flush=True)

    by_app = defaultdict(set)
    app_fam = {}
    for f, sha, k in entries:
        by_app[sha].add(k)
        app_fam[sha] = f
    # forward transitive slices from the entry points, per application
    slice_all = 0
    slice_nonlib = 0
    per_app = {}
    for sha, roots in sorted(by_app.items()):
        nl, su, _ = H.load_graph(app_fam[sha], sha, "fcg")
        lab = {}
        for n, r in nl.items():
            lab[n] = G.canonical_label(r, "fcg")
        start = []
        for n, l in lab.items():
            if l in roots:
                start.append(n)
        seen = set(start)
        stack = list(start)
        while stack:
            u = stack.pop()
            for v in su.get(u, ()):
                if v not in seen and v in lab:
                    seen.add(v)
                    stack.append(v)
        meths = set()
        for n in seen:
            meths.add(lab[n])
        nonlib = set()
        for l in meths:
            if not l.startswith(G.LIB_PREFIXES):
                nonlib.add(l)
        per_app[sha] = {"family": app_fam[sha], "entry_points": len(roots), "slice_methods": len(meths),
                        "slice_methods_nonlib": len(nonlib)}
        slice_all += len(meths)
        slice_nonlib += len(nonlib)
        del nl, su, lab
    print(f"slices: {slice_all} methods ({slice_nonlib} outside library prefixes) over {len(by_app)} applications",
          flush=True)
    # summary
    per_fam_sampled = {}
    for f, v in sampled.items():
        per_fam_sampled[f] = len(v)
    is_sdk_flags = []
    for k in ranked:
        hit = False
        for f in fams:
            if k.startswith(A.SDK[f]):
                hit = True
                break
        is_sdk_flags.append(hit)
    per_family_top = {}
    for f in fams:
        top_rows = []
        for k in per_fam[f]:
            top_rows.append({"method": k, "credit": float(agg[f][k]), "is_sdk": k.startswith(A.SDK[f])})
        per_family_top[f] = top_rows
    global_top = []
    for k in glob:
        global_top.append({"method": k, "credit": float(tot[k])})
    res = {"per_fam_sampled": per_fam_sampled, "top_fam": TOP_FAM, "top_global": TOP_GLOBAL,
           "ranked_before_dedup": ranked_raw, "ranked_after_dedup": len(ranked), "entry_points": len(pairs),
           "entry_apps": len(by_app), "slice_methods": slice_all, "slice_methods_nonlib": slice_nonlib,
           "ranked_sdk_share": float(np.mean(is_sdk_flags)),
           "per_family_top": per_family_top,
           "global_top": global_top, "per_app": per_app}
    json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
