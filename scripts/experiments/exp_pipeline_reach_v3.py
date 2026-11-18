#!/usr/bin/env python3
"""Explanation -> code location -> reverse engineering on corpus v3: does starting from the classifier's
explanation lead an analyst to each provider's proxy engine, and at what cost?

For each sampling seed (default 42,0,1,2,3) and each of the eight classifier providers, the ranking is
exp_label_audit's provenance ranking (linear SGD on identifier FCG vectors, d=1,024, exact linear SHAP split,
aggregated by method signature over the provider's 25 sampled applications; the same rankings as the
attribution table). Each of the top TOP_K signatures is instantiated in the sampled application where it
received the most attribution (the pipeline's code location). From there an analyst follows the call graph:
we run a breadth-first search over that application's FCG in both directions (callees and callers, the moves
of an analyst in a decompiler) and record
  hops        call-graph distance from the ranked method to the nearest method in the provider's SDK
              namespaces (exp_label_audit.SDK, established by the reverse engineering); 0 if it is one
  inspected   methods visited before the first SDK method is reached (the inspection cost of that start)
  first_sdk   the SDK class reached first
REACH_MODE=apponly traverses only methods outside the vectorizer's library/platform prefixes, so two methods are
not connected merely because both call the same framework API (pipeline_reach_v3_apponly.json); REACH_MODE=noplatform
blocks only platform APIs and allows bundled third-party libraries (pipeline_reach_v3_noplatform.json).
Per provider and seed: share of the top-k whose start reaches the SDK within 1, 2 and 3 hops, median hops
and inspected among those that reach it, and the SDK classes reached. Results are summarized as medians over
seeds. Output: results/datasets/proxy_v3/pipeline_reach_v3.json (resumable per seed)
"""
import json
import os
import sys
from collections import Counter, defaultdict, deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H
import exp_label_audit as A
import gen_wl_vectors as G
import provenance_localization as P

# REACH_MODE: all (any method), apponly (only methods outside the vectorizer's library/platform prefixes), or
# noplatform (bundled third-party libraries allowed, platform APIs not)
# legacy REACH_APP_ONLY=1 changes the default to apponly
if os.environ.get("REACH_APP_ONLY") == "1":
    _default_mode = "apponly"
else:
    _default_mode = "all"
MODE = os.environ.get("REACH_MODE", _default_mode)
PLATFORM = ("Landroid/", "Landroidx/", "Ljava/", "Ljavax/", "Lkotlin/", "Lkotlinx/", "Ldalvik/", "Lsun/",
            "Lorg/json/", "Lorg/xmlpull/", "Lorg/w3c/", "Lorg/xml/", "Lorg/apache/http/")
BLOCK = {"all": (), "apponly": None, "noplatform": PLATFORM}[MODE]
OUT = os.path.join(H.ROOT, "results", "datasets", "proxy_v3",
                   {"all": "pipeline_reach_v3.json", "apponly": "pipeline_reach_v3_apponly.json",
                    "noplatform": "pipeline_reach_v3_noplatform.json"}[MODE])
if len(sys.argv) > 1:
    _seed_arg = sys.argv[1]
else:
    _seed_arg = "42,0,1,2,3"
SEEDS = []
for _seed_text in _seed_arg.split(","):
    SEEDS.append(int(_seed_text))
TOP_K = 25
MAX_VISIT = 200_000


def blocked(label):
    if BLOCK is None:
        return label.startswith(G.LIB_PREFIXES)
    if not BLOCK:
        return False
    return label.startswith(BLOCK)


def sdk_class(label):
    return label.split(";->")[0] + ";"


def reach(nl, su, pr, roots, sdk):
    """Per root signature: (hops, inspected, first SDK class) of the nearest SDK node by caller/callee BFS, or None."""
    lab = {}
    for n, r in nl.items():
        lab[n] = G.canonical_label(r, "fcg")
    out = {}
    for sig in roots:
        start = []
        for n, l in lab.items():
            if l == sig:
                start.append(n)
        if not start:
            out[sig] = None
            continue
        seen = set(start)
        q = deque()
        for n in start:
            q.append((n, 0))
        res = None
        while q and len(seen) <= MAX_VISIT:
            u, d = q.popleft()
            if lab[u].startswith(sdk):
                res = (d, len(seen), sdk_class(lab[u]))
                break
            # callees, then callers
            neighbours = []
            if u in su:
                neighbours.extend(su[u])
            if u in pr:
                neighbours.extend(pr[u])
            for v in neighbours:
                if v in lab and v not in seen and not blocked(lab[v]):
                    seen.add(v)
                    q.append((v, d + 1))
        out[sig] = res
    return out


def run_seed(seed):
    A.SEED = seed
    fams = A.FAMS_V3
    ids, fam, X = A.labels("v3", fams)
    m = A.train(fam, X)
    Ex = X.mean(0)
    classes = list(m.classes_)
    del X
    rng = np.random.default_rng(seed)
    res = {}
    for f in fams:
        w = m.coef_[classes.index(f)].astype(np.float64)
        lst = sorted(ids[fam == f].tolist())
        rng.shuffle(lst)
        agg = defaultdict(float)
        best = {}
        used = 0
        for sha in lst:
            g = H.load_graph(f, sha, "fcg")
            if g is None or len(g[0]) == 0:
                continue
            x, prov, nrm = P.provenance_vector(*g)
            del g
            if x is None:
                continue
            credit = P.split_bucket_credit(prov, w, Ex, nrm, w * (x - Ex))
            for sig, c in credit.items():
                agg[sig] += c
                # remember the app where this signature got the most credit
                if sig in best:
                    prev = best[sig][0]
                else:
                    prev = -np.inf
                if c > prev:
                    best[sig] = (c, sha)
            used += 1
            if used >= A.PER_FAM:
                break
        top = sorted(agg, key=lambda k: -agg[k])[:TOP_K]
        by_app = defaultdict(list)
        for sig in top:
            by_app[best[sig][1]].append(sig)
        rows = {}
        for sha, sigs in by_app.items():
            nl, su, pr = H.load_graph(f, sha, "fcg")
            reached = reach(nl, su, pr, sigs, A.SDK[f])
            for sig, r in reached.items():
                if r is None:
                    hops = None
                    inspected = None
                    first_sdk = None
                else:
                    hops = r[0]
                    inspected = r[1]
                    first_sdk = r[2]
                rows[sig] = {"app": sha, "rank": top.index(sig) + 1, "is_sdk": sig.startswith(A.SDK[f]),
                             "hops": hops, "inspected": inspected,
                             "first_sdk": first_sdk}
            del nl, su, pr
        top_rows = []
        for s in top:
            entry = dict(rows[s])
            entry["method"] = s
            top_rows.append(entry)
        res[f] = {"n_apps": used, "top": top_rows}
        rr = res[f]["top"]
        n_sdk = 0
        n_reach2 = 0
        for r in rr:
            n_sdk += r['is_sdk']
            if r['hops'] is not None and r['hops'] <= 2:
                n_reach2 += 1
        print(f"  seed {seed} {f}: SDK at rank<=25 {n_sdk}, reach<=2 hops "
              f"{n_reach2}/{len(rr)}", flush=True)
    return res


def summarize(runs):
    fams = A.FAMS_V3
    summ = {}
    for f in fams:
        per_seed = []
        for s, r in runs.items():
            rows = r[f]["top"]
            hs = []
            is_sdk = []
            within_1 = []
            within_2 = []
            within_3 = []
            reached_any = []
            inspected = []
            first_classes = []
            for x in rows:
                if x["hops"] is not None:
                    hs.append(x["hops"])
                is_sdk.append(x["is_sdk"])
                within_1.append(x["hops"] is not None and x["hops"] <= 1)
                within_2.append(x["hops"] is not None and x["hops"] <= 2)
                within_3.append(x["hops"] is not None and x["hops"] <= 3)
                reached_any.append(x["hops"] is not None)
                if x["inspected"] is not None:
                    inspected.append(x["inspected"])
                if x["first_sdk"]:
                    first_classes.append(x["first_sdk"])
            if hs:
                median_hops = float(np.median(hs))
                median_inspected = float(np.median(inspected))
            else:
                median_hops = None
                median_inspected = None
            per_seed.append({
                "p_at_25": float(np.mean(is_sdk)),
                "reach_1": float(np.mean(within_1)),
                "reach_2": float(np.mean(within_2)),
                "reach_3": float(np.mean(within_3)),
                "reach_any": float(np.mean(reached_any)),
                "median_hops": median_hops,
                "median_inspected": median_inspected,
                "top1_hops": rows[0]["hops"],
                "first_sdk_classes": Counter(first_classes).most_common(5)})
        # median over seeds of each measure (None if no seed has it)
        med = {}
        for k in ("p_at_25", "reach_1", "reach_2", "reach_3", "reach_any", "median_hops", "median_inspected"):
            vals = []
            for p in per_seed:
                if p[k] is not None:
                    vals.append(p[k])
            if vals:
                med[k] = float(np.median(vals))
            else:
                med[k] = None
        summ[f] = {"median_over_seeds": med, "per_seed": per_seed}
        print(f, med, flush=True)
    return summ


def main():
    assert H.V3, "run with DATASET=proxy_v3"
    if os.path.exists(OUT):
        st = json.load(open(OUT))
    else:
        st = {"runs": {}}
    for seed in SEEDS:
        if str(seed) in st["runs"]:
            continue
        st["runs"][str(seed)] = run_seed(seed)
        json.dump(st, open(OUT, "w"))
    runs = {}
    for s in SEEDS:
        runs[s] = st["runs"][str(s)]
    st["summary"] = summarize(runs)
    st["top_k"] = TOP_K
    st["seeds"] = SEEDS
    json.dump(st, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
