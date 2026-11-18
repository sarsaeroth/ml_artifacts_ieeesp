#!/usr/bin/env python3
"""The explanation as a bootstrap, end to end on corpus v3: explanation -> first SDK method -> namespace marker ->
network engine.

For each sampling seed (default 42,0,1,2,3) and each classifier provider:
  1 explanation  exp_label_audit's provenance ranking (corpus-trained linear model, exact split, aggregated by
                 signature over the provider's 25 sampled applications), read from the top; the first method in
                 the provider's SDK namespaces (exp_label_audit.SDK, standing in for the analyst recognizing SDK
                 code) is the first hit. Recorded: its rank, method, class.
  2 marker       the first hit's package (its class name without the last component) is the derived marker; it
                 uses nothing an analyst would not have at that point
  3 marker route in each of the 25 sampled applications: whether the derived marker occurs (coverage across
                 builds), and, where it does, the breadth-first distance from where the analyst enters the marked
                 package (host-code calls into it, else its lifecycle entry methods, else all its methods) to the
                 SDK's network core (methods in the SDK namespaces calling network APIs or a WebSocket library),
                 with the search rules of exp_route_compare_v3 (callers and callees, platform APIs not traversed,
                 asynchronous-callback edges). Applications without a Java network core are counted apart.
Output: results/datasets/proxy_v3/bootstrap_chain_v3.json (resumable per seed)
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
import exp_route_compare_v3 as RC
import gen_wl_vectors as G
import provenance_localization as P

OUT = os.path.join(H.ROOT, "results", "datasets", "proxy_v3", "bootstrap_chain_v3.json")
# seeds from argv[1] (comma-separated), default 42,0,1,2,3
if len(sys.argv) > 1:
    _seed_arg = sys.argv[1]
else:
    _seed_arg = "42,0,1,2,3"
SEEDS = []
for _s in _seed_arg.split(","):
    SEEDS.append(int(_s))


def marker_route(f, sha, marker):
    # call graph of the app: labels, successors, and an undirected adjacency with callback edges
    g = H.load_graph(f, sha, "fcg")
    if g is None or not g[0]:
        return None
    nl, su, pr = g
    lab = {}
    for n, r in nl.items():
        lab[n] = G.canonical_label(r, "fcg")
    adj = defaultdict(list)
    for u, vs in su.items():
        for v in vs:
            if u in lab and v in lab and u != v:
                adj[u].append(v)
                adj[v].append(u)
    # asynchronous callbacks: constructing a class links the caller to that class's callback methods
    cb = defaultdict(list)
    for n, l in lab.items():
        is_callback = False
        for c in RC.CALLBACKS:
            if c in l:
                is_callback = True
                break
        if is_callback:
            cb[l.split(";->")[0]].append(n)
    for u, vs in su.items():
        if u not in lab:
            continue
        for v in vs:
            l = lab.get(v)
            if l and ";-><init>(" in l:
                init_cls = l.split(";->")[0]
                if init_cls in cb:
                    targets = cb[init_cls]
                else:
                    targets = ()
                for t in targets:
                    if t != u:
                        adj[u].append(t)
                        adj[t].append(u)
    del g, nl, pr, cb  # only lab, su and adj are used below
    sdk = A.SDK[f]
    marked = set()
    for n, l in lab.items():
        if l.startswith(marker):
            marked.add(n)
    # network core: SDK methods calling a network API or a WebSocket library
    core = set()
    for u, l in lab.items():
        if not l.startswith(sdk):
            continue
        calls_net = False
        for v in su.get(u, ()):
            if v in lab and (lab[v].startswith(RC.NET_API) or lab[v].startswith(RC.WS_LIBS)):
                calls_net = True
                break
        if calls_net:
            core.add(u)
    out = {"marker_present": bool(marked), "core_methods": len(core)}
    if not marked or not core:
        return out
    # entry points: host calls into the marked package, else its lifecycle methods, else all of it
    host = set()
    for u, vs in su.items():
        if u in lab and not lab[u].startswith(marker):
            for v in vs:
                if v in marked:
                    host.add(v)
    life = set()
    for n in marked:
        for e in RC.LIFECYCLE:
            if e in lab[n]:
                life.add(n)
                break
    if host:
        starts = host
        out["entry"] = "host_call"
    elif life:
        starts = life
        out["entry"] = "lifecycle"
    else:
        starts = marked
        out["entry"] = "all_marked"
    out["hops"] = RC.bfs(lab, adj, starts, core)
    return out


def run_seed(seed):
    A.SEED = seed
    ids, fam, X = A.labels("v3", A.FAMS_V3)
    m = A.train(fam, X)
    Ex = X.mean(0)
    classes = list(m.classes_)
    del X
    rng = np.random.default_rng(seed)
    res = {}
    for f in A.FAMS_V3:
        w = m.coef_[classes.index(f)].astype(np.float64)
        lst = sorted(ids[fam == f].tolist())
        rng.shuffle(lst)
        # provenance ranking aggregated by signature over the sampled apps
        agg = defaultdict(float)
        apps = []
        for sha in lst:
            g = H.load_graph(f, sha, "fcg")
            if g is None or len(g[0]) == 0:
                continue
            x, prov, nrm = P.provenance_vector(g[0], g[1], g[2])
            del g
            if x is None:
                continue
            phi = w * (x - Ex)
            for sig, c in P.split_bucket_credit(prov, w, Ex, nrm, phi).items():
                agg[sig] += c
            apps.append(sha)
            if len(apps) >= A.PER_FAM:
                break
        order = sorted(agg, key=lambda k: -agg[k])
        # first SDK method from the top of the ranking
        first = None
        for i, s in enumerate(order):
            if s.startswith(A.SDK[f]):
                first = (i + 1, s)
                break
        if first is None:
            r = {"candidates": len(order), "first_rank": None, "first_method": None}
        else:
            r = {"candidates": len(order), "first_rank": first[0], "first_method": first[1]}
        if first is not None:
            # marker = package of the first hit (class name without its last component)
            first_cls = first[1].split(";->")[0]
            marker = first_cls.rsplit("/", 1)[0] + "/"
            r["marker"] = marker
            routes = []
            for sha in apps:
                route = marker_route(f, sha, marker)
                if route is not None:
                    routes.append(route)
            present = []
            for route in routes:
                if route["marker_present"]:
                    present.append(route)
            scored = []
            for route in present:
                if route["core_methods"] > 0:
                    scored.append(route)
            hop_list = []
            entries = []
            for route in scored:
                if route.get("hops") is not None:
                    hop_list.append(route["hops"])
                else:
                    hop_list.append(np.inf)
                entries.append(route.get("entry"))
            hops = np.array(hop_list, dtype=float)
            if len(hops):
                reached = float(np.mean(np.isfinite(hops)))
                within_3 = float(np.mean(hops <= 3))
            else:
                reached = None
                within_3 = None
            if np.isfinite(hops).any():
                median_hops = float(np.median(hops[np.isfinite(hops)]))
            else:
                median_hops = None
            r.update({"apps": len(routes), "marker_coverage": len(present) / max(1, len(routes)),
                      "apps_scored": len(scored), "apps_without_core": len(present) - len(scored),
                      "reached": reached,
                      "within_3": within_3,
                      "median_hops": median_hops,
                      "entry": dict(Counter(entries))})
        res[f] = r
        print(f"  seed {seed} {f}: first SDK rank {r['first_rank']} {r.get('first_method', '')[:80]} | marker "
              f"{r.get('marker')} coverage {r.get('marker_coverage')} reached {r.get('reached')} "
              f"median hops {r.get('median_hops')}", flush=True)
    return res


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
    # per provider: median over seeds of each metric, and the markers / first classes found
    summ = {}
    for f in A.FAMS_V3:
        rows = []
        for s in SEEDS:
            rows.append(st["runs"][str(s)][f])
        summ[f] = {}
        for k in ("first_rank", "marker_coverage", "reached", "within_3", "median_hops"):
            vals = []
            for r in rows:
                if r.get(k) is not None:
                    vals.append(r[k])
            if vals:
                summ[f][k] = float(np.median(vals))
            else:
                summ[f][k] = None
        markers = []
        first_classes = []
        for r in rows:
            markers.append(r.get("marker"))
            first_method = r.get("first_method")
            if not first_method:
                first_method = ""
            first_classes.append(first_method.split(";->")[0])
        summ[f]["markers"] = dict(Counter(markers))
        summ[f]["first_classes"] = dict(Counter(first_classes))
        print(f, json.dumps(summ[f]), flush=True)
    st["summary"] = summ
    json.dump(st, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)

if __name__ == "__main__":
    main()
