#!/usr/bin/env python3
"""Two routes to a provider's proxy engine on corpus v3: the classifier explanation (pipeline) and the marker scan.

Target (the same for both routes and every provider): the SDK's network core, the methods in the provider's SDK
namespaces (exp_label_audit.SDK) that call a network API (exp_analyst_heuristics.NET_API: sockets, socket
channels, selectors, TLS sockets, java.net.Proxy, URL connections) or a bundled WebSocket library
(Java-WebSocket, AndroidAsync, OkHttp WebSocket).
  explanation route  start = each top-ranked signature in the application where it received the most
                     attribution (the starts of exp_pipeline_reach_v3, REACH_MODE=noplatform, all seeds)
  marker route       start = where a marker byte scan puts the analyst: the SDK methods that host code calls
                     (host-to-SDK call targets); in builds with no such call (a dormant SDK, or an application
                     whose own code lies in the SDK namespace), the SDK's lifecycle entry methods (onCreate,
                     onStartCommand, onReceive, onBind, onHandleIntent), where the system enters it
Asynchronous dispatch: the FCG records the construction of a task object and the platform call that schedules
it, not the later callback, so a method that constructs class C is also linked to C's run, call, handleMessage,
doInBackground and onPostExecute methods (applied to both routes).
Both routes: breadth-first search over callers and callees in that application's FCG, not traversing platform
APIs (exp_pipeline_reach_v3.PLATFORM); hops = distance to the nearest network-core method. An engine whose network
code is native (Go, C, C++, Rust) has no Java network core; such applications are counted and reported apart.
Per provider: share of starts reaching the core within 1, 2, 3 hops and at all, median hops, for both routes over
the same applications. Output: results/datasets/proxy_v3/route_compare_v3.json
"""
import json
import os
import sys
from collections import defaultdict, deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H
import exp_label_audit as A
import gen_wl_vectors as G
from exp_analyst_heuristics import NET_API

ROOT = H.ROOT
REACH = os.path.join(ROOT, "results", "datasets", "proxy_v3", "pipeline_reach_v3_noplatform.json")
OUT = os.path.join(ROOT, "results", "datasets", "proxy_v3", "route_compare_v3.json")
PLATFORM = ("Landroid/", "Landroidx/", "Ljava/", "Ljavax/", "Lkotlin/", "Lkotlinx/", "Ldalvik/", "Lsun/",
            "Lorg/json/", "Lorg/xmlpull/", "Lorg/w3c/", "Lorg/xml/", "Lorg/apache/http/")
WS_LIBS = ("Lorg/java_websocket/", "Lcom/koushikdutta/async/", "Lokhttp3/WebSocket", "Lokhttp3/OkHttpClient;->newWebSocket")
MAX_VISIT = 200_000
LIFECYCLE = (";->onCreate(", ";->onStartCommand(", ";->onReceive(", ";->onBind(", ";->onHandleIntent(")
CALLBACKS = (";->run()", ";->call()", ";->handleMessage(", ";->doInBackground(", ";->onPostExecute(")


def bfs(lab, adj, starts, target):
    if not starts:
        return None
    seen = set(starts)
    q = deque()
    for n in starts:
        q.append((n, 0))
    while q and len(seen) <= MAX_VISIT:
        u, d = q.popleft()
        if u in target:
            return d
        for v in adj.get(u, ()):
            if v not in seen and not lab[v].startswith(PLATFORM):
                seen.add(v)
                q.append((v, d + 1))
    return None


def app_routes(f, sha, starts_by_sig):
    g = H.load_graph(f, sha, "fcg")
    if g is None:
        return None
    nl, su, pr = g
    lab = {}
    for n, r in nl.items():
        lab[n] = G.canonical_label(r, "fcg")
    sdk = A.SDK[f]
    # undirected call adjacency, no self loops
    adj = defaultdict(list)
    for u, vs in su.items():
        for v in vs:
            if u in lab and v in lab and u != v:
                adj[u].append(v)
                adj[v].append(u)
    # asynchronous dispatch: constructing class C reaches C's callback methods
    cb = defaultdict(list)
    for n, l in lab.items():
        is_callback = False
        for c in CALLBACKS:
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
                for t in cb.get(l.split(";->")[0], ()):
                    if t != u:
                        adj[u].append(t)
                        adj[t].append(u)

    # SDK methods, the network core, and the marker-route starts
    sdk_nodes = set()
    for n, l in lab.items():
        if l.startswith(sdk):
            sdk_nodes.add(n)
    core = set()
    for u in sdk_nodes:
        calls_network = False
        for v in su.get(u, ()):
            if v in lab:
                if lab[v].startswith(NET_API) or lab[v].startswith(WS_LIBS):
                    calls_network = True
                    break
        if calls_network:
            core.add(u)
    host_targets = set()
    for u, vs in su.items():
        if u in lab and not lab[u].startswith(sdk):
            for v in vs:
                if v in sdk_nodes:
                    host_targets.add(v)
    lifecycle = set()
    for n in sdk_nodes:
        is_entry = False
        for e in LIFECYCLE:
            if e in lab[n]:
                is_entry = True
                break
        if is_entry:
            lifecycle.add(n)
    if host_targets:
        marker_starts = host_targets
        marker_start = "host_call"
    elif lifecycle:
        marker_starts = lifecycle
        marker_start = "lifecycle"
    else:
        marker_starts = sdk_nodes
        marker_start = "all_sdk"

    # hops from each route's starts to the core
    if core:
        marker_hops = bfs(lab, adj, marker_starts, core)
    else:
        marker_hops = None
    out = {"sdk_methods": len(sdk_nodes), "core_methods": len(core), "host_entry": bool(host_targets),
           "marker_start": marker_start,
           "marker_hops": marker_hops, "explanation": {}}
    for sig in starts_by_sig:
        st = []
        for n, l in lab.items():
            if l == sig:
                st.append(n)
        if core:
            out["explanation"][sig] = bfs(lab, adj, st, core)
        else:
            out["explanation"][sig] = None
    return out


def rates(h):
    hops = []
    for x in h:
        if x is not None:
            hops.append(x)
        else:
            hops.append(np.inf)
    a = np.array(hops, dtype=float)
    if np.isfinite(a).any():
        median_hops = float(np.median(a[np.isfinite(a)]))
    else:
        median_hops = None
    return {"n": int(len(a)), "within_1": float(np.mean(a <= 1)), "within_2": float(np.mean(a <= 2)),
            "within_3": float(np.mean(a <= 3)), "reached": float(np.mean(np.isfinite(a))),
            "median_hops": median_hops}


def main():
    # explanation starts: family -> app -> signatures, over all seeds
    reach = json.load(open(REACH))
    starts = defaultdict(lambda: defaultdict(set))   # family -> app -> signatures
    for seed, run in reach["runs"].items():
        for f, r in run.items():
            for x in r["top"]:
                starts[f][x["app"]].add(x["method"])

    # per-application routes (resumable)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {"apps": {}}
    for f in A.FAMS_V3:
        for sha, sigs in sorted(starts[f].items()):
            key = f"{f}/{sha}"
            if key in res["apps"]:
                continue
            r = app_routes(f, sha, sigs)
            if r is not None:
                res["apps"][key] = r
                json.dump(res, open(OUT, "w"))
        print(f"[{f}] {len(starts[f])} applications", flush=True)

    # per-provider summary over applications with a Java network core
    summ = {}
    for f in A.FAMS_V3:
        apps = {}
        for k, v in res["apps"].items():
            if k.startswith(f + "/"):
                apps[k.split("/", 1)[1]] = v
        with_core = {}
        for s, v in apps.items():
            if v["core_methods"] > 0:
                with_core[s] = v
        expl = []
        mark = []
        for seed, run in reach["runs"].items():
            for x in run[f]["top"]:
                v = with_core.get(x["app"])
                if v is not None:
                    expl.append(v["explanation"].get(x["method"]))
                    mark.append(v["marker_hops"])
        with_host_entry = 0
        for v in apps.values():
            with_host_entry += v["host_entry"]
        marker_start_counts = {}
        for k in ("host_call", "lifecycle", "all_sdk"):
            count = 0
            for v in apps.values():
                if v.get("marker_start") == k:
                    count += 1
            marker_start_counts[k] = count
        summ[f] = {"apps": len(apps), "apps_with_java_core": len(with_core),
                   "apps_with_host_entry": with_host_entry,
                   "marker_start": marker_start_counts,
                   "explanation": rates(expl), "marker": rates(mark)}
        print(f, json.dumps(summ[f]), flush=True)
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
