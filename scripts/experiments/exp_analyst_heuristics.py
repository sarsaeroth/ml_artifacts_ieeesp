#!/usr/bin/env python3
"""Provenance attribution against the triage an analyst would do by hand, on the same apps.

For each seed, family and label set (exp_label_audit.labels: old, v3), the same 25 apps that
exp_label_audit.run_one samples are ranked four ways over the same candidate methods (FCG methods left
after the vectorizer's standard-library filter, the candidate set of the provenance ranking):
  provenance   exp_label_audit's ranking: exact linear SHAP split over each bucket's contributors
  namespace    methods outside the app's own package (manifest package name) ranked by the number of
               sampled apps they occur in: "foreign code that recurs across the family"
  network_api  methods ranked by the number of distinct network APIs they call (sockets, socket
               channels, TLS sockets, java.net.Proxy, URL connections), summed over apps; ties broken
               by recurrence
  ns_api       network_api restricted to namespace candidates, ties broken by recurrence
  strings      methods ranked by the number of analyst-style strings they load (const-string): proxy
               vocabulary (proxy, socks, tunnel, backconnect, peer, exit node, bandwidth, relay, residential)
               and endpoints (IPv4 addresses, host:port, ws/wss/tcp URLs), summed over apps; ties by recurrence.
               Strings are read from the app's DEX files with androguard (parsing only) and cached per app
               in work/method_strings/
  ns_str       strings restricted to namespace candidates
  no_provenance  every method inherits the largest coefficient of the buckets it hashes to
               (provenance_baselines.py raw_bucket): what the provenance records add
  doc_freq     the number of sampled apps a method occurs in, own package included
               (provenance_baselines.py namespace_freq)
  random       a shuffled order
Ground truth and metrics are exp_label_audit's (SDK namespaces; precision@k, AUC).
Output: results/datasets/proxy_v3/analyst_heuristics.json
Usage: DATASET=proxy_v3 exp_analyst_heuristics.py [SEEDS]   (default 42,0,1,2,3)
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H
import exp_label_audit as A
import gen_wl_vectors as G
import provenance_localization as P

ROOT = H.ROOT
OUT = os.path.join(ROOT, "results", "datasets", "proxy_v3", "analyst_heuristics.json")
# seeds from argv[1] (comma-separated), default 42,0,1,2,3
if len(sys.argv) > 1:
    _seed_arg = sys.argv[1]
else:
    _seed_arg = "42,0,1,2,3"
SEEDS = []
for _s in _seed_arg.split(","):
    SEEDS.append(int(_s))
NET_API = ("Ljava/net/Socket;", "Ljava/net/ServerSocket;", "Ljava/net/InetSocketAddress;", "Ljava/net/Proxy;",
           "Ljava/net/DatagramSocket;", "Ljava/net/URL;->openConnection", "Ljava/net/HttpURLConnection;",
           "Ljava/nio/channels/SocketChannel;", "Ljava/nio/channels/ServerSocketChannel;",
           "Ljava/nio/channels/Selector;", "Ljavax/net/ssl/SSLSocket;", "Ljavax/net/ssl/SSLSocketFactory;",
           "Ljavax/net/SocketFactory;")
METHODS = ("provenance", "no_provenance", "doc_freq", "namespace", "network_api", "ns_api", "strings", "ns_str",
           "random")
STR_PAT = re.compile(r"proxy|socks|tunnel|backconnect|exit.?node|bandwidth|relay|residential|\bpeer"
                     r"|\b\d{1,3}(?:\.\d{1,3}){3}\b|^[a-z0-9.-]+\.[a-z]{2,}:\d{2,5}$|^(?:wss?|tcp)://", re.I)
STR_DIR = os.path.join(ROOT, "work", "method_strings")
MAN_PATH = {}


def run(run_name, seed, pkg_of):
    A.SEED = seed
    if run_name == "v3":
        fams = A.FAMS_V3
    else:
        fams = A.FAMS  # the old corpus has only the four
    ids, fam, X = A.labels(run_name, fams)
    m = A.train(fam, X)
    Ex = X.mean(0)
    classes = list(m.classes_)
    rng = np.random.default_rng(seed)
    res = {}
    for f in fams:
        w = m.coef_[classes.index(f)].astype(np.float64)
        lst = sorted(ids[fam == f].tolist())
        rng.shuffle(lst)  # same order as exp_label_audit.run_one
        prov_s = defaultdict(float)
        df = defaultdict(int)
        df_foreign = defaultdict(int)
        api = defaultdict(int)
        raw_b = defaultdict(float)
        strs = defaultdict(int)
        n_str = 0
        used = 0
        for sha in lst:
            g = H.load_graph(f, sha, "fcg")
            if g is None or len(g[0]) == 0:
                continue
            x, prov, nrm = P.provenance_vector(g[0], g[1], g[2])
            if x is None:
                continue
            phi = w * (x - Ex)
            for meth, c in P.split_bucket_credit(prov, w, Ex, nrm, phi).items():
                prov_s[meth] += c
            for b, s, meth in prov:
                raw_b[meth] = max(raw_b[meth], w[b])

            # {method: number of analyst-style strings it loads}, from the app's DEX files; cached
            ms = None
            cache = os.path.join(STR_DIR, sha + ".json")
            if os.path.exists(cache):
                ms = json.load(open(cache))
            else:
                from androguard.core.apk import APK
                from androguard.core.dex import DEX
                from loguru import logger
                logger.remove()  # androguard logs every parsed class
                out_strings = defaultdict(int)
                strings_failed = False
                try:
                    for raw in APK(os.path.join(ROOT, MAN_PATH[sha])).get_all_dex():
                        for cls in DEX(raw).get_classes():
                            for dex_method in cls.get_methods():
                                code = dex_method.get_code()
                                if code is None:
                                    continue
                                for ins in code.get_bc().get_instructions():
                                    if not ins.get_name().startswith("const-string"):
                                        continue
                                    loaded = ins.get_raw_string()
                                    if not loaded:
                                        loaded = ""
                                    if STR_PAT.search(loaded):
                                        name = f"{dex_method.get_class_name()}->{dex_method.get_name()}{dex_method.get_descriptor()}"
                                        out_strings[name] += 1
                except Exception as e:
                    print(f"    strings {sha[:10]}: {type(e).__name__}", flush=True)
                    strings_failed = True
                if not strings_failed:
                    os.makedirs(STR_DIR, exist_ok=True)
                    json.dump(out_strings, open(cache, "w"))
                    ms = out_strings
            if ms is not None:
                n_str += 1
                for meth, k in ms.items():
                    strs[meth] += k

            nl, su, _ = g
            canon = {}
            for n, raw in nl.items():
                canon[n] = G.canonical_label(raw, "fcg")
            # the app's own package prefix, e.g. Lcom/example/app/
            pkg = pkg_of.get(sha)
            if pkg:
                own = "L" + "/".join(pkg.split(".")[:3]) + "/"
            else:
                own = None
            cands = set()
            for l in canon.values():
                if not l.startswith(G.LIB_PREFIXES):
                    cands.add(l)
            for l in cands:
                df[l] += 1
                if not (own and l.startswith(own)):
                    df_foreign[l] += 1
            # number of distinct successors that are network APIs
            for n, l in canon.items():
                if l in cands:
                    k = 0
                    for s in set(su.get(n, [])):
                        if s not in nl:
                            continue
                        calls_api = False
                        for a in NET_API:
                            if nl[s].startswith(a) or canon[s].startswith(a):
                                calls_api = True
                                break
                        if calls_api:
                            k += 1
                    if k:
                        api[l] += k
            used += 1
            del g, nl, su, canon
            if used >= A.PER_FAM:
                break
        meths = sorted(prov_s)
        truth_list = []
        dfv_list = []
        frv_list = []
        apv_list = []
        stv_list = []
        prov_list = []
        raw_list = []
        for k in meths:
            truth_list.append(k.startswith(A.SDK[f]))
            dfv_list.append(df.get(k, 0))
            frv_list.append(df_foreign.get(k, 0))
            apv_list.append(api.get(k, 0))
            stv_list.append(strs.get(k, 0))
            prov_list.append(prov_s[k])
            raw_list.append(raw_b[k])
        truth = np.array(truth_list)
        dfv = np.array(dfv_list, dtype=float)
        frv = np.array(frv_list, dtype=float)
        apv = np.array(apv_list, dtype=float)
        stv = np.array(stv_list, dtype=float)
        tie = 1.0 / (A.PER_FAM + 1)
        scores = {"provenance": np.array(prov_list),
                  "namespace": frv,
                  "network_api": apv + tie * dfv,
                  "ns_api": np.where(frv > 0, apv + tie * frv, -1.0),
                  "strings": stv + tie * dfv,
                  "ns_str": np.where(frv > 0, stv + tie * frv, -1.0),
                  "no_provenance": np.array(raw_list),
                  "doc_freq": dfv,
                  "random": np.random.default_rng(seed).permutation(len(meths)).astype(float)}
        res[f] = {"n_samples": used, "n_samples_with_strings": n_str, "n_methods": len(meths), "n_sdk": int(truth.sum())}
        # precision@k and AUC per ranking
        for k, v in scores.items():
            order = np.argsort(-v, kind="stable")
            scored = {}
            for top in A.TOP_N:
                scored["p@" + str(top)] = float(truth[order[:top]].mean())
            n_pos = truth.sum()
            if 0 < n_pos and n_pos < len(truth):
                scored["auc"] = float(roc_auc_score(truth, v))
            else:
                scored["auc"] = None
            res[f][k] = scored
        parts = []
        for k in METHODS:
            parts.append(f"{k}={res[f][k]['p@25']:.2f}")
        print(f"  {run_name} seed {seed} {f}: " + " ".join(parts), flush=True)
    return res


def main():
    man = []
    for l in open(os.path.join(ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
        man.append(json.loads(l))
    pkg_of = {}
    for r in man:
        pkg_of[r["sha256"]] = r["package"]
    for r in man:
        if r.get("path"):
            MAN_PATH[r["sha256"]] = r["path"]
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    else:
        out = {"per_family": A.PER_FAM, "runs": {}}
    for run_name in ("v3", "old"):
        for seed in SEEDS:
            key = f"{run_name}/{seed}"
            if key in out["runs"]:
                continue
            out["runs"][key] = run(run_name, seed, pkg_of)
            json.dump(out, open(OUT, "w"), indent=1)
    # median/min/max precision@25 over seeds
    summ = {}
    for run_name in ("v3", "old"):
        if run_name == "v3":
            run_fams = A.FAMS_V3
        else:
            run_fams = A.FAMS
        for f in run_fams:
            for meth in METHODS:
                v = []
                for k in out["runs"]:
                    if k.startswith(run_name + "/"):
                        v.append(out["runs"][k][f][meth]["p@25"])
                summ[f"{run_name}/{f}/{meth}"] = {"median_p@25": float(np.median(v)), "min": min(v), "max": max(v)}
    out["summary_p@25"] = summ
    json.dump(out, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)

if __name__ == "__main__":
    main()
