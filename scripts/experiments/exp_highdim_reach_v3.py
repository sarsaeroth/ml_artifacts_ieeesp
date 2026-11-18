#!/usr/bin/env python3
"""Does reach to the SDK's network core follow precision when the hash dimension grows?

Starts: each provider's top-25 methods at d=2^10 and d=2^20 and the sample in which each received the most credit
(highdim_localization.py with HD_DUMP_TOP=25, results/datasets/proxy_v3/highdim_localization.top.json; 60
applications per provider). Reach: exp_route_compare_v3.app_routes (callers and callees, platform APIs not
traversed, asynchronous-callback edges) to the SDK's network core. Starts whose application has no Java network
core are reported apart. Output: results/datasets/proxy_v3/highdim_reach_v3.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H
import exp_route_compare_v3 as RC

D = os.path.join(H.ROOT, "results", "datasets", "proxy_v3")
OUT = os.path.join(D, "highdim_reach_v3.json")


def main():
    top = json.load(open(os.path.join(D, "highdim_localization.top.json")))["results"]
    res = {}
    for dim, e in top.items():
        res[dim] = {}
        for f, v in e["families"].items():
            by_app = defaultdict(set)
            for x in v.get("top", []):
                by_app[x["app"]].add(x["method"])
            # hops from each start to the network core
            hops = []
            no_core = 0
            for sha, sigs in by_app.items():
                r = RC.app_routes(f, sha, sigs)
                for s in sigs:
                    if r is None or r["core_methods"] == 0:
                        no_core += 1
                    else:
                        hops.append(r["explanation"].get(s))
            hop_vals = []
            for h in hops:
                if h is not None:
                    hop_vals.append(h)
                else:
                    hop_vals.append(np.inf)  # unreachable
            a = np.array(hop_vals, dtype=float)
            if len(a):
                within_1 = float(np.mean(a <= 1))
                within_3 = float(np.mean(a <= 3))
                reached = float(np.mean(np.isfinite(a)))
            else:
                within_1 = None
                within_3 = None
                reached = None
            if np.isfinite(a).any():
                median_hops = float(np.median(a[np.isfinite(a)]))
            else:
                median_hops = None
            res[dim][f] = {"p_at_25": v.get("p@25"), "starts_scored": int(len(a)), "starts_without_core": no_core,
                           "within_1": within_1,
                           "within_3": within_3,
                           "reached": reached,
                           "median_hops": median_hops}
            print(dim, f, json.dumps(res[dim][f]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
