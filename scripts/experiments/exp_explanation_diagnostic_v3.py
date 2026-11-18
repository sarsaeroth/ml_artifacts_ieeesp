#!/usr/bin/env python3
"""Can an analyst tell in advance whether a provider's explanation will reach its SDK? Per-provider predictors
computable without reverse engineering the SDK's protocol, against the observed outcomes, on corpus v3.

Predictors (exp_confound_v3, per-provider F1 under identical DEX-grouped folds, 961 applications):
  sdk_only_f1       how well the SDK namespaces alone separate the provider
  sdk_necessity     full F1 minus SDK-masked F1: how much separation is lost without the SDK
  sdk_share         median share of the provider's non-library methods that lie in its SDK namespaces
Outcomes:
  p_at_25           median precision@25 of the provenance ranking over five seeds (analyst_heuristics.json)
  core_within_3     share of explanation starts reaching the SDK's network core within 3 hops (route_compare_v3)
  core_reached      share of explanation starts reaching it at all
Reported: the per-provider table and Spearman's rank correlation of each predictor with each outcome. With eight
providers the correlations illustrate the relationship rather than establish it.
Output: results/datasets/proxy_v3/explanation_diagnostic_v3.json
"""
import json
import os
import statistics
import sys

from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H
import exp_label_audit as A

D = os.path.join(H.ROOT, "results", "datasets", "proxy_v3")
OUT = os.path.join(D, "explanation_diagnostic_v3.json")


def main():
    conf = json.load(open(os.path.join(D, "confound_v3.json")))
    heur = json.load(open(os.path.join(D, "analyst_heuristics.json")))["runs"]
    route = json.load(open(os.path.join(D, "route_compare_v3.json")))["summary"]
    c = conf["conditions"]
    rows = {}
    for f in A.FAMS_V3:
        # median precision@25 over the corpus-v3 seed runs
        p25 = []
        for k, v in heur.items():
            if k.startswith("v3/"):
                p25.append(v[f]["provenance"]["p@25"])
        rows[f] = {"sdk_only_f1": c["sdk_only"]["per_family_f1"][f],
                   "sdk_necessity": c["full"]["per_family_f1"][f] - c["sdk_masked"]["per_family_f1"][f],
                   "sdk_share": conf["share_median"][f]["sdk"],
                   "p_at_25": statistics.median(p25),
                   "core_within_3": route[f]["explanation"]["within_3"],
                   "core_reached": route[f]["explanation"]["reached"]}
    corr = {}
    for p in ("sdk_only_f1", "sdk_necessity", "sdk_share"):
        for o in ("p_at_25", "core_within_3", "core_reached"):
            pred_vals = []
            outcome_vals = []
            for f in rows:
                pred_vals.append(rows[f][p])
                outcome_vals.append(rows[f][o])
            r = spearmanr(pred_vals, outcome_vals)
            corr[f"{p}~{o}"] = {"rho": float(r.correlation), "p": float(r.pvalue)}
    for f, r in rows.items():
        parts = []
        for k, v in r.items():
            parts.append(f"{k}={v:.3f}")
        print(f"{f:15s} " + " ".join(parts), flush=True)
    for k, v in corr.items():
        print(f"{k:35s} rho={v['rho']:+.2f} p={v['p']:.3f}", flush=True)
    json.dump({"providers": rows, "spearman": corr}, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
