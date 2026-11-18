#!/usr/bin/env python3
"""Validate provenance-guided ranking against simpler baselines.

The claim under test is that provenance-aware inversion of hashed WL features ranks
proxy SDK methods above host code well enough to prioritise reverse engineering.
That claim is only meaningful relative to what simpler procedures achieve, so this
scores four rankings over the SAME candidate method set, per family:

  provenance  signed SHAP credit split across a bucket's contributing tokens,
              q(m) = sum_a sum_i phi_i^(a) * c_{a,m,i} / n_{a,i}
              with c the signed contribution of m to bucket i and n_{a,i} the
              number of tokens in that bucket. This is the paper's method.
  raw_bucket  no provenance: every method inherits the coefficient of the bucket
              it hashes to, max over its buckets. Isolates what provenance adds.
  namespace   rank by how many samples of the family a method appears in
              (document frequency). Needs no model at all.
  random      shuffled order, the floor.

Metrics are precision@k and ROC-AUC against SDK membership, where ground truth is
the authoritative SDK namespace per family. Oxylabs is reported but flagged: its
identifiers are rewritten per build, so namespace membership under-counts its SDK.

Outputs: results/provenance_baselines.json
"""
import json
import os
import sys
from collections import defaultdict, Counter

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import gen_wl_vectors as G
import train_wl_only_classifier as T
import provenance_localization as P
import graph_io as GI

PROXY = ["ipninja", "luminati", "monkeysocks", "oxylabs"]
SDK_PREFIXES = P.SDK_PREFIXES
DIM = P.DIM
SEED = P.SEED
PER_FAM = P.PER_FAM
MAX_FCG = P.MAX_FCG
MODEL = P.MODEL
TOP_N = [10, 25, 50, 100]
OUT = os.path.join(ROOT, "results", "provenance_baselines.json")
if os.environ.get("PROVENANCE_SPLIT", "exact") != "legacy":
    OUT = OUT.replace(".json", "_exactsplit.json")


def main():
    rng = np.random.default_rng(SEED)
    d = joblib.load(MODEL)
    clf = d["clf"]
    le = d["label_encoder"]
    classes = []
    for c in le.classes_:
        classes.append(str(c))

    byfam = defaultdict(list)
    for hdir, gid, lab in T.build_samples():
        if lab in PROXY:
            byfam[lab].append((hdir, gid))

    Xall = []
    for fam in PROXY:
        for hdir, gid in byfam[fam]:
            p = os.path.join(hdir, f"{gid}.fcg_wl_{DIM}")
            if os.path.isfile(p):
                try:
                    Xall.append(np.load(p, allow_pickle=False))
                except Exception:
                    pass
    Ex = np.vstack(Xall).mean(0).astype(np.float64)
    print(f"E[x] from {len(Xall)} vectors", flush=True)

    result = {"dim": DIM, "per_family_samples": PER_FAM, "seed": SEED,
              "top_n": TOP_N, "families": {}}

    for fam in PROXY:
        ci = classes.index(fam)
        w = clf.coef_[ci].astype(np.float64)
        lst = byfam[fam][:]
        rng.shuffle(lst)

        failures = Counter()
        prov_agg = defaultdict(float)
        raw_agg = defaultdict(float)
        docfreq = defaultdict(int)
        used = 0
        for hdir, gid in lst:
            fcg = os.path.join(hdir, f"{gid}.fcg")
            if not GI.exists(fcg) or GI.size(fcg) > MAX_FCG:
                continue
            try:
                nl, su, pr = G.parse_gml_lightweight(fcg)
                x, prov, nrm = P.provenance_vector(nl, su, pr)
            except Exception as e:
                failures[type(e).__name__] += 1
                continue
            if x is None:
                failures["empty_vector"] += 1
                continue
            phi = w * (x - Ex)
            # split rule: P.SPLIT ("exact" default, PROVENANCE_SPLIT=legacy for the published rule)
            for m, c in P.split_bucket_credit(prov, w, Ex, nrm, phi).items():
                prov_agg[m] += c
            for b, s, m in prov:
                # no-provenance baseline: method inherits the bucket coefficient
                raw_agg[m] = max(raw_agg[m], w[b])
            prov_methods = set()
            for p in prov:
                prov_methods.add(p[2])
            for m in prov_methods:
                docfreq[m] += 1
            used += 1
            if used >= PER_FAM:
                break

        if failures:
            print(f"  [{fam}] skipped samples: {dict(failures)}", flush=True)
        labels = sorted(prov_agg)
        if not labels:
            print(f"  [{fam}] NO CANDIDATES - not scored", flush=True)
            continue
        # ground truth: SDK namespace membership
        truth_list = []
        for m in labels:
            if m.startswith(SDK_PREFIXES[fam]):
                truth_list.append(1)
            else:
                truth_list.append(0)
        truth = np.array(truth_list)
        base = float(truth.mean())

        # the four rankings over the same candidate set
        prov_scores = []
        raw_scores = []
        ns_scores = []
        for m in labels:
            prov_scores.append(prov_agg[m])
            raw_scores.append(raw_agg[m])
            ns_scores.append(float(docfreq[m]))
        rankings = {
            "provenance": np.array(prov_scores),
            "raw_bucket": np.array(raw_scores),
            "namespace_freq": np.array(ns_scores),
            "random": rng.permutation(len(labels)).astype(float),
        }
        fam_out = {"samples_used": used, "candidate_methods": len(labels),
                   "sdk_methods": int(truth.sum()), "base_rate": round(base, 5),
                   "rankings": {}}
        # precision@k and ROC-AUC of each ranking
        for name, sc in rankings.items():
            order = np.argsort(-sc)
            scored = {}
            for k in TOP_N:
                n_top = min(k, len(labels))
                scored[f"p@{k}"] = float(truth[order[:n_top]].sum() / n_top)
            n_sdk = int(truth.sum())
            if 0 < n_sdk and n_sdk < len(truth):
                scored["auc"] = float(roc_auc_score(truth, sc))
            else:
                scored["auc"] = None
            fam_out["rankings"][name] = scored
        if fam == "oxylabs":
            fam_out["caveat"] = ("SDK identifiers are rewritten per build, so namespace "
                                 "membership under-counts true SDK methods; treat as a "
                                 "lower bound rather than a clean ground truth")
        result["families"][fam] = fam_out

        print(f"\n=== {fam}  (n={used} samples, {len(labels)} methods, "
              f"base rate {base:.4f}) ===", flush=True)
        print(f"  {'ranking':16}{'p@10':>8}{'p@25':>8}{'p@100':>8}{'AUC':>8}", flush=True)
        for name in ["provenance", "raw_bucket", "namespace_freq", "random"]:
            r = fam_out["rankings"][name]
            if r["auc"] is not None:
                auc = f"{r['auc']:.3f}"
            else:
                auc = "n/a"
            print(f"  {name:16}{r['p@10']:>8.3f}{r['p@25']:>8.3f}"
                  f"{r['p@100']:>8.3f}{auc:>8}", flush=True)

    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
