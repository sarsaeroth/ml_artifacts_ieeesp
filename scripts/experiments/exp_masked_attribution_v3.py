#!/usr/bin/env python3
"""Does co-occurring code displace the SDK in the explanation? Attribution with and without advertising code,
on corpus v3, plus the depth at which the first SDK method appears in each ranking.

Two models, trained on the same applications (up to 200 per provider, seed 42, the sample of exp_confound_v3):
  full        identifier WL over every method of the FCG (d=1,024)
  ad_masked   the same with advertising/analytics/tracking methods removed (ad_confound_check.AD_PREFIXES)
Vectors are built as in exp_confound_v3 (gen_wl_vectors.wl_hashed_vector on the node-filtered graph); the
classifier is exp_label_audit.train (linear SGD, alpha 1e-4, upsampled, seed 42); E[x] over the training apps.
For each sampling seed (default 42,0,1,2,3) and provider, the 25 sampled applications of exp_label_audit are
attributed under both models (provenance_localization.provenance_vector on the same node-filtered graph, exact
split), aggregated by signature, and ranked. Reported per provider, condition and seed:
  p@25          share of the top 25 in the provider's SDK namespaces
  first_sdk     rank of the first SDK method (time to first hit; None if no SDK method is ranked)
  sdk_in_100    SDK methods among the top 100
  reach         from each top-25 method in the application where it received the most credit, the
                breadth-first distance to the SDK's network core (exp_route_compare_v3.app_routes: callers and
                callees, platform APIs not traversed, asynchronous-callback edges): within 3 hops, reached
Output: results/datasets/proxy_v3/masked_attribution_v3.json (resumable per seed)
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import ad_confound_check as AC
import exp_harness as H
import exp_label_audit as A
import exp_route_compare_v3 as RC
import gen_wl_vectors as G
import provenance_localization as P

OUT = os.path.join(H.ROOT, "results", "datasets", "proxy_v3", "masked_attribution_v3.json")
if len(sys.argv) > 1:
    seed_arg = sys.argv[1]
else:
    seed_arg = "42,0,1,2,3"
SEEDS = []
for seed_text in seed_arg.split(","):
    SEEDS.append(int(seed_text))
DIM = 1024
TRAIN_PER_FAM = 200
TOP_K = 25
CONDS = {"full": lambda l: True, "ad_masked": lambda l: not l.startswith(AC.AD_PREFIXES)}


def class_apps():
    man = []
    for l in open(os.path.join(H.ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
        man.append(json.loads(l))
    by = defaultdict(list)
    for m in sorted(man, key=lambda m: m["sha256"]):
        if m["status"] == "kept" and m.get("split") == "class":
            by[m["family"]].append(m["sha256"])
    return by


def filtered(nl, keep):
    out = {}
    for n, raw in nl.items():
        if keep(G.canonical_label(raw, "fcg")):
            out[n] = raw
    return out


def train_models(by):
    # vectors under every condition for up to TRAIN_PER_FAM apps per provider
    rng = np.random.default_rng(42)
    X = {}
    for c in CONDS:
        X[c] = []
    fam = []
    for f in A.FAMS_V3:
        lst = by[f][:]
        rng.shuffle(lst)
        used = 0
        for sha in lst:
            g = H.load_graph(f, sha, "fcg")
            if g is None or not g[0]:
                continue
            nl, su, pr = g
            for c, keep in CONDS.items():
                sub = filtered(nl, keep)
                if sub:
                    X[c].append(G.wl_hashed_vector(sub, su, pr, kind="fcg", dim=DIM))
                else:
                    X[c].append(np.zeros(DIM))
            fam.append(f)
            used += 1
            del g, nl, su, pr
            if used >= TRAIN_PER_FAM:
                break
        print(f"  train {f}: {used}", flush=True)

    # one model and training mean per condition
    fam = np.array(fam)
    models = {}
    for c in CONDS:
        Xc = np.array(X[c], dtype=np.float64)
        models[c] = (A.train(fam, Xc), Xc.mean(0))
    return models


def run_seed(seed, models, by):
    rng = np.random.default_rng(seed)
    res = {}
    for f in A.FAMS_V3:
        # aggregate signature credit over the sampled applications
        lst = sorted(by[f])
        rng.shuffle(lst)
        agg = {}
        for c in CONDS:
            agg[c] = defaultdict(float)
        best = {}
        for c in CONDS:
            best[c] = {}
        used = 0
        for sha in lst:
            g = H.load_graph(f, sha, "fcg")
            if g is None or not g[0]:
                continue
            nl, su, pr = g
            ok = False
            for c, keep in CONDS.items():
                m, Ex = models[c]
                w = m.coef_[list(m.classes_).index(f)].astype(np.float64)
                x, prov, nrm = P.provenance_vector(filtered(nl, keep), su, pr)
                if x is None:
                    continue
                ok = True
                credit = P.split_bucket_credit(prov, w, Ex, nrm, w * (x - Ex))
                for sig, cr in credit.items():
                    agg[c][sig] += cr
                    if sig in best[c]:
                        best_cr = best[c][sig][0]
                    else:
                        best_cr = -np.inf
                    if cr > best_cr:
                        best[c][sig] = (cr, sha)
            del g, nl, su, pr
            used += ok
            if used >= A.PER_FAM:
                break

        # rank, first SDK hit, and reach of the top signatures
        res[f] = {}
        for c in CONDS:
            order = sorted(agg[c], key=lambda k: -agg[c][k])
            is_sdk = []
            for s in order:
                is_sdk.append(s.startswith(A.SDK[f]))
            first = None
            for i, v in enumerate(is_sdk):
                if v:
                    first = i + 1
                    break
            top = order[:TOP_K]
            by_app = defaultdict(set)
            for s in top:
                by_app[best[c][s][1]].add(s)
            hops = []
            for sha, sigs in by_app.items():
                r = RC.app_routes(f, sha, sigs)
                for s in sigs:
                    if r is None or r["core_methods"] == 0:
                        hops.append(None)
                    else:
                        hops.append(r["explanation"].get(s))
            hop_values = []
            for h in hops:
                if h is not None:
                    hop_values.append(h)
                else:
                    hop_values.append(np.inf)
            a = np.array(hop_values, dtype=float)
            sdk_in_100 = 0
            for v in is_sdk[:100]:
                if v:
                    sdk_in_100 += 1
            res[f][c] = {"p_at_25": float(np.mean(is_sdk[:TOP_K])), "first_sdk": first,
                         "sdk_in_100": int(sdk_in_100), "candidates": len(order),
                         "reach_within_3": float(np.mean(a <= 3)), "reach_any": float(np.mean(np.isfinite(a)))}
        parts = []
        for c in CONDS:
            parts.append(f"{c} p@25 {res[f][c]['p_at_25']:.2f} first {res[f][c]['first_sdk']} reach3 {res[f][c]['reach_within_3']:.2f}")
        print(f"  seed {seed} {f}: " + "; ".join(parts), flush=True)
    return res


def main():
    assert H.V3, "run with DATASET=proxy_v3"
    # per-seed runs (resumable)
    if os.path.exists(OUT):
        st = json.load(open(OUT))
    else:
        st = {"runs": {}}
    by = class_apps()
    models = train_models(by)
    for seed in SEEDS:
        if str(seed) in st["runs"]:
            continue
        st["runs"][str(seed)] = run_seed(seed, models, by)
        json.dump(st, open(OUT, "w"))

    # medians over seeds
    summ = {}
    for f in A.FAMS_V3:
        summ[f] = {}
        for c in CONDS:
            rows = []
            for s in SEEDS:
                rows.append(st["runs"][str(s)][f][c])
            fs = []
            for r in rows:
                if r["first_sdk"] is not None:
                    fs.append(r["first_sdk"])
            summ[f][c] = {}
            for k in ("p_at_25", "sdk_in_100", "reach_within_3", "reach_any"):
                values = []
                for r in rows:
                    values.append(r[k])
                summ[f][c][k] = float(np.median(values))
            if fs:
                summ[f][c]["first_sdk_median"] = float(np.median(fs))
            else:
                summ[f][c]["first_sdk_median"] = None
            summ[f][c]["first_sdk_found_seeds"] = len(fs)
        print(f, json.dumps(summ[f]), flush=True)
    st["summary"] = summ
    json.dump(st, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
