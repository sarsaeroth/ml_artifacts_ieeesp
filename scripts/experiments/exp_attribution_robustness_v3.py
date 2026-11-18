#!/usr/bin/env python3
"""Two robustness checks on the provenance ranking, corpus v3: the background-allocation convention and
in-sample versus out-of-fold attribution.

For each sampling seed (default 42,0,1,2,3) and classifier provider, the 25 applications of exp_label_audit are
attributed four ways and each ranking is aggregated by method signature:
  exact        the paper's rule (provenance_localization.split_bucket_credit, exact): token term
               w_b s_t / ||v|| minus the background term w_b E[x_b] shared by |s_t| / sum|s|; model trained on the
               whole corpus (exp_label_audit.train), E[x] over the corpus (in-sample, as in the attribution table)
  uncentered   the token term alone, w_b s_t / ||v||, no background share (same model)
  legacy       the earlier proportional rule, phi_b s_t / sum|s| (same model)
  out_of_fold  exact rule, but each application is attributed by the model of the DEX-grouped fold
               (exp_harness.folds, StratifiedGroupKFold(5), repeat 0) in whose test part it lies, with E[x] over
               that fold's training applications
Reported per provider, ranking, and seed: precision@25 against the SDK namespaces, the rank of the first SDK method,
and the overlap of the top 100 with the exact ranking's top 100.
Output: results/datasets/proxy_v3/attribution_robustness_v3.json (resumable per seed)
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
import provenance_localization as P

OUT = os.path.join(H.ROOT, "results", "datasets", "proxy_v3", "attribution_robustness_v3.json")
# seeds from argv[1] (comma-separated), default 42,0,1,2,3
if len(sys.argv) > 1:
    _seed_arg = sys.argv[1]
else:
    _seed_arg = "42,0,1,2,3"
SEEDS = []
for _s in _seed_arg.split(","):
    SEEDS.append(int(_s))
RULES = ("exact", "uncentered", "legacy", "out_of_fold")


def metrics(agg, sdk, ref_top):
    order = sorted(agg, key=lambda k: -agg[k])
    is_sdk = []
    for s in order:
        is_sdk.append(s.startswith(sdk))
    top100 = set(order[:100])
    # 1-based rank of the first SDK method, None if there is none
    first_sdk = None
    for i, v in enumerate(is_sdk):
        if v:
            first_sdk = i + 1
            break
    if ref_top is not None:
        overlap = len(top100 & ref_top) / 100.0
    else:
        overlap = 1.0
    return {"p_at_25": float(np.mean(is_sdk[:25])),
            "first_sdk": first_sdk,
            "overlap_top100": overlap}, order


def main():
    assert H.V3, "run with DATASET=proxy_v3"
    if os.path.exists(OUT):
        st = json.load(open(OUT))
    else:
        st = {"runs": {}}
    ids, fam, X = A.labels("v3", A.FAMS_V3)
    y, _ = H.labels(fam)
    groups = H.groups_for(ids)
    # out-of-fold models: each app maps to the model of the fold whose test part holds it
    fold_models = {}
    for r, k, tr, te in H.folds(y, groups, 1):
        fm = A.train(fam[tr], X[tr])
        fEx = X[tr].mean(0)
        for j in te:
            fold_models[ids[j]] = (fm, fEx)
        print(f"  fold {k}: trained on {len(tr)}, tests {len(te)}", flush=True)
    for seed in SEEDS:
        if str(seed) in st["runs"]:
            continue
        # one seed: corpus-trained model, 25 sampled apps per provider, four attribution rules
        A.SEED = seed
        m = A.train(fam, X)
        Ex = X.mean(0)
        classes = list(m.classes_)
        rng = np.random.default_rng(seed)
        res = {}
        for f in A.FAMS_V3:
            w = m.coef_[classes.index(f)].astype(np.float64)
            lst = sorted(ids[fam == f].tolist())
            rng.shuffle(lst)
            agg = {}
            for rule in RULES:
                agg[rule] = defaultdict(float)
            used = 0
            for sha in lst:
                g = H.load_graph(f, sha, "fcg")
                if g is None or len(g[0]) == 0:
                    continue
                x, prov, nrm = P.provenance_vector(g[0], g[1], g[2])
                del g
                if x is None:
                    continue
                phi = w * (x - Ex)
                for k, v in P.split_bucket_credit(prov, w, Ex, nrm, phi, rule="exact").items():
                    agg["exact"][k] += v
                # uncentered: token term w_b s_t / ||v|| only, summed per app before aggregating
                credit = defaultdict(float)
                for b, s, meth in prov:
                    credit[meth] += w[b] * (s / nrm)
                for k, v in credit.items():
                    agg["uncentered"][k] += v
                for k, v in P.split_bucket_credit(prov, w, Ex, nrm, phi, rule="legacy").items():
                    agg["legacy"][k] += v
                fm, fEx = fold_models[sha]
                fw = fm.coef_[list(fm.classes_).index(f)].astype(np.float64)
                for k, v in P.split_bucket_credit(prov, fw, fEx, nrm, fw * (x - fEx), rule="exact").items():
                    agg["out_of_fold"][k] += v
                used += 1
                if used >= A.PER_FAM:
                    break
            exact_m, exact_order = metrics(agg["exact"], A.SDK[f], None)
            ref = set(exact_order[:100])
            res[f] = {"exact": exact_m}
            for rule in RULES[1:]:
                res[f][rule] = metrics(agg[rule], A.SDK[f], ref)[0]
            parts = []
            for rule in RULES:
                parts.append(f"{rule} p@25 {res[f][rule]['p_at_25']:.2f} first {res[f][rule]['first_sdk']} "
                             f"ov {res[f][rule]['overlap_top100']:.2f}")
            print(f"  seed {seed} {f}: " + "; ".join(parts), flush=True)
        st["runs"][str(seed)] = res
        json.dump(st, open(OUT, "w"))
    # medians over seeds per provider and rule
    summ = {}
    for f in A.FAMS_V3:
        summ[f] = {}
        for r in RULES:
            rows = []
            for s in SEEDS:
                rows.append(st["runs"][str(s)][f][r])
            fs = []
            p25 = []
            overlaps = []
            for x in rows:
                if x["first_sdk"] is not None:
                    fs.append(x["first_sdk"])
                p25.append(x["p_at_25"])
                overlaps.append(x["overlap_top100"])
            if fs:
                first_sdk_median = float(np.median(fs))
            else:
                first_sdk_median = None
            summ[f][r] = {"p_at_25": float(np.median(p25)),
                          "first_sdk": first_sdk_median,
                          "overlap_top100": float(np.median(overlaps))}
        print(f, json.dumps(summ[f]), flush=True)
    st["summary"] = summ
    json.dump(st, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)

if __name__ == "__main__":
    main()
