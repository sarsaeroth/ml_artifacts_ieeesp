#!/usr/bin/env python3
"""Deletion experiment on corpus v3: are the attributed methods necessary for the decision?

localization_faithfulness.py on the eight corpus-v3 classes, with the attribution the paper uses. A linear
FCG model (identifier WL, d=1024, the packed fcg_wl_1024 vectors; SGD log loss, alpha 1e-5, in-fold
upsampling) is trained per fold of one DEX-grouped StratifiedGroupKFold(5) split, so every test
application is scored by a model that did not see it or its DEX group. For up to PER_FAM test applications
per class, every non-library method is credited with its provenance attribution toward the true class
(provenance_localization.provenance_vector + split_bucket_credit, the exact SHAP split). The top fraction p
of methods by credit is deleted from the FCG, the graph re-vectorized with the same function, and the
application re-classified; the control deletes a random fraction p. Accuracy is the share still assigned
to the true class.
Output: results/datasets/proxy_v3/deletion_v3.json
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
os.environ.setdefault("GRAPHS", "fcg")
import exp_harness as H
import provenance_localization as P

OUT = os.environ.get("DEL_OUT")
if not OUT:  # unset or empty: default path
    OUT = os.path.join(H.ROOT, "results", "datasets", "proxy_v3", "deletion_v3.json")
P_GRID = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
PER_FAM = int(os.environ.get("DEL_PER_FAM", "25"))
SEED = 42


def main():
    assert H.V3, "run with DATASET=proxy_v3"
    rnd = random.Random(SEED)
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    idx, _, Xall = H._load("fcg_wl_1024")
    rows_idx = []
    for i in ids:
        rows_idx.append(idx[i])
    X = Xall[rows_idx].astype(np.float64)
    folds = H.folds(y, H.groups_for(ids), 1)
    # test applications: PER_FAM per class, spread over the folds in the fold order
    order = defaultdict(list)
    for _, k, tr, te in folds:
        for j in te:
            order[fam[j]].append((k, j))
    pick = set()
    for f, lst in order.items():
        rnd.shuffle(lst)
        for k, j in lst[:PER_FAM]:
            pick.add((k, j))
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {"p_grid": P_GRID, "apps": {}}
    for _, k, tr, te in folds:
        todo = []
        for j in te:
            if (k, j) in pick and ids[j] not in res["apps"]:
                todo.append(j)
        if not todo:
            continue
        up = H.upsample(tr, y, 1000 + k)
        m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
        m.fit(X[up], y[up])
        Ex = X[tr].mean(0)
        for j in todo:
            f = fam[j]
            c = y[j]
            g = H.load_graph(f, ids[j], "fcg")
            if g is None or not g[0]:
                continue
            nl, su, pr = g
            x, prov, nrm = P.provenance_vector(nl, su, pr)
            if x is None:
                continue
            w = m.coef_[c]
            credit = P.split_bucket_credit(prov, w, Ex[: len(w)], nrm, w * (x - Ex))
            canon = {}
            for n, raw in nl.items():
                canon[n] = P.G.canonical_label(raw, "fcg")
            dev = []
            for n, l in canon.items():
                if l in credit:
                    dev.append(n)
            by_attr = sorted(dev, key=lambda n: credit[canon[n]], reverse=True)
            rand = dev[:]
            rnd.shuffle(rand)
            row = {"family": f, "n_dev": len(dev), "p0_matches_packed": bool(np.allclose(x, X[j], atol=1e-6)),
                   "attr": [], "rand": []}
            for p in P_GRID:
                kk = int(round(p * len(dev)))
                for name, ordr in (("attr", by_attr), ("rand", rand)):
                    drop = set(ordr[:kk])
                    kept_nodes = {}
                    for n, r in nl.items():
                        if n not in drop:
                            kept_nodes[n] = r
                    v, _, _ = P.provenance_vector(kept_nodes, su, pr)
                    ok = v is not None and int(m.predict(v.reshape(1, -1))[0]) == c
                    row[name].append(int(ok))
            res["apps"][ids[j]] = row
            del g, nl, su, pr, prov
            if len(res["apps"]) % 20 == 0:
                json.dump(res, open(OUT, "w"))
                print(f"  {len(res['apps'])} apps", flush=True)
        json.dump(res, open(OUT, "w"))
    # summary: accuracy per deletion fraction, overall and per family
    apps = list(res["apps"].values())
    n_match = 0
    for a in apps:
        n_match += a["p0_matches_packed"]
    summ = {"n": len(apps), "p0_matches_packed": n_match}
    for name in ("attr", "rand"):
        curve = []
        for i in range(len(P_GRID)):
            vals = []
            for a in apps:
                vals.append(a[name][i])
            curve.append(float(np.mean(vals)))
        summ[name] = curve
    per = defaultdict(list)
    for a in apps:
        per[a["family"]].append(a)
    per_family = {}
    for f, v in per.items():
        fam_row = {"n": len(v)}
        for nm in ("attr", "rand"):
            curve = []
            for i in range(len(P_GRID)):
                vals = []
                for a in v:
                    vals.append(a[nm][i])
                curve.append(float(np.mean(vals)))
            fam_row[nm] = curve
        per_family[f] = fam_row
    summ["per_family"] = per_family
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)
    header = []
    for p in P_GRID:
        header.append(f"{p:>6}")
    print("p     " + " ".join(header))
    for nm in ("attr", "rand"):
        cells = []
        for v in summ[nm]:
            cells.append(f"{v:6.3f}")
        print(f"{nm:5} " + " ".join(cells), flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
