#!/usr/bin/env python3
"""Run provenance attribution end to end on the shipped sample.

1. Trains the attributed model from the dataset artifact's data/vectors/ (STATIC_ARTIFACT) exactly as the paper's ranking does (exp_label_audit.train:
   linear SGD, log loss, alpha 1e-4, upsampled to the largest provider, seed 42) on the evaluated
   applications of the eight providers (manifest rows with a dex_group).
2. For each sample FCG in data/samples/, re-extracts the identifier WL vector with provenance (lib/attribution.py,
   the code the paper's experiments run), checks it against the released vector, and splits the linear SHAP
   value of each bucket exactly over the methods that produced it.
3. Prints, per application, the cosine to the released vector, the predicted provider, and the ten methods
   with the most attribution toward the true provider, marking those inside the provider's SDK namespaces.
Needs numpy and scikit-learn; Python 3.14 (graph_io reads .zst through the standard library's compression).
Usage: python3 run_attribution.py [--top N]
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDClassifier

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
# corpus manifest and vectors come from the dataset paper's artifact (STATIC_ARTIFACT, set by verify.sh;
# else the sibling directory that holds its corpus manifest)
STATIC = os.environ.get("STATIC_ARTIFACT", "")
if STATIC == "":
    parent = os.path.dirname(HERE)
    for name in sorted(os.listdir(parent)):
        if os.path.isfile(os.path.join(parent, name, "data", "manifests", "manifest_v3.csv")):
            STATIC = os.path.join(parent, name)
            break
if STATIC == "":
    STATIC = os.path.join(os.path.dirname(HERE), "static")
import attribution as P
import gen_wl_vectors as G

SEED = 42
SDK = {"ipninja": ("Lcom/ipninja/", "Lnet/ipninja/"),
       "luminati": ("Lio/lum/", "Lio/topvpn/", "Lorg/hola/", "Lio/luminati/", "Llumnet/sdk/", "Lbrdat/sdk/",
                    "Lcom/android/eapx/"),
       "monkeysocks": ("Lmonkeynode/", "Lcom/monkeyttf/", "Lmonkeysocks/"),
       "oxylabs": ("Lframett/", "Lbackmybraces/", "Lcom/backmybraces/re/",
                   "Lcom/vinockal01/bw/", "Lcom/vinockal02/bw/", "Lcom/vinockal03/bw/", "Lcom/vinockal04/bw/", "Lcom/vinockal05/bw/",
                   "Lcom/vinockal06/bw/", "Lcom/vinockal07/bw/", "Lcom/vinockal08/bw/", "Lcom/vinockal09/bw/", "Lcom/vinockal10/bw/", "Lcom/vinockal11/bw/",
                   "Lcom/vinobalk/wb/", "Lcom/gummlist/ink/", "Lrep/click/", "Ldigison/eye/", "Lgo/locy/arm/",
                   "Lrefraglione/", "Lcom/microchev/fo/", "Lvolumea/", "Linstantcoffee/"),
       "cashraven": ("Lio/cashraven/", "Lcom/sprious/"),
       "ipidea": ("Lcom/packet/sdk/", "Lcom/castarsdk/", "Lcom/caster/", "Lcom/through/", "Lcom/debby/"),
       "popa": ("Lio/popanet/", "Lio/netas/", "Lio/hopanet/", "Lio/nn/neunative/", "Lio/nn/lp/", "Lio/nn/alpha/",
                "Lcom/amz/appsuite/", "Lio/moneytise/", "Lio/ninjamon/", "Lio/hopmon", "Lcom/mon/app_bandwidth",
                "Lcom/amazon/streammassive/"),
       "traffmonetizer": ("Lcom/traffmonetizer/", "Lcom/tm/", "Lcom/base/")}


def main():
    # parse args
    if "--top" in sys.argv:
        k = int(sys.argv[sys.argv.index("--top") + 1])
    else:
        k = 10

    # train the attributed model on the eight providers' evaluated applications
    rows = []
    for r in csv.DictReader(open(os.path.join(STATIC, "data", "manifests", "manifest_v3.csv"))):
        if r["provider"] in SDK and r["status"] == "kept" and r["dex_group"] != "":
            rows.append(r)
    keep = set()
    for r in rows:
        keep.add(r["sha256"])
    z = np.load(os.path.join(STATIC, "data", "vectors", "fcg_wl_1024.npz"))
    sel_list = []
    for i in z["ids"]:
        sel_list.append(i in keep)
    sel = np.array(sel_list)
    ids = z["ids"][sel]
    fam = z["family"][sel]
    X = z["X"][sel].astype(np.float64)
    rng = np.random.default_rng(SEED)
    by = defaultdict(list)
    for j, f in enumerate(fam):
        by[f].append(j)
    sizes = []
    for v in by.values():
        sizes.append(len(v))
    top_n = max(sizes)
    # upsample every provider to the largest one
    parts = []
    for v in by.values():
        extra = rng.choice(v, top_n - len(v))
        parts.append(np.r_[v, extra])
    idx = np.concatenate(parts)
    m = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-5, random_state=SEED)
    m.fit(X[idx], fam[idx])
    Ex = X.mean(0)
    released = {}
    for sha, row in zip(ids, X):
        released[sha] = row
    n = len(ids)

    classes = []
    for c in m.classes_:
        classes.append(str(c))
    print(f"model trained on {n} applications, classes {classes}\n")

    # attribute each shipped sample
    picked = json.load(open(os.path.join(HERE, "data", "samples", "samples.json")))
    for f, shas in picked.items():
        w = m.coef_[classes.index(f)].astype(np.float64)
        for sha in shas:
            d = os.path.join(HERE, "data", "samples", f)
            found = None
            for p in os.listdir(d):
                if p.startswith(sha):
                    found = p
                    break
            if found is None:
                raise StopIteration
            path = os.path.join(d, found)
            node_labels, successors, predecessors = G.parse_gml_lightweight(path.removesuffix(".zst"))
            x, prov, nrm = P.provenance_vector(node_labels, successors, predecessors)
            r = released.get(sha)
            if r is not None:
                cos = float(x @ r / (np.linalg.norm(x) * np.linalg.norm(r)))
            else:
                cos = float("nan")
            pred = m.predict(x[None])[0]
            credit = P.split_bucket_credit(prov, w, Ex, nrm, w * (x - Ex))
            top = sorted(credit.items(), key=lambda t: -t[1])[:k]
            n_sdk = 0
            for meth, _ in top:
                if meth.startswith(SDK[f]):
                    n_sdk += 1
            print(f"[{f}] {sha[:16]}  cosine to released vector {cos:.6f}  predicted {pred}  "
                  f"SDK methods in top {k}: {n_sdk}")
            for meth, c in top:
                if meth.startswith(SDK[f]):
                    tag = "SDK "
                else:
                    tag = "    "
                print(f"    {tag}{c:+.5f}  {meth[:110]}")
            print()


if __name__ == "__main__":
    main()
