#!/usr/bin/env python3
"""MKLDroid-style localization (Narayanan et al.) against provenance attribution, on corpus v3.

MKLDroid represents an app by contextual subgraph features in several views of its dependency graph,
learns one weight per view (multiple kernel learning, linear kernels), and localizes code by scoring
each node with the classifier weights of the features rooted at it. Here:
  views     WL subtree features (2 iterations) over the FCG under three node colorings: identifier
            (the method signature), fuzzy (LibScout-style signature with app types erased) and api (the
            set of framework APIs the method calls), as in exp_h_ablation / gen_wl_hardened
  features  hashed at d = 2^20 per view, where the dimension sweep shows SDK tokens are almost
            collision-free, i.e. close to MKLDroid's explicit features
  MKL       SimpleMKL-style alternation for l1-MKL with linear kernels: fit a log-loss linear model on
            [sqrt(b_v) X_v], set b_v proportional to ||w_v||, 5 rounds
  localize  per app, each token's weighted signed contribution (its bucket's SHAP term split over the
            bucket's tokens by occupancy) is credited to the method whose node it is rooted at, summed
            over views and apps
The comparison row is single-view identifier provenance on the same apps, model and dimension (the
mkl run with b fixed to the identifier view), so the two differ only in the views and their weights.
Samples: the dimension sweep's v3 selection (highdim_localization.py with HD_V3=1: 60 per family, seed
42, eight classes; IPNinja has 37, CashRaven 39, Traffmonetizer 17). Ground truth: exp_label_audit.SDK; candidates: non-library identifier methods.
Output: results/datasets/proxy_v3/mkldroid.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
os.environ["HD_V3"] = "1"
import exp_h_ablation as HA
import exp_harness as H
import exp_label_audit as A
import gen_wl_vectors as G
import provenance_localization as P
from wl_hash import wl_digest

ROOT = H.ROOT
OUT = os.environ.get("MK_OUT") or os.path.join(ROOT, "results", "datasets", "proxy_v3", "mkldroid.json")
VIEWS = ("identifier", "fuzzy", "api")
DIM = 1 << 20
ITERS = 2
ROUNDS = 5
SEED = 42
PER_FAM = int(os.environ.get("MK_PER_FAM", "60"))
FAMS = A.FAMS_V3  # the eight corpus-v3 classes
TOP_N = (10, 25, 50, 100)


def view_tokens(g):
    """{view: [(bucket, sign, origin identifier or None)]} for one FCG."""
    nl, su, pr = g
    ident = {}
    for n, r in nl.items():
        ident[n] = G.canonical_label(r, "fcg")
    out = {}
    for v in VIEWS:
        cols = HA.colors(v, nl, su, pr, None)
        if not cols:
            out[v] = []
            continue
        toks = []

        def add(tok, n):
            h = G.stable_hash_u32(tok)
            if (h & 0x80000000) == 0:
                sign = 1.0
            else:
                sign = -1.0
            # origin: the node's identifier, unless missing or a library method
            o = ident.get(n)
            if o and not o.startswith(G.LIB_PREFIXES):
                origin = o
            else:
                origin = None
            toks.append((h % DIM, sign, origin))

        for n, c in cols.items():
            add(f"0|{c}", n)
        cur = cols
        for it in range(1, ITERS + 1):
            nxt = {}
            for n in cur:
                i_l = []
                for p in pr.get(n, []):
                    if p in cur:
                        i_l.append(cur[p])
                i_l.sort()
                o_l = []
                for s in su.get(n, []):
                    if s in cur:
                        o_l.append(cur[s])
                o_l.sort()
                nxt[n] = wl_digest(cur[n], i_l, o_l)
            cur = nxt
            for n in cur:
                add(f"{it}|{cur[n]}", n)
        out[v] = toks
    return out


def fit(Xc, block, y, beta):
    # [sqrt(b_v) X_v] without building a scaled copy: the view blocks of the one combined matrix are
    # rescaled in place for the fit and restored afterwards (memory: one matrix instead of up to nine)
    betas = []
    for v in VIEWS:
        betas.append(beta[v])
    sc = np.sqrt(np.array(betas, dtype=np.float64))

    def scale(up):  # in 8M-entry chunks, so no array of the matrix's size is created
        for a in range(0, len(Xc.data), 1 << 23):
            f = sc[block[a:a + (1 << 23)]]
            if up:
                Xc.data[a:a + (1 << 23)] *= f
            else:
                Xc.data[a:a + (1 << 23)] /= f
    scale(True)
    m = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4, random_state=SEED)
    m.fit(Xc, y)
    scale(False)
    return m


def view_coefs(m, beta, k):
    w = m.coef_[k]
    coefs = {}
    for i, v in enumerate(VIEWS):
        coefs[v] = w[i * DIM:(i + 1) * DIM] * np.sqrt(beta[v])
    return coefs


def main():
    # sample selection, identical to highdim_localization.main with HD_V3=1
    # (named samples, not P: that name is the provenance_localization module)
    rng = np.random.default_rng(SEED)
    man = []
    for l in open(os.path.join(ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
        man.append(json.loads(l))
    by = defaultdict(list)
    for m in sorted(man, key=lambda m: m["sha256"]):
        if m["status"] == "kept" and m.get("split") == "class" and m["family"] in FAMS:
            by[m["family"]].append(m["sha256"])
    samples = []
    for f in FAMS:
        lst = by[f][:]
        rng.shuffle(lst)
        for s in lst[:PER_FAM]:
            samples.append((f, s))
    print(f"{len(samples)} samples", flush=True)
    # one combined CSR over the three views (view i in columns [i*DIM, (i+1)*DIM)), each view block
    # L2-normalized per application as before, built directly from index/value arrays
    indptr = [0]
    ind_parts = []
    val_parts = []
    keep = []
    for i, (f, sha) in enumerate(samples, 1):
        g = H.load_graph(f, sha, "fcg")
        if g is None or not g[0]:
            continue
        vt = view_tokens(g)
        del g
        n_row = 0
        for i_v, v in enumerate(VIEWS):
            # L2-normalized 1 x DIM row of this view's summed signed tokens
            acc = defaultdict(float)
            for b, s, _ in vt[v]:
                acc[b] += s
            if not acc:
                r = csr_matrix((1, DIM))
            else:
                idx = np.fromiter(acc, dtype=np.int64)
                val = np.fromiter(acc.values(), dtype=np.float64)
                n = np.linalg.norm(val)
                if not n:
                    n = 1.0
                r = csr_matrix((val / n, (np.zeros(len(idx), dtype=np.int64), idx)), shape=(1, DIM))
            ind_parts.append(r.indices.astype(np.int64) + i_v * DIM)
            val_parts.append(r.data)
            n_row += r.nnz
        indptr.append(indptr[-1] + n_row)
        keep.append((f, sha))
        if i % 40 == 0:
            print(f"  vectorized {i}/{len(samples)}", flush=True)
    Xc = csr_matrix((np.concatenate(val_parts), np.concatenate(ind_parts), np.array(indptr, dtype=np.int64)),
                    shape=(len(keep), DIM * len(VIEWS)))
    Xc.sort_indices()
    del ind_parts, val_parts, indptr
    block = (Xc.indices // DIM).astype(np.int8)
    le = LabelEncoder()
    keep_fams = []
    for f, _ in keep:
        keep_fams.append(f)
    y = le.fit_transform(keep_fams)

    # MKL: alternate fitting with b_v proportional to ||w_v||
    beta = {}
    for v in VIEWS:
        beta[v] = 1 / len(VIEWS)
    hist = []
    for r in range(ROUNDS):
        m = fit(Xc, block, y, beta)
        norms = {}
        for v in VIEWS:
            sq = 0
            for k in range(len(le.classes_)):
                sq += np.sum(view_coefs(m, beta, k)[v] ** 2) / beta[v]
            norms[v] = float(np.sqrt(sq))
        tot = sum(norms.values())
        new_beta = {}
        for v in VIEWS:
            new_beta[v] = max(norms[v] / tot, 1e-6)
        beta = new_beta
        hist.append(dict(beta))
        print(f"  MKL round {r}: beta {beta}", flush=True)
    models = {"mkl": (fit(Xc, block, y, beta), beta),
              "identifier_only": (fit(Xc, block, y, {"identifier": 1.0, "fuzzy": 1e-12, "api": 1e-12}),
                                  {"identifier": 1.0, "fuzzy": 1e-12, "api": 1e-12})}
    Exc = np.asarray(Xc.mean(0)).ravel()
    Ex = {}
    for i, v in enumerate(VIEWS):
        Ex[v] = Exc[i * DIM:(i + 1) * DIM]

    out = {"n_samples": len(keep), "dim": DIM, "views": VIEWS, "beta_history": hist, "results": {}}
    for name, (m, b) in models.items():
        out["results"][name] = {"beta": b}
        for f in FAMS:
            k = list(le.classes_).index(f)
            W = view_coefs(m, b, k)
            agg = defaultdict(float)
            for j, (ff, sha) in enumerate(keep):
                if ff != f:
                    continue
                g = H.load_graph(ff, sha, "fcg")
                vt = view_tokens(g)
                del g
                xrow = Xc[j]
                for i_v, v in enumerate(VIEWS):
                    if b[v] < 1e-9:
                        continue
                    x = np.zeros(DIM)
                    sel = (xrow.indices >= i_v * DIM) & (xrow.indices < (i_v + 1) * DIM)
                    x[xrow.indices[sel] - i_v * DIM] = xrow.data[sel]
                    phi = W[v] * (x - Ex[v])
                    # provenance_localization.split_bucket_credit (exact by default), as in every attribution
                    # experiment; tokens without a candidate origin (library methods) take part in the split
                    # but are not credited
                    raw = defaultdict(float)
                    for bk, s, _ in vt[v]:
                        raw[bk] += s
                    sq = 0
                    for q in raw.values():
                        sq += q * q
                    nrm = float(np.sqrt(sq))
                    if not nrm:
                        nrm = 1.0
                    prov = []
                    for bk, s, o in vt[v]:
                        if o is not None:
                            prov.append((bk, s, o))
                        else:
                            prov.append((bk, s, "\0lib"))
                    for o, c in P.split_bucket_credit(prov, W[v], Ex[v], nrm, phi).items():
                        if o != "\0lib":
                            agg[o] += c
                del vt
            # rank candidate methods; precision@n and AUC against the SDK namespaces
            meths = sorted(agg)
            sc_list = []
            truth_list = []
            for t in meths:
                sc_list.append(agg[t])
                truth_list.append(t.startswith(A.SDK[f]))
            sc = np.array(sc_list)
            truth = np.array(truth_list)
            order = np.argsort(-sc)
            res = {"candidates": len(meths), "sdk": int(truth.sum())}
            for n in TOP_N:
                res[f"p@{n}"] = float(truth[order[:n]].mean())
            n_true = truth.sum()
            if 0 < n_true and n_true < len(truth):
                res["auc"] = float(roc_auc_score(truth, sc))
            else:
                res["auc"] = None
            out["results"][name][f] = res
            print(f"  {name} {f}: p@25={res['p@25']:.2f} p@100={res['p@100']:.2f} auc={res['auc']}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
