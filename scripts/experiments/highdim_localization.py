#!/usr/bin/env python3
"""Does raising the hash dimension recover provenance localization?

The purity sweep shows SDK tokens become bucket-isolable as d grows (0% at
d=1,024; 82-93% at d=2^20), and the rule d* = M/ln(1/p) predicts it. That is a
statement about bucket occupancy, not ranking quality. This tests end-to-end:
re-vectorize at several d, retrain, re-run provenance attribution, and compare
precision@k and AUC against a d=1,024 baseline computed on the SAME samples.

MEMORY. An earlier version cached every parsed graph so it could reuse them
across dimensions. At ~40 MB per parsed FCG that reached 17 GB for 395 samples,
tripped the kernel OOM killer and took the machine down. Parsed graphs and
provenance records are both far larger than the vectors they produce, so nothing
graph-sized is held across samples now:

  phase A  parse each sample once, emit sparse vectors for every d, discard
  phase B  train one model per d from the stored vectors
  phase C  per (d, family), RE-PARSE each sample, accumulate attribution from
           its provenance, discard immediately

Phase C pays ~0.45 s/sample/dim in re-parsing to keep peak memory at roughly one
graph. The process also caps its own address space, so a regression fails this
job rather than the host.

Controls: the classifier's macro-F1 at each d is reported, because attribution
uses its coefficients and a degraded model would confound the comparison; the
kernel saturates near d=512, so high d is heavily over-parameterized.

Outputs: results/highdim_localization.json (written incrementally per dimension)

HD_V3=1 runs the same sweep on corpus v3 (DATASET=proxy_v3) over its eight classes: samples are the validated apps
(datasets/corpus_v3/manifest.jsonl, status kept), graphs come from exp_harness.load_graph, CV groups are
the DEX groups (exp_harness.groups_for, LEAK_GROUPS default), and the SDK ground truth is exp_label_audit.SDK (the published
namespaces plus the later Bright Data ones). Output: results/datasets/proxy_v3/highdim_localization.json
"""
import json, os, resource, sys, time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

MEM_CAP_GB = int(os.environ.get("HD_MEM_CAP_GB", "8"))
if not (os.environ.get("GPU") == "1" or ".venv-rocm" in sys.prefix):  # ROCm needs its address-space reservation
    resource.setrlimit(resource.RLIMIT_AS,
                       (MEM_CAP_GB * 1024**3, MEM_CAP_GB * 1024**3))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from wl_hash import wl_digest
import gen_wl_vectors as G
import train_wl_only_classifier as T
import provenance_localization as P
import graph_io as GI

PROXY = ["ipninja", "luminati", "monkeysocks", "oxylabs"]
DIMS = []
for _x in os.environ.get("HD_DIMS", "1024,65536,262144,1048576").split(","):
    DIMS.append(int(_x))
del _x
PER_FAM = int(os.environ.get("HD_PER_FAM", "60"))
TOP_N = [10, 25, 50, 100]
SEED = 42
OUT = os.path.join(ROOT, "results", "highdim_localization.json")
V3 = os.environ.get("HD_V3") == "1"
if V3:
    import exp_harness as H
    import exp_label_audit as A
    OUT = os.path.join(ROOT, "results", "datasets", "proxy_v3", "highdim_localization.json")
# HD_DUMP_TOP=K: also record each provider's top-K methods and the sample where each received the most credit,
# so reach from the ranking can be scored per dimension; writes OUT with suffix .top.json and leaves OUT untouched
DUMP_TOP = int(os.environ.get("HD_DUMP_TOP", "0"))
if DUMP_TOP:
    OUT = OUT.replace(".json", ".top.json")
    SDK_PREFIXES = A.SDK
    PROXY = A.FAMS_V3  # the eight corpus-v3 classes
else:
    SDK_PREFIXES = P.SDK_PREFIXES


def parse(f):
    """f is a graph path, or (family, sha) in v3 mode."""
    if V3:
        g = H.load_graph(f[0], f[1], "fcg")
        if g is None:
            raise FileNotFoundError(f[1])
        return g
    return G.parse_gml_lightweight(f)


def walk_tokens(nl, su, pr):
    """Yield (token, origin_method) for every WL token. Streaming; nothing kept."""
    lab = {}
    for n, raw in nl.items():
        lab[n] = G.canonical_label(raw, "fcg")
    drop = []
    for n, l in lab.items():
        if l.startswith(G.LIB_PREFIXES):
            drop.append(n)
    if drop and len(drop) < len(lab):
        for n in drop:
            del lab[n]
    if not lab:
        return
    for n, l in lab.items():
        yield l, l
    cur = dict(lab)
    for it in range(1, 3):
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
        for n, v in cur.items():
            yield f"{it}|{v}", lab[n]


def sparse_vec(path, dim):
    nl, su, pr = parse(path)
    acc = defaultdict(float)
    for tok, _ in walk_tokens(nl, su, pr):
        h = G.stable_hash_u32(tok)
        if (h & 0x80000000) == 0:
            acc[h % dim] += 1.0
        else:
            acc[h % dim] += -1.0
    del nl, su, pr
    if not acc:
        return None
    idx = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
    val = np.fromiter(acc.values(), dtype=np.float64, count=len(acc))
    n = np.linalg.norm(val)
    if n:
        return idx, val / n
    return None


def main():
    rng = np.random.default_rng(SEED)
    byfam = defaultdict(list)
    picks = []
    # sample selection
    if V3:
        man = []
        for l in open(os.path.join(ROOT, "datasets", "corpus_v3", "manifest.jsonl")):
            man.append(json.loads(l))
        for m in sorted(man, key=lambda m: m["sha256"]):
            if m["status"] == "kept" and m.get("split") == "class" and m["family"] in PROXY:
                byfam[m["family"]].append(m["sha256"])
        for fam in PROXY:
            lst = byfam[fam][:]
            rng.shuffle(lst)
            for sha in lst[:PER_FAM]:
                picks.append((fam, sha, (fam, sha)))
        pick_shas = []
        for p in picks:
            pick_shas.append(p[1])
        group_input = []
        for p in picks:
            group_input.append(p[1])
        dexg = {}
        for sha, grp in zip(pick_shas, H.groups_for(group_input)):
            dexg[sha] = grp
    else:
        dexg = json.load(open(os.path.join(ROOT, "results", "full_sweep", "dex_groups.json")))
        for hdir, gid, lab in T.build_samples():
            if lab in PROXY:
                byfam[lab].append((hdir, gid))
    if not V3:
        for fam in PROXY:
            lst = byfam[fam][:]
            rng.shuffle(lst)
            n = 0
            for hdir, gid in lst:
                f = os.path.join(hdir, f"{gid}.fcg")
                if GI.exists(f) and GI.size(f) <= P.MAX_FCG:
                    picks.append((fam, gid, f))
                    n += 1
                if n >= PER_FAM:
                    break
    print(f"{len(picks)} samples, dims={DIMS}, mem cap {MEM_CAP_GB} GB", flush=True)

    out = {"per_family_samples": PER_FAM, "dims": DIMS,
           "n_samples": len(picks), "results": {}}
    if os.path.exists(OUT):
        try:
            out = json.load(open(OUT))
        except Exception:
            pass

    for dim in DIMS:
        if str(dim) in out.get("results", {}):
            print(f"d={dim}: cached, skipping", flush=True)
            continue
        t0 = time.time()
        # phase A: vectors only
        # CSR built directly (indptr/indices/data), not as one-row matrices stacked afterwards: avoids a
        # transient second copy of X, which at d >= 2^18 is the largest object in the job
        indptr = [0]
        ind_parts = []
        val_parts = []
        labels = []
        groups = []
        keep = []
        for i, (fam, gid, f) in enumerate(picks, 1):
            try:
                v = sparse_vec(f, dim)
            except Exception:
                v = None
            if v is None:
                continue
            idx, val = v
            o = np.argsort(idx)
            ind_parts.append(idx[o].astype(np.int32))
            val_parts.append(val[o])
            indptr.append(indptr[-1] + len(idx))
            labels.append(fam)
            groups.append(dexg.get(gid, f"solo_{gid}"))
            keep.append((fam, gid, f))
            if i % 60 == 0:
                print(f"  [d={dim}] vectorized {i}/{len(picks)} ({time.time()-t0:.0f}s)", flush=True)
        X = csr_matrix((np.concatenate(val_parts), np.concatenate(ind_parts), np.array(indptr, dtype=np.int64)),
                       shape=(len(indptr) - 1, dim))
        del ind_parts, val_parts, indptr
        le = LabelEncoder()
        y = le.fit_transform(labels)
        groups = np.array(groups)

        # phase B: model
        f1s = []
        ng = len(set(groups.tolist()))
        for tr, te in GroupKFold(min(5, ng)).split(X, y, groups):
            c = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4, random_state=SEED)
            c.fit(X[tr], y[tr])
            f1s.append(f1_score(y[te], c.predict(X[te]), average="macro", zero_division=0))
        full = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4, random_state=SEED)
        full.fit(X, y)
        Ex = np.asarray(X.mean(0)).ravel()
        # the attribution phase rebuilds each application's row from its provenance records (the same tokens,
        # hashing and L2 normalization as sparse_vec), so the matrix is freed before it
        del X
        entry = {"macro_f1": round(float(np.mean(f1s)), 4), "families": {}}
        print(f"  [d={dim}] macro_f1={entry['macro_f1']}  ({time.time()-t0:.0f}s)", flush=True)

        # phase C: attribution, re-parsing one sample at a time
        for fam in PROXY:
            if fam not in list(le.classes_):
                continue
            w = full.coef_[list(le.classes_).index(fam)]
            agg = defaultdict(float)
            best = {}
            for (fm, gid, f) in keep:
                if fm != fam:
                    continue
                try:
                    nl, su, pr = parse(f)
                except Exception:
                    continue
                prov = []
                for tok, m in walk_tokens(nl, su, pr):
                    h = G.stable_hash_u32(tok)
                    if (h & 0x80000000) == 0:
                        prov.append((h % dim, 1.0, m))
                    else:
                        prov.append((h % dim, -1.0, m))
                del nl, su, pr
                # provenance_localization.split_bucket_credit (exact by default; PROVENANCE_SPLIT=legacy for
                # the published rule), so every attribution experiment uses the same split
                raw = defaultdict(float)
                for b, s, _ in prov:
                    raw[b] += s
                sq = 0
                for v in raw.values():
                    sq += v * v
                nrm = float(np.sqrt(sq))
                if not nrm:
                    nrm = 1.0
                xd = np.zeros(dim)
                for b, v in raw.items():
                    xd[b] = v / nrm
                phi = w * (xd - Ex)
                del xd
                for m_, c in P.split_bucket_credit(prov, w, Ex, nrm, phi).items():
                    agg[m_] += c
                    if DUMP_TOP:
                        if m_ in best:
                            prev = best[m_][0]
                        else:
                            prev = -np.inf
                        if c > prev:
                            best[m_] = (c, gid)
                del prov, phi, raw
            if not agg:
                continue
            meths = sorted(agg)
            sc_list = []
            truth_list = []
            for m in meths:
                sc_list.append(agg[m])
                if m.startswith(SDK_PREFIXES[fam]):
                    truth_list.append(1)
                else:
                    truth_list.append(0)
            sc = np.array(sc_list)
            truth = np.array(truth_list)
            del sc_list, truth_list
            if truth.sum() == 0:
                entry["families"][fam] = {"note": "no SDK methods present in sample"}
                continue
            order = np.argsort(-sc)
            fam_entry = {"candidates": len(meths), "sdk": int(truth.sum()),
                         "base_rate": round(float(truth.mean()), 5)}
            for k in TOP_N:
                kk = min(k, len(meths))
                fam_entry[f"p@{k}"] = round(float(truth[order[:kk]].sum() / kk), 4)
            fam_entry["auc"] = round(float(roc_auc_score(truth, sc)), 4)
            entry["families"][fam] = fam_entry
            if DUMP_TOP:
                top = []
                for i in order[:DUMP_TOP]:
                    top.append({"method": meths[i], "app": best[meths[i]][1], "is_sdk": bool(truth[i])})
                entry["families"][fam]["top"] = top
            r = entry["families"][fam]
            print(f"    {fam:12} p@10={r['p@10']:.3f} p@25={r['p@25']:.3f} "
                  f"p@100={r['p@100']:.3f} AUC={r['auc']:.3f} (base {r['base_rate']:.5f})", flush=True)
            del agg, sc, truth
        if "results" not in out:
            out["results"] = {}
        out["results"][str(dim)] = entry
        json.dump(out, open(OUT, "w"), indent=2)
        del full, Ex
        print(f"  [d={dim}] done in {time.time()-t0:.0f}s\n", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
