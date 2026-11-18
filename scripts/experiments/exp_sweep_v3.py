#!/usr/bin/env python3
"""Dimension and split sensitivity of identifier WL on corpus v3 (the appendix sweep figures).

Configurations: graph block {CFG+FCG, FCG, CFG} x classifier {SGD (log loss, alpha 1e-5), Random Forest
(100 trees), XGBoost (hist, 200 trees, depth 6)}, on the eight classes (3,777 applications), with the packed
vectors {cfg,fcg}_wl_{128,256,512,1024}.
  dimension  macro-F1 under DEX-grouped StratifiedGroupKFold(5), one repeat, in-fold upsampling, for every
             configuration at d in {128, 256, 512, 1024}
  split      at d=1024, stratified holdout (70/30, 80/20, 90/10; seed 0, upsampled training part) against the
             DEX-grouped cross-validation, for every configuration
Also: the nine-class setting (with the background class), CFG+FCG at d=1024, stratified holdouts.
The figures plot the four configurations with the highest DEX-grouped macro-F1 at d=1024.
Run with the .venv-cuda interpreter (XGBoost). Outputs: results/datasets/proxy_v3/sweep_v3.json (resumable)
and papers/mlguided/figures/f1_vs_dimension_v3.png, f1_vs_split_v3.png
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H

OUT = os.path.join(H.ROOT, "results", "datasets", "proxy_v3", "sweep_v3.json")
FIGDIR = os.path.join(H.ROOT, "papers", "mlguided", "figures")
DIMS = [128, 256, 512, 1024]
SPLITS = [0.3, 0.2, 0.1]
BLOCKS = {"CFG+FCG": ("cfg", "fcg"), "FCG": ("fcg",), "CFG": ("cfg",)}


def model(name):
    if name == "SGD":
        return SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, n_jobs=6, random_state=42)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist", n_jobs=6, random_state=42)


def matrix(ids, block, d):
    parts = []
    for g in BLOCKS[block]:
        idx, _, X = H._load(f"{g}_wl_{d}")
        rows = []
        for i in ids:
            rows.append(idx[i])
        parts.append(X[rows])
    return np.hstack(parts).astype(np.float32)


def main():
    assert H.V3, "run with DATASET=proxy_v3"
    ids, fam = H.sample_ids("four")
    y, le = H.labels(fam)
    folds = H.folds(y, H.groups_for(ids), 1)
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    for block in BLOCKS:
        for d in DIMS:
            X = None
            for clf in ("SGD", "RF", "XGB"):
                key = f"{block}|{clf}|{d}|grouped"
                if key in res:
                    continue
                if X is None:
                    X = matrix(ids, block, d)
                f1 = []
                for r, k, tr, te in folds:
                    up = H.upsample(tr, y, 1000 * r + k)
                    m = model(clf).fit(X[up], y[up])
                    f1.append(f1_score(y[te], m.predict(X[te]), average="macro"))
                res[key] = float(np.mean(f1))
                print(key, round(res[key], 4), flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
                if d == 1024:
                    for t in SPLITS:
                        ks = f"{block}|{clf}|{d}|holdout{int(round((1 - t) * 100))}"
                        if ks in res:
                            continue
                        tr, te = train_test_split(np.arange(len(y)), test_size=t, stratify=y, random_state=0)
                        up = H.upsample(tr, y, 7)
                        m = model(clf).fit(X[up], y[up])
                        res[ks] = float(f1_score(y[te], m.predict(X[te]), average="macro"))
                        print(ks, round(res[ks], 4), flush=True)
                        json.dump(res, open(OUT, "w"), indent=1)
    # nine classes (with the background class), CFG+FCG at d=1024: stratified holdouts, for "best stratified"
    ids9, fam9 = H.sample_ids("five")
    y9, _ = H.labels(fam9)
    X9 = None
    for clf in ("SGD", "RF", "XGB"):
        for t in SPLITS:
            ks = f"nine|CFG+FCG|{clf}|1024|holdout{int(round((1 - t) * 100))}"
            if ks in res:
                continue
            if X9 is None:
                X9 = matrix(ids9, "CFG+FCG", 1024)
            tr, te = train_test_split(np.arange(len(y9)), test_size=t, stratify=y9, random_state=0)
            up = H.upsample(tr, y9, 7)
            m = model(clf).fit(X9[up], y9[up])
            res[ks] = float(f1_score(y9[te], m.predict(X9[te]), average="macro"))
            print(ks, round(res[ks], 4), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # the four block|classifier configurations with the highest grouped macro-F1 at d=1024
    cfg_set = set()
    for k in res:
        if not k.startswith("nine|"):
            cfg_set.add(k.rsplit("|", 2)[0])
    cfgs = sorted(cfg_set, key=lambda c: -res[f"{c}|1024|grouped"])[:4]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for c in cfgs:
        f1_by_dim = []
        for d in DIMS:
            f1_by_dim.append(res[f"{c}|{d}|grouped"])
        ax.plot(DIMS, f1_by_dim, marker="o", label=c.replace("|", ", "))
    ax.set_xscale("log", base=2)
    dim_labels = []
    for d in DIMS:
        dim_labels.append(str(d))
    ax.set_xticks(DIMS, dim_labels)
    ax.set_xlabel("feature dimension per graph type")
    ax.set_ylabel("macro-F1 (DEX-grouped 5-fold)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "f1_vs_dimension_v3.png"), dpi=200)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    conds = ["holdout70", "holdout80", "holdout90", "grouped"]
    for c in cfgs:
        f1_by_cond = []
        for s in conds:
            f1_by_cond.append(res[f"{c}|1024|{s}"])
        ax.plot(range(len(conds)), f1_by_cond, marker="o", label=c.replace("|", ", "))
    ax.set_xticks(range(len(conds)), ["70/30", "80/20", "90/10", "DEX-grouped CV"])
    ax.set_ylabel("macro-F1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "f1_vs_split_v3.png"), dpi=200)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
