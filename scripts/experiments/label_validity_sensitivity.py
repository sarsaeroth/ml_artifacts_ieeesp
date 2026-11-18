#!/usr/bin/env python3
"""Attribution with the corpus-v3 builds whose provider label the code does not support removed.

datasets/corpus_v3/label_validity.json lists, from the static reverse engineering
(results/c2_analysis/{MONKEYSOCKS,OXYLABS,IPIDEA}_NOTES.md):
  monkeysocks_dormant   SDK present, never declared in the manifest nor called (1,095; 1,088 AppsGeyser)
  monkeysocks_no_code   no Monkeysocks namespace (1)
  oxylabs_no_code       none of the 20 Oxylabs engine namespaces (15)
  ipidea_ninevpn_only   only the com.through ninevpn client, no IPIDEA proxy engine (88)
Each arm drops one set (or all of them) and repeats the headline evaluation of
exp_e2a_standardized.py: identifier-seeded WL (CFG + FCG), linear SGD on raw vectors, in-fold
upsampling, 3 repeats of StratifiedGroupKFold(5). Run once per grouping (LEAK_GROUPS=dex or neardup).
Reports macro-F1 and per-class F1 per arm.
Output: results/datasets/proxy_v3/label_validity_<groups>.json (resumable per arm)
"""
import gc
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATASET", "proxy_v3")
import exp_harness as H

REPEATS = 3
SETS = json.load(open(os.path.join(H.ROOT, "datasets", "corpus_v3", "label_validity.json")))
KEYS = []
for _set_name in SETS:
    if _set_name != "note":
        KEYS.append(_set_name)
# arms: nothing dropped, each set dropped alone, all sets dropped
ARMS = {"full": []}
for _set_name in KEYS:
    ARMS[f"minus_{_set_name}"] = [_set_name]
ARMS["minus_all"] = KEYS
OUT = os.path.join(H.ROOT, "results", "datasets", "proxy_v3", f"label_validity_{H.GROUPS}.json")


def run(ids, fam):
    y, le = H.labels(fam)
    g = H.groups_for(ids)
    X = np.asarray(H.matrix("labeled", ids), dtype=np.float32)
    macro = []
    per = []
    for r, f, tr, te in H.folds(y, g, REPEATS):
        up = H.upsample(tr, y, 1000 * r + f)
        m = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-5, random_state=42)
        m.fit(X[up], y[up])
        p = m.predict(X[te])
        macro.append(f1_score(y[te], p, average="macro"))
        per.append(f1_score(y[te], p, average=None, labels=range(len(le.classes_))))
        del m
        gc.collect()
    per_family_n = {}
    for c in le.classes_:
        per_family_n[c] = int((fam == c).sum())
    mean_per = np.mean(per, 0)
    per_family_f1 = {}
    for i in range(min(len(le.classes_), len(mean_per))):
        per_family_f1[le.classes_[i]] = float(mean_per[i])
    return {"n": int(len(ids)), "per_family_n": per_family_n,
            "macro_f1": float(np.mean(macro)), "macro_f1_sd": float(np.std(macro)),
            "per_family_f1": per_family_f1}


def main():
    assert H.V3, "run with DATASET=proxy_v3"
    ids0, fam0 = H.sample_ids("four")
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = {}
    for arm, drop in ARMS.items():
        if arm in res:
            continue
        bad = set()
        for k in drop:
            for s in SETS[k]["sha256"]:
                bad.add(s)
        keep_list = []
        for i in ids0:
            keep_list.append(i not in bad)
        keep = np.array(keep_list)
        arm_res = {"dropped": int((~keep).sum())}
        evaluated = run(ids0[keep], fam0[keep])
        for k, v in evaluated.items():
            arm_res[k] = v
        res[arm] = arm_res
        json.dump(res, open(OUT, "w"), indent=1)
        f1_parts = []
        for k, v in res[arm]["per_family_f1"].items():
            f1_parts.append(f"{k}={v:.2f}")
        print(f"{H.GROUPS} {arm}: dropped {res[arm]['dropped']}, n {res[arm]['n']}, macro-F1 {res[arm]['macro_f1']:.3f} "
              + " ".join(f1_parts), flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
