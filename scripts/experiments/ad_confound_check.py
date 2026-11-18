#!/usr/bin/env python3
"""Does the classifier key on the proxy SDK, or on co-bundled ad/analytics SDKs?

The localization study (results/discriminative_methods.json) found that for the
two families whose proxy SDK is obfuscated, the most family-discriminative
classes are co-bundled advertising libraries (mopub, facebook/ads). That raises a
dataset-confound objection a TPL-aware reviewer will make: maybe the model is
recognising the host-app bundle, not the proxy behaviour.

We test it directly. From a single parse of each call graph we build three WL
vectors and cross-validate each under identical DEX-grouped folds:

  full       : all developer methods (the deployed setting)
  ad_masked  : every ad/analytics/tracking library method REMOVED
  ads_only   : ONLY ad/analytics library methods kept

Interpretation:
  ad_masked ~= full   -> the signal does not depend on ad libraries (objection
                         answered; the proxy code carries the decision).
  ads_only  high      -> the bundle alone is predictive, i.e. a real confound
                         that must be disclosed even if ad_masked holds up.

The ad prefix list is deliberately generous: over-masking is the conservative
direction for this test.

Outputs: results/ad_confound.json  (+ vector cache results/ad_confound_vecs.npz)
"""
import json, os, sys
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_wl_vectors as G
import train_wl_only_classifier as T
import graph_io as GI

PROXY = ["ipninja", "luminati", "monkeysocks", "oxylabs"]
DEX_GROUPS = os.path.join(T.PROJECT_ROOT, "results", "full_sweep", "dex_groups.json")
OUT = os.path.join(T.PROJECT_ROOT, "results", "ad_confound.json")
CACHE = os.path.join(T.PROJECT_ROOT, "results", "ad_confound_vecs.npz")
DIM = 1024
PER_FAM = 200
MAX_FCG = 50_000_000
SEED = 42
N_FOLDS = 5

# Ad networks, mediation, attribution and analytics SDKs. Generous on purpose.
AD_PREFIXES = (
    "Lcom/mopub/", "Lcom/facebook/", "Lcom/vungle/", "Lcom/fyber/",
    "Lcom/adcolony/", "Lcom/tappx/", "Lcom/appnext/", "Lcom/startapp/",
    "Lcom/moat/", "Lcom/tapjoy/", "Lcom/integralads/", "Lcom/flurry/",
    "Lcom/applovin/", "Lcom/chartboost/", "Lcom/inmobi/", "Lcom/unity3d/",
    "Lcom/ironsource/", "Lcom/smaato/", "Lcom/mintegral/", "Lcom/mobvista/",
    "Lcom/my/target/", "Lcom/pubmatic/", "Lcom/openx/", "Lcom/criteo/",
    "Lcom/supersonic/", "Lcom/heyzap/", "Lcom/millennialmedia/",
    "Lcom/nativex/", "Lcom/revmob/", "Lcom/leadbolt/", "Lcom/airpush/",
    "Lcom/appbrain/", "Lcom/amazon/device/ads/", "Lcom/mobfox/",
    "Lcom/adincube/", "Lcom/duapps/ad/", "Lcom/batmobi/", "Lcom/altamob/",
    "Lcom/avocarrot/", "Lcom/loopme/", "Lcom/smartadserver/", "Lcom/pollfish/",
    "Lcom/yandex/", "Lcom/tutelatechnologies/", "Lcom/oneaudience/",
    "Lcom/onesignal/", "Lcom/appsflyer/", "Lcom/adjust/", "Lcom/mixpanel/",
    "Lcom/amplitude/", "Lcom/kochava/", "Lcom/localytics/", "Lcom/umeng/",
    "Lcom/crashlytics/", "Lio/fabric/", "Lcom/segment/", "Lio/branch/",
    "Lcom/bytedance/", "Lcom/qq/e/", "Lcom/baidu/mobads/",
)


def wl(raw_labels, su, pr, keep):
    """keep: callable(canonical_label) -> bool"""
    sub = {}
    for n, raw in raw_labels.items():
        if keep(G.canonical_label(raw, "fcg")):
            sub[n] = raw
    if not sub:
        return np.zeros(DIM, dtype=np.float32)
    return G.wl_hashed_vector(sub, su, pr, kind="fcg", dim=DIM)


def build():
    rng = np.random.default_rng(SEED)
    with open(DEX_GROUPS) as f:
        dex_group = json.load(f)
    byfam = defaultdict(list)
    for hdir, gid, label in T.build_samples():
        if label in PROXY:
            byfam[label].append((hdir, gid))

    rows = {"full": [], "ad_masked": [], "ads_only": []}
    labels = []
    groups = []
    ad_frac = []
    for fam in PROXY:
        lst = byfam[fam][:]
        rng.shuffle(lst)
        used = 0
        for hdir, gid in lst:
            fcg = os.path.join(hdir, f"{gid}.fcg")
            if not GI.exists(fcg) or GI.size(fcg) > MAX_FCG:
                continue
            try:
                nl, su, pr = G.parse_gml_lightweight(fcg)
            except Exception:
                continue
            rows["full"].append(wl(nl, su, pr, lambda l: True))
            rows["ad_masked"].append(wl(nl, su, pr, lambda l: not l.startswith(AD_PREFIXES)))
            rows["ads_only"].append(wl(nl, su, pr, lambda l: l.startswith(AD_PREFIXES)))
            # share of developer (non-library) methods that are ad/analytics methods
            dev = []
            for r in nl.values():
                l = G.canonical_label(r, "fcg")
                if not l.startswith(G.LIB_PREFIXES):
                    dev.append(l)
            n_ad = 0
            for l in dev:
                if l.startswith(AD_PREFIXES):
                    n_ad += 1
            ad_frac.append(n_ad / max(1, len(dev)))
            labels.append(fam)
            groups.append(dex_group.get(gid, f"solo_{gid}"))
            used += 1
            if used % 25 == 0:
                print(f"  [{fam}] {used}", flush=True)
            if used >= PER_FAM:
                break
        print(f"[{fam}] {used} FCGs", flush=True)

    np.savez_compressed(CACHE, full=np.vstack(rows["full"]),
                        ad_masked=np.vstack(rows["ad_masked"]),
                        ads_only=np.vstack(rows["ads_only"]),
                        labels=np.array(labels), groups=np.array(groups),
                        ad_frac=np.array(ad_frac))
    return rows, labels, groups, ad_frac


def main():
    if os.path.exists(CACHE):
        print("[*] using cached vectors", flush=True)
        z = np.load(CACHE, allow_pickle=False)
        rows = {}
        for k in ("full", "ad_masked", "ads_only"):
            rows[k] = z[k]
        labels = list(z["labels"])
        groups = z["groups"]
        ad_frac = z["ad_frac"]
    else:
        rows, labels, groups, ad_frac = build()
        for k in rows:
            rows[k] = np.vstack(rows[k])
        groups = np.array(groups)
        ad_frac = np.array(ad_frac)

    y = LabelEncoder().fit_transform(labels)
    print(f"\nsamples: {len(y)}  {dict(Counter(labels))}", flush=True)
    print(f"ad-library share of developer methods: median "
          f"{100*np.median(ad_frac):.1f}%, mean {100*ad_frac.mean():.1f}%\n", flush=True)

    counts = {}
    for k, v in Counter(labels).items():
        counts[k] = int(v)
    result = {"n": int(len(y)), "counts": counts,
              "ad_frac_median": float(np.median(ad_frac)),
              "ad_frac_mean": float(ad_frac.mean()), "conditions": {}}
    for cond in ("full", "ad_masked", "ads_only"):
        # grouped CV, upsampling inside each training fold
        X = np.asarray(rows[cond], dtype=np.float32)
        scores = []
        for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
            pairs = []
            for j in tr:
                pairs.append((j, y[j]))
            up = T.upsample(pairs, seed=SEED)
            idx_list = []
            for j, _ in up:
                idx_list.append(j)
            idx = np.array(idx_list)
            clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4,
                                random_state=SEED, n_jobs=-1)
            clf.fit(X[idx], y[idx])
            scores.append(f1_score(y[te], clf.predict(X[te]), average="macro",
                                   zero_division=0))
        m = float(np.mean(scores))
        s = float(np.std(scores))
        result["conditions"][cond] = {"macro_f1": m, "std": s}
        print(f"{cond:11} macro-F1 = {m:.4f} +/- {s:.4f}", flush=True)
    d = result["conditions"]
    result["delta_full_minus_masked"] = d["full"]["macro_f1"] - d["ad_masked"]["macro_f1"]
    print(f"\nfull - ad_masked = {result['delta_full_minus_masked']:+.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
