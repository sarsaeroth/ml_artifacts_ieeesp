"""provenance_vector and split_bucket_credit, copied verbatim from provenance_localization.py."""
from collections import defaultdict
import numpy as np
from wl_hash import wl_digest
import gen_wl_vectors as G
DIM, ITERS = 1024, 2
SPLIT = "exact"  # the split every attribution experiment in the paper uses

def provenance_vector(nl, su, pr):
    """WL vector for an FCG plus, per emitted token, its (bucket, sign) and iteration-0 origin method; mirrors gen_wl_vectors.wl_hashed_vector."""
    # canonical labels, library nodes dropped unless that would drop everything
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
        return None, None, None
    origin = dict(lab)
    vec = np.zeros(DIM, dtype=np.float64)
    prov = []                       # (bucket, sign, origin_method)
    prov_iter = []                  # WL iteration that emitted each token

    # iteration 0 tokens
    for n, l in lab.items():
        h = G.stable_hash_u32(f"0|{l}")
        b = h % DIM
        if (h & 0x80000000) == 0:
            s = 1.0
        else:
            s = -1.0
        vec[b] += s
        prov.append((b, s, origin[n]))
        prov_iter.append(0)

    # WL refinement tokens
    cur = lab
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
            h = G.stable_hash_u32(f"{it}|{cur[n]}")
            b = h % DIM
            if (h & 0x80000000) == 0:
                s = 1.0
            else:
                s = -1.0
            vec[b] += s
            prov.append((b, s, origin[n]))
            prov_iter.append(it)
    nrm = np.linalg.norm(vec)
    provenance_vector.last_iters = prov_iter
    if nrm == 0:
        return None, None, None
    return vec / nrm, prov, nrm

def split_bucket_credit(prov, w, Ex, nrm, phi, rule=None):
    """Split linear SHAP phi_b = w_b (x_b - E[x_b]) over the tokens of each bucket ("exact" or the published "legacy" rule)."""
    # exact: token t in bucket b gets w_b * s_t / ||phi|| minus the background term w_b E[x_b] split by
    # occupancy |s_t| / sum|s|, so credits in a bucket sum to phi_b. legacy: phi_b * s_t / sum|s|, which
    # does not sum to phi_b when a bucket holds tokens of both signs.
    rule = rule or SPLIT
    tot = defaultdict(float)
    for b, s, _ in prov:
        tot[b] += abs(s)
    credit = defaultdict(float)
    if rule == "legacy":
        for b, s, m in prov:
            if tot[b] > 0:
                credit[m] += phi[b] * (s / tot[b])
        return credit
    for b, s, m in prov:
        if tot[b] > 0:
            credit[m] += w[b] * (s / nrm) - w[b] * Ex[b] * (abs(s) / tot[b])
    return credit
