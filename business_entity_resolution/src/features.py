"""Stage 2 (baseline): pairwise features for the GBDT matcher. No country/one-hot features (France is unseen)."""
import re
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

_NUM = re.compile(r"\d+")


def _acr(a, b):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0
    ia, ib = "".join(w[0] for w in ta), "".join(w[0] for w in tb)
    return int((len(tb) == 1 and tb[0] == ia and len(ia) > 1) or (len(ta) == 1 and ta[0] == ib and len(ib) > 1))


def _jac(x, y):
    x, y = set(x), set(y)
    return len(x & y) / len(x | y) if (x | y) else np.nan


def build_features(pairs, s1, s2, s3):
    cols = ["entity_id", "n", "nc", "a", "pc"]
    q = s1[cols].add_suffix("_q").rename(columns={"entity_id_q": "s1_id"})
    t = pd.concat([s2, s3])[cols].add_suffix("_t").rename(columns={"entity_id_t": "t_id"})
    p = pairs.merge(q, on="s1_id").merge(t, on="t_id").reset_index(drop=True)
    n1, n2, c1, c2, a1, a2, p1, p2 = (p[c].tolist() for c in
                                       ["n_q", "n_t", "nc_q", "nc_t", "a_q", "a_t", "pc_q", "pc_t"])
    F = pd.DataFrame({"name_cos": p.name_cos, "joint_cos": p.joint_cos, "addr_cos": p.addr_cos})
    F["n_ratio"] = [fuzz.ratio(x, y) for x, y in zip(n1, n2)]
    F["n_tsort"] = [fuzz.token_sort_ratio(x, y) for x, y in zip(n1, n2)]
    F["n_tset"] = [fuzz.token_set_ratio(x, y) for x, y in zip(n1, n2)]
    F["n_partial"] = [fuzz.partial_ratio(x, y) for x, y in zip(n1, n2)]
    F["nc_ratio"] = [fuzz.ratio(x, y) for x, y in zip(c1, c2)]
    F["nc_jw"] = [JaroWinkler.similarity(x, y) for x, y in zip(c1, c2)]
    F["first_eq"] = [int(x.split()[:1] == y.split()[:1]) for x, y in zip(c1, c2)]
    F["acr"] = [_acr(x, y) for x, y in zip(c1, c2)]
    F["len_ratio"] = [min(len(x), len(y)) / max(len(x), len(y), 1) for x, y in zip(c1, c2)]
    F["ntok_q"] = [len(x.split()) for x in c1]
    F["ntok_t"] = [len(y.split()) for y in c2]
    am = np.array([(not x) or (not y) for x, y in zip(a1, a2)])
    F["a_missing"] = am.astype(int)
    F["a_tset"] = [fuzz.token_set_ratio(x, y) for x, y in zip(a1, a2)]
    F["a_partial"] = [fuzz.partial_ratio(x, y) for x, y in zip(a1, a2)]
    F.loc[am, ["a_tset", "a_partial", "addr_cos"]] = np.nan
    both = np.array([bool(x) and bool(y) for x, y in zip(p1, p2)])
    F["pc_both"] = both.astype(int)
    F["pc_eq"] = np.where(both, [int(x == y) for x, y in zip(p1, p2)], np.nan)
    F["pc_pre"] = np.where(both, [int(x[:3] == y[:3]) for x, y in zip(p1, p2)], np.nan)
    F["num_jac"] = [_jac([w for w in _NUM.findall(x) if w != pa], [w for w in _NUM.findall(y) if w != pb])
                    for x, y, pa, pb in zip(a1, a2, p1, p2)]
    F["is_s3"] = (p.src == "S3").astype(int)
    # competition features: rank/gap of a pair among its S1's candidates and among the S1s claiming the same target
    g = p[["s1_id", "t_id", "src"]].assign(base=0.5 * p.name_cos + 0.5 * p.joint_cos)
    gq, gt = g.groupby(["s1_id", "src"]).base, g.groupby("t_id").base
    F["rk_q"] = gq.rank(ascending=False, method="min")
    F["gap_q"] = gq.transform("max") - g.base
    F["n_cand"] = gq.transform("size")
    F["rk_t"] = gt.rank(ascending=False, method="min")
    F["gap_t"] = gt.transform("max") - g.base
    return p[["s1_id", "t_id", "src"]], F
