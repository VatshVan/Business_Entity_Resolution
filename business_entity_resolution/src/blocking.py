"""Stage 1: candidate generation. Two sparse TF-IDF kNN channels (name char-ngrams, name+address words),
run per country (open set), unioned. Swap/add a dense-embedding + FAISS channel here for scale."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _knn(Q, T, k, chunk=1000):
    TT = T.T.tocsr()
    qi, tj = [], []
    for s in range(0, Q.shape[0], chunk):
        S = (Q[s:s + chunk] @ TT).tocsr()
        for i in range(S.shape[0]):
            a, b = S.indptr[i], S.indptr[i + 1]
            d, ix = S.data[a:b], S.indices[a:b]
            if len(d) > k:
                ix = ix[np.argpartition(-d, k - 1)[:k]]
            qi.append(np.full(len(ix), s + i)); tj.append(ix)
    if not qi:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(qi), np.concatenate(tj)


def _cos(A, B, i, j):
    return np.asarray(A[i].multiply(B[j]).sum(1)).ravel()


def candidates(q, t, k=20, max_df=0.2):
    if len(q) == 0 or len(t) == 0:
        return pd.DataFrame()
    both = pd.concat([q, t])
    jt = lambda d: d.nc + " " + d.a
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=max_df, sublinear_tf=True,
                         dtype=np.float32).fit(both.nc)
    vj = TfidfVectorizer(max_df=0.5, sublinear_tf=True, dtype=np.float32).fit(jt(both))
    va = TfidfVectorizer(ngram_range=(1, 2), max_df=0.5, sublinear_tf=True, dtype=np.float32).fit(both.a)
    An, Bn = vn.transform(q.nc), vn.transform(t.nc)
    Aj, Bj = vj.transform(jt(q)), vj.transform(jt(t))
    Aa, Ba = va.transform(q.a), va.transform(t.a)
    i1, j1 = _knn(An, Bn, k)
    i2, j2 = _knn(Aj, Bj, k)
    ij = np.unique(np.stack([np.concatenate([i1, i2]), np.concatenate([j1, j2])], 1), axis=0)
    i, j = ij[:, 0], ij[:, 1]
    return pd.DataFrame({"i": i, "j": j, "name_cos": _cos(An, Bn, i, j),
                         "joint_cos": _cos(Aj, Bj, i, j), "addr_cos": _cos(Aa, Ba, i, j)})


def build_pairs(s1, s2, s3, k=20):
    out = []
    for c in sorted(s1.country.unique()):
        q = s1[s1.country == c].reset_index(drop=True)
        for name, t in (("S2", s2), ("S3", s3)):
            t = t[t.country == c].reset_index(drop=True)
            p = candidates(q, t, k)
            if p.empty:
                continue
            p["s1_id"] = q.entity_id.values[p.i]
            p["t_id"] = t.entity_id.values[p.j]
            p["src"] = name
            out.append(p.drop(columns=["i", "j"]))
    return pd.concat(out, ignore_index=True)
