"""Small-model probes on a subset. python -m src.probe --data dataset_2pct --k 50 --out eda_out
(1) blocking recall@K by channel and by noise class  (2) feature-group ablation with LightGBM  (3) gain importance
(4) false-positive / false-negative samples for reading."""
import argparse, os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from src.common import load, read_gt, f05_macro
from src.blocking import build_pairs
from src.features import build_features
from src.pipeline import decide
from src.eda_deep import rec_dicts, classify, RN, RA

NAME = ["name_cos", "n_ratio", "n_tsort", "n_tset", "n_partial", "nc_ratio", "nc_jw", "first_eq", "acr", "len_ratio", "ntok_q", "ntok_t"]
ADDR = ["addr_cos", "a_missing", "a_tset", "a_partial", "pc_both", "pc_eq", "pc_pre", "num_jac"]


def fit_eval(X, y, pk, cols, val_ids, truth_val, name):
    v = pk.s1_id.isin(val_ids).values
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, verbose=-1).fit(X.loc[~v, cols], y[~v])
    p = m.predict_proba(X.loc[v, cols])[:, 1]
    best = max((f05_macro(truth_val, decide(pk[v], p, t)), t) for t in np.arange(0.3, 0.96, 0.05))
    print(f"{name:12s} AUC {roc_auc_score(y[v], p):.4f}  AP {average_precision_score(y[v], p):.4f}  macroF0.5 {best[0]:.4f} @thr {best[1]:.2f}")
    return m, p, v, best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_2pct"); ap.add_argument("--k", type=int, default=50); ap.add_argument("--out", default="eda_out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.RandomState(0)
    s1, s2, s3 = load(a.data, "train"); gt = read_gt(f"{a.data}/train/train_ground_truth.tsv")
    truth = {s: gt.get(s, set()) for s in s1.entity_id}
    pairs = build_pairs(s1, s2, s3, a.k); pk, X = build_features(pairs, s1, s2, s3)
    pos = {(s, t) for s, v in truth.items() for t in v}
    y = np.array([(s, t) in pos for s, t in zip(pk.s1_id, pk.t_id)], int)
    print(f"candidate pairs {len(pk)} | GT pairs {len(pos)} | union recall @k={a.k}: {y.sum() / len(pos):.4f} | avg cands/S1 {len(pk) / len(s1):.1f}")

    print("\n--- (1a) recall@K when ranking the candidate union by one channel ---")
    g = pk.assign(**{c: X[c] for c in ("name_cos", "joint_cos", "addr_cos")}); rows = {}
    for ch in ("name_cos", "joint_cos", "addr_cos"):
        rk = g.groupby(["s1_id", "src"])[ch].rank(ascending=False, method="min")
        rows[ch] = {f"@{K}": y[(rk <= K).values].sum() / len(pos) for K in (1, 3, 5, 10, 20, 50)}
    print(pd.DataFrame(rows).T.round(4))

    print("\n--- (1b) blocking recall by noise class of the true pair ---")
    qd, td = rec_dicts(s1), rec_dicts(pd.concat([s2, s3]))
    items = list(pos); items = [items[i] for i in rng.choice(len(items), min(200000, len(items)), replace=False)]
    cand = set(zip(pk.s1_id, pk.t_id))
    cl = pd.DataFrame([(*classify(qd[s], td[t]), (s, t) in cand, t[:2]) for s, t in items], columns=["name_cls", "addr_cls", "found", "src"])
    for c in ("name_cls", "addr_cls", "src"):
        print(cl.groupby(c).found.agg(["mean", "size"]).round(4).T)
    print(cl.groupby(["name_cls", "addr_cls"]).found.agg(["mean", "size"]).query("size>=20 and mean<0.9").round(3))

    print("\n--- (2) feature-group ablation (LightGBM, S1-level 80/20 split) ---")
    val_ids = set(rng.permutation(s1.entity_id.values)[: int(0.2 * len(s1))]); tv = {s: t for s, t in truth.items() if s in val_ids}
    for nm, cols in (("name only", NAME), ("addr only", ADDR), ("name+addr", NAME + ADDR), ("all (+rank)", list(X.columns))):
        m, p, v, thr = fit_eval(X, y, pk, cols, val_ids, tv, nm)

    print("\n--- (3) gain importance, full model ---")
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print((imp / imp.sum()).round(4).head(20)); imp.to_csv(f"{a.out}/feature_importance.csv")

    print("\n--- (4) errors at best threshold (val S1s) ---")
    pv = pk[v].assign(p=p).sort_values("p", ascending=False).drop_duplicates("t_id")
    fp = pv[(pv.p >= thr) & ~pv.apply(lambda r: (r.s1_id, r.t_id) in pos, axis=1)].head(12)
    print(f"FALSE POSITIVES (showing {len(fp)}):")
    for r in fp.itertuples():
        print(f"  p={r.p:.2f} {qd[r.s1_id][RN]!r} {qd[r.s1_id][RA]!r}  <>  {td[r.t_id][RN]!r} {td[r.t_id][RA]!r}")
    kept = set(zip(pv[pv.p >= thr].s1_id, pv[pv.p >= thr].t_id)); pdict = dict(zip(zip(pk[v].s1_id, pk[v].t_id), p))
    fn = [(s, t) for s, t in pos if s in val_ids and (s, t) not in kept]
    print(f"FALSE NEGATIVES: {len(fn)} (of which not in candidates: {sum((s, t) not in cand for s, t in fn)}); sample:")
    for s, t in [fn[i] for i in rng.choice(len(fn), min(12, len(fn)), replace=False)]:
        print(f"  p={pdict.get((s, t), float('nan')):.2f} {qd[s][RN]!r} {qd[s][RA]!r}  <>  {td[t][RN]!r} {td[t][RA]!r}")


if __name__ == "__main__":
    main()