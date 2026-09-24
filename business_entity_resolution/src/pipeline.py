"""python -m src.pipeline eda|fit|predict --data dataset --art artifacts --out output"""
import argparse, os
from collections import Counter
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from src.common import load, read_gt, f05_macro
from src.blocking import build_pairs
from src.features import build_features


def decide(pk, prob, thr, assign=True):
    """Each S2/S3 record is assigned to at most one S1 (S1 is deduplicated), then threshold."""
    d = pk[["s1_id", "t_id"]].assign(p=prob)
    if assign:
        d = d.sort_values("p", ascending=False).drop_duplicates("t_id")
    d = d[d.p >= thr]
    return d.groupby("s1_id").t_id.apply(set).to_dict()


def write_lists(path, s1_ids, d, col):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            f.write(s + "\t" + ",".join(sorted(d.get(s, ()))) + "\n")


def eda(a):
    s1, s2, s3 = load(a.data, "train")
    gt = read_gt(f"{a.data}/train/train_ground_truth.tsv")
    print("sizes", len(s1), len(s2), len(s3), "| countries", s1.country.value_counts().to_dict())
    print("S1 singleton rate", np.mean([not v for v in gt.values()]))
    print("matches/S1", pd.Series([len(v) for v in gt.values()]).describe().to_dict())
    owner = Counter(t for v in gt.values() for t in v)
    print("targets claimed by >1 S1 (assignment constraint valid if 0):", sum(c > 1 for c in owner.values()))
    cty = pd.concat([s1, s2, s3]).set_index("entity_id").country
    cross = sum(cty[s] != cty[t] for s, v in gt.items() for t in v)
    print("cross-country GT pairs (country blocking valid if 0):", cross)
    for d, n in ((s1, "S1"), (s2, "S2"), (s3, "S3")):
        print(n, "empty addr", (d.a == "").mean().round(3), "| has postal", (d.pc != "").mean().round(3))
    for s in list(gt)[:5]:
        print("\n", s1.set_index("entity_id").loc[s, ["business_name", "business_address"]].tolist())
        allr = pd.concat([s2, s3]).set_index("entity_id")
        for t in gt[s]:
            print("   ->", t, allr.loc[t, ["business_name", "business_address"]].tolist())


def fit(a):
    s1, s2, s3 = load(a.data, "train")
    gt = read_gt(f"{a.data}/train/train_ground_truth.tsv")
    truth = {s: gt.get(s, set()) for s in s1.entity_id}
    pairs = build_pairs(s1, s2, s3, a.k)
    pk, X = build_features(pairs, s1, s2, s3)
    pos = {(s, t) for s, v in truth.items() for t in v}
    y = np.array([(s, t) in pos for s, t in zip(pk.s1_id, pk.t_id)], int)
    print("pairs", len(pk), "| candidate recall", round(y.sum() / max(len(pos), 1), 4), "| pos rate", y.mean().round(4))
    rng = np.random.RandomState(0)
    ids = s1.entity_id.values
    val = set(rng.permutation(ids)[: int(0.2 * len(ids))])
    v = pk.s1_id.isin(val).values
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, verbose=-1)
    m.fit(X[~v], y[~v], eval_set=[(X[v], y[v])], callbacks=[lgb.early_stopping(50, verbose=False)])
    pr = m.predict_proba(X[v])[:, 1]
    tv = {s: t for s, t in truth.items() if s in val}
    best = max(((f05_macro(tv, decide(pk[v], pr, t)), t) for t in np.arange(0.3, 0.96, 0.05)))
    print("val macro F0.5 %.4f @ thr %.2f" % best)
    os.makedirs(a.art, exist_ok=True)
    joblib.dump({"model": m, "cols": list(X.columns), "thr": float(best[1])}, f"{a.art}/model.joblib")


def predict(a):
    s1, s2, s3 = load(a.data, "test")
    art = joblib.load(f"{a.art}/model.joblib")
    pairs = build_pairs(s1, s2, s3, a.k)
    pk, X = build_features(pairs, s1, s2, s3)
    prob = art["model"].predict_proba(X[art["cols"]])[:, 1]
    res = decide(pk, prob, art["thr"])
    cand = pk.groupby("s1_id").t_id.apply(set).to_dict()
    write_lists(f"{a.out}/matching_results.tsv", s1.entity_id, res, "matched_entity_ids")
    write_lists(f"{a.out}/candidate_pairs.tsv", s1.entity_id, cand, "candidate_entity_ids")
    print("matched S1:", len(res), "/", len(s1), "| wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eda", "fit", "predict"])
    ap.add_argument("--data", default="dataset"); ap.add_argument("--art", default="artifacts")
    ap.add_argument("--out", default="output"); ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    {"eda": eda, "fit": fit, "predict": predict}[a.cmd](a)
