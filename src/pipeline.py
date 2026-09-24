"""python -m src.pipeline eda|fit|predict --data dataset --art artifacts --out output"""
import argparse, os
from collections import Counter
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from src.common import load, read_gt, f05_macro
from src.blocking import build_pairs
from src.features import build_features


def decide(pk, prob, thr, assign=True):
    """Each S2/S3 record is assigned to at most one S1 (S1 is deduplicated), then threshold.
    If assign is True, each S2/S3 record is assigned to the S1 with the highest probability.
    If assign is False, all S2/S3 records are considered for matching.
    So, to explain in detail with an example:
    Suppose we have the following pairs with their probabilities:
    S1_id | S2/S3_id | Probability
    A     | X        | 0.9
    A     | Y        | 0.8
    B     | X        | 0.7
    If assign is True, we will assign X to A (highest probability) and Y to A. Then we will apply the 
    threshold, and if the threshold is 0.85, we will keep only the pair (A, X) because it meets the threshold. 
    The final result will be:
    {
        'A': {'X'},
        'B': set()
    }
    If assign is False, we will not assign X to A or B, and we will consider all pairs for matching. 
    Then we will apply the threshold, and if the threshold is 0.85, we will keep only the pair (A, X) 
    because it meets the threshold. The final result will be:
    {
        'A': {'X'},
        'B': set()
    }
    """
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
    """Exploratory Data Analysis (EDA) on the training dataset.
    This function loads the training data, reads the ground truth and prints various statistics about the dataset.
    It calculates the sizes of the datasets, the singleton rate of S1, the number of matches per S1, the number 
    of targets claimed by more than one S1 and the number of cross-country ground truth pairs. It also checks for 
    empty addresses and the presence of postal codes in the datasets. Finally, it prints a few examples of S1 
    records and their corresponding matches in S2 and S3.
    """
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
    # print of adress errors, like Corp and Corporation, Inc and Incorporated, etc.
    print("\nAddress errors (e.g., Corp vs. Corporation, Inc vs. Incorporated):")
    for s in list(gt)[:5]:
        s1_addr = s1.set_index("entity_id").loc[s, "business_address"]
        allr = pd.concat([s2, s3]).set_index("entity_id")
        for t in gt[s]:
            t_addr = allr.loc[t, "business_address"]
            if s1_addr != t_addr:
                print(f"   -> S1: {s1_addr} | S2/S3: {t_addr}")


def fit(a):
    """
    Train a LightGBM model on the training dataset.
    This function loads the training data, reads the ground truth, builds candidate pairs, extracts features, 
    and trains a LightGBM model. It evaluates the model on a validation set and saves the trained model and its 
    parameters to disk.
    """
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
    """
    Predict matches on the test dataset using a trained LightGBM model.
    This function loads the test data, loads the trained model, builds candidate pairs, extracts features, and 
    predicts the probabilities of matches. It then applies a decision threshold to determine the final matches 
    and writes the results to output files.
    """
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
