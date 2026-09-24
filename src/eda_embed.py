"""Zero-shot embedding probe (needs torch + sentence-transformers; fits 4GB VRAM with a small model).
python -m src.eda_embed --data dataset_2pct --model intfloat/multilingual-e5-small --n 5000
Reports recall@1/10/50 of nearest-neighbour retrieval (S1 -> S2/S3 pool, same country) by name noise class."""
import argparse
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from src.common import load, read_gt
from src.eda_deep import rec_dicts, name_class, RN, RA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_2pct"); ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args(); rng = np.random.RandomState(0)
    s1, s2, s3 = load(a.data, "train"); gt = read_gt(f"{a.data}/train/train_ground_truth.tsv")
    pool = pd.concat([s2, s3]).reset_index(drop=True); qd, td = rec_dicts(s1), rec_dicts(pool)
    m = SentenceTransformer(a.model, device="cuda"); m.half()
    enc = lambda df: m.encode(("query: " + df.business_name + " | " + df.business_address).tolist(), batch_size=a.bs,
                              convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    res = []
    for c in sorted(s1.country.unique()):
        q = s1[(s1.country == c) & s1.entity_id.map(lambda e: bool(gt.get(e)))]
        q = q.sample(min(a.n, len(q)), random_state=0); p = pool[pool.country == c].reset_index(drop=True)
        Q, P = enc(q), enc(p); pid = p.entity_id.values
        top = (Q @ P.T).topk(50, dim=1).indices.cpu().numpy()
        for i, s in enumerate(q.entity_id):
            ranked = pid[top[i]]
            for t in gt[s]:
                r = np.where(ranked == t)[0]
                res.append((c, name_class(qd[s], td[t]), r[0] if len(r) else 999))
    df = pd.DataFrame(res, columns=["country", "name_cls", "rank"])
    for K in (1, 10, 50):
        df[f"r@{K}"] = df["rank"] < K
    print(df.groupby("country")[["r@1", "r@10", "r@50"]].mean().round(4))
    print(df.groupby("name_cls")[["r@1", "r@10", "r@50"]].agg("mean").assign(n=df.groupby("name_cls").size()).round(4))


if __name__ == "__main__":
    main()