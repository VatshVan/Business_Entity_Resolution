"""Consistent mini-world of the train set: sample S1 at --frac (per country), keep ALL their matches, and sample
un-owned S2/S3 distractors at the same rate.  python -m src.make_subset --data dataset --frac 0.02 --out dataset_2pct
Then use e.g.  --data dataset_2pct  with every other script."""
import argparse, os
import numpy as np
from src.common import read_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset"); ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--out", default="dataset_2pct"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    s1 = read_tsv(f"{a.data}/train/train_source1.tsv")
    gt = read_tsv(f"{a.data}/train/train_ground_truth.tsv")
    keep = s1.groupby("country", group_keys=False).sample(frac=a.frac, random_state=a.seed)
    g = gt[gt.source1_entity_id.isin(keep.entity_id)]
    matched = {x for v in g.matched_entity_ids for x in v.split(",") if x}
    owned = {x for v in gt.matched_entity_ids for x in v.split(",") if x}
    os.makedirs(f"{a.out}/train", exist_ok=True)
    keep.to_csv(f"{a.out}/train/train_source1.tsv", sep="\t", index=False)
    g.to_csv(f"{a.out}/train/train_ground_truth.tsv", sep="\t", index=False)
    for i in (2, 3):
        d = read_tsv(f"{a.data}/train/train_source{i}.tsv")
        m = d.entity_id.isin(matched) | (~d.entity_id.isin(owned) & (rng.random_sample(len(d)) < a.frac))
        d[m].to_csv(f"{a.out}/train/train_source{i}.tsv", sep="\t", index=False)
        print(f"S{i}: {m.sum()} of {len(d)}")
    print(f"S1: {len(keep)} of {len(s1)} -> {a.out}")


if __name__ == "__main__":
    main()