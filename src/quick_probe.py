import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.common import load, read_gt

def run_quick_probe(data_root="dataset"):
    print("--- Quick Probe: GT Similarity ---")
    s1, s2, s3 = load(data_root, "train")
    gt = read_gt(f"{data_root}/train/train_ground_truth.tsv")

    # Combine all targets
    t_all = pd.concat([s2, s3]).reset_index(drop=True)

    # Sample GT pairs to check
    gt_list = []
    for s1_id, matches in gt.items():
        for t_id in matches:
            gt_list.append((s1_id, t_id))

    sampled_gt = np.random.choice(len(gt_list), min(len(gt_list), 1000), replace=False)

    # Setup Vectorizer (mirroring src.blocking.candidates)
    # using nc (core name)
    both = pd.concat([s1.nc, t_all.nc])
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, dtype=np.float32).fit(both)

    s1_vecs = vn.transform(s1.nc)
    t_vecs = vn.transform(t_all.nc)

    # Map IDs to indices
    s1_map = {id: idx for idx, id in enumerate(s1.entity_id)}
    t_map = {id: idx for idx, id in enumerate(t_all.entity_id)}

    similarities = []
    for idx in sampled_gt:
        s1_id, t_id = gt_list[idx]
        if s1_id in s1_map and t_id in t_map:
            v1 = s1_vecs[s1_map[s1_id]]
            v2 = t_vecs[t_map[t_id]]
            sim = (v1 * v2).sum()
            similarities.append(sim)

    print(f"Sampled GT Pairs: {len(similarities)}")
    print(f"Mean TF-IDF Similarity: {np.mean(similarities):.4f}")
    print(f"Median TF-IDF Similarity: {np.median(similarities):.4f}")
    print(f"Min TF-IDF Similarity: {np.min(similarities):.4f}")
    print(f"Max TF-IDF Similarity: {np.max(similarities):.4f}")
    print(f"Similarity < 0.1: {sum(1 for s in similarities if s < 0.1)} ({sum(1 for s in similarities if s < 0.1)/len(similarities):.2%})")

if __name__ == "__main__":
    run_quick_probe()
