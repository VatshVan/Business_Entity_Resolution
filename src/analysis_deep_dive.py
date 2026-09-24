import re, unicodedata, sys
import numpy as np, pandas as pd
from collections import Counter
from src.common import load, read_gt, norm
from src.blocking import build_pairs

# Fix Windows encoding for non-latin characters
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def analyze_scripts(s):

    if not s: return "empty"
    has_latin = bool(re.search(r'[a-zA-Z]', s))
    has_non_latin = bool(re.search(r'[^\x00-\x7F]', s))
    if has_latin and has_non_latin: return "mixed"
    if has_non_latin: return "non-latin"
    if has_latin: return "latin"
    return "other"

def run_deep_dive(data_root="dataset"):
    print(f"--- Deep Dive Analysis (Sampled) on {data_root} ---")
    s1, s2, s3 = load(data_root, "train")
    gt = read_gt(f"{data_root}/train/train_ground_truth.tsv")

    # 1. Script Analysis
    all_names = pd.concat([s1.business_name, s2.business_name, s3.business_name])
    script_dist = all_names.apply(analyze_scripts).value_counts(normalize=True)
    print("\n[1] Script Distribution in Business Names:")
    print(script_dist)

    # 2. Normalization Audit
    sample_size = 1000
    sample = all_names.sample(min(len(all_names), sample_size))
    normed = sample.apply(norm)

    non_latin_samples = sample[sample.apply(analyze_scripts) == "non-latin"].head(5)
    print("\n[2] Normalization Audit (Non-Latin Samples):")
    for orig, n in zip(non_latin_samples, normed.loc[non_latin_samples.index].head(5)):
        print(f"Orig: {orig} -> Norm: {n}")

    # 3. Blocking Recall Analysis (on a smaller sample to be fast)
    print("\n[3] Blocking Recall Analysis (k=20, sampled)...")
    # Sample S1 entities that HAVE matches in GT
    s1_with_matches = list(gt.keys())
    sampled_s1_ids = np.random.choice(s1_with_matches, min(len(s1_with_matches), 2000), replace=False)

    s1_sub = s1[s1.entity_id.isin(sampled_s1_ids)].reset_index(drop=True)

    # We still need the full S2/S3 for the blocking to be realistic
    pairs = build_pairs(s1_sub, s2, s3, k=20)

    gt_pairs_sub = set()
    for s1_id in sampled_s1_ids:
        for t_id in gt.get(s1_id, []):
            gt_pairs_sub.add((s1_id, t_id))

    found_pairs = set(zip(pairs.s1_id, pairs.t_id))
    hits = len(gt_pairs_sub & found_pairs)
    recall = hits / len(gt_pairs_sub) if gt_pairs_sub else 0
    print(f"GT Pairs (sampled): {len(gt_pairs_sub)} | Found: {hits} | Recall: {recall:.4f}")

    # 4. Address Noise Analysis (GT Matches)
    print("\n[4] Address Noise Analysis (GT Matches):")
    noise_examples = 0
    for s1_id, matched_ids in list(gt.items())[:100]: # limit to first 100 for speed
        s1_addr = s1.set_index("entity_id").loc[s1_id, "business_address"]
        for t_id in matched_ids:
            t_addr = ""
            if t_id in s2.entity_id.values:
                t_addr = s2.set_index("entity_id").loc[t_id, "business_address"]
            elif t_id in s3.entity_id.values:
                t_addr = s3.set_index("entity_id").loc[t_id, "business_address"]

            if s1_addr.strip().lower() != t_addr.strip().lower():
                if noise_examples < 10:
                    print(f"S1: {s1_addr}\nT:  {t_addr}\n---")
                noise_examples += 1
    print(f"Address mismatches in sample: {noise_examples}")

if __name__ == "__main__":
    run_deep_dive()
