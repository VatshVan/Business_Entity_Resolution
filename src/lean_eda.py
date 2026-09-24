import pandas as pd, numpy as np, re, sys
from collections import Counter

def analyze_scripts(s):
    if not s or pd.isna(s): return "empty"
    s = str(s)
    has_latin = bool(re.search(r'[a-zA-Z]', s))
    has_non_latin = bool(re.search(r'[^\x00-\x7F]', s))
    if has_latin and has_non_latin: return "mixed"
    if has_non_latin: return "non-latin"
    if has_latin: return "latin"
    return "other"

def run_lean_eda(data_root="dataset"):
    results = {}
    all_script_counts = Counter()

    # 1. Script Distribution (Chunked)
    print("Analyzing scripts in chunks...")
    for split in ["train_source1", "train_source2", "train_source3"]:
        path = f"{data_root}/train/{split}.tsv"
        try:
            # Read only the business_name column in chunks
            for chunk in pd.read_csv(path, sep="\t", usecols=["business_name"], chunksize=100_000, dtype=str):
                all_script_counts.update(chunk["business_name"].apply(analyze_scripts))
        except Exception as e:
            print(f"Error reading {path}: {e}")

    total_names = sum(all_script_counts.values())
    script_dist = {k: v/total_names for k, v in all_script_counts.items()}
    results["script_distribution"] = script_dist

    # 2. Address Noise Sampling (from GT)
    print("Analyzing address noise from GT...")
    gt_path = f"{data_root}/train/train_ground_truth.tsv"
    try:
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
        # Sample 1000 GT pairs to analyze
        sampled_gt = gt_df.sample(min(len(gt_df), 1000))

        # To avoid loading all sources, we'll just note the target IDs
        # and then look them up in the sources selectively.
        # For now, we'll just report the GT size.
        results["gt_size"] = len(gt_df)
    except Exception as e:
        print(f"Error reading GT: {e}")

    # Write results to file to avoid encoding issues
    with open("eda_results.txt", "w", encoding="utf-8") as f:
        f.write("--- LEAN EDA RESULTS ---\n")
        f.write(f"Total Names Analyzed: {total_names}\n")
        f.write("\nScript Distribution:\n")
        for k, v in script_dist.items():
            f.write(f"{k}: {v:.4%}\n")
        if "gt_size" in results:
            f.write(f"\nGround Truth size: {results['gt_size']}\n")

    print("\nAnalysis complete. Results written to eda_results.txt")

if __name__ == "__main__":
    run_lean_eda()
