# Business Entity Resolution — baseline (stage 1 + GBDT matcher)

    pip install -r requirements.txt
    python -m src.pipeline eda     --data dataset          # checks assumptions (assignment constraint, country blocking)
    python -m src.pipeline fit     --data dataset --art artifacts
    python -m src.pipeline predict --data dataset --art artifacts --out output
    python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

Pipeline: normalise -> per-country TF-IDF kNN blocking (name char-ngrams + name/address words) -> rapidfuzz/cosine/postal/number
features + rank-competition features -> LightGBM -> one-owner-per-S2/S3-record assignment -> threshold tuned for macro F0.5.

Roadmap: (1) dense multilingual bi-encoder blocking channel (FAISS, GPU); (2) fine-tuned cross-encoder / weight-tied looped
transformer pair scorer feeding LightGBM as a feature; (3) open-LLM (Apache/MIT, <=8B) adjudication of borderline band only;
(4) graph consistency pass over S1-S2-S3 (S2<->S3 agreement).
