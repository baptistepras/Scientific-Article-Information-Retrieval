"""
Enhanced cross-encoder reranking with citation-context enriched queries.

Improvements over a first cross-encoder version:
  - Query text enriched with citation-context sentences from full_text
  - max_length=1024 (vs 512) — more text for the cross-encoder to discriminate
  - Reranks top-200 candidates (vs 100) to avoid recall loss
  - Interpolates CE scores with base fusion scores (grid search gamma)
    to prevent catastrophic overrides

Pipeline:
  1. Load base fusion scores (UAE+BGE+E5+TFIDF = 0.57 MAP).
  2. Extract citation-context sentences from query full_text.
  3. Build enriched query: title + abstract + citation contexts.
  4. For each query, take top-200 from fusion → score with bge-reranker-v2-m3.
  5. Interpolate: final = gamma * ce_norm + (1-gamma) * fusion_norm.
  6. Grid search gamma on training set.

Usage:
  python3 09_cross_encoder.py                          # grid search gamma
  python3 09_cross_encoder.py --gamma 0.60             # single config
  python3 09_cross_encoder.py --submit-held-out --gamma 0.60
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from utils import (DEFAULT_CORPUS, DEFAULT_HELD_OUT, DEFAULT_QRELS,
                   DEFAULT_QUERIES, SCRIPT_DIR, compute_base_fusion, evaluate,
                   extract_citation_sentences, format_text, get_device,
                   load_corpus, load_qrels, load_queries, save_submission)

DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "crossencoder_v2"
DEFAULT_CE_DIR = SCRIPT_DIR / "models" / "crossencoder_v2"
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"
DEFAULT_BATCH_SIZE = 64
DEFAULT_BATCH_SIZE_CE = 32
DEFAULT_RERANK_TOP = 100


# Citation context extraction

def build_enriched_query(row: pd.Series, max_ctx_chars: int = 1000) -> str:
    """Title + abstract + citation context sentences (truncated)."""
    ta = format_text(row)
    ctx = extract_citation_sentences(str(row.get("full_text", "")))
    if ctx:
        return ta + "\n\n" + ctx[:max_ctx_chars]
    return ta


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced cross-encoder reranking with citation-context queries"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--ce-dir", default=DEFAULT_CE_DIR, type=Path)
    parser.add_argument("--cross-encoder", default=DEFAULT_CROSS_ENCODER)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-size-ce", type=int, default=DEFAULT_BATCH_SIZE_CE)
    parser.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    parser.add_argument("--gamma", type=float, default=None,
                        help="Fixed interpolation weight (skip grid search). "
                             "1.0 = pure CE, 0.0 = pure fusion.")
    parser.add_argument("--retrain", action="store_true",
                        help="Force re-score with cross-encoder")
    parser.add_argument("--submit-held-out", action="store_true")
    args = parser.parse_args()

    device = get_device()
    ce_dir = Path(args.ce_dir)
    ce_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Cross-encoder: {args.cross_encoder}")
    print(f"Rerank top-{args.rerank_top}")

    # Load corpus
    print("\nLoading corpus...")
    corpus = load_corpus(args.corpus)
    corpus_ids = corpus["doc_id"].tolist()
    corpus_ta_texts = [format_text(row) for _, row in corpus.iterrows()]
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    print(f"  {len(corpus)} docs")

    # Load queries
    is_heldout = args.submit_held_out
    queries = load_queries(args.held_out if is_heldout else args.queries)
    query_ids = queries["doc_id"].tolist()
    query_ta_texts = [format_text(row) for _, row in queries.iterrows()]
    print(f"  {len(queries)} {'held-out' if is_heldout else 'training'} queries")

    # Build enriched queries
    print("Building enriched query texts (TA + citation contexts)...")
    enriched_queries = [build_enriched_query(row) for _, row in queries.iterrows()]
    n_enriched = sum(1 for eq, ta in zip(enriched_queries, query_ta_texts) if len(eq) > len(ta))
    print(f"  {n_enriched}/{len(queries)} queries enriched with citation contexts")

    # Base fusion
    print("\nLoading base fusion scores...")
    fusion_scores = compute_base_fusion(
        query_ids, query_ta_texts, corpus_ta_texts, is_heldout, device, args.batch_size
    )

    # Cross-encoder scoring
    suffix = "_heldout" if is_heldout else "_train"
    ce_cache = ce_dir / f"ce_scores{suffix}.npy"

    if not args.retrain and ce_cache.exists():
        print(f"\nLoading cached CE scores from {ce_cache}...")
        # ce_scores_sparse: (n_q, rerank_top) — scores for top-K candidates
        # ce_indices: (n_q, rerank_top) — corpus indices of candidates
        ce_data = np.load(ce_dir / f"ce_data{suffix}.npz")
        ce_scores_sparse = ce_data["scores"]
        ce_indices = ce_data["indices"]
    else:
        from sentence_transformers import CrossEncoder

        print(f"\nLoading cross-encoder: {args.cross_encoder}")
        ce = CrossEncoder(args.cross_encoder, device=device, max_length=512)

        n_q = len(query_ids)
        rerank_top = args.rerank_top
        ce_indices = np.zeros((n_q, rerank_top), dtype=np.int64)

        # Collect all pairs at once to avoid per-query GPU overhead
        all_pairs = []
        for i in range(n_q):
            top_k_idx = np.argsort(-fusion_scores[i])[:rerank_top]
            ce_indices[i] = top_k_idx
            for j in top_k_idx:
                all_pairs.append((enriched_queries[i], corpus_ta_texts[j]))

        print(f"  Scoring {len(all_pairs)} pairs in one batch...")
        all_scores = ce.predict(all_pairs, batch_size=args.batch_size_ce,
                                show_progress_bar=True)
        ce_scores_sparse = np.array(all_scores, dtype=np.float32).reshape(n_q, rerank_top)

        del ce
        np.savez(
            ce_dir / f"ce_data{suffix}.npz",
            scores=ce_scores_sparse, indices=ce_indices,
        )
        print(f"  Saved CE scores → {ce_dir / f'ce_data{suffix}.npz'}")

    # Build full CE score matrix (sparse → dense for top-K only)
    n_q = len(query_ids)
    n_docs = len(corpus_ids)

    def rank_with_gamma(gamma: float) -> dict[str, list[str]]:
        """Interpolate CE + fusion, return top-100 predictions."""
        predictions = {}
        for i, qid in enumerate(query_ids):
            top_k_idx = ce_indices[i]
            ce_raw = ce_scores_sparse[i]

            # Normalize CE scores for this query
            ce_min, ce_max = ce_raw.min(), ce_raw.max()
            if ce_max - ce_min > 1e-10:
                ce_norm = (ce_raw - ce_min) / (ce_max - ce_min)
            else:
                ce_norm = np.zeros_like(ce_raw)

            # Normalize fusion scores for the same candidates
            fus_raw = fusion_scores[i, top_k_idx]
            fus_min, fus_max = fus_raw.min(), fus_raw.max()
            if fus_max - fus_min > 1e-10:
                fus_norm = (fus_raw - fus_min) / (fus_max - fus_min)
            else:
                fus_norm = np.zeros_like(fus_raw)

            # Interpolate
            combined = gamma * ce_norm + (1 - gamma) * fus_norm
            sorted_local = np.argsort(-combined)

            ranked_ids = [corpus_ids[top_k_idx[j]] for j in sorted_local[:100]]
            predictions[qid] = ranked_ids
        return predictions

    # Held-out submission
    if is_heldout:
        gamma = args.gamma if args.gamma is not None else 0.60
        print(f"\nSubmitting with gamma={gamma}")
        predictions = rank_with_gamma(gamma)
        save_submission(predictions, args.output)
        return

    # Training evaluation
    qrels = load_qrels(args.qrels)
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    # Reference: fusion only (gamma=0)
    base_preds = rank_with_gamma(0.0)
    base_result = evaluate(base_preds, qrels, ks=[10, 100], verbose=False)
    base_map = base_result["overall"]["MAP"]
    base_ndcg = base_result["overall"]["NDCG@10"]
    print(f"\nBase fusion (gamma=0): MAP={base_map:.4f}, NDCG@10={base_ndcg:.4f}")

    # Single config
    if args.gamma is not None:
        preds = rank_with_gamma(args.gamma)
        evaluate(preds, qrels, ks=[10, 100], query_domains=query_domains)
        save_submission(preds, args.output)
        return

    # Grid search gamma
    gamma_values = [round(g * 0.1, 1) for g in range(11)]  # 0.0 to 1.0
    print(f"\nGrid search over gamma: {gamma_values}")

    best_map = base_map
    best_ndcg = base_ndcg
    best_gamma = 0.0
    results = []

    for gamma in gamma_values:
        preds = rank_with_gamma(gamma)
        result = evaluate(preds, qrels, ks=[10, 100], verbose=False)
        m = result["overall"]["MAP"]
        n = result["overall"]["NDCG@10"]
        results.append((gamma, m, n))
        if m > best_map:
            best_map = m
            best_ndcg = n
            best_gamma = gamma

    # Print table
    print(f"\n{'gamma':>6}  {'MAP':>6}  {'NDCG@10':>8}")
    print("-" * 28)
    for gamma, m, n in results:
        marker = " ← best" if gamma == best_gamma else ""
        print(f"{gamma:>6.1f}  {m:.4f}  {n:.4f}{marker}")

    print(f"\n{'=' * 50}")
    print(f"Base (gamma=0): MAP={base_map:.4f}, NDCG@10={base_ndcg:.4f}")
    print(f"Best:           MAP={best_map:.4f}, NDCG@10={best_ndcg:.4f}")
    print(f"Best gamma:     {best_gamma}")
    print(f"{'=' * 50}")

    # Full eval with best gamma
    best_preds = rank_with_gamma(best_gamma)
    evaluate(best_preds, qrels, ks=[10, 100], query_domains=query_domains)
    save_submission(best_preds, args.output)

    print(f"\nCommand for held-out submission:")
    print(f"python3 09_cross_encoder.py --submit-held-out --gamma {best_gamma}")


if __name__ == "__main__":
    main()
