"""
Score-level fusion: BGE + BM25 with normalized weighted sum.

Normalizes each retriever's scores to [0,1] and takes a weighted sum.
BGE dominates (high alpha) while BM25 provides a small lexical bonus
for exact-term matches.

Formula: final_score = alpha * bge_score_norm + (1 - alpha) * bm25_score_norm

Without --submit-held-out, runs a grid search over alpha on training queries
and prints the best value. BGE query embeddings are cached for reuse.

Usage:
  python3 10_score_fusion.py                    # grid search + eval
  python3 10_score_fusion.py --alpha 0.85       # fixed alpha + eval
  python3 10_score_fusion.py --submit-held-out --alpha 0.85
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from utils import (evaluate, format_text, get_device, load_corpus,
                   load_embeddings, load_qrels, load_queries, save_submission)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.parquet"
DEFAULT_CORPUS = DATA_DIR / "corpus.parquet"
DEFAULT_QRELS = DATA_DIR / "qrels.json"
DEFAULT_HELD_OUT = SCRIPT_DIR / "held_out_queries.parquet"
DEFAULT_BM25_DIR = SCRIPT_DIR / "models" / "bm25"
DEFAULT_BGE_DIR = SCRIPT_DIR / "models" / "bge"
DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "score_fusion"
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_BATCH_SIZE = 64
DEFAULT_ALPHA = 0.85   # weight for BGE; (1-alpha) for BM25

# BGE retrieval models use this prefix for queries (not corpus docs)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_STOPWORDS = None


def get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words("english"))
        except LookupError:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def tokenize(text: str) -> list:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = text.split()
    sw = get_stopwords()
    return [t for t in tokens if t not in sw]


def load_bm25(model_dir):
    with open(model_dir / "index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    print(f"Loaded BM25 index from {model_dir / 'index.pkl'}")
    return bm25


def build_bm25(corpus_texts, model_dir):
    print("Tokenizing corpus (clean_nostop)...")
    tokenized = [tokenize(t) for t in tqdm(corpus_texts)]
    bm25 = BM25Okapi(tokenized, k1=1.0, b=1.0)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved → {model_dir / 'index.pkl'}")
    return bm25


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Normalize a 1D array to [0, 1] using min-max scaling."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def encode_queries_bge(model, texts, batch_size):
    prefixed = [QUERY_INSTRUCTION + t for t in texts]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="BGE + BM25 score-level fusion"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--bm25-dir", default=DEFAULT_BM25_DIR, type=Path)
    parser.add_argument("--bge-dir", default=DEFAULT_BGE_DIR, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--alpha", type=float, default=None,
                        help="BGE score weight (0-1). If omitted, grid search is run on training queries.")
    parser.add_argument("--retrain", action="store_true",
                        help="Re-encode BGE queries even if cached")
    parser.add_argument("--rebuild-bm25", action="store_true",
                        help="Rebuild BM25 index")
    parser.add_argument("--submit-held-out", action="store_true",
                        help="Run inference on held-out queries (requires --alpha)")
    args = parser.parse_args()

    if args.submit_held_out and args.alpha is None:
        parser.error("--submit-held-out requires --alpha (run without --submit-held-out first to find best alpha)")

    device = get_device()
    print(f"Using device: {device}")

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    corpus_ids = corpus["doc_id"].tolist()
    corpus_texts = [format_text(row) for _, row in corpus.iterrows()]
    print(f"  {len(corpus)} docs")

    # BM25
    index_path = Path(args.bm25_dir) / "index.pkl"
    if not args.rebuild_bm25 and index_path.exists():
        bm25 = load_bm25(Path(args.bm25_dir))
    else:
        bm25 = build_bm25(corpus_texts, Path(args.bm25_dir))

    bge_dir = Path(args.bge_dir)
    print("Loading BGE corpus embeddings...")
    bge_corpus_embs, _ = load_embeddings(
        bge_dir / "corpus_embeddings.npy",
        bge_dir / "corpus_ids.json",
    )
    print(f"  BGE corpus: {bge_corpus_embs.shape}")

    bge_q_emb_path = bge_dir / "query_embeddings.npy"
    bge_q_ids_path = bge_dir / "query_ids.json"

    # ── Load / encode queries ────────────────────────────────────
    if args.submit_held_out:
        print("Loading held-out queries...")
        queries = load_queries(args.held_out)
        query_ids = queries["doc_id"].tolist()
        query_texts = [format_text(row) for _, row in queries.iterrows()]
        print(f"  {len(queries)} held-out queries")

        from sentence_transformers import SentenceTransformer
        print(f"Loading BGE model: {args.model_name} ...")
        bge_model = SentenceTransformer(args.model_name, device=device)
        print("Encoding held-out queries with BGE prefix...")
        bge_query_embs = encode_queries_bge(bge_model, query_texts, args.batch_size)
        del bge_model

    else:
        queries = load_queries(args.queries)
        query_ids = queries["doc_id"].tolist()
        query_texts = [format_text(row) for _, row in queries.iterrows()]
        print(f"  {len(queries)} training queries")

        if not args.retrain and bge_q_emb_path.exists():
            print("Loading cached BGE query embeddings...")
            bge_query_embs, _ = load_embeddings(bge_q_emb_path, bge_q_ids_path)
            print(f"  BGE queries: {bge_query_embs.shape}")
        else:
            from sentence_transformers import SentenceTransformer
            print(f"Loading BGE model: {args.model_name} ...")
            bge_model = SentenceTransformer(args.model_name, device=device)
            print("Encoding queries with BGE prefix...")
            bge_query_embs = encode_queries_bge(bge_model, query_texts, args.batch_size)
            del bge_model
            np.save(bge_q_emb_path, bge_query_embs)
            with open(bge_q_ids_path, "w") as f:
                json.dump(query_ids, f)
            print(f"  BGE queries: {bge_query_embs.shape} → saved")

    # ── Similarity matrix ────────────────────────────────────────
    print("Computing BGE similarity matrix...")
    bge_sim = bge_query_embs @ bge_corpus_embs.T   # (n_queries, n_docs)

    print("Computing BM25 scores for all queries...")
    bm25_matrix = np.zeros((len(query_ids), len(corpus_ids)), dtype=np.float32)
    for i, qtext in enumerate(tqdm(query_texts, desc="BM25")):
        bm25_matrix[i] = np.array(bm25.get_scores(tokenize(qtext)), dtype=np.float32)

    # ── Held-out submission ──────────────────────────────────────
    if args.submit_held_out:
        alpha = args.alpha
        print(f"Score fusion: {alpha:.2f} * BGE + {1 - alpha:.2f} * BM25")
        predictions = {}
        for i, qid in enumerate(tqdm(query_ids, desc="Ranking")):
            fused = alpha * normalize_minmax(bge_sim[i]) + (1 - alpha) * normalize_minmax(bm25_matrix[i])
            top_idx = np.argsort(-fused)[:100]
            predictions[qid] = [corpus_ids[j] for j in top_idx]
        save_submission(predictions, args.output)
        return

    # ── Training eval ────────────────────────────────────────────
    qrels = load_qrels(args.qrels)
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    if args.alpha is not None:
        # Fixed alpha: single eval
        alpha = args.alpha
        print(f"\nScore fusion: {alpha:.2f} * BGE + {1 - alpha:.2f} * BM25")
        predictions = {}
        for i, qid in enumerate(tqdm(query_ids, desc="Ranking")):
            fused = alpha * normalize_minmax(bge_sim[i]) + (1 - alpha) * normalize_minmax(bm25_matrix[i])
            top_idx = np.argsort(-fused)[:100]
            predictions[qid] = [corpus_ids[j] for j in top_idx]
        evaluate(predictions, qrels, ks=[10, 100], query_domains=query_domains)
        save_submission(predictions, args.output)
        return

    # ── Grid search over alpha ───────────────────────────────────
    print("\nGrid searching alpha (BGE weight)...")
    alpha_values = [round(a * 0.05, 2) for a in range(1, 20)]  # 0.05 to 0.95 step 0.05

    best_map = 0.0
    best_alpha = DEFAULT_ALPHA
    results_log = []

    for alpha in alpha_values:
        predictions = {}
        for i, qid in enumerate(query_ids):
            fused = alpha * normalize_minmax(bge_sim[i]) + (1 - alpha) * normalize_minmax(bm25_matrix[i])
            top_idx = np.argsort(-fused)[:100]
            predictions[qid] = [corpus_ids[j] for j in top_idx]
        result = evaluate(predictions, qrels, ks=[10, 100], verbose=False)
        map_score = result["overall"]["MAP"]
        results_log.append((alpha, map_score))
        if map_score > best_map:
            best_map = map_score
            best_alpha = alpha

    print("\nGrid search results:")
    print(f"  {'alpha':>6}  MAP")
    for alpha, map_score in results_log:
        marker = " ← BEST" if alpha == best_alpha else ""
        print(f"  {alpha:>6.2f}  {map_score:.4f}{marker}")

    print(f"\nBest alpha={best_alpha} → MAP={best_map:.4f}")
    print(f"\nTo submit held-out:")
    print(f"  python3 10_score_fusion.py --submit-held-out --alpha {best_alpha}")

    # Full eval with best alpha
    print("\nFull evaluation with best alpha:")
    predictions = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Ranking")):
        fused = best_alpha * normalize_minmax(bge_sim[i]) + (1 - best_alpha) * normalize_minmax(bm25_matrix[i])
        top_idx = np.argsort(-fused)[:100]
        predictions[qid] = [corpus_ids[j] for j in top_idx]
    evaluate(predictions, qrels, ks=[10, 100], query_domains=query_domains)
    save_submission(predictions, args.output)


if __name__ == "__main__":
    main()
