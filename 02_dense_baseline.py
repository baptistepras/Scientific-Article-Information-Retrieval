"""
Dense retrieval baseline using pre-computed MiniLM embeddings.
Embeddings are already L2-normalised, so cosine similarity = dot product.
No training needed — just load and retrieve.

Usage:
  python3 02_dense_baseline.py
  python3 02_dense_baseline.py --submit-held-out
"""

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import (evaluate, format_text, load_corpus, load_embeddings,
                   load_qrels, load_queries, save_submission)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.parquet"
DEFAULT_CORPUS = DATA_DIR / "corpus.parquet"
DEFAULT_QRELS = DATA_DIR / "qrels.json"
DEFAULT_HELD_OUT = SCRIPT_DIR / "held_out_queries.parquet"
DEFAULT_EMB_DIR = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "dense_baseline"

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(description="Dense baseline retrieval with pre-computed MiniLM embeddings")
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMB_DIR, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--submit-held-out", action="store_true",
                        help="Run inference on held-out queries instead of training queries")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)

    # For held-out queries we need to re-encode them since pre-computed embeddings
    # only cover the 100 training queries
    if args.submit_held_out:
        print("Loading held-out queries...")
        queries = load_queries(args.held_out)
        query_ids = queries["doc_id"].tolist()

        print(f"Loading corpus embeddings from {emb_dir}...")
        corpus_embs, corpus_ids = load_embeddings(
            emb_dir / "corpus_embeddings.npy",
            emb_dir / "corpus_ids.json",
        )
        print(f"  corpus embeddings: {corpus_embs.shape}")

        print(f"Encoding held-out queries with {MINILM_MODEL}...")
        model = SentenceTransformer(MINILM_MODEL)
        query_texts = [format_text(row) for _, row in queries.iterrows()]
        query_embs = model.encode(
            query_texts, batch_size=64, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        print(f"  query embeddings: {query_embs.shape}")
    else:
        print(f"Loading pre-computed embeddings from {emb_dir}...")
        query_embs, query_ids = load_embeddings(
            emb_dir / "query_embeddings.npy",
            emb_dir / "query_ids.json",
        )
        corpus_embs, corpus_ids = load_embeddings(
            emb_dir / "corpus_embeddings.npy",
            emb_dir / "corpus_ids.json",
        )
        print(f"  query embeddings : {query_embs.shape}")
        print(f"  corpus embeddings: {corpus_embs.shape}")

    # dot product = cosine similarity since embeddings are already L2-normalised
    print("Ranking by dot product similarity...")
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :100]
    predictions = {
        qid: [corpus_ids[j] for j in top_indices[i]]
        for i, qid in enumerate(query_ids)
    }

    if not args.submit_held_out:
        queries = load_queries(args.queries)
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        evaluate(predictions, qrels, ks=[10, 100], query_domains=query_domains)

    save_submission(predictions, args.output)


if __name__ == "__main__":
    main()
