"""
BM25 retrieval on title + abstract using rank_bm25 (BM25Okapi).
BM25 typically outperforms TF-IDF for IR tasks.

Usage:
  python3 02_bm25.py
  python3 02_bm25.py --retrain
  python3 02_bm25.py --submit-held-out
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from utils import (DEFAULT_CORPUS, DEFAULT_HELD_OUT, DEFAULT_QRELS,
                   DEFAULT_QUERIES, SCRIPT_DIR, build_bm25, evaluate,
                   format_text, load_bm25, load_corpus, load_qrels,
                   load_queries, save_submission, tokenize)

DEFAULT_MODEL_DIR = SCRIPT_DIR / "models" / "bm25"
DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "bm25"


def retrieve(bm25: Any, query_texts: list[str], corpus_ids: list[str],
             top_k: int = 100) -> dict[int, list[str]]:
    predictions = {}
    for i, qtext in enumerate(tqdm(query_texts, desc="Retrieving")):
        tokens = tokenize(qtext)
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(-scores)[:top_k]
        # query_id will be filled by caller, store by index for now
        predictions[i] = [corpus_ids[j] for j in top_indices]
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 improved sparse retrieval")
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--retrain", action="store_true",
                        help="Rebuild BM25 index even if one is already saved")
    parser.add_argument("--submit-held-out", action="store_true",
                        help="Run inference on held-out queries instead of training queries")
    args = parser.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    corpus_ids = corpus["doc_id"].tolist()
    corpus_texts = [format_text(row) for _, row in corpus.iterrows()]
    print(f"  {len(corpus)} docs")

    index_path = Path(args.model_dir) / "index.pkl"
    if not args.retrain and index_path.exists():
        bm25 = load_bm25(Path(args.model_dir))
    else:
        bm25 = build_bm25(corpus_texts, Path(args.model_dir))

    if args.submit_held_out:
        print("Loading held-out queries...")
        queries = load_queries(args.held_out)
    else:
        print("Loading queries...")
        queries = load_queries(args.queries)
    query_ids = queries["doc_id"].tolist()
    query_texts = [format_text(row) for _, row in queries.iterrows()]
    print(f"  {len(queries)} queries")

    raw = retrieve(bm25, query_texts, corpus_ids)
    predictions = {query_ids[i]: docs for i, docs in raw.items()}

    if not args.submit_held_out:
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        evaluate(predictions, qrels, ks=[10, 100], query_domains=query_domains)

    save_submission(predictions, args.output)


if __name__ == "__main__":
    main()
