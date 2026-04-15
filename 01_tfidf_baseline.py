"""
TF-IDF baseline retrieval on title + abstract.
Fits a TfidfVectorizer on the corpus, then ranks docs by cosine similarity.

Usage:
  python3 01_tfidf_baseline.py
  python3 01_tfidf_baseline.py --retrain
  python3 01_tfidf_baseline.py --submit-held-out
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (evaluate, format_text, load_corpus, load_qrels,
                   load_queries, save_submission)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.parquet"
DEFAULT_CORPUS = DATA_DIR / "corpus.parquet"
DEFAULT_QRELS = DATA_DIR / "qrels.json"
DEFAULT_HELD_OUT = SCRIPT_DIR / "held_out_queries.parquet"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models" / "tfidf"
DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "tfidf"


def build_vectorizer(corpus_texts, model_dir):
    print("Fitting TF-IDF vectorizer on corpus...")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=1,
        max_df=0.85,
        ngram_range=(1, 1),
        stop_words=None,
    )
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved → {model_dir / 'vectorizer.pkl'}")
    return vectorizer, corpus_matrix


def load_vectorizer(model_dir):
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print(f"Loaded vectorizer from {model_dir / 'vectorizer.pkl'}")
    return vectorizer


def main():
    parser = argparse.ArgumentParser(description="TF-IDF baseline retrieval")
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--retrain", action="store_true",
                        help="Refit the vectorizer even if one is already saved")
    parser.add_argument("--submit-held-out", action="store_true",
                        help="Run inference on held-out queries instead of training queries")
    args = parser.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    corpus_ids = corpus["doc_id"].tolist()
    corpus_texts = [format_text(row) for _, row in corpus.iterrows()]
    print(f"  {len(corpus)} docs")

    vectorizer_path = Path(args.model_dir) / "vectorizer.pkl"
    if not args.retrain and vectorizer_path.exists():
        vectorizer = load_vectorizer(Path(args.model_dir))
        corpus_matrix = vectorizer.transform(corpus_texts)
    else:
        vectorizer, corpus_matrix = build_vectorizer(corpus_texts, Path(args.model_dir))

    print(f"Corpus TF-IDF matrix: {corpus_matrix.shape}")

    if args.submit_held_out:
        print("Loading held-out queries...")
        queries = load_queries(args.held_out)
    else:
        print("Loading queries...")
        queries = load_queries(args.queries)
    query_ids = queries["doc_id"].tolist()
    query_texts = [format_text(row) for _, row in queries.iterrows()]
    print(f"  {len(queries)} queries")

    query_matrix = vectorizer.transform(query_texts)

    print("Computing similarities and ranking...")
    sim_matrix = cosine_similarity(query_matrix, corpus_matrix)
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :100]
    predictions = {
        qid: [corpus_ids[j] for j in top_indices[i]]
        for i, qid in enumerate(query_ids)
    }

    if not args.submit_held_out:
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        evaluate(predictions, qrels, ks=[10, 100], query_domains=query_domains)

    save_submission(predictions, args.output)


if __name__ == "__main__":
    main()
