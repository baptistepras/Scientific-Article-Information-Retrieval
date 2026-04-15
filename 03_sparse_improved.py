"""
BM25 retrieval on title + abstract using rank_bm25 (BM25Okapi).
BM25 typically outperforms TF-IDF for IR tasks.

Usage:
  python3 03_sparse_improved.py
  python3 03_sparse_improved.py --retrain
  python3 03_sparse_improved.py --submit-held-out
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from utils import (evaluate, format_text, load_corpus, load_qrels,
                   load_queries, save_submission)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.parquet"
DEFAULT_CORPUS = DATA_DIR / "corpus.parquet"
DEFAULT_QRELS = DATA_DIR / "qrels.json"
DEFAULT_HELD_OUT = SCRIPT_DIR / "held_out_queries.parquet"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models" / "bm25"
DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "bm25"

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
    # clean_nostop: remove punctuation + filter English stopwords (best from tuning)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = text.split()
    sw = get_stopwords()
    return [t for t in tokens if t not in sw]


def build_bm25(corpus_texts, model_dir):
    print("Tokenizing corpus...")
    tokenized = [tokenize(t) for t in tqdm(corpus_texts)]
    print("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized, k1=1.0, b=1.0)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved → {model_dir / 'index.pkl'}")
    return bm25


def load_bm25(model_dir):
    with open(model_dir / "index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    print(f"Loaded BM25 index from {model_dir / 'index.pkl'}")
    return bm25


def retrieve(bm25, query_texts, corpus_ids, top_k=100):
    predictions = {}
    for i, qtext in enumerate(tqdm(query_texts, desc="Retrieving")):
        tokens = tokenize(qtext)
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(-scores)[:top_k]
        # query_id will be filled by caller, store by index for now
        predictions[i] = [corpus_ids[j] for j in top_indices]
    return predictions


def main():
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
