"""
Generic dense retrieval with configurable model name and query/corpus prefixes.

Allows testing MTEB top retrieval models:
  - WhereIsAI/UAE-Large-V1 (no prefix needed, default)
  - intfloat/e5-large-v2   (--query-prefix "query: " --corpus-prefix "passage: ")
  - BAAI/bge-large-en-v1.5 (--query-prefix "Represent this sentence for searching relevant passages: " --dir-name bge)
  - malteos/scincl         (--dir-name specter2)

Embeddings are cached under models/{model_name_safe}/ where model_name_safe
replaces "/" with "_", or under models/{dir_name}/ with --dir-name. Later scripts
read BGE from models/bge/ and SciNCL from models/specter2/.
Use --force-encode to re-encode corpus, --retrain for queries.

Usage:
  python3 04_dense_encoders.py
  python3 04_dense_encoders.py --model-name intfloat/e5-large-v2 --query-prefix "query: " --corpus-prefix "passage: "
  python3 04_dense_encoders.py --retrain
  python3 04_dense_encoders.py --submit-held-out
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import (evaluate, format_text, get_device, load_corpus,
                   load_embeddings, load_qrels, load_queries, save_submission)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.parquet"
DEFAULT_CORPUS = DATA_DIR / "corpus.parquet"
DEFAULT_QRELS = DATA_DIR / "qrels.json"
DEFAULT_HELD_OUT = SCRIPT_DIR / "held_out_queries.parquet"
DEFAULT_MODEL_NAME = "WhereIsAI/UAE-Large-V1"
DEFAULT_BATCH_SIZE = 64


def model_name_to_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def encode_texts(model, texts, prefix, batch_size):
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Dense retrieval with configurable model and prefixes"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="HuggingFace model name (default: WhereIsAI/UAE-Large-V1)")
    parser.add_argument("--query-prefix", default="",
                        help="Prefix to prepend to each query text (default: none)")
    parser.add_argument("--corpus-prefix", default="",
                        help="Prefix to prepend to each corpus text (default: none)")
    parser.add_argument("--dir-name", default=None,
                        help="Directory under models/ (default: model name with / replaced by _)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--retrain", action="store_true",
                        help="Re-encode queries even if cached")
    parser.add_argument("--force-encode", action="store_true",
                        help="Re-encode corpus even if cached (slow)")
    parser.add_argument("--submit-held-out", action="store_true",
                        help="Run inference on held-out queries")
    args = parser.parse_args()

    safe_name = args.dir_name or model_name_to_dir(args.model_name)
    model_dir = SCRIPT_DIR / "models" / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir = SCRIPT_DIR / "submissions" / safe_name

    corpus_emb_path = model_dir / "corpus_embeddings.npy"
    corpus_ids_path = model_dir / "corpus_ids.json"
    query_emb_path = model_dir / "query_embeddings.npy"
    query_ids_path = model_dir / "query_ids.json"

    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    if args.query_prefix:
        print(f"Query prefix: {args.query_prefix!r}")
    if args.corpus_prefix:
        print(f"Corpus prefix: {args.corpus_prefix!r}")

    # ── Corpus embeddings ──────────────────────────────────────
    if not args.force_encode and corpus_emb_path.exists():
        print("Loading cached corpus embeddings...")
        corpus_embs, corpus_ids = load_embeddings(corpus_emb_path, corpus_ids_path)
        print(f"  corpus: {corpus_embs.shape}")
    else:
        print("Loading corpus...")
        corpus = load_corpus(args.corpus)
        corpus_ids = corpus["doc_id"].tolist()
        corpus_texts = [format_text(row) for _, row in corpus.iterrows()]
        print(f"  {len(corpus)} docs")

        print(f"Loading model: {args.model_name} ...")
        model = SentenceTransformer(args.model_name, device=device)

        print("Encoding corpus...")
        corpus_embs = encode_texts(model, corpus_texts, args.corpus_prefix, args.batch_size)
        np.save(corpus_emb_path, corpus_embs)
        with open(corpus_ids_path, "w") as f:
            json.dump(corpus_ids, f)
        print(f"  corpus: {corpus_embs.shape} → saved")

    # ── Query embeddings ───────────────────────────────────────
    if args.submit_held_out:
        print("Loading held-out queries...")
        queries = load_queries(args.held_out)
        query_ids = queries["doc_id"].tolist()
        query_texts = [format_text(row) for _, row in queries.iterrows()]
        print(f"  {len(queries)} held-out queries")

        if "model" not in dir():
            print(f"Loading model: {args.model_name} ...")
            model = SentenceTransformer(args.model_name, device=device)
        print("Encoding held-out queries...")
        query_embs = encode_texts(model, query_texts, args.query_prefix, args.batch_size)
    else:
        queries = load_queries(args.queries)
        query_ids = queries["doc_id"].tolist()
        query_texts = [format_text(row) for _, row in queries.iterrows()]

        if not args.retrain and not args.force_encode and query_emb_path.exists():
            print("Loading cached query embeddings...")
            query_embs, query_ids = load_embeddings(query_emb_path, query_ids_path)
            print(f"  queries: {query_embs.shape}")
        else:
            if "model" not in dir():
                print(f"Loading model: {args.model_name} ...")
                model = SentenceTransformer(args.model_name, device=device)
            print("Encoding queries...")
            query_embs = encode_texts(model, query_texts, args.query_prefix, args.batch_size)
            np.save(query_emb_path, query_embs)
            with open(query_ids_path, "w") as f:
                json.dump(query_ids, f)
            print(f"  queries: {query_embs.shape} → saved")

    # ── Ranking ────────────────────────────────────────────────
    print("Ranking by dot product similarity...")
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :100]
    predictions = {
        qid: [corpus_ids[j] for j in top_indices[i]]
        for i, qid in enumerate(query_ids)
    }

    if not args.submit_held_out:
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        evaluate(predictions, qrels, ks=[10, 100], query_domains=query_domains)

    save_submission(predictions, output_dir)


if __name__ == "__main__":
    main()
