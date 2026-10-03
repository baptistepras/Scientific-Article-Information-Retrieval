"""
Exhaustive multi-model score fusion with grid-searched weights.

Tests every subset of 2..N active models and every weight assignment
(each weight ≥ 0.10, step 0.10, sum = 1.0). Reports the globally best MAP
and prints the exact command to submit held-out queries.

Active model pool (uncomment gte/stella after encoding their corpus with 04_dense_encoders.py):
  uae    — WhereIsAI/UAE-Large-V1        (MAP 0.57)
  bge    — BAAI/bge-large-en-v1.5        (MAP 0.57 with BM25)
  bm25   — BM25 Okapi
  tfidf  — TF-IDF cosine
  e5     — intfloat/e5-large-v2          (MAP 0.46)
  # gte    — Alibaba-NLP/gte-large-en-v1.5
  # stella — dunzhang/stella_en_1.5B_v5

Training run (no arguments):
  python3 06_multi_fusion.py

Held-out submission (use command printed at end of training run):
  python3 06_multi_fusion.py --submit-held-out --models uae bge bm25 --weights 0.40 0.50 0.10
"""

import argparse
import itertools
import json
import pickle
from pathlib import Path
from collections.abc import Iterator

import numpy as np
from tqdm import tqdm

from utils import (DEFAULT_CORPUS, DEFAULT_HELD_OUT, DEFAULT_QRELS,
                   DEFAULT_QUERIES, SCRIPT_DIR, evaluate, format_text,
                   get_device, load_corpus, load_embeddings, load_qrels,
                   load_queries, normalize_rows, save_submission, tokenize)

DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "combined_v2"
DEFAULT_BATCH_SIZE = 64

# Model pool
# type="dense": needs model_name, safe_name (dir under models/), query_prefix,
#               trust_remote_code (for special models like stella).
# type="bm25" / "tfidf": no extra fields.
# active=True: included in the grid search.
# Activate gte/stella by uncommenting after running 04_dense_encoders.py for them.
MODELS = {
    "uae": {
        "type": "dense",
        "model_name": "WhereIsAI/UAE-Large-V1",
        "safe_name": "WhereIsAI_UAE-Large-V1",
        "query_prefix": "",
        "trust_remote_code": False,
        "active": True,
    },
    "bge": {
        "type": "dense",
        "model_name": "BAAI/bge-large-en-v1.5",
        "safe_name": "bge",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "trust_remote_code": False,
        "active": True,
    },
    "e5": {
        "type": "dense",
        "model_name": "intfloat/e5-large-v2",
        "safe_name": "intfloat_e5-large-v2",
        "query_prefix": "query: ",
        "trust_remote_code": False,
        "active": True,
    },
    "bm25": {"type": "bm25", "active": True},
    "tfidf": {"type": "tfidf", "active": True},
    # Uncomment after: python3 04_dense_encoders.py --model-name Alibaba-NLP/gte-large-en-v1.5
    # "gte": {
    #     "type": "dense",
    #     "model_name": "Alibaba-NLP/gte-large-en-v1.5",
    #     "safe_name": "Alibaba-NLP_gte-large-en-v1.5",
    #     "query_prefix": "",
    #     "trust_remote_code": False,
    #     "active": True,
    # },
    # Uncomment after: python3 04_dense_encoders.py --model-name dunzhang/stella_en_1.5B_v5
    # Note: stella requires trust_remote_code=True. Also add it to 04_dense_encoders.py:
    #   model = SentenceTransformer(args.model_name, device=device, trust_remote_code=True)
    # "stella": {
    #     "type": "dense",
    #     "model_name": "dunzhang/stella_en_1.5B_v5",
    #     "safe_name": "dunzhang_stella_en_1.5B_v5",
    #     "query_prefix": "Instruct: Retrieve relevant scientific papers.\nQuery: ",
    #     "trust_remote_code": True,
    #     "active": True,
    # },
}

# Helpers

def simplex_grid(n: int, total: int = 10, min_val: int = 1) -> Iterator[tuple[int, ...]]:
    """Yield all integer tuples of length n summing to total, each ≥ min_val."""
    if n == 1:
        if total >= min_val:
            yield (total,)
        return
    for first in range(min_val, total - (n - 1) * min_val + 1):
        for rest in simplex_grid(n - 1, total - first, min_val):
            yield (first,) + rest


def weight_combos(n: int, steps: int = 10) -> Iterator[tuple[float, ...]]:
    """Yield weight tuples of length n summing to 1.0, step=1/steps, each ≥ 1/steps."""
    for ints in simplex_grid(n, steps, min_val=1):
        yield tuple(i / steps for i in ints)


# Score matrix loaders

def load_dense_scores(key: str, cfg: dict, query_ids: list, query_texts: list,
                      is_heldout: bool, device: str, batch_size: int) -> np.ndarray:
    """Load corpus embeddings + encode queries → dot-product matrix (n_q, n_docs)."""
    from sentence_transformers import SentenceTransformer

    model_dir = SCRIPT_DIR / "models" / cfg["safe_name"]
    corpus_emb_path = model_dir / "corpus_embeddings.npy"

    if not corpus_emb_path.exists():
        raise FileNotFoundError(
            f"[{key}] Corpus embeddings not found at {corpus_emb_path}.\n"
            f"  Run: python3 04_dense_encoders.py --model-name {cfg['model_name']}"
        )
    corpus_embs, _ = load_embeddings(corpus_emb_path, model_dir / "corpus_ids.json")

    q_emb_path = model_dir / "query_embeddings.npy"
    q_ids_path = model_dir / "query_ids.json"

    if not is_heldout and q_emb_path.exists():
        q_embs, _ = load_embeddings(q_emb_path, q_ids_path)
    else:
        print(f"  [{key}] Encoding {len(query_texts)} queries with {cfg['model_name']}...")
        model = SentenceTransformer(
            cfg["model_name"], device=device,
            trust_remote_code=cfg.get("trust_remote_code", False),
        )
        prefix = cfg.get("query_prefix", "")
        texts = [prefix + t for t in query_texts] if prefix else query_texts
        q_embs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        del model
        if not is_heldout:
            np.save(q_emb_path, q_embs)
            with open(q_ids_path, "w") as f:
                json.dump(query_ids, f)
            print(f"  [{key}] Saved query embeddings → {q_emb_path}")

    return q_embs @ corpus_embs.T   # (n_q, n_docs)


def load_bm25_scores(query_texts: list, corpus_texts: list,
                     is_heldout: bool) -> np.ndarray:
    bm25_dir = SCRIPT_DIR / "models" / "bm25"
    cache_path = bm25_dir / ("heldout_scores.npy" if is_heldout else "train_scores.npy")

    if not is_heldout and cache_path.exists():
        print("  [bm25] Loading cached training scores...")
        return np.load(cache_path).astype(np.float32)

    index_path = bm25_dir / "index.pkl"
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {index_path}.")
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)

    print(f"  [bm25] Scoring {len(query_texts)} queries against {len(corpus_texts)} docs...")
    matrix = np.zeros((len(query_texts), len(corpus_texts)), dtype=np.float32)
    for i, qt in enumerate(tqdm(query_texts, desc="BM25", leave=False)):
        matrix[i] = np.array(bm25.get_scores(tokenize(qt)), dtype=np.float32)

    if not is_heldout:
        bm25_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
        print(f"  [bm25] Saved training scores → {cache_path}")
    return matrix


def load_tfidf_scores(query_texts: list, corpus_texts: list,
                      is_heldout: bool) -> np.ndarray:
    tfidf_dir = SCRIPT_DIR / "models" / "tfidf"
    cache_path = tfidf_dir / ("heldout_scores.npy" if is_heldout else "train_scores.npy")
    vect_path = tfidf_dir / "vectorizer.pkl"

    if not is_heldout and cache_path.exists():
        print("  [tfidf] Loading cached training scores...")
        return np.load(cache_path).astype(np.float32)

    from sklearn.feature_extraction.text import TfidfVectorizer

    if vect_path.exists() and not is_heldout:
        print("  [tfidf] Loading cached vectorizer...")
        with open(vect_path, "rb") as f:
            vect = pickle.load(f)
        corpus_vecs = vect.transform(corpus_texts)
    else:
        print(f"  [tfidf] Fitting TF-IDF on {len(corpus_texts)} corpus docs...")
        vect = TfidfVectorizer(max_features=100_000, sublinear_tf=True)
        corpus_vecs = vect.fit_transform(corpus_texts)
        if not is_heldout:
            tfidf_dir.mkdir(parents=True, exist_ok=True)
            with open(vect_path, "wb") as f:
                pickle.dump(vect, f)

    print(f"  [tfidf] Scoring {len(query_texts)} queries...")
    query_vecs = vect.transform(query_texts)
    matrix = (query_vecs @ corpus_vecs.T).toarray().astype(np.float32)

    if not is_heldout:
        np.save(cache_path, matrix)
        print(f"  [tfidf] Saved training scores → {cache_path}")
    return matrix


def get_scores(key: str, query_ids: list, query_texts: list, corpus_texts: list,
               is_heldout: bool, device: str, batch_size: int) -> np.ndarray:
    """Return normalized (n_q, n_docs) score matrix for a model."""
    cfg = MODELS[key]
    if cfg["type"] == "dense":
        raw = load_dense_scores(key, cfg, query_ids, query_texts,
                                is_heldout, device, batch_size)
    elif cfg["type"] == "bm25":
        raw = load_bm25_scores(query_texts, corpus_texts, is_heldout)
    elif cfg["type"] == "tfidf":
        raw = load_tfidf_scores(query_texts, corpus_texts, is_heldout)
    else:
        raise ValueError(f"Unknown model type: {cfg['type']}")
    return normalize_rows(raw)


def fuse_and_rank(score_mats: list, weights: tuple, query_ids: list,
                  corpus_ids: list) -> dict:
    """Weighted sum of normalized matrices → top-100 predictions."""
    fused = sum(w * m for w, m in zip(weights, score_mats))
    top_idx = np.argsort(-fused, axis=1)[:, :100]
    return {qid: [corpus_ids[j] for j in top_idx[i]] for i, qid in enumerate(query_ids)}


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaustive multi-model score fusion — grid search or held-out submission"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--submit-held-out", action="store_true",
                        help="Submit held-out queries using --models and --weights")
    parser.add_argument("--models", nargs="+",
                        help="Model keys to fuse (e.g. uae bge bm25)")
    parser.add_argument("--weights", nargs="+", type=float,
                        help="Fusion weights matching --models order, must sum to 1")
    args = parser.parse_args()

    if args.submit_held_out:
        if not args.models or not args.weights:
            parser.error("--submit-held-out requires --models and --weights")
        if len(args.models) != len(args.weights):
            parser.error("--models and --weights must have the same length")
        if abs(sum(args.weights) - 1.0) > 1e-4:
            parser.error(f"--weights must sum to 1.0 (got {sum(args.weights):.4f})")
        for key in args.models:
            if key not in MODELS:
                parser.error(f"Unknown model: {key!r}. Available: {list(MODELS)}")

    device = get_device()
    print(f"Using device: {device}")

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    corpus_ids = corpus["doc_id"].tolist()
    corpus_texts = [format_text(row) for _, row in corpus.iterrows()]
    print(f"  {len(corpus)} docs")

    # HELD-OUT SUBMISSION
    if args.submit_held_out:
        queries = load_queries(args.held_out)
        query_ids = queries["doc_id"].tolist()
        query_texts = [format_text(row) for _, row in queries.iterrows()]
        print(f"  {len(queries)} held-out queries")

        score_mats = []
        for key in args.models:
            print(f"\nLoading scores for [{key}]...")
            score_mats.append(get_scores(key, query_ids, query_texts, corpus_texts,
                                         is_heldout=True, device=device,
                                         batch_size=args.batch_size))

        predictions = fuse_and_rank(score_mats, args.weights, query_ids, corpus_ids)
        save_submission(predictions, args.output)
        return

    # TRAINING GRID SEARCH
    queries = load_queries(args.queries)
    query_ids = queries["doc_id"].tolist()
    query_texts = [format_text(row) for _, row in queries.iterrows()]
    print(f"  {len(queries)} training queries")

    qrels = load_qrels(args.qrels)
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    active_keys = [k for k, cfg in MODELS.items() if cfg.get("active", True)]
    n_active = len(active_keys)
    print(f"\nActive models ({n_active}): {active_keys}")

    # Pre-load all score matrices
    print("\n── Loading score matrices ─────────────────────────────────────────")
    all_scores: dict[str, np.ndarray] = {}
    skipped = []
    for key in active_keys:
        try:
            print(f"\n[{key}]")
            all_scores[key] = get_scores(key, query_ids, query_texts, corpus_texts,
                                          is_heldout=False, device=device,
                                          batch_size=args.batch_size)
            print(f"  [{key}] OK — shape {all_scores[key].shape}")
        except FileNotFoundError as e:
            print(f"  [{key}] SKIPPED: {e}")
            skipped.append(key)

    available_keys = [k for k in active_keys if k not in skipped]
    n = len(available_keys)
    if n < 2:
        print("Not enough models available for fusion (need ≥ 2). Exiting.")
        return
    if skipped:
        print(f"\nSkipped models (missing embeddings): {skipped}")
    print(f"\nRunning grid search over {n} models: {available_keys}")

    # Count total evaluations for reporting
    total_evals = sum(
        sum(1 for _ in itertools.combinations(available_keys, size))
        * sum(1 for _ in weight_combos(size))
        for size in range(2, n + 1)
    )
    print(f"Total combinations to evaluate: {total_evals}\n")

    best_map = 0.0
    best_combo: tuple = ()
    best_weights: tuple = ()
    n_evals = 0

    for size in range(2, n + 1):
        combos = list(itertools.combinations(available_keys, size))
        w_list = list(weight_combos(size))
        print(f"Size {size}: {len(combos)} model combos × {len(w_list)} weight combos"
              f" = {len(combos) * len(w_list)} evals")

        for combo in combos:
            mats = [all_scores[k] for k in combo]
            for weights in w_list:
                preds = fuse_and_rank(mats, weights, query_ids, corpus_ids)
                result = evaluate(preds, qrels, ks=[10, 100], verbose=False)
                map_score = result["overall"]["MAP"]
                n_evals += 1
                if map_score > best_map:
                    best_map = map_score
                    best_combo = combo
                    best_weights = weights

    # Final report
    print(f"\n{'=' * 60}")
    print(f"Grid search complete — {n_evals} combinations evaluated")
    print(f"{'=' * 60}")
    print(f"Best MAP:     {best_map:.4f}")
    print(f"Best models:  {list(best_combo)}")
    w_str = "  ".join(f"{k}={w:.2f}" for k, w in zip(best_combo, best_weights))
    print(f"Best weights: {w_str}")

    # Full detailed eval with best configuration
    print("\nFull evaluation with best configuration:")
    best_preds = fuse_and_rank(
        [all_scores[k] for k in best_combo], best_weights, query_ids, corpus_ids
    )
    evaluate(best_preds, qrels, ks=[10, 100], query_domains=query_domains)
    save_submission(best_preds, args.output)

    # Print the exact held-out command
    models_arg = " ".join(best_combo)
    weights_arg = " ".join(f"{w:.2f}" for w in best_weights)
    print(f"\n{'=' * 60}")
    print("Command to submit held-out queries:")
    print(f"{'=' * 60}")
    print(f"python3 06_multi_fusion.py --submit-held-out \\")
    print(f"    --models {models_arg} \\")
    print(f"    --weights {weights_arg}")


if __name__ == "__main__":
    main()
