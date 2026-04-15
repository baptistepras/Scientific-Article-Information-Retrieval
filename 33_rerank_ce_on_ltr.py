"""
Cross-encoder reranker on top of LTR+CE predictions.

Takes the top-K candidates from the LTR ranking and rescores them with a
strong cross-encoder (BGE-reranker-v2-m3 by default). Final score is an
interpolation between the reranker score and the LTR rank position.

Usage:
  python3 33_rerank_ce_on_ltr.py
  python3 33_rerank_ce_on_ltr.py --rerank-top 50 --gamma 0.7
  python3 33_rerank_ce_on_ltr.py --submit-held-out
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (evaluate, format_text, get_device, load_corpus,
                   load_qrels, load_queries, save_submission)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.parquet"
DEFAULT_CORPUS = DATA_DIR / "corpus.parquet"
DEFAULT_QRELS = DATA_DIR / "qrels.json"
DEFAULT_HELD_OUT = SCRIPT_DIR / "held_out_queries.parquet"

DEFAULT_LTR_SUB = SCRIPT_DIR / "submissions" / "ltr_ce" / "submission_data.json"
DEFAULT_LTR_HELDOUT_SUB = SCRIPT_DIR / "submissions" / "ltr_ce" / "submission_data.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "rerank_ce_on_ltr"

DEFAULT_CE_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BATCH_SIZE = 32
DEFAULT_RERANK_TOP = 50
DEFAULT_GAMMA = 0.7
DEFAULT_MAX_LENGTH = 512


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def rerank_query(ce_model, query_text, candidate_texts, batch_size, max_length):
    pairs = [[query_text, t] for t in candidate_texts]
    scores = ce_model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(scores, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-encoder rerank of LTR+CE predictions"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--predictions", default=DEFAULT_LTR_SUB, type=Path,
                        help="Path to script 24's submission_data.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--ce-model", default=DEFAULT_CE_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP,
                        help="Number of top LTR candidates to rerank")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help="Weight on CE score vs LTR rank (0 = keep LTR, 1 = pure CE)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--submit-held-out", action="store_true")
    args = parser.parse_args()

    device = get_device()
    is_heldout = args.submit_held_out
    print(f"Device: {device}")
    print(f"CE model: {args.ce_model}")
    print(f"Rerank top-{args.rerank_top}, gamma={args.gamma}")

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"ERROR: LTR predictions not found at {pred_path}")
        print("Run script 24 first (with --submit-held-out if needed).")
        return
    with open(pred_path) as f:
        ltr_preds = json.load(f)
    print(f"Loaded {len(ltr_preds)} LTR rankings from {pred_path}")

    print("Loading corpus / queries...")
    corpus = load_corpus(args.corpus)
    corpus_texts = {row["doc_id"]: format_text(row) for _, row in corpus.iterrows()}

    queries = load_queries(args.held_out if is_heldout else args.queries)
    query_ids = queries["doc_id"].tolist()
    query_texts = {row["doc_id"]: format_text(row) for _, row in queries.iterrows()}

    missing = [q for q in query_ids if q not in ltr_preds]
    if missing:
        print(f"WARNING: {len(missing)} queries missing from LTR predictions "
              f"(they will be skipped/kept empty).")

    from sentence_transformers import CrossEncoder
    print(f"Loading cross-encoder on {device}...")
    ce = CrossEncoder(args.ce_model, max_length=args.max_length, device=device)

    predictions = {}
    for qid in tqdm(query_ids, desc="Reranking"):
        ltr_ranked = ltr_preds.get(qid, [])
        if not ltr_ranked:
            predictions[qid] = []
            continue

        top_candidates = ltr_ranked[: args.rerank_top]
        tail = ltr_ranked[args.rerank_top:]

        cand_texts = [corpus_texts.get(d, "") for d in top_candidates]
        q_text = query_texts.get(qid, "")

        ce_scores = rerank_query(ce, q_text, cand_texts, args.batch_size,
                                 args.max_length)

        ltr_rank_score = np.array(
            [1.0 / (60 + r + 1) for r in range(len(top_candidates))],
            dtype=np.float32,
        )

        ce_norm = normalize_minmax(ce_scores)
        ltr_norm = normalize_minmax(ltr_rank_score)
        final = args.gamma * ce_norm + (1.0 - args.gamma) * ltr_norm

        order = np.argsort(-final)
        reranked_top = [top_candidates[i] for i in order]

        merged = reranked_top + tail
        seen = set()
        deduped = []
        for d in merged:
            if d not in seen:
                deduped.append(d)
                seen.add(d)
        predictions[qid] = deduped[:100]

    if not is_heldout:
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        evaluate(predictions, qrels, ks=[10, 100], query_domains=query_domains)

    save_submission(predictions, args.output)


if __name__ == "__main__":
    main()
