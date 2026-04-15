import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ── Device selection ─────────────────────────────────────────

def get_device():
    """Return the best available device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ── Data loaders ─────────────────────────────────────────────

def load_queries(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_corpus(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_qrels(path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_embeddings(emb_path, ids_path):
    """Load pre-computed embeddings and their IDs. Returns (np.ndarray float32, list)."""
    embeddings = np.load(emb_path).astype(np.float32)
    with open(ids_path) as f:
        ids = json.load(f)
    assert len(embeddings) == len(ids), "Embedding count mismatch"
    return embeddings, ids


# ── Text formatting ───────────────────────────────────────────

def format_text(row) -> str:
    """Return title + abstract as a single string."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " " + abstract
    return title or abstract


# ── Chunk extraction ──────────────────────────────────────────

def get_chunks(full_text: str, chunk_meta_json) -> list:
    """
    Extract all text sections from a paper using pre-computed chunk metadata.
    Returns list of dicts: [{"type": "ta"|"body", "text": str, "char_start": int, "char_end": int}]
    """
    meta = json.loads(chunk_meta_json) if isinstance(chunk_meta_json, str) else chunk_meta_json
    chunks = []
    for i, entry in enumerate(meta):
        char_start = entry["char_start"]
        if entry["type"] == "ta":
            char_end = entry["char_end"]
        else:
            char_end = meta[i + 1]["char_start"] if i + 1 < len(meta) else len(full_text)
        text = full_text[char_start:char_end].strip()
        chunks.append({"type": entry["type"], "text": text,
                       "char_start": char_start, "char_end": char_end})
    return chunks


def get_ta(row) -> str:
    """Return the pre-extracted title+abstract string from a paper row."""
    return str(row.get("ta", "") or "").strip()


def get_body_chunks(row, min_chars: int = 100) -> list:
    """Return all body section texts for a paper, filtering out very short sections."""
    chunks = get_chunks(row["full_text"], row["chunk_meta"])
    return [c["text"] for c in chunks if c["type"] == "body" and len(c["text"]) >= min_chars]


# ── Per-query metrics ─────────────────────────────────────────

def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc in ranked[:k] if doc in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for doc in ranked[:k] if doc in relevant)
    return hits / k


def mrr_at_k(ranked: list, relevant: set, k: int) -> float:
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(ranked[:k], start=1)
        if doc in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked: list, relevant: set) -> float:
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for rank, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)


# ── Aggregate evaluation ──────────────────────────────────────

def evaluate(submission: dict, qrels: dict, ks: list = None,
             query_domains: dict = None, verbose: bool = True) -> dict:
    """
    Evaluate a retrieval submission against ground-truth qrels.
    submission: {query_id: [top-100 doc_ids]}
    qrels:      {query_id: [relevant doc_ids]}
    """
    if ks is None:
        ks = [10, 100]

    per_query = {}
    for qid, rel_list in qrels.items():
        relevant = set(rel_list)
        ranked = submission.get(qid, [])
        q = {}
        for k in ks:
            q[f"Recall@{k}"] = recall_at_k(ranked, relevant, k)
            q[f"Precision@{k}"] = precision_at_k(ranked, relevant, k)
            q[f"MRR@{k}"] = mrr_at_k(ranked, relevant, k)
            q[f"NDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
        q["AP"] = average_precision(ranked, relevant)
        per_query[qid] = q

    metric_keys = list(next(iter(per_query.values())).keys()) if per_query else []
    overall = {}
    for key in metric_keys:
        vals = [per_query[qid][key] for qid in per_query]
        overall[key] = float(np.mean(vals))
    overall["MAP"] = overall.pop("AP", 0.0)
    overall["num_queries"] = len(per_query)

    result = {"overall": overall, "per_query": per_query}

    if query_domains:
        per_domain = {}
        for domain in sorted(set(query_domains.values())):
            dqids = [q for q in per_query if query_domains.get(q) == domain]
            if not dqids:
                continue
            dm = {}
            for key in metric_keys:
                dm[key] = float(np.mean([per_query[q][key] for q in dqids]))
            dm["MAP"] = dm.pop("AP", 0.0)
            dm["num_queries"] = len(dqids)
            per_domain[domain] = dm
        result["per_domain"] = per_domain

    if verbose:
        _print_results(result, ks)

    return result


def _print_results(results: dict, ks: list):
    o = results["overall"]
    print("\n" + "=" * 68)
    print("OVERALL RESULTS")
    print("=" * 68)
    for label, keys in [
        ("Recall",    [f"Recall@{k}"    for k in ks]),
        ("Precision", [f"Precision@{k}" for k in ks]),
        ("MRR",       [f"MRR@{k}"       for k in ks]),
        ("NDCG",      [f"NDCG@{k}"      for k in ks]),
    ]:
        row = f"{label:<14}"
        for k, key in zip(ks, keys):
            row += f"  @{k:>3}: {o.get(key, 0):.4f}"
        print(row)
    print(f"{'MAP':<14}  {o.get('MAP', 0):.4f}")
    print(f"{'Queries':<14}  {int(o.get('num_queries', 0))}")

    if "per_domain" in results:
        print("\n" + "-" * 68)
        print("PER-DOMAIN  (first k only)")
        print("-" * 68)
        k = ks[0]
        print(f"  {'Domain':<28} R@{k:<3} P@{k:<3} MRR@{k:<3} NDCG@{k:<3}  MAP    n")
        for domain, dm in sorted(results["per_domain"].items()):
            print(
                f"  {domain:<28}"
                f" {dm.get(f'Recall@{k}', 0):.3f}"
                f" {dm.get(f'Precision@{k}', 0):.3f}"
                f" {dm.get(f'MRR@{k}', 0):.3f}  "
                f" {dm.get(f'NDCG@{k}', 0):.3f}"
                f"  {dm.get('MAP', 0):.3f}"
                f"  {int(dm.get('num_queries', 0))}"
            )
    print()


# ── Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(rankings: list, k: int = 60) -> dict:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    rankings: list of lists of doc_ids (each list is one ranked result)
    k: RRF smoothing constant (standard value is 60)
    Returns dict {doc_id: rrf_score} — higher is better.
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


# ── Submission I/O ────────────────────────────────────────────

def save_submission(predictions: dict, output_path: str):
    """
    Save predictions as submission_data.json and zip it.
    output_path: path without extension, e.g. 'submissions/tfidf'
    Writes: <output_path>/submission_data.json and <output_path>/submission.zip
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "submission_data.json"
    zip_path = out_dir / "submission.zip"

    with open(json_path, "w") as f:
        json.dump(predictions, f)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, "submission_data.json")

    print(f"Saved → {json_path}")
    print(f"Saved → {zip_path}")
