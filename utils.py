import json
import math
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


# Paths

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.parquet"
DEFAULT_CORPUS = DATA_DIR / "corpus.parquet"
DEFAULT_QRELS = DATA_DIR / "qrels.json"
DEFAULT_HELD_OUT = SCRIPT_DIR / "held_out_queries.parquet"


# Device selection

def get_device() -> str:
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


# Data loaders

def load_queries(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_corpus(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_qrels(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_embeddings(emb_path: str | Path,
                    ids_path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load pre-computed embeddings and their IDs. Returns (np.ndarray float32, list)."""
    embeddings = np.load(emb_path).astype(np.float32)
    with open(ids_path) as f:
        ids = json.load(f)
    assert len(embeddings) == len(ids), "Embedding count mismatch"
    return embeddings, ids


# Text formatting

def format_text(row: pd.Series) -> str:
    """Return title + abstract as a single string."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " " + abstract
    return title or abstract


# Chunk extraction

def get_chunks(full_text: str, chunk_meta_json: str | list) -> list:
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


def get_ta(row: pd.Series) -> str:
    """Return the pre-extracted title+abstract string from a paper row."""
    return str(row.get("ta", "") or "").strip()


def get_body_chunks(row: pd.Series, min_chars: int = 100) -> list:
    """Return all body section texts for a paper, filtering out very short sections."""
    chunks = get_chunks(row["full_text"], row["chunk_meta"])
    return [c["text"] for c in chunks if c["type"] == "body" and len(c["text"]) >= min_chars]


# Per-query metrics

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


# Aggregate evaluation

def evaluate(submission: dict, qrels: dict, ks: list | None = None,
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


def _print_results(results: dict, ks: list) -> None:
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


# Fusion

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


# Submission I/O

def save_submission(predictions: dict, output_path: str | Path) -> None:
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


# Tokenizer

_STOPWORDS = None


def get_stopwords() -> set[str]:
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


# Normalization

def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Min-max normalize each row (per-query) to [0, 1]."""
    mins = matrix.min(axis=1, keepdims=True)
    maxs = matrix.max(axis=1, keepdims=True)
    denom = np.where(maxs - mins < 1e-10, 1.0, maxs - mins)
    return np.where(maxs - mins < 1e-10, 0.0, (matrix - mins) / denom)


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Normalize a 1D array to [0, 1] using min-max scaling."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# BM25 index

def build_bm25(corpus_texts: list[str], model_dir: Path) -> Any:
    from rank_bm25 import BM25Okapi

    print("Tokenizing corpus...")
    tokenized = [tokenize(t) for t in tqdm(corpus_texts)]
    print("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized, k1=1.0, b=1.0)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved → {model_dir / 'index.pkl'}")
    return bm25


def load_bm25(model_dir: Path, verbose: bool = True) -> Any:
    with open(model_dir / "index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    if verbose:
        print(f"Loaded BM25 index from {model_dir / 'index.pkl'}")
    return bm25


# Score matrix loaders

def load_dense_scores(safe_name: str, model_name: str, query_prefix: str,
                      query_ids: list[str], query_texts: list[str], is_heldout: bool,
                      device: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model_dir = SCRIPT_DIR / "models" / safe_name
    corpus_embs, _ = load_embeddings(
        model_dir / "corpus_embeddings.npy", model_dir / "corpus_ids.json"
    )
    q_emb_path = model_dir / "query_embeddings.npy"
    q_ids_path = model_dir / "query_ids.json"

    if not is_heldout and q_emb_path.exists():
        q_embs, _ = load_embeddings(q_emb_path, q_ids_path)
    else:
        print(f"    Encoding queries with {model_name}...")
        model = SentenceTransformer(model_name, device=device)
        texts = [query_prefix + t for t in query_texts] if query_prefix else query_texts
        q_embs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        del model
        if not is_heldout:
            np.save(q_emb_path, q_embs)
            with open(q_ids_path, "w") as f:
                json.dump(query_ids, f)
    return q_embs @ corpus_embs.T


def load_tfidf_scores(query_texts: list[str], corpus_texts: list[str],
                      is_heldout: bool) -> np.ndarray:
    tfidf_dir = SCRIPT_DIR / "models" / "tfidf"
    cache_path = tfidf_dir / ("heldout_scores.npy" if is_heldout else "train_scores.npy")
    vect_path = tfidf_dir / "vectorizer.pkl"

    if not is_heldout and cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    from sklearn.feature_extraction.text import TfidfVectorizer
    if vect_path.exists():
        with open(vect_path, "rb") as f:
            vect = pickle.load(f)
        corpus_vecs = vect.transform(corpus_texts)
    else:
        vect = TfidfVectorizer(max_features=100_000, sublinear_tf=True)
        corpus_vecs = vect.fit_transform(corpus_texts)
        if not is_heldout:
            tfidf_dir.mkdir(parents=True, exist_ok=True)
            with open(vect_path, "wb") as f:
                pickle.dump(vect, f)
    query_vecs = vect.transform(query_texts)
    matrix = (query_vecs @ corpus_vecs.T).toarray().astype(np.float32)
    if not is_heldout:
        np.save(cache_path, matrix)
    return matrix


def load_dense_sim(safe_name: str, model_name: str, query_prefix: str,
                   query_ids: list[str], query_texts: list[str], is_heldout: bool,
                   device: str, batch_size: int) -> np.ndarray | None:
    """Load or compute dense cosine similarity matrix (n_q, n_docs)."""
    from sentence_transformers import SentenceTransformer
    model_dir = SCRIPT_DIR / "models" / safe_name
    corpus_emb_path = model_dir / "corpus_embeddings.npy"
    corpus_ids_path = model_dir / "corpus_ids.json"

    if not corpus_emb_path.exists():
        print(f"    [SKIP] {safe_name}: no corpus embeddings found")
        return None

    corpus_embs, _ = load_embeddings(corpus_emb_path, corpus_ids_path)
    q_emb_path = model_dir / "query_embeddings.npy"
    q_ids_path = model_dir / "query_ids.json"

    if not is_heldout and q_emb_path.exists():
        q_embs, _ = load_embeddings(q_emb_path, q_ids_path)
    else:
        print(f"    Encoding queries with {model_name}...")
        model = SentenceTransformer(model_name, device=device)
        texts = [query_prefix + t for t in query_texts] if query_prefix else query_texts
        q_embs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        del model
        if not is_heldout:
            np.save(q_emb_path, q_embs)
            with open(q_ids_path, "w") as f:
                json.dump(query_ids, f)
    return q_embs @ corpus_embs.T


def load_bm25_ta_scores(query_texts: list[str], is_heldout: bool) -> np.ndarray | None:
    """Load BM25 TA scores."""
    bm25_dir = SCRIPT_DIR / "models" / "bm25"
    cache_path = bm25_dir / ("heldout_scores.npy" if is_heldout else "train_scores.npy")
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    index_path = bm25_dir / "index.pkl"
    if not index_path.exists():
        print("    [SKIP] BM25 TA: no index found")
        return None
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)
    n_docs = bm25.corpus_size
    matrix = np.zeros((len(query_texts), n_docs), dtype=np.float32)
    for i, qt in enumerate(tqdm(query_texts, desc="BM25 TA", leave=False)):
        matrix[i] = np.array(bm25.get_scores(tokenize(qt)), dtype=np.float32)
    if not is_heldout:
        np.save(cache_path, matrix)
    return matrix


def load_bm25_fulltext_scores(is_heldout: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load full-text BM25 scores (title + abstract + body)."""
    ft_dir = SCRIPT_DIR / "models" / "bm25_fulltext"
    suffix = "_heldout" if is_heldout else "_train"

    cite_path = ft_dir / f"cite_ctx_scores{suffix}.npy"
    ta_ft_path = ft_dir / f"ta_fulltext_scores{suffix}.npy"

    cite_scores = None
    ta_ft_scores = None

    if cite_path.exists():
        cite_scores = np.load(cite_path).astype(np.float32)
    else:
        print("    [SKIP] Citation-context BM25: run 07_citation_context.py first")

    if ta_ft_path.exists():
        ta_ft_scores = np.load(ta_ft_path).astype(np.float32)
    else:
        print("    [SKIP] TA full-text BM25: run 07_citation_context.py first")

    return cite_scores, ta_ft_scores


# Base fusion

FUSION_MODELS = {
    "uae": {
        "safe_name": "WhereIsAI_UAE-Large-V1",
        "model_name": "WhereIsAI/UAE-Large-V1",
        "query_prefix": "",
        "weight": 0.60,
    },
    "bge": {
        "safe_name": "bge",
        "model_name": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "weight": 0.10,
    },
    "e5": {
        "safe_name": "intfloat_e5-large-v2",
        "model_name": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "weight": 0.10,
    },
}
TFIDF_WEIGHT = 0.20


def compute_base_fusion(query_ids: list[str], query_texts: list[str],
                        corpus_texts: list[str], is_heldout: bool, device: str,
                        batch_size: int) -> np.ndarray:
    """Reproduce the 0.57 base fusion (UAE+BGE+E5+TFIDF)."""
    fused = None
    for key, cfg in FUSION_MODELS.items():
        print(f"  [{key}] loading scores...")
        raw = load_dense_scores(
            cfg["safe_name"], cfg["model_name"], cfg["query_prefix"],
            query_ids, query_texts, is_heldout, device, batch_size,
        )
        normed = normalize_rows(raw) * cfg["weight"]
        fused = normed if fused is None else fused + normed
    print("  [tfidf] loading scores...")
    tfidf_raw = load_tfidf_scores(query_texts, corpus_texts, is_heldout)
    fused += TFIDF_WEIGHT * normalize_rows(tfidf_raw)
    return fused


# Citation context extraction

CITE_PATTERNS = [
    re.compile(r'\[[\d,;\s\-]+\]'),                                      # [1], [1,2], [1-3]
    re.compile(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?\)'),   # (Author et al., 2020)
    re.compile(r'\([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,?\s*\d{4}\)'),       # (Smith and Jones, 2020)
    re.compile(r'\([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,?\s*\d{4}\)'),         # (Smith & Jones, 2020)
]


def extract_citation_sentences(full_text: str) -> str:
    """Extract sentences containing citation markers from full_text."""
    if not full_text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    cite_sents = []
    for sent in sentences:
        if any(p.search(sent) for p in CITE_PATTERNS):
            # Strip the citation markers themselves to keep content words
            cleaned = sent
            for p in CITE_PATTERNS:
                cleaned = p.sub('', cleaned)
            cleaned = cleaned.strip()
            if len(cleaned) > 20:  # skip near-empty sentences
                cite_sents.append(cleaned)
    return " ".join(cite_sents)


# Learning to rank

# Model directories for cached embeddings

DENSE_MODELS = {
    "uae": {
        "safe_name": "WhereIsAI_UAE-Large-V1",
        "model_name": "WhereIsAI/UAE-Large-V1",
        "query_prefix": "",
    },
    "bge": {
        "safe_name": "bge",
        "model_name": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "e5": {
        "safe_name": "intfloat_e5-large-v2",
        "model_name": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
    },
    "scincl": {
        "safe_name": "specter2",
        "model_name": "malteos/scincl",
        "query_prefix": "",
    },
}
TOP_K_CANDIDATES = 200  # union of top-K from each system


def fill_labels(labels: np.ndarray, pair_info: list, query_ids: list, corpus_ids: list,
                qrels: dict) -> np.ndarray:
    """Fill binary labels from qrels."""
    for idx, (qi, di) in enumerate(pair_info):
        qid = query_ids[qi]
        doc_id = corpus_ids[di]
        if doc_id in set(qrels.get(qid, [])):
            labels[idx] = 1.0
    return labels


def predict_rankings(model: Any, features: np.ndarray, pair_info: list[tuple[int, int]],
                     query_ids: list[str], corpus_ids: list[str],
                     groups: list[int]) -> dict[str, list[str]]:
    """Run XGBRanker prediction, return {qid: [top-100 doc_ids]}."""
    scores = model.predict(features)
    predictions = {}
    offset = 0
    for qi, g in enumerate(groups):
        qid = query_ids[qi]
        group_scores = scores[offset:offset + g]
        group_pairs = pair_info[offset:offset + g]
        sorted_local = np.argsort(-group_scores)
        ranked_ids = [corpus_ids[group_pairs[j][1]] for j in sorted_local[:100]]
        predictions[qid] = ranked_ids
        offset += g
    return predictions
