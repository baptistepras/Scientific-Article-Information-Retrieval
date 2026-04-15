"""
Citation context mining + full-text BM25.

Key insight: query full_text contains ~50 sentences with citation markers
([1], (Author et al., 2020)) that directly describe cited papers. This signal
is completely unused by all existing models (which only encode title+abstract).

Pipeline:
  1. Build a full-text BM25 index on corpus (title + abstract + first 5000 chars
     of body) instead of the current TA-only index.
  2. Extract citation-context sentences from query full_text.
  3. Score each query's citation-context text against the full-text corpus index.
  4. Also score the standard TA query against the full-text index.
  5. Fuse these 2 new signals with the existing 0.57 base via grid search.

Usage:
  python3 21_cite_context_bm25.py                        # grid search
  python3 21_cite_context_bm25.py --retrain               # rebuild BM25 index
  python3 21_cite_context_bm25.py --submit-held-out --alpha-cite 0.10 --alpha-ft 0.10
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from utils import (evaluate, format_text, get_body_chunks, get_device,
                   load_corpus, load_embeddings, load_qrels, load_queries,
                   save_submission)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.parquet"
DEFAULT_CORPUS = DATA_DIR / "corpus.parquet"
DEFAULT_QRELS = DATA_DIR / "qrels.json"
DEFAULT_HELD_OUT = SCRIPT_DIR / "held_out_queries.parquet"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models" / "bm25_fulltext"
DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "cite_context_bm25"
DEFAULT_BATCH_SIZE = 64

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

# ── Citation context extraction ────────────────────────────────────────────

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


# ── Tokenizer ──────────────────────────────────────────────────────────────

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
    return [t for t in text.split() if t not in get_stopwords()]


# ── Normalization ──────────────────────────────────────────────────────────

def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    mins = matrix.min(axis=1, keepdims=True)
    maxs = matrix.max(axis=1, keepdims=True)
    denom = np.where(maxs - mins < 1e-10, 1.0, maxs - mins)
    return np.where(maxs - mins < 1e-10, 0.0, (matrix - mins) / denom)


# ── Full-text corpus text ─────────────────────────────────────────────────

def format_fulltext(row, max_body_chars: int = 5000) -> str:
    """Title + abstract + first 5000 chars of body text."""
    ta = format_text(row)
    try:
        body_chunks = get_body_chunks(row, min_chars=50)
        if body_chunks:
            body = " ".join(body_chunks)[:max_body_chars]
            return ta + " " + body
    except Exception:
        pass
    return ta


# ── BM25 index ─────────────────────────────────────────────────────────────

def build_fulltext_bm25(corpus, model_dir: Path):
    """Build BM25 index on full-text corpus."""
    print("Building full-text corpus texts...")
    corpus_texts = [format_fulltext(row) for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Corpus text")]
    print("Tokenizing corpus...")
    tokenized = [tokenize(t) for t in tqdm(corpus_texts, desc="Tokenize")]
    print("Building BM25Okapi index on full text...")
    bm25 = BM25Okapi(tokenized, k1=1.0, b=1.0)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"Full-text BM25 index saved → {model_dir / 'index.pkl'}")
    return bm25


def load_bm25(model_dir: Path):
    with open(model_dir / "index.pkl", "rb") as f:
        return pickle.load(f)


# ── Score matrices ─────────────────────────────────────────────────────────

def score_bm25(bm25, query_texts: list, n_corpus: int) -> np.ndarray:
    """Score each query against the BM25 index, return (n_q, n_docs) matrix."""
    matrix = np.zeros((len(query_texts), n_corpus), dtype=np.float32)
    for i, qt in enumerate(tqdm(query_texts, desc="BM25 scoring", leave=False)):
        tokens = tokenize(qt)
        if tokens:
            matrix[i] = np.array(bm25.get_scores(tokens), dtype=np.float32)
    return matrix


# ── Base fusion loader ─────────────────────────────────────────────────────

def load_dense_scores(safe_name, model_name, query_prefix, query_ids, query_texts,
                      is_heldout, device, batch_size):
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


def load_tfidf_scores(query_texts, corpus_texts, is_heldout):
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


def compute_base_fusion(query_ids, query_texts, corpus_texts, is_heldout, device, batch_size):
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


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Citation context mining + full-text BM25 fusion"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--retrain", action="store_true",
                        help="Rebuild full-text BM25 index")
    parser.add_argument("--alpha-cite", type=float, default=None,
                        help="Fixed weight for citation-context BM25 (skip grid search)")
    parser.add_argument("--alpha-ft", type=float, default=None,
                        help="Fixed weight for full-text BM25 (skip grid search)")
    parser.add_argument("--submit-held-out", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ── Load corpus ────────────────────────────────────────────────────────
    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    corpus_ids = corpus["doc_id"].tolist()
    corpus_ta_texts = [format_text(row) for _, row in corpus.iterrows()]
    n_corpus = len(corpus)
    print(f"  {n_corpus} docs")

    # ── Build or load full-text BM25 index ─────────────────────────────────
    index_path = Path(args.model_dir) / "index.pkl"
    if args.retrain or not index_path.exists():
        bm25 = build_fulltext_bm25(corpus, args.model_dir)
    else:
        print("Loading cached full-text BM25 index...")
        bm25 = load_bm25(args.model_dir)

    # ── Load queries ───────────────────────────────────────────────────────
    is_heldout = args.submit_held_out
    queries = load_queries(args.held_out if is_heldout else args.queries)
    query_ids = queries["doc_id"].tolist()
    query_ta_texts = [format_text(row) for _, row in queries.iterrows()]
    print(f"  {len(queries)} {'held-out' if is_heldout else 'training'} queries")

    # ── Extract citation contexts ──────────────────────────────────────────
    print("Extracting citation-context sentences from query full_text...")
    cite_ctx_texts = []
    n_with_ctx = 0
    for _, row in queries.iterrows():
        ctx = extract_citation_sentences(str(row.get("full_text", "")))
        cite_ctx_texts.append(ctx)
        if ctx:
            n_with_ctx += 1
    avg_len = np.mean([len(t) for t in cite_ctx_texts if t])
    print(f"  {n_with_ctx}/{len(queries)} queries have citation contexts")
    print(f"  avg citation-context length: {avg_len:.0f} chars")

    # ── Score matrices ─────────────────────────────────────────────────────
    suffix = "_heldout" if is_heldout else "_train"

    # Citation-context BM25 scores
    cite_cache = Path(args.model_dir) / f"cite_ctx_scores{suffix}.npy"
    if not is_heldout and cite_cache.exists() and not args.retrain:
        print("Loading cached citation-context BM25 scores...")
        cite_scores = np.load(cite_cache).astype(np.float32)
    else:
        print("Scoring citation contexts against full-text BM25 index...")
        cite_scores = score_bm25(bm25, cite_ctx_texts, n_corpus)
        if not is_heldout:
            np.save(cite_cache, cite_scores)
            print(f"  Saved → {cite_cache}")

    # TA query against full-text BM25 scores
    ta_ft_cache = Path(args.model_dir) / f"ta_fulltext_scores{suffix}.npy"
    if not is_heldout and ta_ft_cache.exists() and not args.retrain:
        print("Loading cached TA full-text BM25 scores...")
        ta_ft_scores = np.load(ta_ft_cache).astype(np.float32)
    else:
        print("Scoring TA queries against full-text BM25 index...")
        ta_ft_scores = score_bm25(bm25, query_ta_texts, n_corpus)
        if not is_heldout:
            np.save(ta_ft_cache, ta_ft_scores)
            print(f"  Saved → {ta_ft_cache}")

    # ── Base fusion ────────────────────────────────────────────────────────
    print("\nLoading base fusion scores (UAE+BGE+E5+TFIDF)...")
    base_scores = compute_base_fusion(
        query_ids, query_ta_texts, corpus_ta_texts, is_heldout, device, args.batch_size
    )

    # Normalize the new signals
    norm_cite = normalize_rows(cite_scores)
    norm_ta_ft = normalize_rows(ta_ft_scores)

    # ── Held-out submission ────────────────────────────────────────────────
    if is_heldout:
        ac = args.alpha_cite if args.alpha_cite is not None else 0.10
        af = args.alpha_ft if args.alpha_ft is not None else 0.10
        base_w = 1.0 - ac - af
        print(f"Submitting: base={base_w:.2f}, cite_ctx={ac:.2f}, ta_ft={af:.2f}")
        fused = base_w * base_scores + ac * norm_cite + af * norm_ta_ft
        top_idx = np.argsort(-fused, axis=1)[:, :100]
        predictions = {qid: [corpus_ids[j] for j in top_idx[i]]
                       for i, qid in enumerate(query_ids)}
        save_submission(predictions, args.output)
        return

    # ── Training grid search ──────────────────────────────────────────────
    qrels = load_qrels(args.qrels)
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    # Reference: base only
    base_top = np.argsort(-base_scores, axis=1)[:, :100]
    base_preds = {qid: [corpus_ids[j] for j in base_top[i]]
                  for i, qid in enumerate(query_ids)}
    base_result = evaluate(base_preds, qrels, ks=[10, 100], verbose=False)
    base_map = base_result["overall"]["MAP"]
    base_ndcg = base_result["overall"]["NDCG@10"]
    print(f"\nBase fusion MAP: {base_map:.4f}, NDCG@10: {base_ndcg:.4f}")

    # Single-config eval
    if args.alpha_cite is not None and args.alpha_ft is not None:
        ac, af = args.alpha_cite, args.alpha_ft
        base_w = 1.0 - ac - af
        fused = base_w * base_scores + ac * norm_cite + af * norm_ta_ft
        top_idx = np.argsort(-fused, axis=1)[:, :100]
        preds = {qid: [corpus_ids[j] for j in top_idx[i]]
                 for i, qid in enumerate(query_ids)}
        evaluate(preds, qrels, ks=[10, 100], query_domains=query_domains)
        save_submission(preds, args.output)
        return

    # Grid search over alpha_cite and alpha_ft
    alpha_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    print(f"\nGrid search: {len(alpha_values)}² = {len(alpha_values)**2} combos")

    best_map = base_map
    best_ndcg = base_ndcg
    best_ac = 0.0
    best_af = 0.0
    results = []

    for ac in alpha_values:
        for af in alpha_values:
            if ac + af > 0.60:  # keep base weight >= 0.40
                continue
            base_w = 1.0 - ac - af
            fused = base_w * base_scores + ac * norm_cite + af * norm_ta_ft
            top_idx = np.argsort(-fused, axis=1)[:, :100]
            preds = {qid: [corpus_ids[j] for j in top_idx[i]]
                     for i, qid in enumerate(query_ids)}
            result = evaluate(preds, qrels, ks=[10, 100], verbose=False)
            m = result["overall"]["MAP"]
            n = result["overall"]["NDCG@10"]
            results.append((ac, af, m, n))
            if m > best_map:
                best_map = m
                best_ndcg = n
                best_ac = ac
                best_af = af

    # Print results table
    print(f"\n{'ac':>6} {'af':>6}  {'MAP':>6}  {'NDCG@10':>8}")
    print("-" * 32)
    for ac, af, m, n in sorted(results, key=lambda x: -x[2])[:15]:
        marker = " ← best" if ac == best_ac and af == best_af else ""
        print(f"{ac:>6.2f} {af:>6.2f}  {m:.4f}  {n:.4f}{marker}")

    print(f"\n{'=' * 50}")
    print(f"Base MAP:  {base_map:.4f}  NDCG@10: {base_ndcg:.4f}")
    print(f"Best MAP:  {best_map:.4f}  NDCG@10: {best_ndcg:.4f}")
    print(f"Best:      alpha_cite={best_ac}, alpha_ft={best_af}")
    print(f"{'=' * 50}")

    # Full eval with best config
    base_w = 1.0 - best_ac - best_af
    fused = base_w * base_scores + best_ac * norm_cite + best_af * norm_ta_ft
    top_idx = np.argsort(-fused, axis=1)[:, :100]
    best_preds = {qid: [corpus_ids[j] for j in top_idx[i]]
                  for i, qid in enumerate(query_ids)}
    evaluate(best_preds, qrels, ks=[10, 100], query_domains=query_domains)
    save_submission(best_preds, args.output)

    print(f"\nCommand for held-out submission:")
    print(f"python3 21_cite_context_bm25.py --submit-held-out "
          f"--alpha-cite {best_ac} --alpha-ft {best_af}")


if __name__ == "__main__":
    main()
