"""
LTR + Cross-Encoder features.

Extends script 22 by adding cross-encoder scores (from script 23's cached data)
as features in the XGBRanker. The CE captures token-level query-doc interaction
that bi-encoder cosine similarity misses.

New features (3):
  - ce_score: raw cross-encoder score (0 if candidate not in CE top-100)
  - has_ce_score: binary indicator (1 if candidate was scored by CE)
  - ce_rank: reciprocal rank among CE-scored candidates (0 if not scored)

Usage:
  python3 24_ltr_ce_features.py                    # CV evaluation
  python3 24_ltr_ce_features.py --retrain           # force recompute features
  python3 24_ltr_ce_features.py --submit-held-out   # generate submission
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
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
DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "ltr_ce"
DEFAULT_LTR_DIR = SCRIPT_DIR / "models" / "ltr_ce"
DEFAULT_CE_DIR = SCRIPT_DIR / "models" / "crossencoder_v2"
DEFAULT_BATCH_SIZE = 64
TOP_K_CANDIDATES = 200

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

# ── Helpers ────────────────────────────────────────────────────────────────

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


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    mins = matrix.min(axis=1, keepdims=True)
    maxs = matrix.max(axis=1, keepdims=True)
    denom = np.where(maxs - mins < 1e-10, 1.0, maxs - mins)
    return np.where(maxs - mins < 1e-10, 0.0, (matrix - mins) / denom)


# ── Score matrix loaders ───────────────────────────────────────────────────

def load_dense_sim(safe_name, model_name, query_prefix, query_ids, query_texts,
                   is_heldout, device, batch_size):
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


def load_bm25_ta_scores(query_texts, is_heldout):
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


def load_tfidf_scores(query_texts, corpus_texts, is_heldout):
    tfidf_dir = SCRIPT_DIR / "models" / "tfidf"
    cache_path = tfidf_dir / ("heldout_scores.npy" if is_heldout else "train_scores.npy")
    if not is_heldout and cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vect_path = tfidf_dir / "vectorizer.pkl"
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


def load_bm25_fulltext_scores(is_heldout):
    ft_dir = SCRIPT_DIR / "models" / "bm25_fulltext"
    suffix = "_heldout" if is_heldout else "_train"

    cite_path = ft_dir / f"cite_ctx_scores{suffix}.npy"
    ta_ft_path = ft_dir / f"ta_fulltext_scores{suffix}.npy"

    cite_scores = None
    ta_ft_scores = None

    if cite_path.exists():
        cite_scores = np.load(cite_path).astype(np.float32)
    else:
        print("    [SKIP] Citation-context BM25: run script 21 first")

    if ta_ft_path.exists():
        ta_ft_scores = np.load(ta_ft_path).astype(np.float32)
    else:
        print("    [SKIP] TA full-text BM25: run script 21 first")

    return cite_scores, ta_ft_scores


def load_ce_data(ce_dir: Path, is_heldout: bool):
    """Load cached cross-encoder scores and indices."""
    suffix = "_heldout" if is_heldout else "_train"
    ce_data_path = ce_dir / f"ce_data{suffix}.npz"

    if not ce_data_path.exists():
        print(f"    [SKIP] Cross-encoder scores: run script 23 first")
        return None, None

    ce_data = np.load(ce_data_path)
    return ce_data["scores"], ce_data["indices"]


# ── Feature construction ──────────────────────────────────────────────────

def build_features(score_matrices: dict, query_ids: list, corpus_ids: list,
                   queries_df, corpus_df, ce_scores_sparse, ce_indices,
                   n_candidates: int = TOP_K_CANDIDATES):
    """
    Build (query, doc) feature vectors for LTR, including CE features.

    Returns:
      features, labels, groups, pair_info, feature_names
    """
    q_years = dict(zip(queries_df["doc_id"], queries_df["year"]))
    q_domains = dict(zip(queries_df["doc_id"], queries_df["domain"]))
    c_years = dict(zip(corpus_df["doc_id"], corpus_df["year"]))
    c_domains = dict(zip(corpus_df["doc_id"], corpus_df["domain"]))

    score_names = list(score_matrices.keys())
    feature_names = (
        [f"score_{name}" for name in score_names] +
        [f"rank_{name}" for name in score_names] +
        ["year_proximity", "domain_match"]
    )

    # CE features
    has_ce = ce_scores_sparse is not None and ce_indices is not None
    if has_ce:
        feature_names.extend(["ce_score", "has_ce_score", "ce_rank"])
        # Build per-query lookup: corpus_doc_idx -> (ce_score, ce_rank)
        ce_lookups = []
        for qi in range(len(query_ids)):
            lookup = {}
            # Compute ranks within CE candidates for this query
            ce_raw = ce_scores_sparse[qi]
            ce_sorted_indices = np.argsort(-ce_raw)
            for rank_pos, local_idx in enumerate(ce_sorted_indices):
                corpus_idx = int(ce_indices[qi, local_idx])
                score = float(ce_raw[local_idx])
                recip_rank = 1.0 / (60 + rank_pos + 1)
                lookup[corpus_idx] = (score, recip_rank)
            ce_lookups.append(lookup)
    else:
        ce_lookups = None

    # Precompute rank matrices
    rank_matrices = {}
    for name, mat in score_matrices.items():
        if mat is not None:
            ranks = np.argsort(np.argsort(-mat, axis=1), axis=1)
            rank_matrices[name] = 1.0 / (60 + ranks + 1)

    all_features = []
    all_labels = []
    groups = []
    pair_info = []

    for qi, qid in enumerate(tqdm(query_ids, desc="Building features")):
        candidate_set = set()
        for name, mat in score_matrices.items():
            if mat is not None:
                top_k = np.argsort(-mat[qi])[:n_candidates]
                candidate_set.update(top_k.tolist())
        candidates = sorted(candidate_set)

        q_year = q_years.get(qid, 2020)
        q_domain = q_domains.get(qid, "")

        for di in candidates:
            doc_id = corpus_ids[di]

            # Score features
            row = []
            for name in score_names:
                mat = score_matrices[name]
                row.append(float(mat[qi, di]) if mat is not None else 0.0)

            # Rank features
            for name in score_names:
                rmat = rank_matrices.get(name)
                row.append(float(rmat[qi, di]) if rmat is not None else 0.0)

            # Metadata features
            d_year = c_years.get(doc_id, 2020)
            year_prox = 1.0 / (1.0 + abs(q_year - d_year))
            d_domain = c_domains.get(doc_id, "")
            domain_match = 1.0 if q_domain == d_domain and q_domain else 0.0
            row.extend([year_prox, domain_match])

            # CE features
            if has_ce and ce_lookups is not None:
                ce_info = ce_lookups[qi].get(di)
                if ce_info is not None:
                    row.extend([ce_info[0], 1.0, ce_info[1]])
                else:
                    row.extend([0.0, 0.0, 0.0])

            all_features.append(row)
            pair_info.append((qi, di))

        groups.append(len(candidates))

    features = np.array(all_features, dtype=np.float32)
    labels_arr = np.zeros(len(all_features), dtype=np.float32)

    return features, labels_arr, groups, pair_info, feature_names


def fill_labels(labels: np.ndarray, pair_info: list, query_ids: list,
                corpus_ids: list, qrels: dict):
    for idx, (qi, di) in enumerate(pair_info):
        qid = query_ids[qi]
        doc_id = corpus_ids[di]
        if doc_id in set(qrels.get(qid, [])):
            labels[idx] = 1.0
    return labels


def predict_rankings(model, features, pair_info, query_ids, corpus_ids, groups):
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


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LTR with Cross-Encoder features (5-fold GroupKFold)"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--ltr-dir", default=DEFAULT_LTR_DIR, type=Path)
    parser.add_argument("--ce-dir", default=DEFAULT_CE_DIR, type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--submit-held-out", action="store_true")
    args = parser.parse_args()

    device = get_device()
    ltr_dir = Path(args.ltr_dir)
    ltr_dir.mkdir(parents=True, exist_ok=True)

    try:
        from xgboost import XGBRanker
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        return

    from sklearn.model_selection import GroupKFold

    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    corpus_ids = corpus["doc_id"].tolist()
    corpus_ta_texts = [format_text(row) for _, row in corpus.iterrows()]
    print(f"  {len(corpus)} docs")

    is_heldout = args.submit_held_out
    queries = load_queries(args.held_out if is_heldout else args.queries)
    query_ids = queries["doc_id"].tolist()
    query_ta_texts = [format_text(row) for _, row in queries.iterrows()]
    print(f"  {len(queries)} {'held-out' if is_heldout else 'training'} queries")

    # ── Load all score matrices ────────────────────────────────────────────
    print("\nLoading score matrices...")
    score_matrices = {}

    for key, cfg in DENSE_MODELS.items():
        print(f"  [{key}]")
        mat = load_dense_sim(
            cfg["safe_name"], cfg["model_name"], cfg["query_prefix"],
            query_ids, query_ta_texts, is_heldout, device, args.batch_size,
        )
        score_matrices[key] = mat

    print("  [bm25_ta]")
    score_matrices["bm25_ta"] = load_bm25_ta_scores(query_ta_texts, is_heldout)

    print("  [tfidf]")
    score_matrices["tfidf"] = load_tfidf_scores(query_ta_texts, corpus_ta_texts, is_heldout)

    print("  [bm25_fulltext / cite_ctx]")
    cite_scores, ta_ft_scores = load_bm25_fulltext_scores(is_heldout)
    score_matrices["cite_ctx_bm25"] = cite_scores
    score_matrices["bm25_fulltext"] = ta_ft_scores

    available = {k: v for k, v in score_matrices.items() if v is not None}
    print(f"\nAvailable score matrices: {list(available.keys())}")
    print(f"Missing (will use 0): {[k for k, v in score_matrices.items() if v is None]}")

    # ── Load cross-encoder scores ──────────────────────────────────────────
    print("\nLoading cross-encoder scores...")
    ce_scores_sparse, ce_indices = load_ce_data(args.ce_dir, is_heldout)
    if ce_scores_sparse is not None:
        print(f"  CE scores shape: {ce_scores_sparse.shape}")
    else:
        print("  WARNING: No CE scores available. Running without CE features.")

    # ── Build features ─────────────────────────────────────────────────────
    feature_cache = ltr_dir / ("features_heldout.npz" if is_heldout else "features_train.npz")

    if not args.retrain and feature_cache.exists():
        print(f"\nLoading cached features from {feature_cache}...")
        data = np.load(feature_cache, allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        groups = data["groups"].tolist()
        pair_info = [tuple(x) for x in data["pair_info"]]
        feature_names = data["feature_names"].tolist()
    else:
        print("\nBuilding feature matrix...")
        features, labels, groups, pair_info, feature_names = build_features(
            score_matrices, query_ids, corpus_ids, queries, corpus,
            ce_scores_sparse, ce_indices,
        )
        if not is_heldout:
            qrels = load_qrels(args.qrels)
            labels = fill_labels(labels, pair_info, query_ids, corpus_ids, qrels)
        np.savez(
            feature_cache,
            features=features, labels=labels,
            groups=np.array(groups), pair_info=np.array(pair_info),
            feature_names=np.array(feature_names),
        )
        print(f"  Saved features -> {feature_cache}")

    print(f"  Features shape: {features.shape}")
    print(f"  Feature names: {feature_names}")
    print(f"  Total pairs: {len(features)}, queries: {len(groups)}")
    print(f"  Avg candidates/query: {len(features)/len(groups):.0f}")
    if not is_heldout:
        print(f"  Positive pairs: {int(labels.sum())} ({100*labels.mean():.2f}%)")

    # ── Held-out submission ────────────────────────────────────────────────
    if is_heldout:
        model_path = ltr_dir / "model.json"
        if not model_path.exists():
            print(f"ERROR: No trained model found at {model_path}")
            print("Run without --submit-held-out first to train.")
            return

        model = XGBRanker()
        model.load_model(model_path)
        predictions = predict_rankings(model, features, pair_info, query_ids, corpus_ids, groups)
        save_submission(predictions, args.output)
        return

    # ── Cross-validation ───────────────────────────────────────────────────
    qrels = load_qrels(args.qrels)
    if labels.sum() == 0:
        labels = fill_labels(labels, pair_info, query_ids, corpus_ids, qrels)

    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    print(f"\n{'=' * 60}")
    print("5-fold GroupKFold cross-validation")
    print(f"{'=' * 60}")

    gkf = GroupKFold(n_splits=5)
    pair_groups = np.array([qi for qi, _ in pair_info])

    fold_maps = []
    fold_ndcgs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(features, labels, pair_groups)):
        train_qi_set = sorted(set(pair_groups[train_idx]))
        val_qi_set = sorted(set(pair_groups[val_idx]))

        train_groups = []
        for qi in train_qi_set:
            train_groups.append(int(np.sum(pair_groups[train_idx] == qi)))

        val_groups = []
        for qi in val_qi_set:
            val_groups.append(int(np.sum(pair_groups[val_idx] == qi)))

        model = XGBRanker(
            objective="rank:ndcg",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )

        model.fit(
            features[train_idx], labels[train_idx],
            group=train_groups,
            eval_set=[(features[val_idx], labels[val_idx])],
            eval_group=[val_groups],
            verbose=False,
        )

        # Predict on validation fold
        val_pair_info = [pair_info[i] for i in val_idx]
        remapped = {}
        offset = 0
        for qi_local, g in enumerate(val_groups):
            actual_qi = val_qi_set[qi_local]
            qid = query_ids[actual_qi]
            group_scores = model.predict(features[val_idx][offset:offset+g])
            group_pairs = val_pair_info[offset:offset+g]
            sorted_local = np.argsort(-group_scores)
            ranked_ids = [corpus_ids[group_pairs[j][1]] for j in sorted_local[:100]]
            remapped[qid] = ranked_ids
            offset += g

        result = evaluate(remapped, qrels, ks=[10, 100], verbose=False)
        fold_map = result["overall"]["MAP"]
        fold_ndcg = result["overall"]["NDCG@10"]
        fold_maps.append(fold_map)
        fold_ndcgs.append(fold_ndcg)
        print(f"  Fold {fold+1}: MAP={fold_map:.4f}, NDCG@10={fold_ndcg:.4f} "
              f"(train={len(train_qi_set)}q, val={len(val_qi_set)}q)")

    mean_map = np.mean(fold_maps)
    mean_ndcg = np.mean(fold_ndcgs)
    std_map = np.std(fold_maps)
    std_ndcg = np.std(fold_ndcgs)

    print(f"\n{'=' * 60}")
    print(f"CV MAP:     {mean_map:.4f} +/- {std_map:.4f}")
    print(f"CV NDCG@10: {mean_ndcg:.4f} +/- {std_ndcg:.4f}")
    print(f"{'=' * 60}")

    # ── Train final model on all data ──────────────────────────────────────
    print("\nTraining final model on all training data...")
    final_model = XGBRanker(
        objective="rank:ndcg",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    final_model.fit(features, labels, group=groups)

    model_path = ltr_dir / "model.json"
    final_model.save_model(model_path)
    print(f"Model saved -> {model_path}")

    # Full training eval
    print("\nFull training evaluation:")
    train_preds = predict_rankings(final_model, features, pair_info, query_ids, corpus_ids, groups)
    evaluate(train_preds, qrels, ks=[10, 100], query_domains=query_domains)
    save_submission(train_preds, args.output)

    # Feature importance
    importance = final_model.feature_importances_
    sorted_idx = np.argsort(-importance)
    print("\nFeature importance:")
    for i in sorted_idx:
        if importance[i] > 0:
            print(f"  {feature_names[i]:<25s} {importance[i]:.4f}")

    print(f"\nCommand for held-out submission:")
    print(f"python3 24_ltr_ce_features.py --submit-held-out")


if __name__ == "__main__":
    main()
