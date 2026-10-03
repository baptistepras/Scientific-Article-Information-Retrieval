"""
Learning to Rank with XGBRanker.

Replaces simple weighted-sum fusion with a gradient-boosted tree ranker that
learns non-linear feature interactions across all retrieval signals + metadata.

Features per (query, doc) pair:
  - Dense cosine similarities: UAE, BGE, E5, SciNCL (4)
  - Sparse scores: BM25 TA, BM25 full-text, citation-context BM25, TF-IDF (4)
  - Metadata: year proximity, domain match (2)
  - Reciprocal ranks from each retrieval system (6)
  Total: ~16 features

Training: 5-fold GroupKFold (group = query_id) to avoid overfitting on 100 queries.
Model: XGBRanker with objective='rank:ndcg', shallow trees (max_depth=4).

Usage:
  python3 08_learning_to_rank.py                    # CV evaluation
  python3 08_learning_to_rank.py --retrain           # force recompute features
  python3 08_learning_to_rank.py --submit-held-out   # generate submission
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (DEFAULT_CORPUS, DEFAULT_HELD_OUT, DEFAULT_QRELS,
                   DEFAULT_QUERIES, DENSE_MODELS, SCRIPT_DIR, TOP_K_CANDIDATES,
                   evaluate, fill_labels, format_text, get_device,
                   load_bm25_fulltext_scores, load_bm25_ta_scores, load_corpus,
                   load_dense_sim, load_qrels, load_queries, load_tfidf_scores,
                   predict_rankings, save_submission)

DEFAULT_OUTPUT = SCRIPT_DIR / "submissions" / "ltr"
DEFAULT_LTR_DIR = SCRIPT_DIR / "models" / "ltr"
DEFAULT_BATCH_SIZE = 64


# Feature construction

def build_features(score_matrices: dict, query_ids: list, corpus_ids: list,
                   queries_df: pd.DataFrame, corpus_df: pd.DataFrame,
                   n_candidates: int = TOP_K_CANDIDATES) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[int, int]], list[str]]:
    """
    Build (query, doc) feature vectors for LTR.

    Returns:
      features: np.ndarray (n_pairs, n_features)
      labels: np.ndarray (n_pairs,) — 1 if relevant, 0 otherwise
      groups: list[int] — number of candidates per query (for XGBRanker)
      pair_info: list of (query_idx, doc_idx) for reconstruction
      feature_names: list of str
    """
    # Build metadata lookups
    q_years = dict(zip(queries_df["doc_id"], queries_df["year"]))
    q_domains = dict(zip(queries_df["doc_id"], queries_df["domain"]))
    c_years = dict(zip(corpus_df["doc_id"], corpus_df["year"]))
    c_domains = dict(zip(corpus_df["doc_id"], corpus_df["domain"]))

    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    # Feature names
    score_names = list(score_matrices.keys())
    feature_names = (
        [f"score_{name}" for name in score_names] +
        [f"rank_{name}" for name in score_names] +
        ["year_proximity", "domain_match"]
    )

    # Precompute rank matrices for each scoring system
    rank_matrices = {}
    for name, mat in score_matrices.items():
        if mat is not None:
            ranks = np.argsort(np.argsort(-mat, axis=1), axis=1)  # 0-indexed rank
            rank_matrices[name] = 1.0 / (60 + ranks + 1)  # reciprocal rank feature

    all_features = []
    all_labels = []
    groups = []
    pair_info = []

    for qi, qid in enumerate(tqdm(query_ids, desc="Building features")):
        # Collect candidate documents: union of top-K from each system
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
            all_features.append(row)
            pair_info.append((qi, di))

        groups.append(len(candidates))

    features = np.array(all_features, dtype=np.float32)
    labels_arr = np.zeros(len(all_features), dtype=np.float32)

    return features, labels_arr, groups, pair_info, feature_names


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learning to Rank with XGBRanker (5-fold GroupKFold)"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--ltr-dir", default=DEFAULT_LTR_DIR, type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--retrain", action="store_true",
                        help="Force recompute features")
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

    # Load data
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

    # Load all score matrices
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

    # Remove None entries for candidate generation but keep them as 0 features
    available = {k: v for k, v in score_matrices.items() if v is not None}
    print(f"\nAvailable score matrices: {list(available.keys())}")
    print(f"Missing (will use 0): {[k for k, v in score_matrices.items() if v is None]}")

    # Build features
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
            score_matrices, query_ids, corpus_ids, queries, corpus
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
        print(f"  Saved features → {feature_cache}")

    print(f"  Features shape: {features.shape}")
    print(f"  Feature names: {feature_names}")
    print(f"  Total pairs: {len(features)}, queries: {len(groups)}")
    print(f"  Avg candidates/query: {len(features)/len(groups):.0f}")
    if not is_heldout:
        print(f"  Positive pairs: {int(labels.sum())} ({100*labels.mean():.2f}%)")

    # Held-out submission
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

    # Cross-validation
    qrels = load_qrels(args.qrels)
    if labels.sum() == 0:
        labels = fill_labels(labels, pair_info, query_ids, corpus_ids, qrels)

    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    print(f"\n{'=' * 60}")
    print("5-fold GroupKFold cross-validation")
    print(f"{'=' * 60}")

    gkf = GroupKFold(n_splits=5)
    # Build group labels for each pair (the query index)
    pair_groups = np.array([qi for qi, _ in pair_info])

    fold_maps = []
    fold_ndcgs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(features, labels, pair_groups)):
        # Build group sizes for train and val
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
        val_query_ids = [query_ids[qi] for qi in val_qi_set]
        val_predictions = predict_rankings(
            model, features[val_idx], val_pair_info,
            query_ids, corpus_ids, val_groups,
        )

        # Remap predictions to use actual query_ids from val set
        # (predict_rankings uses sequential qi, need the actual qids)
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
    print(f"CV MAP:     {mean_map:.4f} ± {std_map:.4f}")
    print(f"CV NDCG@10: {mean_ndcg:.4f} ± {std_ndcg:.4f}")
    print(f"{'=' * 60}")

    # Train final model on all data
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
    print(f"Model saved → {model_path}")

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
    print(f"python3 08_learning_to_rank.py --submit-held-out")


if __name__ == "__main__":
    main()
