# Scientific Article Information Retrieval

A progressive information retrieval pipeline for scientific citation prediction, built for the [Codabench](https://www.codabench.org/) challenge.

**Task:** Given a query article (title + abstract + full text), retrieve the 100 most likely cited articles from a corpus of 20,000 scientific papers.

**Metrics:** MAP (primary), NDCG@10, Recall@100.

**Data:** 100 training queries with ground-truth relevance judgments (`qrels.json`) and a corpus of 20,000 documents.

---

## Results Overview

| Script | Method | MAP | Δ vs Previous |
|--------|--------|-----|---------------|
| `01` | TF-IDF baseline | 0.45 | — |
| `03` | BM25 baseline | ~0.48 | +0.03 |
| `02` | Dense MiniLM baseline | 0.50 | +0.02 |
| `10` | Score fusion (BGE + BM25) | 0.57 | +0.07 |
| `21` | Citation context BM25 | 0.59 | +0.02 |
| `22` | Learning to Rank (XGBRanker) | 0.67 | **+0.08** |
| `24` | LTR + Cross-Encoder features | ≈0.67+ | +marginal |
| `33` | CE reranking on LTR | <0.67 | regression |
| `34` | LLM reranking on LTR | <0.67 | regression |

---

## Repository Structure

```
├── utils.py                  # Shared utilities: data loading, metrics, submission I/O
├── 01_tfidf_baseline.py      # TF-IDF sparse retrieval baseline
├── 02_dense_baseline.py      # Dense retrieval with MiniLM embeddings
├── 03_sparse_improved.py     # BM25 (Okapi) sparse retrieval baseline
├── 10_score_fusion.py        # BGE dense + BM25 score-level fusion
├── 21_cite_context_bm25.py   # Citation context mining + full-text BM25
├── 22_learning_to_rank.py    # XGBRanker Learning to Rank (best result)
├── 24_ltr_ce_features.py     # LTR extended with cross-encoder features
├── 33_rerank_ce_on_ltr.py    # Cross-encoder reranking on LTR output
├── 34_rerank_llm_on_ltr.py   # LLM-based reranking on LTR output
├── resume.md                 # Detailed summary of all approaches (French)
└── presentation.md           # Presentation plan (French)
```

---

## Approach Details

### 1. Baselines

#### 01 — TF-IDF (MAP 0.45)

Classic sparse retrieval. Title and abstract of each document are encoded with scikit-learn's `TfidfVectorizer` (sublinear TF, `max_df=0.85`). At inference, queries are transformed the same way and documents are ranked by cosine similarity.

**Limitation:** Exact term matching only — no semantic understanding.

#### 03 — BM25 (MAP ~0.48)

Improved sparse retrieval using `BM25Okapi` from `rank_bm25` on title + abstract. Tokenization includes lowercasing, punctuation removal, and English stopword filtering. Parameters `k1=1.0, b=1.0` are selected via grid search.

**Improvement over TF-IDF:** BM25 saturates the impact of very frequent terms and applies better document-length normalization.

#### 02 — Dense Retrieval with MiniLM (MAP 0.50)

Dense retrieval with pre-computed embeddings from `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalized). Cosine similarity reduces to a dot product. No training is required.

**Improvement over sparse methods:** Captures semantic similarity beyond exact word matches.

---

### 2. Score Fusion

#### 10 — BGE + BM25 Score Fusion (MAP 0.57)

Fuses two complementary retrieval signals using a weighted normalized sum:

- **BM25 (lexical):** BM25Okapi index on title + abstract.
- **BGE-large (semantic):** `BAAI/bge-large-en-v1.5` (1024 dimensions), a much more powerful bi-encoder than MiniLM.

**Formula:** `final_score = α × BGE_normalized + (1 − α) × BM25_normalized`

Each score is min-max normalized to [0, 1]. A grid search over α (0.05 to 0.95, step 0.05) finds the optimal weight → **α = 0.85** (BGE strongly dominates).

**Why it works:** BGE-large is fundamentally stronger than MiniLM (7× larger, trained specifically for semantic search). BM25 adds a small bonus for exact matches on rare technical terms.

---

### 3. Citation Context Mining

#### 21 — Citation Context + Full-Text BM25 (MAP 0.59)

Exploits the `full_text` of query articles (completely ignored until this point).

**Key insight:** Citing articles contain *citation sentences* — phrases that directly describe the cited papers: `"As shown by [1]"`, `"(Smith et al., 2020) demonstrated..."`. These sentences are direct descriptions of relevant documents.

**Pipeline:**
1. Build a full-text BM25 index on the corpus (title + abstract + first 5,000 characters of the body).
2. Extract citation contexts from each query's `full_text` using regex patterns to detect markers like `[1]`, `(Author et al., 2020)`, etc., then strip the markers to keep the descriptive text.
3. Score citation contexts against the full-text index.
4. Fuse with the base dense model scores via 2D grid search on weights `(alpha_cite, alpha_ft)`.

**Why it helps:** Citation sentences contain precise technical keywords that match titles and abstracts of cited documents exactly.

---

### 4. Learning to Rank (Best Result)

#### 22 — XGBRanker LTR (MAP 0.67)

Replaces the linear weighted sum with a gradient-boosted tree ranking model that *learns* the interactions between signals.

**Features per (query, document) pair (~16 features):**

| Category | Features |
|----------|----------|
| Dense similarity scores | UAE, BGE, E5, SciNCL (4) |
| Sparse scores | BM25 on title+abstract, BM25 on full text, citation-context BM25, TF-IDF (4) |
| Reciprocal ranks | From each retrieval system (6) |
| Metadata | Year proximity, domain match (2) |

**Training:**
- `XGBRanker(objective='rank:ndcg')`, `max_depth=4`, 200 estimators.
- **5-fold GroupKFold** cross-validation with `groups=query_id` to prevent data leakage across queries.
- Candidates: union of the top-200 from each retrieval system (~600–800 candidates per query).

**Why it's a major jump (+0.08):**
- Non-linear interactions between signals are crucial (e.g., high UAE score *and* high BM25 score is much stronger than their sum).
- Reciprocal rank features let the model weight signals by their relative reliability.
- Year proximity captures the tendency to cite recent or temporally close papers.
- Domain matching filters out cross-disciplinary false positives.

#### 24 — LTR + Cross-Encoder Features (MAP ≈0.67+)

Extends script 22 with features from a strong cross-encoder (`bge-reranker-v2-m3`):

| Feature | Description |
|---------|-------------|
| `ce_score` | Raw cross-encoder score (0 if not in CE top-200) |
| `has_ce_score` | Binary indicator of CE presence |
| `ce_rank` | Reciprocal rank among CE-scored candidates |

**Marginal gain:** The cross-encoder captures token-level query-document interaction that bi-encoders miss, but the LTR from script 22 was already near saturation on the available signals.

---

### 5. Failed Reranking Attempts

#### 33 — Cross-Encoder Reranking on LTR (Regression)

Applies `bge-reranker-v2-m3` directly on the top-K candidates from the LTR pipeline. The final score interpolates CE score with LTR rank position.

**Why it fails:** The LTR model already incorporates cross-encoder features (script 24). Re-applying the same signal on top provides no new information.

#### 34 — LLM Reranking on LTR (Regression)

Uses a small instruct LLM (Qwen2.5-1.5B-Instruct) for pointwise scoring. The model rates each (query, candidate) pair from 0 to 10 for citation relevance. The score is extracted via logit expectations over digit tokens.

**Why it fails:** The 1.5B-parameter LLM is too small to reason about the nuances of citation relationships in specialized scientific domains. The LTR ensemble is already well-calibrated and cannot be outperformed by a small generative model.

---

## Key Dependencies

- [sentence-transformers](https://www.sbert.net/) — Bi-encoder embeddings (MiniLM, BGE-large, UAE, E5, SciNCL)
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — BM25Okapi sparse retrieval
- [xgboost](https://xgboost.readthedocs.io/) — XGBRanker for Learning to Rank
- [scikit-learn](https://scikit-learn.org/) — TF-IDF, GroupKFold cross-validation
- [transformers](https://huggingface.co/docs/transformers/) — LLM inference (Qwen2.5)
- [numpy](https://numpy.org/) / [pandas](https://pandas.pydata.org/) — Data manipulation
- [nltk](https://www.nltk.org/) — Stopwords and tokenization

---

## Utility Module (`utils.py`)

Provides shared functionality across all scripts:

- **Device detection** — Automatic CUDA / MPS / CPU selection.
- **Data loaders** — Queries, corpus, qrels, and embeddings (Parquet, JSON, NumPy).
- **Text processing** — Title + abstract formatting, body text chunking.
- **Evaluation metrics** — Recall@k, Precision@k, MRR@k, NDCG@k, MAP, with per-domain breakdown.
- **Reciprocal Rank Fusion** — Combines multiple ranked lists.
- **Submission I/O** — JSON export + ZIP compression for Codabench.
