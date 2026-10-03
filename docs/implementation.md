# Implementation

What each step does, and why it helps or fails. Commands are in [usage.md](usage.md).

## Task and evaluation

For each query paper, return a ranked list of 100 corpus papers that it is likely to cite. The corpus has 20,000 papers, and the training set has 100 queries with their true citations. `utils.py` computes Recall@k, Precision@k, MRR@k, NDCG@k, and MAP, overall and per domain. All grid searches pick the configuration with the best MAP on the training queries, so the scores of the fused models are slightly optimistic. The learning to rank models are evaluated with 5-fold `GroupKFold` split by query, so no query is seen in training and in evaluation.

## Baselines

**`01_tfidf.py` (MAP 0.45).** `TfidfVectorizer` on title and abstract (sublinear term frequency, `max_df=0.85`), ranked by cosine similarity. It only matches exact terms.

**`02_bm25.py` (MAP 0.48).** `BM25Okapi` on title and abstract, after lowercasing, punctuation removal, and English stopword filtering. A grid search gives `k1=1.0` and `b=1.0`. BM25 saturates the weight of frequent terms and normalizes document length better than TF-IDF.

**`03_dense_minilm.py` (MAP 0.50).** Cosine similarity between the MiniLM embeddings (`all-MiniLM-L6-v2`, 384 dimensions) provided with the challenge. It captures meaning beyond exact words, but MiniLM is small and not trained on scientific text.

## Dense encoders

**`04_dense_encoders.py` (MAP 0.57).** Encodes the corpus and the queries with any Sentence Transformers model, then ranks by cosine similarity. UAE-Large-V1 reaches 0.57 alone, and E5-large-v2 0.46 with its `query:` and `passage:` prefixes. The script also caches the corpus embeddings of each encoder in `models/<encoder>/`, which every later dense signal reuses: UAE-Large-V1, E5-large-v2, BGE-large (with its retrieval instruction on queries), and SciNCL.

## Score fusion

**`05_score_fusion.py` (MAP 0.57).** BGE-large (`BAAI/bge-large-en-v1.5`, 1024 dimensions, with its retrieval instruction on queries) and BM25 scores are min-max normalized per query, then combined:

`score = alpha * BGE + (1 - alpha) * BM25`

A grid search over alpha from 0.05 to 0.95 gives alpha = 0.85. BGE does most of the work, and BM25 adds a bonus for rare technical terms.

**`06_multi_fusion.py` (MAP 0.57).** The same idea with more signals. The pool holds three dense encoders (UAE-Large-V1, BGE-large, E5-large-v2), BM25, and TF-IDF. The script tries every subset of at least two signals and every weighting with weights of at least 0.10 in steps of 0.10 that sum to 1. The best fusion uses UAE (0.60), BGE (0.10), E5 (0.10), and TF-IDF (0.20). It does not beat UAE alone or script 05, but its four signals are more diverse, and it serves as the base of scripts 07 and 09, which rebuild it from the cached scores.

## Citation contexts

**`07_citation_context.py` (MAP 0.59).** The full text of a query paper describes its references in sentences such as "as shown by [1]" or "(Smith et al., 2020) demonstrated". These sentences are extracted with regular expressions, their markers are removed, and they are scored with BM25 against a full-text index of the corpus (title, abstract, and the first 5,000 characters of the body). The title and abstract of the query are scored against the same index. Both signals are fused with the base of script 06 through a 2D grid search on their weights. The gain is modest because BM25 remains a sparse signal.

## Learning to rank

**`08_learning_to_rank.py` (MAP 0.67).** An `XGBRanker` (objective `rank:ndcg`, 200 trees of depth 4) replaces the fixed weighted sum. Candidates are the union of the top 200 of each retriever, about 600 to 800 per query. Each query and candidate pair has 16 features:

| Group | Features |
| --- | --- |
| Dense similarities | UAE-Large-V1, BGE-large, E5-large-v2, SciNCL |
| Sparse scores | BM25 on title and abstract, BM25 on full text, citation context BM25, TF-IDF |
| Reciprocal ranks | one per retrieval system (6) |
| Metadata | year proximity, domain match |

This is the largest single gain (+0.08). Trees learn interactions that a sum cannot express: a paper ranked high by both a dense and a sparse retriever is much more likely to be cited than either score suggests. Reciprocal ranks let the model weight each retriever by its reliability, year proximity captures the tendency to cite recent work, and domain match removes cross-field false positives.

## Cross-encoder and rerankers

**`09_cross_encoder.py`.** `BAAI/bge-reranker-v2-m3` reads each query and candidate together (up to 1,024 tokens), on the top 200 candidates of the fusion of script 06. Queries are enriched with their citation sentences. The cross-encoder score is interpolated with the fusion score, and the scores are cached for script 10.

**`10_ltr_cross_encoder.py` (MAP about 0.67).** Script 08 with three more features: the cross-encoder score, a flag telling whether the candidate was scored, and its reciprocal rank among the scored candidates. The gain is marginal, because the ranker was already close to saturation on the available signals.

**`11_rerank_cross_encoder.py` (below 0.67).** Rescores the top candidates of script 10 with the same cross-encoder, then interpolates with the learning to rank position. It brings no new information, since script 10 already uses this cross-encoder.

**`12_rerank_llm.py` (below 0.67).** Pointwise scoring with Qwen2.5-7B-Instruct: the LLM rates each pair from 0 to 10, read from the first number of its answer or, if that fails, from the expected value over the digit tokens. Only the top candidates are rescored. The LLM is too imprecise on specialized citation links to improve a well calibrated ranker. An OpenAI backend is also available.

## Files

| File | Role |
| --- | --- |
| `utils.py` | Device selection, data loaders, text formatting, metrics, reciprocal rank fusion, and submission export (JSON and ZIP for Codabench). |
| `01_tfidf.py` to `12_rerank_llm.py` | One step each, as described above. Each script evaluates on the training queries by default, and writes a Codabench submission with `--submit-held-out`. |
