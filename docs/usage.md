# Usage

Setup, data, then the order of the scripts and their options. What each script does is in [implementation.md](implementation.md).

## Setup

The project has its own environment, `citation-ir`, defined in [`environment.yml`](../environment.yml).

```bash
mamba env create -f environment.yml          # create the environment once
mamba activate citation-ir                   # activate it in every new terminal
mamba env update -f environment.yml --prune  # after environment.yml changes
```

## Data

For data confidentiality reasons, the challenge data is not included in this repository. The scripts expect this layout:

```
data/queries.parquet
data/corpus.parquet
data/qrels.json
data/embeddings/sentence-transformers_all-MiniLM-L6-v2/   MiniLM embeddings provided with the challenge
held_out_queries.parquet                                  queries of the Codabench submission
```

Scripts 05 to 10 read the corpus embeddings of each dense encoder from `models/<encoder>/corpus_embeddings.npy` and `models/<encoder>/corpus_ids.json`. `04_dense_encoders.py` builds them, once per encoder, with the prefixes and folder names that the later scripts expect:

```bash
python 04_dense_encoders.py                                                  # UAE-Large-V1
python 04_dense_encoders.py --model-name intfloat/e5-large-v2 --query-prefix "query: " --corpus-prefix "passage: "
python 04_dense_encoders.py --model-name BAAI/bge-large-en-v1.5 --dir-name bge \
    --query-prefix "Represent this sentence for searching relevant passages: "
python 04_dense_encoders.py --model-name malteos/scincl --dir-name specter2
```

Query embeddings are computed and cached by the scripts themselves.

## Order of the scripts

Every script runs from the project root. Later scripts reuse what earlier ones save in `models/`:

| Script | Needs | Saves |
| --- | --- | --- |
| `01_tfidf.py` | data | `models/tfidf/` |
| `02_bm25.py` | data | `models/bm25/` |
| `03_dense_minilm.py` | data and the provided MiniLM embeddings | |
| `04_dense_encoders.py` | data | `models/<encoder>/` corpus and query embeddings |
| `05_score_fusion.py` | `models/bm25/`, BGE corpus embeddings | BGE query embeddings |
| `06_multi_fusion.py` | `models/bm25/`, `models/tfidf/`, UAE, BGE, and E5 corpus embeddings | score caches in `models/bm25/` and `models/tfidf/` |
| `07_citation_context.py` | `models/tfidf/`, UAE, BGE, and E5 corpus embeddings | `models/bm25_fulltext/` and its score caches |
| `08_learning_to_rank.py` | scripts 01, 02, and 07, all corpus embeddings | `models/ltr/` |
| `09_cross_encoder.py` | `models/tfidf/`, UAE, BGE, and E5 corpus embeddings | `models/crossencoder_v2/` |
| `10_ltr_cross_encoder.py` | everything script 08 needs, and script 09 | `models/ltr/` with cross-encoder features |
| `11_rerank_cross_encoder.py`, `12_rerank_llm.py` | the submission of script 10 | their own submissions |

## Common options

Every script accepts `--queries`, `--corpus`, `--qrels`, `--held-out`, and `--output` to change its paths, and `--submit-held-out` to rank the held-out queries and write a Codabench submission (`submission_data.json` and its ZIP) in `submissions/<method>/`. Scripts that build an index or features accept `--retrain` to rebuild it instead of reading the cache.

## Script options

| Script | Main options |
| --- | --- |
| `04_dense_encoders.py` | `--model-name`, `--query-prefix`, `--corpus-prefix`, `--dir-name` (folder under `models/`), `--force-encode` (encode the corpus again), `--retrain` (encode the queries again) |
| `05_score_fusion.py` | `--alpha` (fixed weight instead of the grid search), `--model-name`, `--batch-size` |
| `06_multi_fusion.py` | `--models` and `--weights` (a fixed fusion instead of the grid search, required with `--submit-held-out`), `--batch-size` |
| `07_citation_context.py` | `--alpha-cite`, `--alpha-ft` (fixed weights instead of the grid search), `--batch-size` |
| `08_learning_to_rank.py` | `--ltr-dir`, `--batch-size` |
| `09_cross_encoder.py` | `--cross-encoder`, `--rerank-top`, `--gamma`, `--batch-size-ce` |
| `10_ltr_cross_encoder.py` | `--ltr-dir`, `--ce-dir` |
| `11_rerank_cross_encoder.py` | `--predictions`, `--ce-model`, `--rerank-top`, `--gamma`, `--max-length` |
| `12_rerank_llm.py` | `--predictions`, `--backend` (`local` or `openai`), `--llm-model`, `--rerank-top`, `--gamma`, `--limit-queries` |

Examples:

```bash
python 05_score_fusion.py --submit-held-out --alpha 0.85
python 06_multi_fusion.py --submit-held-out --models uae bge e5 tfidf --weights 0.60 0.10 0.10 0.20
python 07_citation_context.py --submit-held-out --alpha-cite 0.10 --alpha-ft 0.10
python 11_rerank_cross_encoder.py --rerank-top 50 --gamma 0.7
python 12_rerank_llm.py --backend openai --llm-model gpt-4o-mini   # needs OPENAI_API_KEY
```
