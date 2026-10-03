# Scientific Article Information Retrieval

Citation retrieval on a corpus of 20,000 scientific papers, built step by step for a [Codabench challenge](https://www.codabench.org/competitions/15308/): from a TF-IDF baseline to a learning to rank model that combines lexical, dense, and citation context signals. MAP goes from 0.45 to 0.67.

Given a query paper (title, abstract, and full text), the system must return the 100 papers of the corpus that it most likely cites. The training set has 100 queries with their true citations. Each script adds one idea on top of the previous best, and the scripts are numbered in the order of the results table below.

## Results

MAP on the 100 training queries (the learning to rank scores are 5-fold cross-validated):

| Script | Method | MAP | Gain |
| --- | --- | --- | --- |
| `01_tfidf.py` | TF-IDF baseline (provided by the challenge) | 0.45 | |
| `02_bm25.py` | BM25 with tuned parameters | 0.48 | +0.03 |
| `03_dense_minilm.py` | MiniLM dense baseline (provided by the challenge) | 0.50 | +0.02 |
| `04_dense_encoders.py` | Stronger dense encoders: UAE-Large-V1 alone (E5-large-v2 reaches 0.46) | 0.57 | +0.07 |
| `05_score_fusion.py` | BGE-large and BM25 score fusion | 0.57 | same |
| `06_multi_fusion.py` | Fusion of UAE, BGE, E5, and TF-IDF, with an exhaustive weight search | 0.57 | same |
| `07_citation_context.py` | Citation sentences mined from the full text | 0.59 | +0.02 |
| `08_learning_to_rank.py` | **Learning to rank on 16 features** | **0.67** | **+0.08** |
| `09_cross_encoder.py` | Cross-encoder scores, used as features by script 10 | not reported | |
| `10_ltr_cross_encoder.py` | Learning to rank with cross-encoder features | 0.67 | marginal |
| `11_rerank_cross_encoder.py` | Cross-encoder reranking of script 10 | below 0.67 | regression |
| `12_rerank_llm.py` | LLM reranking of script 10 | below 0.67 | regression |

Three lessons stand out. A stronger encoder matters more than a clever fusion: UAE-Large-V1 alone already reaches 0.57, and no fixed weighting of several retrievers does better. Learning how the signals interact beats any fixed weighting, and gives the largest single gain. Reranking a well calibrated learning to rank model with a cross-encoder or a small LLM only adds noise.

## Environment

The project uses its own environment, `citation-ir`, defined in [`environment.yml`](environment.yml):

- Python 3.12;
- PyTorch, Transformers, and Sentence Transformers for the dense encoders, the cross-encoder, and the LLM;
- rank_bm25, scikit-learn, and NLTK for the sparse retrievers, XGBoost for learning to rank;
- NumPy, pandas, and PyArrow for the data.

The code runs on CUDA, Apple GPUs (MPS), and CPU. A GPU is recommended for the encoders and the cross-encoder, and the LLM reranker (Qwen2.5-7B-Instruct) needs about 16 GB of GPU memory. The first runs download the models from Hugging Face.

```bash
mamba env create -f environment.yml   # create the environment once
mamba activate citation-ir            # activate it in every new terminal
```

## Data

For data confidentiality reasons, the challenge data is not included in this repository. The scripts expect it in `data/` (queries, corpus, relevance judgments, and the provided MiniLM embeddings) and the held-out queries in `held_out_queries.parquet`.

## Quick start

```bash
python 01_tfidf.py && python 02_bm25.py      # sparse indexes reused by later scripts
python 04_dense_encoders.py                  # UAE-Large-V1, then once per other encoder (see docs/usage.md)
python 06_multi_fusion.py                    # base fusion of the next steps
python 07_citation_context.py
python 08_learning_to_rank.py                # cross-validated MAP of the best model
python 08_learning_to_rank.py --submit-held-out
```

The dependencies between scripts and every command are in [docs/usage.md](docs/usage.md).

## Repository layout

```
01_tfidf.py to 12_rerank_llm.py   one script per step of the results table
utils.py                          shared data loading, metrics, and submission export
docs/                             implementation and usage
```

## Documentation

- [Implementation](docs/implementation.md): each method, its features and parameters, and why it helps or fails.
- [Usage](docs/usage.md): setup, data layout, the order of the scripts, and their options.

## References

- S. Robertson and H. Zaragoza. The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 2009.
- N. Reimers and I. Gurevych. Sentence-BERT: sentence embeddings using Siamese BERT-networks. *EMNLP*, 2019.
- W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou. MiniLM: deep self-attention distillation for task-agnostic compression of pre-trained transformers. *NeurIPS*, 2020.
- S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y. Nie. C-Pack: packed resources for general Chinese embeddings. *SIGIR*, 2024.
- L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei. Text embeddings by weakly-supervised contrastive pre-training. arXiv:2212.03533, 2022.
- X. Li and J. Li. AnglE-optimized text embeddings. arXiv:2309.12871, 2023.
- M. Ostendorff, N. Rethmeier, I. Augenstein, B. Gipp, and G. Rehm. Neighborhood contrastive learning for scientific document representations with citation embeddings. *EMNLP*, 2022.
- J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu. BGE M3-Embedding: multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. *Findings of ACL*, 2024.
- T. Chen and C. Guestrin. XGBoost: a scalable tree boosting system. *KDD*, 2016.
- C. J. C. Burges. From RankNet to LambdaRank to LambdaMART: an overview. Microsoft Research Technical Report, 2010.
- Qwen Team. Qwen2.5 technical report. arXiv:2412.15115, 2024.

## Authors

Baptiste PRAS, Vladimir HERRERA-NATIVI, and Pedro BARTOLOMEI.
