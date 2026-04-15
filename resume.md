# Résumé des approches — Challenge de citation retrieval scientifique

**Tâche :** Pour chaque article requête (titre + résumé + texte complet), retrouver parmi 20 000 articles les 100 plus susceptibles d'être *cités* par cet article requête.
**Métriques :** NDCG@10 (primaire), MAP, Recall@100.
**Données :** 100 requêtes d'entraînement avec ground truth (`qrels.json`), corpus de 20 000 documents.

---

## 1. Baselines

### 01 — TF-IDF (`0.45 MAP`)

**Architecture :** Récupération sparse classique. On encode titre + résumé de chaque document via un `TfidfVectorizer` (sublinear TF, `min_df=1`, `max_df=0.85`). À l'inférence, on transforme la requête de la même façon et on classe les documents par similarité cosinus.

**Limite principale :** Correspondance exacte de termes uniquement. Deux papiers sur le même sujet avec un vocabulaire différent auront un score nul. Pas de compréhension sémantique.

---

### 03 — BM25 (`~0.48 MAP`)

**Architecture :** Récupération sparse améliorée avec `BM25Okapi` de `rank_bm25` sur titre + résumé (tokenisation : minuscules + ponctuation retirée + stopwords anglais filtrés). Paramètres `k1=1.0, b=1.0` choisis par grid search sur le dev set.

**Apport par rapport à TF-IDF :** BM25 sature l'impact des termes très fréquents via la fonction `(k1+1)·tf / (k1·(1-b+b·|d|/avg|d|) + tf)`, alors que TF-IDF laisse les termes dominants trop peser. Normalisation de longueur de document plus adaptée. C'est le standard industriel pour la recherche lexicale.

**Limite :** Même faiblesse que TF-IDF — aucune compréhension sémantique, uniquement correspondance exacte.

---

### 02 — Dense retrieval MiniLM (`0.50 MAP`)

**Architecture :** Récupération dense avec des embeddings pré-calculés `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalisés). La similarité cosinus se réduit à un produit scalaire. Aucun entraînement : on charge les embeddings et on classe.

**Apport par rapport à TF-IDF/BM25 :** Capture la similarité sémantique au-delà des mots exacts (+0.05 MAP). Deux phrases paraphrasées peuvent avoir un score élevé.

**Limite :** MiniLM est un modèle généraliste petit (22M params). Il n'est pas spécialisé pour la littérature scientifique, et ses embeddings 384-dim ont moins de capacité représentationnelle que des modèles plus grands.

---

## 2. Première grande amélioration — Score fusion

### 10 — BGE + BM25 score fusion (`0.57 MAP`, +0.07 vs dense baseline)

**Architecture :** Fusion de deux signaux complémentaires avec somme pondérée normalisée.

- **BM25 (lexical)** : index BM25Okapi sur titre + résumé (k1=1.0, b=1.0), avec stopwords retirés.
- **BGE-large** : `BAAI/bge-large-en-v1.5` (1024 dims). Modèle bi-encodeur beaucoup plus puissant que MiniLM. Les requêtes reçoivent un préfixe d'instruction spécifique pour la recherche.

**Formule :** `score_final = α × BGE_normalisé + (1-α) × BM25_normalisé`

Chaque signal est normalisé en [0,1] par min-max avant fusion. Un grid search sur α (0.05 à 0.95 par pas de 0.05) identifie la meilleure pondération → **α = 0.85** (BGE domine fortement).

**Pourquoi ça marche :** BGE-large est fondamentalement plus fort que MiniLM (7× plus grand, entraîné spécifiquement pour la recherche sémantique). BM25 apporte un petit bonus pour les correspondances exactes de termes techniques rares. La normalisation min-max permet de combiner deux espaces de scores hétérogènes sans biais.

---

## 3. Exploitation du texte complet — Franchissement du plateau

### 21 — Citation context + full-text BM25 (`0.59 MAP`, +0.02 vs score fusion)

**Architecture :** Exploitation du `full_text` des requêtes (totalement ignoré jusqu'ici).

**Insight clé :** Les articles citants contiennent des *phrases de citation*, c'est-à-dire des phrases qui décrivent et mentionnent les papiers cités : `"As shown by [1]"`, `"(Smith et al., 2020) demonstrated..."`. Ces phrases sont des descriptions directes des documents pertinents.

**Pipeline :**

1. **Index BM25 full-text** sur le corpus : titre + résumé + 5000 premiers caractères du corps.
2. **Extraction de contextes de citation** depuis `full_text` de la requête : regex pour détecter `[1]`, `(Author et al., 2020)`, etc. → on retire les marqueurs pour garder le texte descriptif.
3. Deux nouveaux signaux BM25 : (a) contextes de citation contre index full-text, (b) titre+résumé de la requête contre index full-text.
4. Grid search 2D sur les poids `(alpha_cite, alpha_ft)` pour fusionner avec la base (UAE+BGE+E5+TF-IDF à 0.57).

**Pourquoi ça marche (un peu) :** Les phrases de citation contiennent des mots-clés techniques précis qui matchent exactement les titres et abstracts des documents cités. C'est un signal lexical très ciblé. Gain modeste (+0.02) car BM25 reste un signal sparse.

---

## 4. Learning to Rank — Explosion du plafond

### 22 — XGBRanker LTR (`0.67 MAP`, +0.08 vs script 21 !)

**Architecture :** Remplace la somme pondérée linéaire par un modèle de ranking à gradient boosted trees qui *apprend* les interactions entre signaux.

**Features par paire (requête, document) ≈ 16 features :**

- **Scores** de similarité dense : UAE, BGE, E5, SciNCL (4 features)
- **Scores** sparses : BM25 TA, BM25 full-text, citation-context BM25, TF-IDF (4 features)
- **Rangs réciproques** de chaque système (6 features : 1/(60+rang+1))
- **Métadonnées** : proximité temporelle (1/|année_q - année_d| + 1), match de domaine (0 ou 1)

**Entraînement :**

- `XGBRanker(objective='rank:ndcg')`, max_depth=4, 200 estimateurs.
- **5-fold GroupKFold** avec groups = query_id (les 100 requêtes sont divisées en 5 groupes de 20). Évite le data leakage inter-requêtes.
- Candidats : union des top-200 de chaque système (pool ≈ 600–800 candidats/requête).

**Pourquoi c'est un saut énorme :**

- Les interactions non-linéaires entre signaux sont cruciales. Par exemple : UAE score élevé *et* BM25 score élevé = beaucoup plus fort que leur somme. XGBoost capture cela automatiquement.
- Les features de rang réciproque permettent de pondérer les signaux selon leur fiabilité relative.
- La proximité temporelle capture le fait que les articles cités sont souvent récents ou proches de l'article requête.
- Le match de domaine filtre les faux positifs interdisciplinaires.

---

### 24 — LTR + Cross-Encoder features (`≈ 0.67+ MAP`, gain marginal)

**Architecture :** Extension de 22 avec des features issues d'un cross-encoder fort (`bge-reranker-v2-m3`).

**Pipeline :**

1. Script 23 pré-calcule les scores cross-encoder sur le top-200 de la fusion de base (enrichissement de la requête avec les contextes de citation, `max_length=1024`). Les scores sont cachés dans `models/crossencoder_v2/ce_data_train.npz`.
2. Script 24 ajoute 3 features par paire (q, d) dans le LTR :
   - `ce_score` : score CE brut (0 si le candidat n'est pas dans le top-200 CE)
   - `has_ce_score` : indicateur binaire de présence dans le top CE
   - `ce_rank` : rang réciproque parmi les candidats scorés par le CE

**Pourquoi c'est marginal (+épsilon) :** Le cross-encoder apporte un signal d'interaction token-à-token query-document que les bi-encoders (UAE/BGE/E5) manquent. Mais le gain est faible car (a) le LTR de 22 était déjà quasi saturé sur les features dense+sparse, (b) le CE n'est calculé que sur le top-200 (pas sur tous les candidats), donc le signal est creux. Utile surtout pour départager les 20 premiers documents.

---

## 5. Tentatives de reranking (échecs)

### 33 — CE reranking sur LTR (`échec, score < 0.67`)

**Architecture :** Application directe d'un cross-encoder fort (`bge-reranker-v2-m3`) sur les top-K candidats du LTR+CE (script 24). Score final interpolé entre le score CE et la position LTR :

`score_final = γ × CE_normalisé + (1-γ) × RangRéciproque_LTR`

**Intuition :** La base de candidats LTR (0.67 MAP) est bien meilleure que les premiers essais de reranking. Le CE avec accès à une liste de qualité devrait pouvoir affiner le top.

**Pourquoi ça échoue :** Le LTR intègre déjà des features de cross-encoder (script 24). Appliquer un CE par-dessus ne fait que redoubler un signal déjà présent dans le modèle, sans apporter d'information nouvelle. Le LTR a déjà appris à pondérer ce signal de façon optimale.

---

### 34 — LLM reranking sur LTR (`échec, score < 0.67`)

**Architecture :** Scoring pointwise avec un petit LLM instruct (Qwen2.5-1.5B-Instruct). Le modèle donne un score de 0 à 10 pour chaque paire (requête, candidat).

**Prompt :** *"You are an expert at judging scientific citation relevance. [...] Rate how likely the candidate is cited by the query paper on a scale from 0 to 10."*

**Score extrait :** Valeur attendue sur les logits des tokens "0"–"10". Interpolé avec le rang LTR (poids γ). N'est appliqué que sur le top-20 pour des raisons de coût.

**Pourquoi ça échoue :** Qwen2.5-1.5B est trop petit pour raisonner finement sur des paires de papiers scientifiques. Le modèle n'a pas la capacité de distinguer les subtilités de la relation de citation dans un domaine académique spécialisé. La base LTR est déjà bien calibrée et un modèle de 1.5B params ne peut pas faire mieux que la combinaison de signaux appris par XGBoost.

---

## Tableau récapitulatif

| Script | Méthode | MAP | vs. précédent |
|--------|---------|-----|---------------|
| 01 | TF-IDF baseline | 0.45 | — |
| 03 | BM25 baseline | ~0.48 | +0.03 |
| 02 | Dense MiniLM baseline | 0.50 | +0.02 |
| 10 | Score fusion BGE+BM25 | 0.57 | +0.07 |
| 21 | Citation context BM25 | 0.59 | +0.02 |
| 22 | LTR XGBRanker | 0.67 | **+0.08** |
| 24 | LTR + CE features | ≈ 0.67+ | +marginal |
| 33 | CE reranking sur LTR | < 0.67 | **régresse** (échec) |
| 34 | LLM reranking sur LTR | < 0.67 | **régresse** (échec) |
