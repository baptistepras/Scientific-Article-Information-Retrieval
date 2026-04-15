# Plan de présentation — Citation Retrieval Challenge

**Durée totale :** 10–13 minutes à 3 personnes  
**Métriques clés :** MAP (principale), NDCG@10, Recall@100

---

## Vue d'ensemble de la progression

| Étape | Méthode | MAP |
|-------|---------|-----|
| Baselines (fournies) | TF-IDF / BM25 / MiniLM | 0.45–0.50 |
| Score fusion | BGE + BM25 | 0.57 |
| Citation context | + BM25 full-text | 0.59 |
| **LTR** | **XGBRanker** | **0.67** |
| Tentatives de reranking | CE / LLM sur LTR | < 0.67 |

---

## Partie 1 — Introduction + baselines `~2 min` · Personne A

### Ce qu'il faut dire
- **Tâche :** pour chaque article requête (titre + résumé + texte complet), retrouver parmi 20 000 articles les 100 plus susceptibles d'être *cités*.
- **Données :** 100 requêtes avec ground truth, 20 000 documents dans le corpus.
- Les 3 baselines (TF-IDF, BM25, MiniLM dense) **ont été fournies par le prof** — les mentionner très brièvement comme point de départ, ne pas s'y attarder.
- Juste poser le fait qu'on part de 0.50 MAP et que l'objectif est d'améliorer.

### Ce qu'il ne faut pas dire
- Ne pas expliquer BM25 ou TF-IDF en détail.
- Ne pas décrire les hyperparamètres des baselines (k1, b, etc.).

---

## Partie 2 — Score fusion BGE + BM25 `~2 min` · Personne B

### Ce qu'il faut dire
- **Idée centrale :** deux signaux complémentaires — le dense capture la sémantique, le lexical capture les termes exacts rares. On les combine.
- Normalisation min-max des deux scores → somme pondérée. Formule simple : `α × BGE + (1−α) × BM25`.
- Choix du modèle : on passe de MiniLM à **BGE-large** (bien plus puissant, 1024 dims vs 384).
- Grid search sur α → 0.85 : BGE domine, BM25 apporte un petit bonus lexical.
- **Résultat : 0.50 → 0.57 MAP (+0.07), gain le plus simple du projet.**

### Ce qu'il ne faut pas dire
- Ne pas détailler le grid search ou afficher tous les résultats.
- Pas besoin de justifier la normalisation min-max techniquement.

---

## Partie 3 — Citation context mining `~1 min 30` · Personne B (ou C)

### Ce qu'il faut dire
- **Insight :** les articles scientifiques contiennent des *phrases de citation* qui décrivent directement les papiers qu'ils citent — `"As shown by [1]..."`, `"(Smith et al., 2020) demonstrated..."`.
- On extrait ces phrases via regex depuis le `full_text` de la requête, on retire les marqueurs `[1]` pour garder le texte descriptif.
- On construit un index BM25 sur le texte complet du corpus (titre + résumé + corps), et on score ces phrases de citation dessus.
- Fusion avec la base par grid search 2D.
- **Résultat : 0.57 → 0.59 MAP (+0.02)** — gain modeste car signal sparse.

### Ce qu'il ne faut pas dire
- Pas la liste complète des regex.
- Pas le détail du grid search 2D.

---

## Partie 4 — Learning to Rank (XGBRanker) `~4 min` · Personne C

C'est le **cœur de la présentation**, le saut le plus important (+0.08).

### Ce qu'il faut dire
1. **Problème avec la somme pondérée :** linéaire et fixe — elle ne peut pas capturer les interactions entre signaux (ex. : UAE score élevé *et* BM25 score élevé = bien plus fort que leur somme).
2. **Architecture LTR :** pour chaque paire (requête, document), on construit un vecteur de ~16 features :
   - 4 scores de similarité dense : UAE, BGE, E5, SciNCL
   - 4 scores sparse : BM25 TA, BM25 full-text, citation-context BM25, TF-IDF
   - Rangs réciproques de chaque système (`1/(60+rang)`)
   - Métadonnées : proximité temporelle, match de domaine
3. **Modèle :** XGBRanker avec `objective='rank:ndcg'`, arbres peu profonds (max_depth=4).
4. **Évaluation :** 5-fold GroupKFold (groups = query_id) pour éviter le data leakage — les requêtes ne se mélangent pas entre train et val.
5. **Candidats :** union des top-200 de chaque système (~600–800 candidats/requête) → le ranker reordonne ce pool.
6. **Résultat : 0.59 → 0.67 MAP (+0.08)**, meilleur score du projet.

### Ce qu'il peut être bon de montrer
- Le tableau d'importance des features si vous l'avez (montre que UAE et les rangs réciproques dominent).
- Souligner que la proximité temporelle et le match de domaine aident.

### Ce qu'il ne faut pas dire
- Pas les hyperparamètres exacts d'XGBoost (subsample, colsample, etc.).
- Pas le code en détail.

---

## Partie 5 — Ce qui n'a pas marché `~1 min 30` · Personne A (ou C)

### Cross-encoder reranking (script 33)
- Idée : réappliquer un cross-encoder fort (`bge-reranker-v2-m3`) sur le top-50 du LTR.
- **Résultat : régression sous 0.67.** Pourquoi : le LTR intègre déjà des scores CE comme features — redoubler ce signal n'apporte rien.

### LLM reranking (script 34)
- Idée : utiliser un petit LLM instruct (Qwen2.5-7B) pour scorer chaque paire (requête, candidat) de 0 à 10.
- **Résultat : régression sous 0.67.** Pourquoi : le LLM n'a pas la capacité de distinguer les subtilités de relation de citation dans un domaine spécialisé, et le LTR est déjà bien calibré.
- Mentionner que c'est appliqué uniquement sur le top-20 pour des raisons de coût.

### Ce qu'il ne faut pas dire
- Pas les détails du prompt LLM.
- Pas les valeurs exactes de gamma ou les résultats intermédiaires.

---

## Partie 6 — Conclusion `~30 sec` · Personne A ou B

- Rappeler la progression globale : **0.45 → 0.67 MAP**.
- Le saut principal vient du passage à un modèle qui apprend les interactions entre signaux (LTR), pas d'un modèle plus fort.
- Piste non explorée : fine-tuning d'un bi-encoder sur les citations du dataset.

---

## Répartition suggérée

| Personne | Parties | Durée |
|----------|---------|-------|
| A | Intro + baselines + Conclusion | ~3 min |
| B | Score fusion + Citation context | ~3–4 min |
| C | LTR (cœur) + Échecs | ~5–6 min |

**Total : ~11–13 min**
