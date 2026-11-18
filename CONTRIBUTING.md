# Contribuer

## Branches et PR
- `main` ne peut être ni réécrit (force-push) ni supprimé.
- Merge commit (titre et description de la PR) : le tracé des branches reste dans l'historique ;
  la branche est supprimée sur GitHub après le merge.

## Écrire du code
Les noms et les types disent *quoi*, la docstring dit le *contrat*, le commentaire dit *pourquoi
c'est fait comme ça*, et l'outillage vérifie que tout reste vrai.

| Pratique | Concrètement | Vérifié par |
|---|---|---|
| Types sur toute fonction publique | Paramètres et retour annotés (`def build(split: str) -> Subset`) ; inutile sur les variables locales | ruff `ANN001`, `ANN201` |
| Une docstring par module et par classe | À quoi sert le fichier, quand l'utiliser, ses pièges | ruff `D100`, `D101` |
| Docstring de fonction seulement si nom et types ne suffisent pas | Le contrat : format des tenseurs (forme, plage, convention), effets de bord (disque, réseau), invariants, erreurs levées. Une ligne suffit souvent ; pas de `Args:` / `Returns:` qui répètent les types | relecture |
| Commentaires rares | Une décision ou un piège (`# graine unique par (split, index)`), jamais une paraphrase du code ; pas de code commenté (git s'en souvient) ; un commentaire faux est corrigé dans le même commit | relecture |
| Pas de `TODO` orphelin | Ouvrir une issue `task` plutôt | relecture |
| Une seule langue : le français | Code, docstrings, commentaires, doc et messages de commit | relecture |
| ASCII seulement dans le code | `.py`, configs YAML, `pyproject.toml` : ni accents ni symboles (`->`, `<=`, pas `→`, `≤`). Sous Windows, une sortie redirigée (fichier, pipe, job) est encodée en cp1252, et un `print` non ASCII fait planter la commande. La doc (`.md`) et les templates GitHub gardent leurs accents | relecture |
| Format et lint | Formatage, imports triés, erreurs probables | ruff (pre-commit, CI `Lint`) |
| Chaque chose à sa place | Architecture d'ensemble dans `docs/` ; pourquoi d'un changement dans le message de commit ; exemples d'usage dans les tests (documentation exécutable, qui ne peut pas devenir fausse sans que la CI le voie) | — |

## Avant de pousser
La CI refait ces vérifications, mais un échec y coûte un aller-retour : mieux vaut le voir en local.

| Quand | Commande | Pourquoi |
|---|---|---|
| Une fois par clone | `uv run pre-commit install` | Format, lint et recherche de secrets à chaque commit (dépôt public : un token HF commité est exposé). Sans lui, la CI `Lint` échoue. |
| Code modifié | `uv run pytest -m "not slow"` | Tests unitaires (~2 min) : contrat Backbone / Neck / Head, splits, noms du Hub… |
| Chaîne `lo`, configs ou dépendances modifiées | `uv run pytest -m slow` | Smoke de bout en bout (pretrain → train → eval → mlflow sync) sur CPU : une chaîne cassée se voit ici, pas au bout d'un job de 14 h sur le DCE. |

Doc seule : rien à lancer, le hook suffit. Sur GitHub, `Lint` tourne à chaque PR ; tests et
smoke (`CI`) seulement si `src/`, `configs/`, `tests/` ou les dépendances changent.

## Expériences
- Une expérience = un fichier dans `configs/experiment/`, mergé sur `main`.
- Pas de gros run sans carte d'expérience : une issue ouverte avec son hypothèse, complétée au fil
  de l'eau (noms des runs, résultats, conclusion), fermée une fois les runs publiés
  (`lo hub push`).
- Smoke test vert avant tout job long : la CI le fait sur `main`, sinon `uv run pytest -m slow`.
- Pas de job long lancé avec un arbre git non commité.
- Commit poussé avant tout run : le run enregistre son commit, qui doit être retrouvable par
  tous. Sinon, `lo pretrain|train|eval` demande s'il faut continuer quand même (y/n).

## Ce qui n'entre jamais dans git
PDF, datasets, checkpoints, exports ONNX. Les poids vont sur Hugging Face Hub.
