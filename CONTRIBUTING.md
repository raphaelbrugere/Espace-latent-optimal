# Contribuer

## Branches et PR
- `main` est protégé : PR obligatoire, au moins une relecture, CI verte.
- Branches courtes : `feat/<nom>` pour le code, `exp/<nom>` pour une expérience.
- Squash merge.
- Toute PR qui touche le contrat Backbone / Head est relue par tous les propriétaires de workstream.

## Expériences
- Une expérience = un fichier dans `configs/experiment/`, mergé sur `main`.
- Pas de gros run sans carte d'expérience (issue : hypothèse, config, lien W&B, conclusion).
- Smoke test (`experiment=smoke`) avant tout job long.
- Pas de job long lancé avec un arbre git non commité.

## Ce qui n'entre jamais dans git
PDF, datasets, checkpoints, exports ONNX. Les poids vont sur Hugging Face Hub.
