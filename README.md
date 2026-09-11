# WODVision

Analyse video de mouvements CrossFit (squat, Phase 1) : squelette, trajectoire de
barre, comptage de reps, metriques par rep. Voir `CLAUDE.md` pour les
specifications completes et la roadmap.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Commandes

```
wodvision probe data/raw/video.mp4
```

## Tests

```
pytest
```
