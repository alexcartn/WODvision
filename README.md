# WODVision

Analyse video de mouvements CrossFit : squelette, trajectoire de barre,
comptage de reps, metriques par rep. Voir `CLAUDE.md` pour les specifications
completes, les contraintes (licences, cout, offline-first) et la roadmap
detaillee (Phase 0 a 5). Ce README documente l'etat reel du depot, pas la
cible finale.

## Pivot rameur (octobre 2026)

Focus actuel : technique d'aviron sur Concept2 (voir `docs/rowing_concept.md`).

```
streamlit run app/streamlit_app.py                      # application locale
wodvision row-analyze data/clip.mp4 --click-x 800 --click-y 450
wodvision row-analyze x --from-cache data/cache/rowing_<hash>.npz   # sans relancer les modeles
```

Version en ligne gratuite (Hugging Face Spaces) : `docs/deploiement_huggingface.md`. Démo instantanée
(exemples simulés) et analyse de vidéos de 30 s maximum.

Couplage PM5 : `wodvision pm5-record` (extra `pip install -e ".[pm5]"`) puis
`wodvision row-analyze video.mp4 --pm5 data/pm5/seance.jsonl`. Détails et limites : `docs/rowing_concept.md` section 5.

Sorties (`data/runs/<video>/<horodatage>/`) : `dashboard.mp4`, `strokes.csv`, `summary.json`, `config.json`.
Tests : `pytest` (rameur synthetique, mouvement de camera simule, aucune video reelle requise).

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Dependances externes (hors pip) : ffmpeg et yt-dlp, pour `io/youtube.py` et
le decoupage de clips. Voir la section 5 de `CLAUDE.md` pour le detail de la
stack et des licences.

## Etat actuel (voir aussi CLAUDE.md section 9 "Roadmap")

**Fait, dans `src/wodvision/` (package reutilisable, teste) :**

- `io/probe.py`, `io/frames.py` : inspection video (PTS reels, VFR, rotation),
  extraction de frames a un instant ou sur une fenetre.
- `pose/` : interface `PoseEstimator` + backend rtmlib (RTMPose, COCO-17),
  detection CPU/CUDA automatique.
- `detection/person.py` : detecteur de personnes (YOLOX fourni par rtmlib).
- `tracking/` : wrapper ByteTrack (`Tracker`), selection d'athlete par clic
  ou plus grande bbox moyenne (`athlete_selector.py`), et re-accrochage
  "collant" d'une cible unique a travers les changements d'ID de ByteTrack
  (`sticky_target.py` — guard par ratio d'aire + seuil d'IoU minimal, voir
  ses docstrings pour le contexte du bug qu'il corrige).
- `signal/` : reechantillonnage uniforme (VFR-safe), filtre Butterworth
  zero-phase, derivees cinematiques.
- `analysis/` : angles articulaires COCO-17 (`angles.py`, y compris
  inclinaison de tronc/tibia par rapport a la verticale), segmentation de
  reps specifique squat (`reps.py`), et un compteur de cycles generique +
  detection des points bas entre sommets (`cycle_counter.py`) reutilisable
  hors squat.
- Tests unitaires synthetiques pour tout ce qui precede (`pytest`, 46 tests,
  aucune video requise pour les faire passer).
- CLI (`wodvision probe|pose-preview|track-preview|movement-count`) : chaque
  brique ci-dessus est exposee et visuellement verifiable en CLI.

**Fait, hors package (`scripts/`, ad hoc, pas dans le perimetre Phase 1) :**

Le porteur a demande d'explorer row + thruster (pas squat) sur des videos
personnelles pour iterer plus vite sur le suivi/l'affichage avant d'attaquer
le squat "officiel". Deux scripts autonomes, non generiques (chemins et
segments codes en dur pour une video precise, a adapter avant reemploi) :

- `scripts/export_performance.py` : suit un athlete (clic + `StickyTarget`),
  dessine son squelette seul (aucune boite sur les autres personnes), calcule
  des metriques par rep (timing de phase montante/descendante, angles cles,
  vitesse et deviation de trajectoire du poignet comme proxy de
  barre/poignee — pas de detection de barre reelle), exporte une video
  annotee + un `data.json`.
- `scripts/export_video_dashboard.py` : recompose `data.json` + les clips
  annotes en une seule video mp4 partageable (video + squelette seul recadre
  + graphique en direct + stats), sans dependre d'un artifact HTML.

Ces deux scripts ont aussi servi a produire un artifact HTML interactif
(video + squelette seul + graphiques + histogrammes synchronises sur la
lecture), publie hors depot (les artifacts Claude ne sont pas versionnes
ici).

**Pas fait — prochaine etape naturelle (squat, perimetre Phase 1 officiel) :**

- `detection/barbell.py` (section 7.5) : detection de la barre par cercles/
  ellipses (Hough + fit d'ellipse + RANSAC), pas encore commence.
- `calibration/plate_scale.py` (section 7.6) : echelle px -> m via le
  diametre du disque, controle qualite par ratio d'ellipse.
- `analysis/squat.py`, extension `analysis/metrics.py` (sections 7.9-7.10) :
  profondeur (hanche vs genou), lockout, MCV/PV, perte de vitesse,
  quality_score. `analysis/angles.py` et `analysis/reps.py` existent deja et
  sont directement reutilisables ici.
- `viz/annotate.py`, `viz/plots.py`, `io/export.py` (section 7.11) : video
  annotee finale (squelette + trajectoire barre + compteur + statut),
  `reps.csv`/`summary.json`, graphiques position/vitesse.
- `datasets/manifest.yaml` est vide : aucun clip squat (YouTube ou perso)
  n'est encore reference. `io/youtube.py` (fetch + decoupe ffmpeg) n'existe
  pas encore non plus.
- `scripts/evaluate.py` (section 8) n'existe pas encore.

## Commandes CLI disponibles aujourd'hui

```
wodvision probe <video.mp4>
wodvision pose-preview <video.mp4> --time 5.0
wodvision track-preview <video.mp4> --start 0 --duration 5
wodvision movement-count <video.mp4> thruster --start 260 --duration 20 --click-x 355 --click-y 160
```

Scripts ad hoc (a adapter : chemins de video et fenetres temporelles sont
codes en dur en tete de fichier) :

```
python scripts/export_performance.py
python scripts/export_video_dashboard.py
```

## Tests

```
pytest
ruff check src tests scripts
```

## Donnees

`data/` est gitignore (jamais de video, poids de modele ou donnee
personnelle commitee — voir CLAUDE.md section 12). Pour retravailler sur ce
depot dans une nouvelle session, il faut donc re-deposer les videos de
travail dans `data/raw/` avant de relancer les scripts ci-dessus.
