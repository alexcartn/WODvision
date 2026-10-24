# WODVision : analyse vidéo de mouvements CrossFit

## 0. Pivot rameur (octobre 2026)

Le périmètre actif est désormais l'analyse de la technique d'aviron sur ergomètre Concept2, en vue de profil. Le squat (sections 3, 7.5, 7.6, 7.8 à 7.10) est mis en pause, pas supprimé. Tout le reste (contraintes, stack, licences, conventions, cache d'inférence) s'applique inchangé.

- Entrée : vidéos MP4, AVI, MOV (smartphone ou clips YouTube), caméra fixe ou mobile (mouvement de caméra compensé).
- Interface : application Streamlit locale (`app/streamlit_app.py`), vidéo composite (athlète annoté, squelette seul, indicateurs en direct) + constats.
- CLI : `wodvision row-analyze <video> [--click-x --click-y] [--from-cache trace.npz]`.
- Code : `analysis/rowing/` (repère machine, coups, métriques, règles), `stabilization/camera_motion.py`, `rowing_pipeline.py`, `viz/rowing_dashboard.py`.
- Vue non profil : mode dégradé explicite (modes full / degraded / minimal), angles absolus masqués, mesures relatives conservées et marquées indicatives (`docs/rowing_concept.md` section 3 bis).
- Synthèse pédagogique : niveaux Débutant / Coach, pastilles par critère, priorités avec conseils (`configs/conseils.yaml`, éditable par un coach), images clés et ralentis du meilleur coup et du coup à améliorer (`coaching.py`, `viz/rowing_highlights.py`, `docs/rowing_concept.md` section 3 quater).
- Mode fantôme : posture de référence (modèle Concept2 mis à l'échelle du rameur) superposée en vue de profil (`analysis/rowing/ghost.py`, `docs/rowing_concept.md` section 3 ter).
- Nom du service : Pimp my row (références à l'émission Pimp my ride). Le paquet et la CLI gardent le nom `wodvision`.
- Fiche PDF récapitulative : `report_pdf.py` (reportlab, BSD), `viz/rowing_curves.py` (courbes de poignée, genou, buste superposées coup par coup, moyenne et écart-type), schéma des phases `viz/rowing_diagram.py` ; CLI écrit `fiche.pdf` et `fiche_coach.pdf`. Ratio récupération/drive dépendant de la cadence (`rowing.ratio_targets`, hypothèses à valider).
- Version en ligne gratuite : profil `configs/hosted.yaml`, démo instantanée, Space Hugging Face (`docs/deploiement_huggingface.md`).
- Règles de technique : explicites, chacune cite métrique, valeur et seuil ; seuils dans `configs/default.yaml` section `rowing`, hypothèses à calibrer (voir `docs/rowing_concept.md`).
- Couplage PM5 : `pm5/` (parseurs BLE d'après la spec Concept2, session, enregistreur `bleak` optionnel, extra `pm5`), `analysis/rowing/pm5_sync.py` (appariement des coups, décalage d'horloge, validation) et `force_curve.py` ; CLI `pm5-record`, `pm5-info`, `row-analyze --pm5`. Dispositions d'octets testées par aller-retour, pas encore vérifiées sur un vrai PM5.
- Phase suivante : test sur matériel réel (enregistrement PM5 + vidéo de profil), calibration des seuils.
- Biomécanique, règles, limites et sources : `docs/rowing_concept.md`.

## 1. Contexte et objectif

Projet personnel de R&D. Outil Python qui analyse une vidéo (fichier ou flux smartphone) d'un athlète CrossFit et produit :

- une vidéo annotée (squelette, trajectoire de barre, compteur de reps, statut de validité),
- des métriques par rep (CSV / JSON),
- un résumé de séance.

Objectif long terme : suivre les progrès dans le temps et fournir des recommandations basées sur des règles explicites et traçables.

Profil du porteur : ingénieur (Python, traitement du signal, model-based design), coach CrossFit. Inutile d'expliquer les bases du filtrage ou de la dérivation ; expliquer en revanche les choix propres à la computer vision.

## 2. Contraintes non négociables

- Coût : outils gratuits et open source uniquement. Aucune plateforme SaaS payante (pas de Roboflow platform, pas d'API cloud). Entraînement éventuel en local ou sur GPU gratuits (Google Colab, Kaggle).
- Licences : privilégier Apache 2.0, MIT, BSD.
  - Ultralytics (AGPL-3.0) : accepté seulement comme backend optionnel isolé dans un extra, jamais comme dépendance obligatoire.
  - RF-DETR : uniquement les modèles sous Apache 2.0 (Nano à Large). Exclure RF-DETR-XL / 2XL et le paquet `rfdetr_plus` (licence PML 1.0).
- Source vidéo : n'importe quel smartphone (Android ou iOS). Aucune dépendance à un SDK constructeur.
- Exécution : doit tourner sur le CPU d'un laptop (plus lent que le temps réel accepté en offline). GPU CUDA utilisé automatiquement s'il est présent.
- Python 3.11+.
- Offline d'abord, temps réel plus tard.

## 3. Périmètre MVP (Phase 1)

Mouvement : back squat et front squat, filmés de profil, caméra fixe.

Sources : segments de vidéos YouTube sélectionnés, puis vidéos personnelles smartphone.

Sorties par rep :

| Champ | Unité |
|---|---|
| rep_index | - |
| t_start, t_bottom, t_end | s |
| depth_status | ok / no / uncertain |
| knee_angle_min, hip_angle_min | deg |
| mean_concentric_velocity (MCV) | m/s |
| peak_velocity (PV) | m/s |
| vertical_displacement | m |
| bar_horizontal_deviation_max | mm |
| eccentric_duration, concentric_duration | s |
| quality_score | 0 à 1 |

Résumé de séance : nombre de reps, reps non valides, perte de vitesse (%), indicateurs de qualité globaux.

Hors périmètre Phase 1 : haltérophilie (snatch, clean), multi-mouvements, temps réel, application mobile, recommandations, plusieurs athlètes analysés simultanément.

## 4. Architecture du pipeline

```
VideoSource (fichier / flux)
   -> FrameReader (timestamps PTS réels)
   -> branche athlète : PersonDetector -> Tracker -> AthleteSelector -> PoseEstimator
   -> branche barre   : BarDetector / BarTracker
   -> Calibration (échelle px -> m)
   -> Cache disque (parquet)            <- fin de la partie "inférence"
   -> SignalProcessing (rééchantillonnage, filtrage, dérivées)
   -> RepSegmenter
   -> MovementAnalyzer (squat)
   -> Metrics
   -> Exporters (CSV, JSON, parquet) + VideoAnnotator + Plots
```

Principe clé : séparer l'inférence (lourde, mise en cache) de l'analyse (légère, itérée souvent). Toute la partie analyse doit pouvoir tourner uniquement à partir du cache, sans relancer les modèles.

## 5. Stack technique

| Besoin | Choix par défaut | Alternative | Licence |
|---|---|---|---|
| Environnement | uv | venv + pip | MIT / Apache 2.0 |
| Lecture vidéo et timestamps | PyAV | OpenCV | BSD |
| Image, dessin, trackers classiques | opencv-contrib-python | - | Apache 2.0 |
| Pose 2D | rtmlib (RTMPose / RTMO via ONNX Runtime) | MediaPipe Pose Landmarker (Tasks API) | Apache 2.0 |
| Pose (optionnel, extra `agpl`) | Ultralytics YOLO pose | - | AGPL-3.0 |
| Détection personne | détecteur fourni par rtmlib | RF-DETR classe person | Apache 2.0 |
| Tracking multi-objets | supervision (ByteTrack) | - | MIT |
| Barre Phase 1 | méthode classique OpenCV (cercles / ellipses) | - | Apache 2.0 |
| Barre Phase 2 | RF-DETR Nano ou Small fine-tuné | - | Apache 2.0 |
| Annotation de données | CVAT ou Label Studio auto-hébergé | - | MIT / Apache 2.0 |
| Signal | numpy, scipy | - | BSD |
| Données | pandas, pyarrow | - | BSD / Apache 2.0 |
| CLI et config | typer, pyyaml, pydantic | - | MIT |
| Graphiques | matplotlib | plotly | PSF / MIT |
| Dashboard (Phase 4) | Streamlit + SQLite | - | Apache 2.0 |
| Téléchargement / découpe | yt-dlp, ffmpeg | - | Unlicense / LGPL |
| Qualité | ruff, pytest | mypy | MIT |

Note : `supervision` et `rfdetr` sont des bibliothèques open source utilisables sans compte ni plateforme Roboflow.

Important : vérifier la version et l'API actuelles de rtmlib, mediapipe, rfdetr et supervision avant d'écrire du code. Ces API évoluent, ne pas se fier à des exemples anciens.

## 6. Structure du dépôt

```
wodvision/
  CLAUDE.md
  README.md
  pyproject.toml
  configs/
    default.yaml
  datasets/
    manifest.yaml            # description des vidéos de test + vérité terrain
  data/                      # gitignored
    raw/
    clips/
    cache/
    runs/
  src/wodvision/
    __init__.py
    cli.py
    config.py
    types.py                 # dataclasses partagées
    pipeline.py
    io/
      video_source.py        # FileSource, StreamSource, WebcamSource
      probe.py
      youtube.py
      export.py
      cache.py
    detection/
      person.py
      barbell.py
    tracking/
      tracker.py
      athlete_selector.py
    pose/
      base.py                # interface PoseEstimator
      rtmlib_backend.py
      mediapipe_backend.py
      ultralytics_backend.py # extra optionnel
    calibration/
      plate_scale.py
    signal/
      resampling.py
      filtering.py
      kinematics.py
    analysis/
      reps.py
      angles.py
      squat.py
      metrics.py
    viz/
      annotate.py
      plots.py
  scripts/
    evaluate.py
  tests/
    synthetic/               # générateurs de trajectoires synthétiques
  notebooks/
  app/                       # Streamlit, Phase 4
```

## 7. Spécifications des modules

### 7.1 Entrée vidéo (`io/`)

- Interface `VideoSource` itérable qui renvoie `Frame(index: int, pts_s: float, image_bgr: np.ndarray)`.
- Implémentations :
  - `FileSource` : mp4 / mov, H.264 et HEVC.
  - `StreamSource` : URL RTSP / HTTP / MJPEG lue par OpenCV (applis smartphone gratuites de type IP Webcam, DroidCam, Larix, à vérifier).
  - `WebcamSource`.
- Timestamps : utiliser les PTS réels via PyAV. Les smartphones, surtout sous Android, enregistrent souvent en fréquence d'images variable (VFR). Ne jamais calculer le temps à partir de index / fps.
- Rotation : appliquer la métadonnée de rotation (vidéos tournées en portrait).
- Ralenti : une vidéo slow-motion fausse toutes les vitesses. Détecter via les métadonnées si possible, sinon lire `slowmo_factor` dans le manifeste.
- `probe(path)` renvoie : résolution, fps nominal, VFR oui / non (dispersion des intervalles PTS), durée, rotation, codec.

### 7.2 Données YouTube (`io/youtube.py`, `datasets/manifest.yaml`)

- Téléchargement avec yt-dlp, meilleure qualité mp4 disponible, stockage dans `data/raw/`.
- Découpe des segments exploitables avec ffmpeg en réencodant (découpe précise à la frame, pas de `-c copy`).
- Format d'une entrée du manifeste :

```yaml
- id: squat_yt_001
  url: https://www.youtube.com/watch?v=XXXX
  movement: back_squat
  view: side_right          # side_left | side_right | front | oblique
  segments: [[12.0, 48.5]]  # secondes
  slowmo_factor: 1.0
  camera_static: true
  plate_visible: true
  plate_diameter_mm: 450
  load_kg: null
  expected_reps: 5          # vérité terrain comptée à la main
  expected_depth: [ok, ok, ok, no, ok]
  notes: "coupure caméra à 50 s"
```

- Critères de sélection : caméra fixe, vue de profil, disque visible, pas de montage ni de ralenti dans le segment, 30 fps minimum.
- Usage strictement personnel de R&D : pas de redistribution, aucune vidéo commitée.

### 7.3 Personne et athlète (`detection/person.py`, `tracking/`)

- Détection de personnes puis tracking ByteTrack (supervision).
- Sélection de l'athlète, par ordre de priorité :
  1. option CLI `--athlete-click` : clic sur l'athlète dans la première frame,
  2. personne dont les poignets sont les plus proches de la barre détectée,
  3. plus grande bounding box moyenne.
- Conserver un seul `track_id`. Occultations courtes : interpolation jusqu'à N frames (configurable), au-delà marquer les données comme manquantes.

### 7.4 Pose (`pose/`)

- Interface : `PoseEstimator.infer(image, bbox) -> Keypoints(xy: ndarray[K, 2], score: ndarray[K])`, format COCO-17.
- Backends : rtmlib par défaut (mode configurable léger / équilibré / performance), MediaPipe mappé vers COCO-17, Ultralytics optionnel dans l'extra `agpl`.
- Côté visible : en vue de profil, choisir le côté gauche ou droit ayant le meilleur score moyen sur toute la vidéo, puis s'y tenir.
- Points utiles pour le squat : épaule, hanche, genou, cheville du côté visible.
- Livrable annexe : script de benchmark rtmlib vs MediaPipe (précision visuelle genou / hanche en vue de profil, fps CPU).

### 7.5 Barre (`detection/barbell.py`)

Phase 1, approche classique sans entraînement :

1. Initialisation : recherche automatique du plus grand cercle proche des poignets (`cv2.HoughCircles`). En cas d'échec, clic utilisateur sur le centre du disque.
2. Suivi : prédiction de position (vitesse constante) puis recherche locale dans une ROI ; tracker CSRT en secours.
3. Raffinement : fit d'ellipse sur les contours (`cv2.fitEllipse`) avec RANSAC pour résister aux occultations par les mains et le corps.
4. Sortie par frame : `BarObservation(center_px, major_axis_px, minor_axis_px, angle_deg, confidence)`.

Phase 2 : détecteur RF-DETR (Nano ou Small) fine-tuné sur disques et barre, annoté dans CVAT, entraîné en local ou sur Colab. Même interface de sortie.

### 7.6 Calibration (`calibration/plate_scale.py`)

- Échelle : `m_per_px = plate_diameter_m / major_axis_px`, calculée sur une médiane glissante robuste.
- Contrôle qualité : ratio `minor_axis / major_axis`. Si inférieur à 0,9 (configurable), avertir que la caméra n'est pas perpendiculaire au plan du mouvement : vitesses et déviation horizontale dégradées.
- Diamètre par défaut 450 mm (bumper standard), paramétrable car les disques techniques légers sont plus petits.
- Fallback sans disque exploitable : taille de l'athlète saisie en CLI, résultat marqué comme moins précis.

### 7.7 Traitement du signal (`signal/`)

- Rééchantillonnage sur une grille temporelle uniforme à partir des PTS (interpolation linéaire), avec masque des points interpolés.
- Rejet des points à faible confiance, comblement des trous courts uniquement.
- Filtrage : Butterworth passe-bas ordre 4 à phase nulle (`scipy.signal.butter` + `filtfilt`), fréquence de coupure configurable (défaut 6 Hz pour le squat). Savitzky-Golay en alternative. Le rééchantillonnage uniforme est obligatoire avant filtrage.
- Dérivées calculées sur le signal filtré (`np.gradient`).
- Repère interne : mètres, axe y orienté vers le haut.

### 7.8 Segmentation des reps (`analysis/reps.py`)

- Signal d'entrée : position verticale filtrée de la barre.
- `scipy.signal.find_peaks` sur les hauts et les bas, avec prominence minimale exprimée en fraction du diamètre du disque (défaut 0,5) et distance temporelle minimale entre reps.
- Machine à états : STANDING -> ECCENTRIC -> BOTTOM -> CONCENTRIC -> LOCKOUT.
- Rejet des cycles incomplets en début et fin de segment et des mouvements partiels (réajustement de barre, rebond).

### 7.9 Analyse squat (`analysis/angles.py`, `analysis/squat.py`)

- Angles 2D : genou (hanche, genou, cheville), hanche (épaule, hanche, genou), inclinaison du tronc par rapport à la verticale.
- Profondeur : le standard est le pli de hanche sous le haut du genou. Le keypoint hanche est un centre articulaire, pas le pli. Utiliser la comparaison hanche_y vs genou_y avec une marge configurable et trois états : `ok`, `no`, `uncertain` quand l'écart tombe dans la marge.
- Lockout : angles genou et hanche au-dessus d'un seuil configurable (défaut 165°).

### 7.10 Métriques (`analysis/metrics.py`)

- MCV = déplacement vertical concentrique / durée concentrique.
- PV = vitesse verticale maximale pendant la phase concentrique.
- Déviation horizontale de barre = amplitude max en x pendant la rep (mm).
- Perte de vitesse = (MCV meilleure rep - MCV dernière rep) / MCV meilleure rep.
- Chaque métrique est accompagnée d'un `quality_score` combinant confiance moyenne des détections, pourcentage de points interpolés et ratio d'ellipse.

### 7.11 Sorties (`io/export.py`, `viz/`)

- Vidéo annotée : squelette côté visible, traîne de trajectoire de barre colorée selon la vitesse, compteur de reps, MCV de la dernière rep, statut de profondeur.
- `reps.csv`, `summary.json`, séries brutes et filtrées en parquet.
- Graphique position et vitesse de barre en fonction du temps avec les reps surlignées.
- Chaque run écrit dans `data/runs/<video_id>/<horodatage>/` avec une copie de la config utilisée.

## 8. Validation

Jeu de test Phase 1 : au moins 10 segments YouTube et 3 vidéos personnelles.

| Critère | Cible |
|---|---|
| Comptage de reps | erreur nulle sur au moins 90 % des segments |
| Profondeur | accord avec l'annotation manuelle d'au moins 85 % (hors incertains) |
| Taux de `uncertain` | inférieur à 20 % |
| Suivi de barre | détection valide sur au moins 95 % des frames des segments propres |
| Vitesses | plausibles (MCV squat typiquement entre 0,2 et 1,2 m/s), aucun pic aberrant |

- `scripts/evaluate.py` compare les sorties au manifeste et produit un rapport.
- Tests unitaires sur données synthétiques : trajectoire de barre connue + bruit + frames manquantes + VFR simulé. Ils valident signal, segmentation et métriques sans aucune vidéo réelle.
- Validation de l'échelle sur vidéo personnelle : comparer le déplacement mesuré à une amplitude mesurée au mètre.

## 9. Roadmap

- Phase 0 : setup. uv, structure du dépôt, ruff + pytest, `probe`, téléchargement YouTube, manifeste initial.
- Phase 1 : squat de profil, offline (sections 3 à 8).
- Phase 2 : haltérophilie (clean, snatch, jerk). Trajectoire, vitesse de pic, détecteur de disques RF-DETR. Exige au moins 60 fps, idéalement 120 à 240 fps.
- Phase 3 : analyse de WOD. Classification de mouvement par règles sur la pose, puis petit classifieur scikit-learn sur séquences de keypoints. Comptage multi-mouvements (air squat, wall ball, thruster, burpee, pull-up, box jump), courbe de pacing, dégradation technique liée à la fatigue (dérive des métriques par rapport aux premières reps).
- Phase 4 : flux smartphone en temps réel (`StreamSource`, traitement en fenêtre glissante, mesure de latence) et dashboard Streamlit d'historique des séances (SQLite).
- Phase 5 : moteur de recommandations à règles explicites. Chaque recommandation cite la métrique, la valeur et le seuil qui la déclenchent.

## 10. Conventions de code

- Layout `src/`, type hints partout, docstrings courtes.
- Dataclasses partagées dans `types.py` : `Frame`, `Keypoints`, `BarObservation`, `RepMetrics`, `SessionSummary`.
- Un seul fichier de config YAML (`configs/default.yaml`), surchargeable en CLI. Aucun seuil codé en dur.
- Module `logging`, pas de `print`.
- Cache d'inférence indexé par hash (fichier vidéo + backend + version du modèle + paramètres).
- Dépendances lourdes en extras : `mediapipe`, `agpl`, `train`, `app`.
- Unités SI en interne (s, m, m/s), angles en degrés.

## 11. Commandes CLI cibles

```
wodvision probe data/raw/video.mp4
wodvision fetch --manifest datasets/manifest.yaml
wodvision analyze data/clips/squat_yt_001.mp4 --movement back_squat --view side
wodvision analyze --from-cache data/runs/squat_yt_001/<horodatage>
wodvision evaluate --manifest datasets/manifest.yaml
```

## 12. Consignes pour Claude Code

- Avant chaque phase ou module important, proposer un plan court et attendre validation.
- Vérifier les versions et API actuelles des bibliothèques avant de les utiliser.
- Signaler explicitement toute dépendance à licence copyleft ou à usage payant avant de l'ajouter.
- Écrire les tests synthétiques du module signal avant de brancher la vidéo réelle.
- Préférer une solution simple et mesurable à une solution sophistiquée. Signaler clairement les limites de la vision 2D.
- Chaque étape livrée doit être visible via une commande CLI (vidéo annotée ou graphique).
- Ne jamais committer de vidéos, de poids de modèles ni de données personnelles.

## 13. Points ouverts

- Backend de pose définitif après benchmark rtmlib vs MediaPipe.
- Initialisation de la barre quand plusieurs barres sont dans le champ (contexte box).
- Conformité RGPD si des vidéos d'adhérents sont utilisées (Phase 4 et au-delà).
