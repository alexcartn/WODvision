# WODVision Rameur : concept, biomécanique et règles

Pivot d'octobre 2026 : l'outil se concentre sur l'analyse de la technique d'aviron sur ergomètre Concept2.
Cible : un rameur se filme (profil), dépose la vidéo (MP4, AVI, MOV) dans une application Streamlit locale,
et obtient cadence, ratio drive/récupération, angles, séquence jambes/dos/bras et alertes de technique.
Phase suivante : coupler les données du PM5 (force, drive length, cadence) à la vision.

## 1. Ce qu'est un bon coup de rame (ergomètre)

Sources principales : Concept2 (pages technique et blog), revue BJSM 2021, Kleshnev (sur l'eau).
Les consignes Concept2 sont qualitatives : les seuils numériques ci-dessous sont des hypothèses à calibrer.

- Séquence du drive : jambes, puis dos, puis bras. Récupération : bras, buste, puis jambes.
- Passage jambes vers dos : vers 80 % de l'extension des jambes (blog Concept2) ou vers la moitié (manuel).
  Les deux sources divergent : seuil configurable (`min_leg_fraction_at_trunk_onset`).
- Catch : bras tendus, épaules basses, tibias verticaux sans dépasser la verticale, poignée à hauteur de genoux
  sans plonger.
- Finish : jambes tendues, léger recul du buste (environ 25° derrière la verticale selon le manuel), poignée sous
  les côtes, poignets plats. Angle de hanche au finish d'environ 100°.
- Récupération : bras tendus, buste incliné, puis flexion des genoux une fois les mains passées au-dessus.
- Rythme : récupération environ deux fois plus longue que le drive.
- Chaîne : trajet de la poignée en ligne droite vers le volant, chaîne horizontale (parallèle au rail).
  Tirer vers le haut ou laisser plonger la poignée dégrade la transmission de force.
- Risque lombaire : la flexion lombaire au catch augmente avec la fatigue, davantage sur ergo que sur l'eau
  (BJSM 2021). D'où l'indicateur de dérive de l'inclinaison du tronc au catch.

## 2. Repère machine (robuste au mouvement de caméra)

- Mouvement de caméra : flot optique sur le fond (hors athlète) puis similitude RANSAC, cumulée vers la première
  frame (`stabilization/camera_motion.py`). Les keypoints stabilisés sont comparables dans le temps.
- Axe du rail : ACP de la trajectoire de la hanche (le siège coulisse à hauteur constante). L'horizontale de
  l'image n'est jamais utilisée : un roulis de caméra ne change pas les angles (testé de 0 à 90°).
- Coordonnées machine (s, h) : s le long du rail (volant côté s négatif), h perpendiculaire vers le haut.
- Sens de l'axe : déduit des données (le drive est plus court que la récupération).
- Hauteurs normalisées par la longueur du tronc : aucune calibration métrique nécessaire.

## 3. Règles (`analysis/rowing/rules.py`)

Un constat est levé si au moins 30 % des coups valides violent la règle. Chaque constat cite la métrique,
la valeur médiane et le seuil. Seuils dans `configs/default.yaml`, section `rowing`.

| Code | Faute | Métrique | Seuil par défaut |
|---|---|---|---|
| trunk_early | dos qui s'ouvre trop tôt | extension des jambes quand le dos démarre | < 0,4 |
| arms_early | bras qui tirent trop tôt | extension des jambes quand les bras démarrent | < 0,75 |
| shin_beyond_vertical | tibias au-delà de la verticale au catch | angle du tibia | < -5° |
| layback_excess / layback_missing | recul du buste au finish | inclinaison du tronc | > 35° ou < 5° |
| handle_drop | poignée qui plonge au catch | hauteur poignée vs genou (longueurs de tronc) | < -0,25 |
| handle_high_finish | poignée trop haute au finish | 0 = hanche, 1 = épaule | > 0,75 |
| chain_misaligned | chaîne non alignée avec le rail | pente de la poignée pendant le drive | > 8° |
| knee_break_early | genoux qui plient avant les mains | flexion déjà faite au passage des mains | > 0,25 |
| recovery_rushed | récupération trop courte | ratio récup / drive | < 1,3 |
| trunk_drift | buste de plus en plus penché au catch | dérive dernier tiers - premier tiers | > 6° |

Non détectable en vision 2D : dos arrondi (COCO-17 ne donne que la ligne épaule-hanche), asymétrie gauche/droite
(vue de face nécessaire), poussée réelle des pieds (force : PM5).

## 3 bis. Modes d'analyse selon la vue

La vue est classée par le rapport largeur d'épaules / longueur du tronc (médiane, ~0,2 en profil strict,
~1 de face ou de dos ; seuils `max_view_shoulder_ratio` = 0,5 et `view_frontal_ratio` = 1,1).

| Mode | Vue | Fiable | Indicatif (marqué `~`) | Masqué |
|---|---|---|---|---|
| full | profil | tout | - | - |
| degraded | oblique | comptage, cadence, durée du drive, ratio récup/drive, régularité du rythme | séquence jambes/dos/bras, raccourcissement du drive avec la fatigue | angles et hauteurs absolus : tibia, layback, hauteur de poignée, chaîne, dérive d'angle du tronc |
| minimal | face / dos | comptage, cadence, ratio, régularité | - | tout le reste |

Principe : une vue non perpendiculaire déforme les angles et hauteurs absolus, mais beaucoup moins les mesures
relatives (progression de chaque segment normalisée par sa propre excursion, comparaison d'une grandeur à
elle-même dans le temps, durées). Ces dernières restent donc exploitables, avec la mention "indicatif".
Aucun seuil absolu n'est affiché en vue dégradée. Un message explique le mode, ce qui reste fiable et comment filmer.

Détection des coups robuste : trois signaux candidats (poignet, hanche, épaule) projetés sur l'axe du rail ; le
plus régulier (CV de la période) l'emporte, ce qui sauve l'analyse quand le poignet est occulté ou bruité.

## 4. Limites assumées

- Vue 2D : angles hors plan faussés. Le profil strict est la configuration de référence. Une vue oblique
  (caméra côté volant, basse) dégrade surtout les angles du tronc et la pente de chaîne.
- Erreur angulaire typique des modèles de pose sans marqueurs dans le plan sagittal : 3 à 15°.
- Aucune validation publiée de pose vidéo contre capture de mouvement sur ergo n'a été trouvée.
- Le poignet sert de proxy de la poignée.
- Seuils non calibrés : à ajuster avec des vidéos annotées par un coach.
- Vidéo en ralenti : les durées sont faussées (non détecté automatiquement).

## 5. Couplage PM5 (phase suivante)

Spec Concept2 "PM CSAFE Communication Definition", révision 0.31 (2025). Service BLE rowing :
- état général (temps, distance, drag factor), état additionnel (vitesse, cadence, allure, puissance, FC),
- données par coup : drive length, drive time, recovery time, force de pic, force moyenne, travail par coup,
  compteur de coups,
- courbe de force complète par coup (PM5v2 uniquement, plusieurs notifications),
- échantillonnage réglable de 100 ms à 1 s.

Apports : vérité terrain gratuite pour valider cadence et ratio drive/récup côté vidéo ; forme de la courbe de
force (pic précoce ou tardif) pour la poussée des jambes. Synchronisation par événements de coup (catch vidéo vs
début de coup PM5). Web Bluetooth : Chrome et Edge uniquement (pas Safari iOS). Spec et SDK fournis "as-is" :
vérifier les conditions avant toute redistribution.

## 6. Existant open source (inspiration, pas de dépendance)

- kelsorj/Concept2-PM5-Rowing-Video-Analysis : MIT mais basé sur Ultralytics YOLO (AGPL), donc non réutilisable
  comme dépendance ici.
- edvale732/stroke-ai : MediaPipe, détection de coups par le poignet, pas de licence affichée.
- ergarcade/pm5-base : lecture du PM5 en Web Bluetooth, MIT.

## 7. Sources

- Concept2, Rowing technique : https://www.concept2.com/technique
- Concept2, catch / drive / finish : https://www.concept2.co.nz/Blog/x_post/Understanding-rowing-technique-catch-drive-and-finish-00054.html
- Concept2, trois erreurs courantes : https://www.concept2.co.nz/Blog/x_post/Three-common-technique-mistakes-on-the-indoor-rower-00085.html
- Concept2, spec PM5 : https://cms.concept2.com/sites/default/files/2026-03/Concept2%20PM%20CSAFE%20Communication%20Definition.pdf
- PLOS ONE 2023, foot-stretcher et cadence : https://pmc.ncbi.nlm.nih.gov/articles/PMC10174490/
- BJSM 2021, lombalgie et biomécanique : https://bjsm.bmj.com/content/55/11/616
- Kleshnev, récupération : https://www.row2k.com/features/5722/the-biomechanics-of-the-recovery-phase-by-dr-valery-kleshnev/
