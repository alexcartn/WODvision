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

## 3 ter. Mode fantôme (posture de référence)

`analysis/rowing/ghost.py`. Un squelette jaune "fantôme" est superposé à la posture réelle, sur la vidéo et dans le
panneau du squelette seul, avec l'écart de buste en direct (degrés, + = plus de recul que la référence).

- Modèle cinématique plan dans le repère machine, mis à l'échelle des segments du rameur (cuisse, tibia, tronc,
  bras), ancré sur sa cheville et à sa hauteur d'assise, synchronisé sur sa phase de coup (progression du drive ou de
  la récupération, tirée des instants de catch et de finish détectés).
- Séquence de référence : drive jambes, puis dos, puis bras ; récupération bras, buste, puis jambes. Tibias
  verticaux au catch, jambes presque tendues et recul de 25° au finish (manuel Concept2), poignée à hauteur de
  genoux avec chaîne horizontale.
- Paramètres dans `configs/default.yaml`, section `ghost` : hypothèses tirées des consignes Concept2, qui donnent
  la séquence et quelques repères mais pas un modèle numérique complet (par exemple l'inclinaison du buste au catch,
  -35°, est une valeur plausible non sourcée). Ce n'est pas une posture "parfaite" validée sur des rameurs d'élite :
  c'est un guide visuel à calibrer avec un coach.
- Disponible en analyse complète (vue de profil) seulement : en vue oblique ou de face, les angles sont déformés et
  une référence serait trompeuse. Désactivable : `--no-ghost` (CLI) ou la case de l'application.

## 3 quater. Synthèse pour amateurs et coachs

Deux niveaux d'affichage dans l'application (Débutant par défaut, Coach) ; la même analyse, deux lectures.

- Contenu éditable : `configs/conseils.yaml` (critères, points forts, et pour chaque faute un titre simple, "pourquoi",
  "ce que tu dois sentir", un exercice). Fichier de contenu à relire par un coach : les conseils sont une première
  rédaction (consignes Concept2 et exercices classiques), marqués `a_valider`. Un test vérifie que chaque règle a son
  conseil.
- Moteur : `coaching.py`. Cinq critères (rythme, enchaînement, positions, tenue dans la durée, force PM5) avec un
  statut vert / orange / rouge / non mesuré. Rouge seulement si une faute fiable touche au moins 60 % des coups ; un
  constat indicatif (vue non profil) ne dépasse jamais l'orange. Pas de note globale : un score unique ferait croire
  à une précision qu'on n'a pas. Un point fort, au plus deux priorités (trois en mode Coach), une par critère avant de
  répéter un critère.
- Images clés (`viz/rowing_highlights.py`) : en analyse complète, le meilleur coup et le coup à améliorer, choisis par
  leur écart moyen au fantôme. Au catch et au finish : posture réelle (vert), référence (jaune), flèche vers
  l'articulation qui s'écarte le plus ; plus un ralenti x4 de chaque coup. Si les deux coups sont presque identiques
  (écart inférieur à 0,03 longueur de tronc), un seul est montré. Absent en vue non profil.
- Cohérence : la posture de référence respecte les règles appliquées au rameur (test dédié), sinon elle serait
  elle-même "en faute".

## 4. Limites assumées

- Vue 2D : angles hors plan faussés. Le profil strict est la configuration de référence. Une vue oblique
  (caméra côté volant, basse) dégrade surtout les angles du tronc et la pente de chaîne.
- Erreur angulaire typique des modèles de pose sans marqueurs dans le plan sagittal : 3 à 15°.
- Aucune validation publiée de pose vidéo contre capture de mouvement sur ergo n'a été trouvée.
- Le poignet sert de proxy de la poignée.
- Seuils non calibrés : à ajuster avec des vidéos annotées par un coach.
- Vidéo en ralenti : les durées sont faussées (non détecté automatiquement).

## 5. Couplage PM5

Spec Concept2 "PM CSAFE Communication Definition", révision 0.31 (2025), section Smart Bluetooth.
UUID = `CE06XXXX-43E5-11E4-916C-0800200C9A66`. Caractéristiques utilisées (`pm5/spec.py`, `pm5/parsers.py`) :

| Id | Contenu | Octets |
|---|---|---|
| 0x0031 | état général : temps, distance, état du coup (drive / dwelling / recovery), drag factor | 19 |
| 0x0032 | cadence, vitesse, allure, fréquence cardiaque | 17 |
| 0x0035 | par coup : drive length, drive time, recovery time, force de pic et moyenne, travail, compteur | 20 |
| 0x0036 | par coup : puissance, calories, compteur | 18 |
| 0x003D | courbe de force (PM5v2 seulement), plusieurs notifications par courbe | 2 à 20 |
| 0x0034 | écriture : cadence d'échantillonnage de l'état (3 = 100 ms) | 1 |

Les dispositions d'octets sont testées par aller-retour avec un encodeur écrit d'après la spec (ce test a déjà
attrapé une erreur de décalage). Elles n'ont pas été vérifiées sur un vrai PM5.

### Chaîne

1. Enregistrer : `wodvision pm5-record --output data/pm5/seance.jsonl` (extra `pip install -e ".[pm5]"`, bibliothèque
   `bleak`, sans-fil du PM5 activé : More Options > Turn Wireless ON, PM5 non connecté à une autre application).
   Journal brut horodaté (heure de l'ordinateur), décodé hors ligne. À défaut : un CSV de coups
   (`pm5/session.py`, colonnes `CSV_COLUMNS`).
2. Coupler : `wodvision row-analyze video.mp4 --pm5 data/pm5/seance.jsonl [--pm5-lag N] [--pm5-phase finish|catch]`
   ou le champ PM5 de l'application Streamlit.
3. Vérifier un fichier : `wodvision pm5-info data/pm5/seance.jsonl`.

### Appariement (`analysis/rowing/pm5_sync.py`)

- Par séquence : décalage entier de coups qui minimise l'écart des périodes et des durées de drive. Trop régulier
  (variation de période inférieure à 0,05 s, ou aucun décalage ne se détache) : appariement déclaré ambigu, avec
  avertissement. `--pm5-lag` permet de l'imposer.
- Décalage d'horloge PM5 = vidéo + offset : médiane des écarts, gigue donnée. Le PM5 notifie un coup à un instant
  fixe de son cycle (fin du drive, ou début du suivant). La phase est déduite des états à 100 ms si présents, sinon
  "fin du drive" est supposé (hypothèse non vérifiée : le PM5 ne connaît force de pic et drive length qu'à la fin
  du drive). Cela décale l'horloge d'au plus un drive et n'affecte pas l'appariement.
- Validation de la vidéo : écart de période, biais et dispersion du temps de drive. Le drive PM5 (volant en
  accélération) est plus court que le drive vidéo : un biais est normal, la dispersion est l'indicateur.
- Contrôle de cohérence : longueur de tronc déduite = drive length PM5 / amplitude de la poignée en longueurs de
  tronc. Proche de 0,45 à 0,60 m si l'appariement est correct et la vue de profil.

### Courbe de force (`analysis/rowing/force_curve.py`)

L'axe est la position de la poignée (échantillons de ~3 cm d'après le forum Concept2) et l'unité de force n'est pas
documentée : seule la forme normalisée est utilisée. Repères Concept2 ("Using the Force Curve") : courbe lisse
= force progressive ; catch explosif = pic aigu décalé vers le début ; plusieurs pics = transition jambes / dos /
bras à travailler. Règles (seuils = hypothèses) :

| Code | Faute | Métrique | Seuil |
|---|---|---|---|
| force_peak_early | pic trop précoce (catch explosif) | position du pic dans le drive | < 0,2 |
| force_multipeak | transition saccadée | pics secondaires (prominence >= 15 % du pic) | > 1 |

Pas de règle de pic tardif : aucune source solide trouvée. Aucune étude évaluée par des pairs ne relie la forme de
la courbe à la performance (seulement des consignes Concept2 et des avis de forum).

### Limites et points à vérifier sur matériel

- Réassemblage des courbes (numéro de séquence repartant de 0 à chaque courbe) : hypothèse, courbe abandonnée en
  cas de perte plutôt que mal reconstruite.
- Moment exact de la notification de coup (voir ci-dessus).
- Export du Logbook Concept2 : format non vérifié, non pris en charge pour l'instant (journal brut ou CSV maison).
- Droits de la spec et du SDK Concept2 : fournis "as-is", à relire avant toute redistribution. `bleak` : licence
  à confirmer dans son dépôt avant de distribuer (MIT à ma connaissance).
- Web Bluetooth n'existe pas sur Safari iOS : l'enregistrement passe par l'ordinateur (Python), pas le navigateur.

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
