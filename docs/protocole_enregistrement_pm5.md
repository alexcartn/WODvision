# Protocole d'enregistrement : vidéo de profil + PM5

Objectif : obtenir une séance qui valide (1) la lecture Bluetooth du PM5, (2) l'appariement des coups vidéo/PM5,
(3) la détection vidéo contre la vérité terrain PM5, (4) des exemples annotés de fautes pour calibrer les seuils.
Durée : environ 20 minutes, dont 12 minutes d'effort. Ce protocole n'a pas encore été exécuté sur un vrai PM5 :
les points marqués "à vérifier" sont ceux que la séance doit trancher.

## 0. Avant la séance (10 minutes, une fois)

1. Installer l'extra Bluetooth : `pip install -e ".[pm5]"` (bibliothèque `bleak`).
2. Ordinateur : Bluetooth activé, à moins de 3 m du PM5. Windows et macOS : autoriser l'accès Bluetooth au
   terminal. Linux : service `bluetooth` actif.
3. PM5 : mettre à jour le firmware (page Concept2 de la série PM5), puis More Options > Turn Wireless ON.
   Vérifier qu'aucun téléphone ni aucune application (ErgData, etc.) n'est connecté au PM5 : une seule connexion
   BLE à la fois.
4. Noter le modèle : PM5 v2 ou v3 (la courbe de force n'existe pas sur PM5v1, voir `Menu > Utilities > About`).
5. Régler le drag factor (par exemple 115) et ne plus y toucher pendant la séance. Il est enregistré.
6. Test à blanc de 2 minutes : lancer `wodvision pm5-record --output data/pm5/test.jsonl --duration 60`,
   ramer 15 coups, puis `wodvision pm5-info data/pm5/test.jsonl`. Attendu : nombre de coups cohérent, courbes de
   force égales au nombre de coups, états 100 ms de plusieurs centaines d'échantillons. Si les courbes sont à 0 :
   voir la section Dépannage.

## 1. Caméra

- Téléphone sur trépied, orientation paysage, 30 images/s minimum (60 si possible), pas de ralenti, pas de zoom
  numérique, pas de stabilisation agressive si le téléphone permet de la couper. Le mouvement de caméra est
  compensé mais un trépied reste la référence.
- Vue de profil stricte : l'axe de l'objectif perpendiculaire au rail, à hauteur de hanche (environ 50 à 60 cm),
  à 3 à 4 m. Le rameur entier doit rester dans l'image, poignée tendue au catch comme au finish.
- Côté : n'importe lequel, mais pas de la place du volant (le volant ne doit pas cacher les jambes).
- Fond sobre, lumière constante, une seule personne dans l'image, vêtements qui contrastent avec le fond et ne
  masquent pas genoux et hanches (pas de sweat large).
- Vérifier avant de commencer : prendre une photo et contrôler que épaule, hanche, genou, cheville, coude et
  poignet du côté caméra sont visibles à toutes les phases.

## 2. Ordre des opérations (à chaque enregistrement)

1. Lancer l'enregistreur PM5 : `wodvision pm5-record --output data/pm5/seance_01.jsonl --duration 1200`
2. Démarrer la vidéo.
3. Rameur assis, immobile, bras tendus pendant 5 s (repère de début, l'état du PM5 est enregistré).
4. Démarrer le PM5 (Just Row) et ramer selon les blocs ci-dessous. Le temps écoulé du PM5 démarre au premier coup :
   c'est normal, le décalage d'horloge est estimé après coup.
5. Après le dernier coup : rester immobile 5 s, arrêter la vidéo, puis arrêter l'enregistreur (Ctrl-C).
6. Copier la vidéo sur l'ordinateur. Ne jamais la committer (fichiers `.mp4` ignorés par git).

## 3. Blocs de la séance

Respirer 60 s entre les blocs, sans arrêter vidéo ni enregistreur. Le PM5 peut passer en pause : sans importance,
chaque coup est enregistré avec son temps. Noter dans le tableau (section 5) l'heure de la vidéo où chaque bloc
commence (chronomètre vidéo ou annonce vocale à voix haute, qui reste dans le son de la vidéo).

| Bloc | Consigne | Coups | Cadence |
|---|---|---|---|
| A | Rythme irrégulier volontaire : 3 coups lents, 2 rapides, 4 lents, 3 rapides, 2 lents, 3 rapides | 17 | 18 à 30 |
| B | Technique propre, régulière | 25 | 22 |
| C | Technique propre, régulière | 25 | 28 |
| D | Dos qui s'ouvre trop tôt (balancer le dos dès le début du drive) | 15 | 22 |
| E | Bras qui tirent trop tôt | 15 | 22 |
| F | Tibias au-delà de la verticale au catch (rouler trop loin) | 15 | 22 |
| G | Poignée qui monte (tirer vers la poitrine, chaîne non horizontale) | 15 | 22 |
| H | Catch explosif (coup sec au début, pic de force précoce) | 15 | 22 |
| I | Recul excessif du buste au finish | 15 | 22 |
| J | Technique propre en fin de séance, effort soutenu (fatigue) | 40 | 26 |

Pourquoi : le bloc A rend l'appariement des coups non ambigu (rythme irrégulier). B et C donnent la référence
propre et la comparaison cadence/ratio avec le PM5. D à I fournissent une vérité terrain par faute pour calibrer
les seuils de `configs/default.yaml` (section `rowing` et `pm5`). J mesure la dérive de fatigue.

## 4. Après la séance

1. Contrôler le fichier : `wodvision pm5-info data/pm5/seance_01.jsonl`
   Attendu : coups PM5 proches du total ramé (environ 170 avec le tableau ci-dessus), courbes de force presque
   toutes présentes, plusieurs milliers d'états 100 ms.
2. Analyser : `wodvision row-analyze data/raw/seance_01.mp4 --pm5 data/pm5/seance_01.jsonl`
3. Lire le bloc "couplage PM5" de la sortie. Critères de réussite :
   - appariement non ambigu (confiance au moins 0,5) ;
   - phase PM5 déduite des états ("status") et non supposée ("default") ;
   - gigue d'horloge inférieure à 0,15 s ;
   - écart de période inférieur à 60 ms ; dispersion du temps de drive inférieure à 100 ms ;
   - longueur de tronc déduite entre 0,45 et 0,60 m, CV inférieur à 15 %.
4. Si un critère échoue, ne pas retoucher les seuils : noter lequel et envoyer les trois fichiers (vidéo, journal
   PM5, tableau d'annotation) pour diagnostic.

## 5. Tableau d'annotation

Copier `datasets/rowing_annotations_template.csv` en `datasets/rowing_annotations_seance_01.csv` et remplir une
ligne par bloc :

- `t_start_video_s`, `t_end_video_s` : bornes dans la vidéo (secondes) ;
- `n_strokes_counted` : coups comptés à la main (vérité terrain du comptage) ;
- `expected_faults` : codes de `docs/rowing_concept.md` séparés par `;` (par exemple `trunk_early`),
  vide pour un bloc propre ;
- `coach_note` : ressenti ou remarque (par exemple "trop exagéré", "pas vraiment tenu").

## 6. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| "PM5 introuvable" | sans-fil éteint, PM5 déjà connecté ailleurs, trop loin | Turn Wireless ON, fermer ErgData et le Bluetooth des téléphones, se rapprocher |
| connexion puis aucun coup | PM5 pas en effort | lancer Just Row et ramer |
| 0 courbe de force | PM5v1, ou notification 0x003D non reçue | vérifier la version, relancer l'enregistrement ; les autres mesures restent valides |
| appariement ambigu | rythme trop régulier | refaire le bloc A, ou passer `--pm5-lag` avec le nombre de coups avant le début de la vidéo |
| tronc déduit hors 0,45 à 0,60 m | vue non perpendiculaire, ou mauvais appariement | contrôler l'angle de caméra, vérifier `--pm5-lag` |
| mode dégradé annoncé | caméra pas de profil (rapport épaules/tronc > 0,5) | refaire la vidéo de profil strict |
| perte de connexion en cours de séance | distance, interférences | rapprocher l'ordinateur ; le journal conserve ce qui a été reçu |

## 7. Confidentialité

La vidéo montre une personne : usage personnel, pas de partage ni de commit. Si d'autres personnes sont filmées,
recueillir leur accord avant toute analyse (voir le point ouvert RGPD de `CLAUDE.md`).
