# Références ergomètre : ce qu'on a lu, ce qu'on en a retenu

Lecture du texte complet des études ci-dessous (pas seulement des résumés), sauf mention. Toutes portent sur des
ergomètres, pas sur l'eau. Échantillons petits : ce sont des ordres de grandeur, pas des normes.

## Sources

1. Kleshnev V. (2005). Comparison of on-water rowing with its simulation on Concept2 and RowPerfect machines.
   ISBS 2005, Beijing, p. 130-133. 5 rameuses de haut niveau (1,80 m), Concept2, environ 20,7 et 32,1 coups/min,
   capteurs de force et de position (25 Hz).
2. Penichet-Tomas A., Calavia-Carbajal S., Pueo B., Villalon-Gasch L. (2025). Kinematic analysis of Olympic and
   traditional rowing mechanics at different stroke rates. Int J Exerc Sci 18(7):610-621. doi:10.70252/RFXJ1471.
   13 rameuses nationales, Concept2 modèle D (siège mobile = "Olympic"), 18, 24 et 30 c/min, vidéo 60 images/s dans le
   plan sagittal, extraction de pose par IA non validée par les auteurs (RowerUp).
3. O'Halloran J., Anderson R. (2007). An examination of intrarower coordination movement coupling using continuous
   relative phase. ISBS 2007, Ouro Preto. 5 rameurs (1 expérimenté, 4 novices), RowPerfect, 200 Hz.
4. Fumoto M. et al. (2020). Body motion and rowing performance: association between hip angle and rowing
   performance, a pilot study. Keio J Med 69(3):66-75. doi:10.2302/kjm.2019-0007-OA. 11 rameurs, Concept2 D, 30 Hz.
5. Bingul B. et al. (2014). Two-dimensional kinematic analysis of catch and finish positions during a 2000 m
   rowing ergometer time trial. S Afr J Res Sport Phys Educ Recreat 36(3):1-10. 5 rameurs, Concept IID. Résumé
   seulement : pas de valeurs d'angles accessibles.
6. Concept2, "How to Row Harder" et blog technique : ratio de 2 pour 1 comme règle empirique, cadence de 24 à 30
   pour la plupart des séances.

## Valeurs retenues

| Donnée | Valeur | Source | Où dans le code |
|---|---|---|---|
| Part du drive dans le cycle | 41,7 % à 20,7 c/min, 51,9 % à 32,1 c/min, soit retour/poussée de 1,40 et 0,93 | 1 | `rowing.ratio_targets` |
| Position du pic de vitesse de poignée | 76,4 % (20,7 c/min) et 74,1 % (32,1 c/min) du drive | 1 | repère jaune des courbes |
| Position du pic de force | 37,2 % et 40,8 % du drive | 1 | repère jaune de la courbe PM5 |
| Vitesse maximale de poignée | 1,70 à 1,89 m/s | 1 | non utilisée (échelle métrique inconnue) |
| Parts de la course du drive | jambes 0,55 m, tronc 0,47 m, bras 0,44 m | 1 | information (séquence) |
| Angle du genou (cuisse-tibia) | catch 56 à 61 deg, finish 167 à 169 deg | 2 | cohérent avec le fantôme |
| Inclinaison du buste (depuis la verticale) | catch -33 à -36 deg, finish +44 à +49 deg (plus de recul à cadence élevée) | 2 | `ghost.lean_finish_deg` 45, `max_layback_deg` 55, `min_layback_deg` 30 |
| Variabilité de l'angle de hanche au catch | corrélée à la performance (rho 0,66) ; moyenne non | 4 | motive la lecture de la régularité |
| Couplage genou-coude | diffère nettement entre expert et novices | 3 | motive la figure d'enchaînement |

## Ce que ça a changé dans notre approche

- Ratio retour/poussée : l'ancienne table (2,0 à 18 c/min jusqu'à 1,0 à 36) n'avait pas de source et était trop
  exigeante. Un coup à 33 c/min avec un ratio de 1,0 est conforme aux mesures. Nouvelle table : deux points mesurés,
  interpolation entre eux, valeur bornée au-delà (le repère ne descend pas sous 0,93 ni ne dépasse 1,40). Concept2
  recommande 2 pour 1 à cadence tranquille : on reste donc volontairement indulgent en dessous de 20 c/min.
- Recul du buste au finish : l'ancien repère de 25 deg (manuel Concept2) était trop faible, un recul de 45 deg est
  la norme chez les rameuses d'élite mesurées. Seuil d'excès relevé de 35 à 55 deg, seuil d'insuffisance de 5 à 30 deg.
- Pic de vitesse : situé aux trois quarts de la poussée sur Concept2, plus tard que sur l'eau (59 à 65 %). La courbe
  de référence dérivée de notre modèle a été retirée (non sourcée) ; on affiche à la place le repère de position.
- Correction d'un propos précédent : la vitesse de poignée mesurée sur la vidéo de Pauline culmine vers 85 % de la
  poussée, ce qui est cohérent avec cette littérature, contrairement à ce que j'avais annoncé.

## Limites

- Échantillons de 5 à 13 personnes, dont 5 rameuses de haut niveau pour les durées, 13 rameuses pour les angles :
  pas de rameurs hommes, pas de débutants dans les références chiffrées.
- La source 2 extrait les angles par IA non validée ; ses définitions (buste depuis la verticale, genou = angle
  cuisse-tibia) sont proches des nôtres mais la correspondance n'est pas garantie au degré près.
- Les angles de la source 2 sont des moyennes de groupe : pas de fourchette individuelle, donc nos seuils (30 à 55 deg
  de recul) sont environ une moyenne moins ou plus deux écarts-types, à affiner avec des mesures sur tes rameurs.
- Non couverts par ces sources : longueurs de segments, début du dos (trunk onset), hauteur de poignée.
