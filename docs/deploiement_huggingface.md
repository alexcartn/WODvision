# Déploiement gratuit sur Hugging Face Spaces

Objectif : un lien public pour que d'autres essaient l'outil, sans rien payer. L'application tourne dans un Space
(SDK Docker, CPU gratuit) avec le profil léger `configs/hosted.yaml`.

## Ce que fait la version en ligne

- Onglet "Démo instantanée" : exemples simulés (profil avec PM5, vue oblique), résultat immédiat sans calcul lourd.
- Onglet "Analyser ma vidéo" : 30 s maximum, modèle de pose léger, images réduites à 640 px de large pour la
  détection et la pose, 15 images/s analysées. Mesure sur un extrait réel de 14 s (2080x900) : 17 s de calcul,
  contre environ 100 s avant allègement.
- Consentement obligatoire (accord des personnes filmées, traitement sur serveur externe).
- Rien n'est conservé : vidéo, fichier PM5, trace d'inférence et vidéo annotée sont supprimés du serveur après
  l'analyse (le résultat reste en mémoire dans la session du visiteur).

## Étapes

1. Créer un compte gratuit sur huggingface.co.
2. New Space : nom au choix, SDK Docker (modèle Blank), matériel CPU basic (gratuit), visibilité publique.
3. Créer un jeton d'accès avec droit d'écriture (Settings > Access Tokens).
4. Pousser l'application, au choix.

   Une commande depuis la racine du dépôt (Git LFS installé, jeton jamais affiché) :
   ```
   HF_TOKEN=hf_xxx HF_SPACE=<ton_compte>/<nom_du_space> bash deploy/huggingface/push_space.sh
   ```
   Automatique à chaque mise à jour de `main` (GitHub Actions, `.github/workflows/sync_hf_space.yml`) : dans
   GitHub, Settings > Secrets and variables > Actions, créer le secret `HF_TOKEN` (le jeton) et la variable
   `HF_SPACE` (`<ton_compte>/<nom_du_space>`). Ensuite chaque push sur `main` met le Space à jour, ou à la main
   via l'onglet Actions (Run workflow). Sans `HF_SPACE`, le job est ignoré.

   Le fichier `.gitattributes` fourni range les `.mp4` et `.npz` de démonstration dans Git LFS (Hugging Face
   refuse les binaires hors LFS).
5. Le Space construit l'image (comptez 5 à 10 minutes : OpenCV, ONNX Runtime, poids des modèles légers intégrés à
   l'image). Ensuite le lien `https://<ton_compte>-<nom_du_space>.hf.space` est partageable.
6. Mise à jour : relancer la même commande (ou pousser sur `main` si l'automatique est configuré).

## Points à vérifier (non testés ici)

- Le build Docker n'a pas pu être exécuté dans l'environnement de développement (pas de démon Docker). Équivalent
  testé : installation du dossier `build/hf_space` dans un environnement vierge, puis démarrage de l'application
  avec `WODVISION_HOSTED=1`, démo comprise, et un test de bout en bout dans un navigateur (envoi d'une vidéo
  de 12 s, analyse en 11 s, aucun fichier résiduel).
- Les ressources de l'offre gratuite (mémoire, processeur) et sa mise en veille après inactivité : à confirmer sur la
  page Hugging Face. Le premier visiteur après une veille attend le redémarrage.
- L'envoi de fichiers dans l'iframe d'un Space demande de désactiver CORS et XSRF (options de la commande de
  lancement dans le Dockerfile). Si l'envoi échoue avec une erreur 403, regarder en premier ce point.
- Streamlit Community Cloud reste utilisable en secours avec le même dépôt (`packages.txt` fourni). Sa ressource est
  plus limitée.

## Vie privée

Les vidéos d'autres personnes ne doivent être déposées qu'avec leur accord. La version en ligne ne conserve rien
mais le traitement a lieu sur un serveur tiers : pour des vidéos confidentielles ou longues, utiliser l'outil en
local (`wodvision row-analyze`).
