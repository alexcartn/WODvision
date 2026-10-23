#!/usr/bin/env bash
# Construit le Space et le pousse sur Hugging Face.
# Usage : HF_TOKEN=hf_xxx HF_SPACE=<compte>/<nom_du_space> bash deploy/huggingface/push_space.sh
# Le jeton (droit d'ecriture) n'est jamais affiche ni stocke dans le depot.
set -euo pipefail

: "${HF_TOKEN:?definir HF_TOKEN (jeton Hugging Face avec droit d'ecriture)}"
: "${HF_SPACE:?definir HF_SPACE, par exemple monCompte/wodvision}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# HF_SDK = gradio (defaut, lanceur Streamlit) ou docker, selon le SDK choisi a la creation du Space.
bash deploy/huggingface/prepare_space.sh build/hf_space >/dev/null
cd build/hf_space

git init -q -b main
git config user.name "${GIT_AUTHOR_NAME:-wodvision-deploy}"
git config user.email "${GIT_AUTHOR_EMAIL:-deploy@users.noreply.huggingface.co}"
git lfs install --local >/dev/null
git add -A
git commit -q -m "Deploy WODVision Rameur ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
# Le jeton passe par l'URL du depot distant, hors des logs (git ne l'affiche pas).
git push --force "https://hf-user:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE}" main 2>&1 \
  | sed "s/${HF_TOKEN}/***/g"
echo "Space pousse : https://huggingface.co/spaces/${HF_SPACE}"
