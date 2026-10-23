#!/usr/bin/env bash
# Assemble le dossier a pousser vers un Space Hugging Face.
# Usage : HF_SDK=gradio|docker bash deploy/huggingface/prepare_space.sh [dossier_de_sortie]
#   gradio (defaut) : Space au SDK Gradio ; app.py lance Streamlit (si Docker n'est pas disponible)
#   docker          : Space au SDK Docker (modeles integres a l'image)
# A lancer depuis la racine du depot.
set -euo pipefail

OUT="${1:-build/hf_space}"
SDK="${HF_SDK:-gradio}"
rm -rf "$OUT"
mkdir -p "$OUT"

cp -r src app configs .streamlit pyproject.toml "$OUT"/
cp deploy/huggingface/gitattributes "$OUT/.gitattributes"
case "$SDK" in
  gradio)
    cp deploy/huggingface/README_gradio.md "$OUT/README.md"
    cp deploy/huggingface/launcher_app.py "$OUT/app.py"
    cp deploy/huggingface/requirements_gradio.txt "$OUT/requirements.txt"
    cp deploy/huggingface/packages_gradio.txt "$OUT/packages.txt"
    ;;
  docker)
    cp deploy/huggingface/Dockerfile "$OUT/Dockerfile"
    cp deploy/huggingface/README.md "$OUT/README.md"
    ;;
  *) echo "HF_SDK inconnu : $SDK (gradio ou docker)" >&2; exit 1 ;;
esac
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$OUT" -name "*.egg-info" -type d -prune -exec rm -rf {} +

echo "Space ($SDK) pret dans : $OUT"
echo "Fichiers : $(find "$OUT" -type f | wc -l), taille : $(du -sh "$OUT" | cut -f1)"
