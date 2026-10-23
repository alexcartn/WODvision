#!/usr/bin/env bash
# Assemble le dossier a pousser vers un Space Hugging Face (SDK Docker).
# Usage : bash deploy/huggingface/prepare_space.sh [dossier_de_sortie]   (depuis la racine du depot)
set -euo pipefail

OUT="${1:-build/hf_space}"
rm -rf "$OUT"
mkdir -p "$OUT"

cp -r src app configs .streamlit pyproject.toml "$OUT"/
cp deploy/huggingface/Dockerfile "$OUT/Dockerfile"
cp deploy/huggingface/README.md "$OUT/README.md"
cp deploy/huggingface/gitattributes "$OUT/.gitattributes"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$OUT" -name "*.egg-info" -type d -prune -exec rm -rf {} +

echo "Space pret dans : $OUT"
echo "Fichiers : $(find "$OUT" -type f | wc -l), taille : $(du -sh "$OUT" | cut -f1)"
echo "Suite : voir docs/deploiement_huggingface.md"
