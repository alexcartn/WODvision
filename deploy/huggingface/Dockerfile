# Space Hugging Face (SDK Docker, CPU gratuit). Construit par deploy/huggingface/prepare_space.sh.
FROM python:3.12-slim-trixie

# Bibliotheques systeme requises par OpenCV (Debian 13 : GLib s'appelle libglib2.0-0t64).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face execute le conteneur avec l'utilisateur 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    WODVISION_HOSTED=1
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir ".[app]"

COPY --chown=user app ./app
COPY --chown=user configs ./configs
COPY --chown=user .streamlit ./.streamlit

# Poids des modeles legers integres a l'image : pas de telechargement au premier visiteur.
RUN python -c "from wodvision.rowing_pipeline import build_default_models; build_default_models('lightweight')"

EXPOSE 7860
# CORS / XSRF desactives : necessaires pour l'envoi de fichiers dans l'iframe d'un Space.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
