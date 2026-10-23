"""Lanceur pour un Space au SDK Gradio : demarre l'application Streamlit sur le port 7860.

Hugging Face execute `python app.py` et attend un serveur sur le port 7860 : n'importe quel
serveur convient, ici Streamlit. CORS et XSRF sont desactives pour l'envoi de fichiers dans
l'iframe d'un Space.
"""

import os
import sys

os.environ.setdefault("WODVISION_HOSTED", "1")
os.execv(sys.executable, [
    sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
    "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true",
    "--server.enableCORS=false", "--server.enableXsrfProtection=false",
])
