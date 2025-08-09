# Dockerfile Optimal pour SunuStat ANSD Backend
# Combine simplicité de votre version + optimisations spécifiques SunuStat

# Utiliser Python 3.11 comme image de base (VOTRE CHOIX - PARFAIT)
FROM python:3.11-slim

# Métadonnées essentielles seulement
LABEL maintainer="ANSD Sénégal"
LABEL description="Backend SunuStat avec système RAG pour statistiques du Sénégal"
LABEL version="1.3.0"

# Définir le répertoire de travail (VOTRE APPROCHE)
WORKDIR /app

# Installer les dépendances système nécessaires (VOTRE BASE + RAG)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installer Poetry (VOTRE MÉTHODE)
RUN pip install poetry

# Configurer Poetry pour ne pas créer d'environnement virtuel (VOTRE CONFIG)
RUN poetry config virtualenvs.create false

# Copier les fichiers de configuration Poetry (VOTRE APPROCHE - OPTIMALE)
COPY pyproject.toml poetry.lock* ./

# Installer les dépendances Python (PRODUCTION SEULEMENT - OPTIMISATION)
RUN poetry install --only=main --no-interaction --no-root

# Copier le code source (VOTRE STRUCTURE)
COPY . .

# Créer un utilisateur non-root pour la sécurité (VOTRE SÉCURITÉ + OPTIMISATION)
RUN useradd --create-home --shell /bin/bash sunustat \
    && mkdir -p /app/{logs,data,temp} \
    && chown -R sunustat:sunustat /app

USER sunustat

# Variables d'environnement par défaut (VOTRE BASE + SPÉCIFIQUE SUNUSTAT)
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8001
ENV ANSD_MODE=production

# Exposer le port (ADAPTÉ POUR SUNUSTAT)
EXPOSE 8001

# Commande de santé (SIMPLE ET EFFICACE)
HEALTHCHECK --interval=30s --timeout=15s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Point d'entrée direct (SIMPLE COMME VOTRE VERSION)
ENTRYPOINT ["python", "backend_server_sunustat.py"]

