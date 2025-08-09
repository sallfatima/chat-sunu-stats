
"""Configuration pour le système RAG simple amélioré pour l'ANSD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class RagConfiguration(BaseConfiguration):
    """Configuration améliorée pour le système RAG ANSD."""

    # Modèle principal - Changé pour utiliser OpenAI par défaut car plus stable
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",  # Changé de anthropic/claude-3-5-sonnet-20240620
        metadata={
            "description": "Le modèle de langage utilisé pour la génération de réponses. Format: provider/model-name."
        },
    )

    # Paramètres de récupération améliorés pour l'ANSD
    retrieval_k: int = field(
        default=15,  # Augmenté de 10 à 15 pour plus de contexte
        metadata={
            "description": "Nombre de documents à récupérer lors de la recherche sémantique."
        },
    )

    retrieval_fetch_k: int = field(
        default=50,  # Augmenté pour un meilleur pool de candidats
        metadata={
            "description": "Nombre de documents candidats à examiner avant sélection finale."
        },
    )

    # Paramètres spécifiques à l'ANSD
    enable_query_preprocessing: bool = field(
        default=True,
        metadata={
            "description": "Activer le prétraitement des requêtes avec synonymes ANSD."
        },
    )

    enable_document_scoring: bool = field(
        default=True,
        metadata={
            "description": "Activer le scoring avancé des documents récupérés."
        },
    )

    max_context_length: int = field(
        default=8000,  # Augmenté pour plus de contexte
        metadata={
            "description": "Longueur maximale du contexte envoyé au modèle (en caractères)."
        },
    )

    # Seuils de qualité
    min_document_score: float = field(
        default=0.1,
        metadata={
            "description": "Score minimum requis pour qu'un document soit considéré comme pertinent."
        },
    )

    # Paramètres de débogage
    enable_debug_logs: bool = field(
        default=True,
        metadata={
            "description": "Activer les logs de débogage pour le processus RAG."
        },
    )

    # Fallback configuration
    enable_fallback_search: bool = field(
        default=True,
        metadata={
            "description": "Essayer une recherche avec la requête originale si aucun document n'est trouvé."
        },
    )

    # Configuration spécifique aux enquêtes ANSD
    prioritize_recent_data: bool = field(
        default=True,
        metadata={
            "description": "Prioriser les données les plus récentes dans les résultats."
        },
    )

    ansd_survey_weights: dict = field(
        default_factory=lambda: {
            "rgph": 1.5,    # Recensement - haute priorité
            "eds": 1.3,     # Enquête démographique et santé
            "esps": 1.3,    # Enquête pauvreté
            "ehcvm": 1.2,   # Conditions de vie ménages
            "enes": 1.2,    # Enquête emploi
        },
        metadata={
            "description": "Poids à appliquer aux différents types d'enquêtes ANSD lors du scoring."
        },
    )