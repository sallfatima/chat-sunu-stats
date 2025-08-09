"""Configuration pour le système RAG simple amélioré pour l'ANSD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Dict, Any

from shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class RagConfiguration(BaseConfiguration):
    """Configuration améliorée pour le système RAG ANSD."""

    # =============================================================================
    # MODÈLE ET PARAMÈTRES DE BASE (VOS PARAMÈTRES EXISTANTS)
    # =============================================================================
    
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "Le modèle de langage utilisé pour la génération de réponses. Format: provider/model-name."
        },
    )

    # Paramètres de récupération (UNIFIÉS avec search_kwargs)
    retrieval_k: int = field(
        default=15,
        metadata={
            "description": "Nombre de documents à récupérer lors de la recherche sémantique."
        },
    )

    retrieval_fetch_k: int = field(
        default=50,
        metadata={
            "description": "Nombre de documents candidats à examiner avant sélection finale."
        },
    )

    # Paramètres spécifiques à l'ANSD (VOS PARAMÈTRES CONSERVÉS)
    enable_query_preprocessing: bool = field(
        default=True,
        metadata={"description": "Activer le prétraitement des requêtes avec synonymes ANSD."},
    )

    enable_document_scoring: bool = field(
        default=True,
        metadata={"description": "Activer le scoring avancé des documents récupérés."},
    )

    max_context_length: int = field(
        default=8000,
        metadata={"description": "Longueur maximale du contexte envoyé au modèle (en caractères)."},
    )

    min_document_score: float = field(
        default=0.1,
        metadata={"description": "Score minimum requis pour qu'un document soit considéré comme pertinent."},
    )

    enable_debug_logs: bool = field(
        default=True,
        metadata={"description": "Activer les logs de débogage pour le processus RAG."},
    )

    enable_fallback_search: bool = field(
        default=True,
        metadata={"description": "Essayer une recherche avec la requête originale si aucun document n'est trouvé."},
    )

    prioritize_recent_data: bool = field(
        default=True,
        metadata={"description": "Prioriser les données les plus récentes dans les résultats."},
    )

    ansd_survey_weights: dict = field(
        default_factory=lambda: {
            "rgph": 1.5,    # Recensement - haute priorité
            "eds": 1.3,     # Enquête démographique et santé
            "esps": 1.3,    # Enquête pauvreté
            "ehcvm": 1.2,   # Conditions de vie ménages
            "enes": 1.2,    # Enquête emploi
        },
        metadata={"description": "Poids à appliquer aux différents types d'enquêtes ANSD lors du scoring."},
    )

    # =============================================================================
    # NOUVEAUX PARAMÈTRES search_kwargs INTÉGRÉS
    # =============================================================================
    
    # Configuration générale de recherche
    search_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            # Utilise les paramètres retrieval_k et retrieval_fetch_k définis ci-dessus
            # Ces valeurs seront mises à jour dynamiquement
            #"lambda_mult": 0.7,         # Balance similarité/diversité
            "score_threshold": 0.1,     # Seuil minimum (correspond à min_document_score)
           # "search_type": "mmr",       # Maximum Marginal Relevance
           
            
            # Recherche hybride
            "hybrid_search": True,
            "alpha": 0.6,              # 60% sémantique, 40% lexicale
            
            # Optimisations vectorielles
            "vector_search_kwargs": {
                "ef": 200,
                "nprobe": 10
            },
            
            # Boost pour champs spécifiques
            "field_weights": {
                "content": 1.0,
                "title": 1.5,
                "pdf_name": 1.3,
                "metadata.survey_type": 2.0
            }
        },
        metadata={"description": "Paramètres de recherche par défaut pour tous types de questions"},
    )

    # Configurations spécialisées par domaine
    demographic_search_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "k_multiplier": 1.3,        # 15 * 1.3 = ~20 documents
            "fetch_k_multiplier": 1.2,  # 50 * 1.2 = 60 candidats
            "priority_surveys": ["rgph", "eds"],
            "boost_keywords": ["population", "habitants", "démographie", "recensement"],
            "field_weights": {
                "content": 1.0,
                "demographic_indicators": 2.0,
                "title": 1.5
            }
        },
        metadata={"description": "Configuration pour questions démographiques"},
    )
    
    economic_search_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "k_multiplier": 0.8,        # 15 * 0.8 = 12 documents (plus ciblé)
            "fetch_k_multiplier": 0.8,  # 50 * 0.8 = 40 candidats
            "priority_surveys": ["ehcvm", "esps", "enes"],
            "boost_keywords": ["emploi", "pauvreté", "économie", "revenus", "ménages"],
            "field_weights": {
                "content": 1.0,
                "economic_indicators": 2.0,
                "employment_data": 1.8
            }
        },
        metadata={"description": "Configuration pour questions économiques"},
    )
    
    health_search_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "k_multiplier": 0.7,        # 15 * 0.7 = ~10 documents (très ciblé)
            "fetch_k_multiplier": 0.7,  # 50 * 0.7 = 35 candidats
            "priority_surveys": ["eds"],
            "boost_keywords": ["santé", "mortalité", "morbidité", "nutrition", "vaccination"],
            "field_weights": {
                "content": 1.0,
                "health_indicators": 2.0,
                "medical_data": 1.8
            }
        },
        metadata={"description": "Configuration pour questions de santé"},
    )

    # =============================================================================
    # PARAMÈTRES AVANCÉS
    # =============================================================================
    
    # Re-ranking des résultats
    enable_reranking: bool = field(
        default=True,
        metadata={"description": "Activer le re-classement des résultats pour améliorer la pertinence"},
    )
    
    # Expansion automatique des requêtes
    enable_query_expansion: bool = field(
        default=True,
        metadata={"description": "Activer l'expansion de requête avec synonymes ANSD"},
    )
    
    # Synonymes ANSD pour l'expansion
    ansd_synonyms: Dict[str, list] = field(
        default_factory=lambda: {
            "population": ["habitants", "démographie", "recensement", "effectif"],
            "pauvreté": ["indigence", "vulnérabilité", "conditions_de_vie", "esps"],
            "emploi": ["travail", "activité_économique", "occupation", "enes"],
            "santé": ["mortalité", "morbidité", "espérance_de_vie", "eds"],
            "éducation": ["alphabétisation", "scolarisation", "enseignement"],
            "rgph": ["recensement_général", "population_habitat"],
            "eds": ["enquête_démographique_santé"],
            "esps": ["enquête_pauvreté", "suivi_pauvreté"],
            "ehcvm": ["conditions_vie_ménages", "budget_consommation"],
            "enes": ["enquête_emploi", "emploi_sénégal"]
        },
        metadata={"description": "Dictionnaire de synonymes pour l'expansion de requêtes"},
    )
    
    # Filtrage intelligent
    enable_smart_filtering: bool = field(
        default=True,
        metadata={"description": "Activer le filtrage intelligent selon le contexte de la question"},
    )
    
    # Cache pour optimiser les performances
    enable_search_cache: bool = field(
        default=True,
        metadata={"description": "Mettre en cache les résultats de recherche fréquents"},
    )
    
    cache_ttl_minutes: int = field(
        default=60,
        metadata={"description": "Durée de vie du cache en minutes"},
    )

    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def get_search_kwargs_for_query(self, query: str) -> Dict[str, Any]:
        """Génère les search_kwargs optimaux selon le type de question."""
        
        query_lower = query.lower()
        
        # Créer une copie des search_kwargs de base
        kwargs = self.search_kwargs.copy()
        
        # Déterminer le domaine et appliquer la configuration spécialisée
        domain_config = None
        domain_type = "général"
        
        if any(term in query_lower for term in 
               ['population', 'habitants', 'démographie', 'recensement', 'rgph']):
            domain_config = self.demographic_search_config
            domain_type = "démographique"
        
        elif any(term in query_lower for term in 
                ['emploi', 'chômage', 'économie', 'pauvreté', 'revenus', 'ehcvm', 'esps', 'enes']):
            domain_config = self.economic_search_config
            domain_type = "économique"
        
        elif any(term in query_lower for term in 
                ['santé', 'mortalité', 'morbidité', 'eds', 'nutrition', 'vaccination']):
            domain_config = self.health_search_config
            domain_type = "santé"
        
        # Appliquer la configuration spécialisée
        if domain_config:
            # Ajuster k et fetch_k selon les multiplicateurs
            k = int(self.retrieval_k * domain_config.get('k_multiplier', 1.0))
            fetch_k = int(self.retrieval_fetch_k * domain_config.get('fetch_k_multiplier', 1.0))
            
            kwargs.update({
                'k': k,
                'fetch_k': fetch_k,
                'field_weights': domain_config.get('field_weights', kwargs.get('field_weights', {}))
            })
            
            # Ajouter filtres pour enquêtes prioritaires si supporté par votre vector store
            if 'priority_surveys' in domain_config:
                kwargs['filter'] = {
                    'survey_type': {'$in': domain_config['priority_surveys']}
                }
        else:
            # Configuration par défaut
            kwargs.update({
                'k': self.retrieval_k,
                'fetch_k': self.retrieval_fetch_k
            })
        
        if self.enable_debug_logs:
            print(f"🎯 Configuration de recherche pour domaine '{domain_type}':")
            print(f"   k={kwargs.get('k')}, fetch_k={kwargs.get('fetch_k')}")
            if 'filter' in kwargs:
                print(f"   Enquêtes prioritaires: {kwargs['filter']['survey_type']['$in']}")
        
        return kwargs
    
    def get_expanded_query(self, original_query: str) -> str:
        """Étend la requête avec des synonymes ANSD si l'expansion est activée."""
        
        if not self.enable_query_expansion:
            return original_query
        
        query_lower = original_query.lower()
        expanded_terms = []
        
        for key, synonyms in self.ansd_synonyms.items():
            if key in query_lower:
                # Ajouter 2 synonymes les plus pertinents
                expanded_terms.extend(synonyms[:2])
        
        if expanded_terms:
            expanded_query = f"{original_query} {' '.join(expanded_terms)}"
            if self.enable_debug_logs:
                print(f"🔧 Requête étendue: {original_query} → {expanded_query}")
            return expanded_query
        
        return original_query
    
    def should_use_survey_weights(self, query: str) -> bool:
        """Détermine si les poids d'enquêtes doivent être appliqués."""
        return any(survey in query.lower() for survey in self.ansd_survey_weights.keys())

# =============================================================================
# EXEMPLE D'UTILISATION DANS LE CODE
# =============================================================================

"""
# Dans votre fonction retrieve, utilisez comme ceci :

async def retrieve(state: GraphState, *, config: RagConfiguration):
    question = extract_question(state.messages)
    
    # Expansion de requête
    expanded_query = config.get_expanded_query(question)
    
    # Configuration adaptative
    search_kwargs = config.get_search_kwargs_for_query(question)
    
    # Configuration du retriever
    retriever_config = {
        'search_kwargs': search_kwargs,
        'enable_debug': config.enable_debug_logs
    }
    
    async with retrieval.make_retriever(retriever_config) as retriever:
        documents = await retriever.ainvoke(expanded_query)
    
    return {"documents": documents, "messages": state.messages}
"""