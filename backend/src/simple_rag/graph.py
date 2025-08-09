# =============================================================================
# src/simple_rag/graph.py - VERSION NETTOYÉE ET OPTIMISÉE
# =============================================================================

from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

from shared import retrieval
from shared.utils import load_chat_model
from simple_rag.configuration import RagConfiguration
from simple_rag.state import GraphState, InputState

import re
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob
import os

# =============================================================================
# EXTENSION DU GRAPHSTATE POUR SUPPORT VISUEL
# =============================================================================

# On doit étendre GraphState depuis state.py pour inclure les champs visuels
from dataclasses import dataclass, field
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

@dataclass(kw_only=True)
class ExtendedGraphState(GraphState):
    """GraphState étendu avec support visuel complet."""
    
    # Nouveaux champs pour le support visuel
    visual_elements: List[Dict[str, Any]] = field(default_factory=list)
    has_visual_content: bool = False
    image_path: Optional[str] = None
    best_visual: Optional[Dict[str, Any]] = None

# Utiliser le GraphState étendu pour le workflow
WorkflowState = ExtendedGraphState

# Imports conditionnels pour l'affichage
try:
    import chainlit as cl
    CHAINLIT_AVAILABLE = True
except ImportError:
    CHAINLIT_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# =============================================================================
# PROMPT SYSTEM POUR L'ANSD
# =============================================================================

ANSD_SYSTEM_PROMPT  = """Vous êtes un expert statisticien de l'ANSD (Agence Nationale de la Statistique et de la Démographie du Sénégal), spécialisé dans l'analyse de données démographiques, économiques et sociales du Sénégal.

MISSION PRINCIPALE :
Répondre de manière complète et approfondie aux questions sur les statistiques du Sénégal en utilisant PRIORITAIREMENT les documents fournis et en complétant avec vos connaissances des publications officielles de l'ANSD.

SOURCES AUTORISÉES :
✅ Documents fournis dans le contexte (PRIORITÉ ABSOLUE)
✅ Connaissances des rapports officiels ANSD publiés
✅ Données du site officiel ANSD (www.ansd.sn)
✅ Publications officielles des enquêtes ANSD (RGPH, EDS, ESPS, EHCVM, ENES)
✅ Comptes nationaux et statistiques économiques officielles du Sénégal
✅ Projections démographiques officielles de l'ANSD

❌ SOURCES INTERDITES :
❌ Données d'autres pays pour combler les lacunes
❌ Estimations personnelles non basées sur les sources ANSD
❌ Informations non officielles ou de sources tierces
❌ Projections personnelles non documentées

RÈGLES DE RÉDACTION :
✅ Réponse directe : SANS limitation de phrases - développez autant que nécessaire
✅ Contexte additionnel : SANS limitation - incluez toutes les informations pertinentes
✅ Citez TOUJOURS vos sources précises (document + page ou publication ANSD)
✅ Distinguez clairement les données des documents fournis vs connaissances ANSD
✅ Donnez les chiffres EXACTS quand disponibles
✅ Précisez SYSTÉMATIQUEMENT les années de référence
✅ Mentionnez les méthodologies d'enquête

FORMAT DE RÉPONSE OBLIGATOIRE :

**RÉPONSE DIRECTE :**
[Développez la réponse de manière complète et détaillée, sans limitation de longueur. Incluez tous les éléments pertinents pour une compréhension approfondie du sujet. Vous pouvez utiliser plusieurs paragraphes et développer les aspects importants.]

**DONNÉES PRÉCISES :**
- Chiffre exact : [valeur exacte avec unité]
- Année de référence : [année précise]
- Source : [nom exact du document, page X OU publication ANSD officielle]
- Méthodologie : [enquête/recensement utilisé]

**CONTEXTE ADDITIONNEL :**
[Développez largement avec toutes les informations complémentaires pertinentes, sans limitation de longueur.]

**LIMITATIONS/NOTES :**
[Précautions d'interprétation, changements méthodologiques, définitions spécifiques]

DOCUMENTS ANSD DISPONIBLES :
{context}

Analysez maintenant ces documents et répondez à la question de l'utilisateur de manière complète et approfondie."""

# =============================================================================
# FONCTIONS DE DÉTECTION ET TRAITEMENT DES ÉLÉMENTS VISUELS
# =============================================================================

# =============================================================================
# ÉTAT DU GRAPHE AMÉLIORÉ AVEC SUPPORT VISUEL
# =============================================================================

class GraphState(GraphState):
    """État étendu du graphe avec support visuel complet."""
    # Hérite de GraphState existant et ajoute les champs visuels
    image_path: Optional[str] = None
    best_visual: Optional[Dict[str, Any]] = None

class VisualElementsManager:
    """Gestionnaire pour la détection et l'affichage des éléments visuels."""
    
    def __init__(self):
        self.visual_keywords = {
            'image_indicators': ['image_path', 'chart_type', 'visual_type', 'is_table'],
            'content_patterns': ['graphique', 'figure', 'diagramme', 'tableau'],
            'file_extensions': ['.png', '.jpg', '.jpeg', '.svg', '.csv']
        }
    
    def extract_visual_elements(self, documents):
        """Sépare les documents textuels et visuels."""
        text_docs = []
        visual_elements = []
        
        print(f"🔍 Analyse de {len(documents)} documents...")
        
        for doc in documents:
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            
            is_visual, element_type = self._detect_visual_element(doc, metadata, content)
            
            if is_visual:
                visual_element = {
                    'type': element_type,
                    'metadata': metadata,
                    'content': content,
                    'document': doc,
                    'relevance_score': 0
                }
                visual_elements.append(visual_element)
            else:
                text_docs.append(doc)
        
        print(f"✅ Résultat: {len(text_docs)} textuels, {len(visual_elements)} visuels")
        return text_docs, visual_elements
    
    def _detect_visual_element(self, doc, metadata, content):
        """Détecte si un document est un élément visuel."""
        
        # Vérification des métadonnées
        for indicator in self.visual_keywords['image_indicators']:
            if indicator in metadata and metadata[indicator]:
                return True, 'visual_chart' if 'table' not in indicator else 'visual_table'
        
        # Vérification du type de document
        doc_type = metadata.get('type', '').lower()
        if any(visual_type in doc_type for visual_type in ['visual', 'image', 'chart', 'table']):
            return True, 'visual_chart' if 'table' not in doc_type else 'visual_table'
        
        # Vérification du contenu
        if content and self._is_table_content(content):
            return True, 'visual_table'
        
        # Vérification des mots-clés dans le contenu
        if content:
            content_lower = content.lower()
            chart_score = sum(1 for keyword in self.visual_keywords['content_patterns'] 
                            if keyword in content_lower)
            if chart_score >= 2:
                return True, 'visual_chart'
        
        return False, None
    
    def _is_table_content(self, content: str) -> bool:
        """Détecte si le contenu est tabulaire."""
        if not content or len(content.strip()) < 50:
            return False
        
        lines = content.split('\n')
        if len(lines) < 3:
            return False
        
        # Compteurs d'indicateurs
        pipe_lines = sum(1 for line in lines if '|' in line)
        tab_lines = sum(1 for line in lines if '\t' in line)
        number_lines = sum(1 for line in lines if re.search(r'\d+', line))
        
        # Score composite
        table_score = 0
        if pipe_lines >= 2: table_score += 3
        if tab_lines >= 2: table_score += 3
        if number_lines >= 3: table_score += 1
        
        return table_score >= 4
    
    def analyze_visual_relevance(self, visual_elements: List[Dict], user_question: str) -> List[Dict]:
        """Analyse la pertinence des éléments visuels."""
        if not visual_elements:
            return []
        
        question_lower = user_question.lower()
        relevant_elements = []
        
        # Mots-clés thématiques ANSD
        theme_keywords = {
            'démographie': ['population', 'habitants', 'démographique', 'rgph'],
            'économie': ['économie', 'pib', 'croissance', 'secteur'],
            'emploi': ['emploi', 'travail', 'chômage', 'enes'],
            'pauvreté': ['pauvreté', 'pauvre', 'esps'],
            'santé': ['santé', 'mortalité', 'eds'],
            'éducation': ['éducation', 'école', 'scolarisation']
        }
        
        for element in visual_elements:
            relevance_score = 0
            content = element['content'].lower()
            
            # Score thématique
            for theme, keywords in theme_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    theme_matches = sum(1 for keyword in keywords if keyword in content)
                    relevance_score += min(theme_matches, 3)
            
            # Score mots-clés directs
            question_words = set(word for word in question_lower.split() if len(word) > 3)
            content_words = set(content.split())
            common_words = question_words.intersection(content_words)
            relevance_score += min(len(common_words), 3)
            
            # Seuil de pertinence
            if relevance_score >= 2:
                element['relevance_score'] = relevance_score
                relevant_elements.append(element)
        
        # Trier et limiter à 3 éléments
        relevant_elements.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return relevant_elements[:3]

# =============================================================================
# CLASSE POUR L'AFFICHAGE DES VISUELS
# =============================================================================

# Supprimer complètement la classe VisualDisplayManager qui n'est plus nécessaire
# et les fonctions create_smooth_transition et _send_message qui sont dupliquées

# =============================================================================
# CLASSE POUR LA GÉNÉRATION DE SUGGESTIONS
# =============================================================================

class SuggestionsManager:
    """Gestionnaire pour la génération de suggestions de questions."""
    
    async def generate_suggestions(self, user_question: str, response_content: str, model) -> str:
        """Génère des suggestions de questions contextuelles."""
        
        suggestions_prompt = ChatPromptTemplate.from_messages([
            ("system", """Générez 4 questions de suivi pertinentes pour ANSD basées sur la question et réponse fournies.

RÈGLES :
✅ Questions complémentaires spécifiques au Sénégal
✅ Utilisez la terminologie ANSD (RGPH, EDS, ESPS, EHCVM, ENES)
✅ Mélangez les angles : temporel, géographique, thématique

FORMAT :
**❓ QUESTIONS SUGGÉRÉES :**

1. [Question sur l'évolution temporelle]
2. [Question sur la répartition géographique]
3. [Question sur un indicateur connexe]
4. [Question d'approfondissement]"""),
            ("user", f"Question originale: {user_question}\n\nRéponse fournie: {response_content[:500]}...")
        ])
        
        try:
            suggestions_chain = suggestions_prompt | model
            response = await suggestions_chain.ainvoke({})
            return f"\n\n{response.content}"
        except Exception as e:
            print(f"❌ Erreur suggestions: {e}")
            return self._generate_fallback_suggestions(user_question)
    
    def _generate_fallback_suggestions(self, user_question: str) -> str:
        """Génère des suggestions de base."""
        question_lower = user_question.lower()
        
        if 'population' in question_lower:
            return """
**❓ QUESTIONS SUGGÉRÉES :**

1. Quelle est l'évolution de la population sénégalaise selon les recensements ?
2. Comment la population se répartit-elle entre les régions ?
3. Quels sont les indicateurs démographiques clés du Sénégal ?
4. Quelle est la structure par âge de la population ?"""
        
        else:
            return """
**❓ QUESTIONS SUGGÉRÉES :**

1. Quels sont les derniers résultats du RGPH-5 ?
2. Comment les indicateurs sociaux ont-ils évolué ?
3. Quelles sont les disparités régionales observées ?
4. Quels défis pose la collecte de données au Sénégal ?"""

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def format_docs_with_metadata(docs) -> str:
    """Formatage des documents avec métadonnées."""
    if not docs:
        return "❌ Aucun document pertinent trouvé."
    
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # En-tête du document
        header = f"\n{'='*50}\n📊 DOCUMENT ANSD #{i}\n"
        
        # Informations source
        if 'pdf_name' in metadata:
            header += f"📋 Document: {metadata['pdf_name']}\n"
        if 'page_num' in metadata:
            header += f"📖 Page: {metadata['page_num']}\n"
        
        header += f"{'='*50}\n"
        
        # Contenu
        content = doc.page_content.strip()
        formatted_parts.append(f"{header}\n{content}\n")
    
    return "\n".join(formatted_parts)

def evaluate_response_quality(response_content: str) -> bool:
    """Évalue la qualité de la réponse."""
    response_lower = response_content.lower()
    
    # Indicateurs d'échec
    failure_indicators = [
        "information non disponible",
        "aucune information", 
        "impossible de répondre"
    ]
    
    if any(indicator in response_lower for indicator in failure_indicators):
        return False
    
    # Critères de succès
    has_numbers = bool(re.search(r'\d+', response_content))
    has_content = len(response_content) > 100
    has_structure = '**' in response_content
    
    return sum([has_numbers, has_content, has_structure]) >= 2

# =============================================================================
# FONCTIONS PRINCIPALES
# =============================================================================

async def retrieve(state, *, config):
    """Récupération avec extraction des éléments visuels."""
    print("🔍 ---RETRIEVE AVEC SUPPORT VISUEL---")
    
    # Gestion hybride du state
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])
    
    # Extraire la question
    question = ""
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            question = msg.content
            break
    
    if not question:
        return {"documents": [], "visual_elements": [], "has_visual_content": False}
    
    print(f"📝 Question: {question}")
    
    try:
        # Configuration
        safe_config = dict(config) if config else {}
        if 'configurable' not in safe_config:
            safe_config['configurable'] = {}
        
        safe_config['configurable']['search_kwargs'] = {"k": 20}
        
        # Récupération des documents
        async with retrieval.make_retriever(safe_config) as retriever:
            documents = await retriever.ainvoke(question, safe_config)
            
            print(f"✅ Documents récupérés: {len(documents)}")
            
            # Gestionnaire d'éléments visuels
            visual_manager = VisualElementsManager()
            text_docs, visual_elements = visual_manager.extract_visual_elements(documents)
            
            # Analyser la pertinence
            relevant_visuals = visual_manager.analyze_visual_relevance(visual_elements, question)
            
            print(f"📄 Documents textuels: {len(text_docs)}")
            print(f"🎨 Éléments visuels pertinents: {len(relevant_visuals)}")
            
            return {
                "documents": text_docs,
                "visual_elements": relevant_visuals,
                "has_visual_content": len(relevant_visuals) > 0
            }
            
    except Exception as e:
        print(f"❌ Erreur récupération: {e}")
        return {"documents": [], "visual_elements": [], "has_visual_content": False}

async def generate(state, *, config):
    """Génération avec intégration fluide des visuels."""
    print("🤖 ---GENERATE AVEC INTÉGRATION FLUIDE---")
    
    # Gestion hybride du state
    if isinstance(state, dict):
        messages = state.get("messages", [])
        documents = state.get("documents", [])
        visual_elements = state.get("visual_elements", [])
        has_visual = state.get("has_visual_content", False)
    else:
        messages = getattr(state, "messages", [])
        documents = getattr(state, "documents", [])
        visual_elements = getattr(state, "visual_elements", [])
        has_visual = getattr(state, "has_visual_content", False)
    
    # Configuration
    try:
        configuration = RagConfiguration.from_runnable_config(config)
        model = load_chat_model(configuration.model)
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return {"messages": [AIMessage(content="❌ Erreur de configuration.")]}
    
    # Extraire la question
    user_question = ""
    for msg in messages:
        if hasattr(msg, 'content'):
            user_question = msg.content
            break
    
    print(f"❓ Question: {user_question}")
    print(f"📄 Documents: {len(documents)}, 🎨 Visuels: {len(visual_elements)}")
    
    try:
        # ÉTAPE 1: Générer la réponse textuelle d'abord
        if documents:
            print("📝 Génération de la réponse textuelle...")
            
            # Adapter le prompt selon la présence de visuels
            visual_note = ""
            if visual_elements:
                visual_note = f"\n\nNote: {len(visual_elements)} graphique(s)/tableau(x) pertinent(s) seront affichés automatiquement avec votre réponse."
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", ANSD_SYSTEM_PROMPT + visual_note),
                ("placeholder", "{messages}")
            ])
            
            context = format_docs_with_metadata(documents)
            rag_chain = prompt | model
            
            response = await rag_chain.ainvoke({
                "context": context,
                "messages": messages
            })
            
            response_content = response.content
            
            print(f"📊 Réponse générée: {len(response_content)} caractères")
            print(f"🔍 Evaluation qualité...")
            
            # Évaluer la qualité de la réponse
            quality_ok = evaluate_response_quality(response_content)
            print(f"✅ Qualité OK: {quality_ok}")
            
            if quality_ok:
                print("✅ Réponse satisfaisante générée - Préparation visuels...")
                
                # ÉTAPE 2: Préparer la réponse avec intégration des visuels
                response_data = await display_response_with_visuals(
                    response_content, visual_elements, user_question, model
                )
                
                print(f"🔍 DEBUG RESPONSE_DATA:")
                print(f"   Content length: {len(response_data['content'])}")
                print(f"   Image path: {response_data['image_path']}")
                print(f"   Has visual: {response_data['has_visual']}")
                
                final_result = {
                    "messages": [AIMessage(content=response_data["content"])],
                    "documents": documents,
                    "visual_elements": visual_elements,
                    "has_visual_content": response_data["has_visual"],
                    "image_path": response_data["image_path"],  # ✅ MAINTENANT SUPPORTÉ
                    "best_visual": response_data["best_visual"]  # ✅ MAINTENANT SUPPORTÉ
                }
                
                print(f"🔍 DEBUG FINAL_RESULT keys: {list(final_result.keys())}")
                print(f"🔍 FINAL_RESULT image_path: {final_result['image_path']}")
                
                return final_result
            
            else:
                print("⚠️ Réponse jugée insuffisante - passage au fallback...")
                # Continuer vers le fallback...
        else:
            print("⚠️ ÉTAPE 2 IGNORÉE : Aucun document disponible - passage direct au fallback")
        
        # Fallback: utiliser les connaissances ANSD
        print("🌐 ÉTAPE 3: Utilisation des connaissances ANSD...")
        
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un expert ANSD. Répondez en utilisant vos connaissances des publications officielles ANSD."),
            ("placeholder", "{messages}")
        ])
        
        fallback_chain = fallback_prompt | model
        fallback_response = await fallback_chain.ainvoke({"messages": messages})
        
        print(f"📊 Fallback réponse générée: {len(fallback_response.content)} caractères")
        
        # Préparer la réponse fallback avec visuels si disponibles
        if visual_elements:
            print("🎨 Fallback avec visuels...")
            response_data = await display_response_with_visuals(
                fallback_response.content, visual_elements, user_question, model
            )
            
            print(f"🔍 DEBUG FALLBACK RESPONSE_DATA:")
            print(f"   Image path: {response_data['image_path']}")
            print(f"   Has visual: {response_data['has_visual']}")
            
            fallback_result = {
                "messages": [AIMessage(content=response_data["content"])],
                "documents": documents,
                "visual_elements": visual_elements,
                "image_path": response_data["image_path"],  # ✅ MAINTENANT SUPPORTÉ
                "has_visual_content": response_data["has_visual"],
                "best_visual": response_data["best_visual"]  # ✅ MAINTENANT SUPPORTÉ
            }
            
            print(f"🔍 DEBUG FALLBACK_RESULT keys: {list(fallback_result.keys())}")
            return fallback_result
        else:
            print("📝 Fallback sans visuels...")
            return {
                "messages": [AIMessage(content=fallback_response.content)],
                "documents": documents,
                "visual_elements": visual_elements,
                "image_path": None,  # ✅ MAINTENANT SUPPORTÉ
                "has_visual_content": False,
                "best_visual": None  # ✅ MAINTENANT SUPPORTÉ
            }
        
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        return {
            "messages": [AIMessage(content="❌ Erreur lors de la génération de la réponse.")],
            "documents": documents,
            "visual_elements": visual_elements,
            "has_visual_content": False,
            "image_path": None,  # ✅ MAINTENANT SUPPORTÉ
            "best_visual": None  # ✅ MAINTENANT SUPPORTÉ
        }

async def display_response_with_visuals(response_content: str, visual_elements: List[Dict], 
                                      user_question: str, model) -> Dict[str, Any]:
    """Prépare la réponse textuelle avec visuels intégrés pour l'app Chainlit."""
    
    # 1. Construire la réponse complète
    full_response = response_content
    
    # 2. Préparer les éléments visuels
    best_visual = None
    image_path = None
    graph_title = ""
    
    if visual_elements:
        print(f"🎨 Préparation de {len(visual_elements)} éléments visuels...")
        
        # Sélectionner le meilleur visuel
        best_visual = max(visual_elements, key=lambda x: x.get('relevance_score', 0))
        
        # Trouver le chemin de l'image
        image_path = find_image_path_for_visual(best_visual)
        
        if image_path and Path(image_path).exists():
            # Préparer le titre du graphique pour la fin
            metadata = best_visual['metadata']
            caption = metadata.get('caption', 'Graphique ANSD')
            graph_title = f"📊 **{caption}**"
            
            print(f"✅ Image préparée avec titre pour la fin: {image_path}")
        else:
            print(f"❌ Image non trouvée pour: {best_visual['metadata']}")
    
    # 3. Générer les suggestions
    try:
        suggestions_manager = SuggestionsManager()
        suggestions = await suggestions_manager.generate_suggestions(
            user_question, response_content, model
        )
        full_response += suggestions
    except Exception as e:
        print(f"❌ Erreur suggestions: {e}")
        full_response += "\n\n❓ Suggestions non disponibles"
    
    # 4. Ajouter les sources
    sources_count = len(visual_elements) + 10  # Estimation
    full_response += f"\n\n📚 **Sources consultées :** {sources_count} document(s) ANSD"
    
    # 5. Ajouter le titre du graphique à la FIN (juste avant l'image)
    if graph_title:
        full_response += f"\n\n{graph_title}"
    
    # 6. RETOURNER les données pour l'app Chainlit
    return {
        "content": full_response,
        "image_path": image_path,
        "best_visual": best_visual,
        "has_visual": image_path is not None
    }

def find_image_path_for_visual(visual_element: Dict) -> Optional[str]:
    """Trouve le chemin de l'image pour un élément visuel."""
    metadata = visual_element['metadata']
    
    print(f"🔍 Recherche image pour: {metadata}")
    
    # Méthode 1: Vérifier les métadonnées directes
    for key in ['image_path', 'source', 'file_path', 'path']:
        if key in metadata and metadata[key]:
            path = str(metadata[key])
            print(f"   Test métadonnée {key}: {path}")
            
            if Path(path).exists():
                print(f"   ✅ Trouvé (direct): {path}")
                return path
                
            # Essayer avec le dossier images/
            filename = Path(path).name
            test_path = Path('images') / filename
            if test_path.exists():
                print(f"   ✅ Trouvé (images/): {test_path}")
                return str(test_path)
    
    # Méthode 2: Recherche par patterns
    pdf_name = metadata.get('pdf_name', '')
    page_num = metadata.get('page_num', metadata.get('page', ''))
    
    if pdf_name and page_num:
        # Pattern basé sur votre CSV charts_index.csv
        pdf_clean = pdf_name.replace('.pdf', '').replace(' ', '*')
        
        patterns = [
            f"images/{pdf_clean}*p{page_num}*.png",
            f"images/*{pdf_clean}*{page_num}*.png",
            f"images/*p{page_num}*.png",
            f"images/*{page_num}*.png"
        ]
        
        for pattern in patterns:
            print(f"   Pattern: {pattern}")
            matches = glob.glob(pattern)
            if matches:
                print(f"   ✅ Trouvé (pattern): {matches[0]}")
                return matches[0]
    
    print(f"   ❌ Aucune image trouvée")
    return None

def create_smooth_transition(visual_element: Dict, user_question: str) -> str:
    """Crée une transition fluide vers le graphique."""
    question_lower = user_question.lower()
    
    if any(word in question_lower for word in ['graphique', 'visualisation', 'courbe']):
        return "📊 **Voici le graphique demandé :**"
    
    elif any(word in question_lower for word in ['évolution', 'tendance']):
        return "📈 **Pour visualiser cette évolution :**"
    
    elif 'population' in question_lower:
        return "📊 **Graphique démographique associé :**"
    
    else:
        return "📊 **Graphique complémentaire :**"

# Supprimer l'ancienne fonction _send_message qui n'est plus nécessaire
async def _send_message(content: str):
    """DEPRECATED - Ne plus utiliser, tout se fait en un seul cl.Message maintenant."""
    pass

# Suppression des fonctions de transition non nécessaires
async def _send_message(content: str):
    """DEPRECATED - Ne plus utiliser, tout se fait en un seul cl.Message maintenant."""
    pass

# =============================================================================
# CONFIGURATION DU WORKFLOW AVEC SUPPORT VISUEL COMPLET
# =============================================================================

workflow = StateGraph(WorkflowState, input=InputState, config_schema=RagConfiguration)

# Définir les noeuds
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Construire le graphe
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compiler
graph = workflow.compile()
graph.name = "CleanedSimpleRagWithVisualsAndSuggestions"

# =============================================================================
# FONCTIONS DE DIAGNOSTIC (OPTIONNELLES)
# =============================================================================

def check_system_status():
    """Vérifie l'état du système."""
    print("🔍 VÉRIFICATION SYSTÈME:")
    print(f"   - Chainlit: {'✅' if CHAINLIT_AVAILABLE else '❌'}")
    print(f"   - Streamlit: {'✅' if STREAMLIT_AVAILABLE else '❌'}")
    print(f"   - Dossier images: {'✅' if Path('images').exists() else '❌'}")
    
    if Path('images').exists():
        img_count = len(list(Path('images').glob('*.png')))
        print(f"   - Images PNG: {img_count} fichiers")

# Exporter les éléments principaux
__all__ = [
    'graph',
    'retrieve', 
    'generate',
    'VisualElementsManager',
    'VisualDisplayManager',
    'SuggestionsManager',
    'check_system_status'
]