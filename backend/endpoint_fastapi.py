# backend_server_sunustat.py - Backend avec mise en forme SunuStat CORRIGÉ
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn
import asyncio
from sse_starlette.sse import EventSourceResponse
import json
import sys
from pathlib import Path

app = FastAPI(title="SunuStat ANSD - LangGraph API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Import amélioré du Simple RAG
RAG_AVAILABLE = False
simple_rag_graph = None
RagConfiguration = None

def setup_rag_system():
    """Configuration améliorée du système RAG"""
    global RAG_AVAILABLE, simple_rag_graph, RagConfiguration
    
    try:
        # Déterminer le chemin vers src/ de manière plus robuste
        current_dir = Path(__file__).parent
        src_path = current_dir / "src"
        
        # Essayer différents chemins possibles
        possible_paths = [
            src_path,
            current_dir.parent / "src",
            Path.cwd() / "src"
        ]
        
        for path in possible_paths:
            if path.exists():
                sys.path.insert(0, str(path))
                print(f"📁 Chemin src ajouté: {path}")
                break
        
        # Import du Simple RAG avec gestion d'erreurs détaillée
        from simple_rag.graph import graph as simple_rag_graph
        from simple_rag.configuration import RagConfiguration
        
        RAG_AVAILABLE = True
        print("✅ Simple RAG chargé avec succès dans le backend SunuStat")
        
        # Vérification optionnelle de la configuration
        default_config = RagConfiguration()
        print(f"⚙️ Configuration RAG: {default_config.model}")
        
    except ImportError as e:
        print(f"❌ Erreur d'import Simple RAG: {e}")
        print(f"📂 Répertoire courant: {Path.cwd()}")
        print(f"🐍 Python path: {sys.path[:3]}...")
        print("⚠️ Le backend fonctionnera en mode démo")
    except Exception as e:
        print(f"❌ Erreur inattendue lors du setup RAG: {e}")

# Initialiser le système RAG au démarrage
setup_rag_system()

# Models Pydantic
class Message(BaseModel):
    content: str
    type: str = "human"

class RunRequest(BaseModel):
    messages: List[Message]
    config: Dict[str, Any] = {}

class Assistant(BaseModel):
    assistant_id: str
    name: str
    graph_id: str
    metadata: Dict[str, Any]

class Thread(BaseModel):
    thread_id: str
    created_at: str

# Fonctions utilitaires améliorées
async def call_simple_rag(user_input: str, chat_history: list):
    """Appelle le Simple RAG avec le message utilisateur - Version backend améliorée"""
    
    if not RAG_AVAILABLE:
        return generate_demo_response(user_input), []
    
    try:
        # Convertir l'historique en messages LangChain
        messages = []
        for user_msg, bot_msg in chat_history:
            if isinstance(user_msg, str) and isinstance(bot_msg, str):
                messages.append(HumanMessage(content=user_msg))
                messages.append(AIMessage(content=bot_msg))
        
        # Ajouter le message actuel
        messages.append(HumanMessage(content=user_input))
        
        print(f"🔍 Backend: Appel Simple RAG avec {len(messages)} messages")
        
        # Configuration par défaut si disponible
        config = None
        if RagConfiguration:
            try:
                config = {"configurable": {"model": "openai/gpt-4o-mini"}}
            except:
                config = None
        
        # Appeler le graphique Simple RAG avec timeout
        result = await asyncio.wait_for(
            simple_rag_graph.ainvoke({"messages": messages}, config=config),
            timeout=60.0  # 60 secondes timeout
        )
        
        # Extraire la réponse
        answer = "❌ Aucune réponse générée par Simple RAG"
        if "messages" in result and result["messages"]:
            answer = result["messages"][-1].content
            print(f"✅ Backend: Réponse générée: {len(answer)} caractères")
        
        # Extraire les documents sources
        sources = result.get("documents", [])
        print(f"📄 Backend: Documents récupérés: {len(sources)}")
        
        # Extraire les informations visuelles si disponibles
        visual_info = {
            "has_visual": result.get("has_visual_content", False),
            "image_path": result.get("image_path"),
            "best_visual": result.get("best_visual")
        }
        
        if visual_info["has_visual"]:
            print(f"🖼️ Backend: Contenu visuel détecté: {visual_info['image_path']}")
        
        return answer, sources, visual_info
        
    except asyncio.TimeoutError:
        print("⏰ Backend: Timeout lors de l'appel Simple RAG")
        return "⏰ La réponse prend trop de temps. Veuillez réessayer avec une question plus spécifique.", []
    except Exception as e:
        print(f"❌ Backend: Erreur Simple RAG: {e}")
        import traceback
        traceback.print_exc()
        return generate_demo_response(user_input), []

def generate_demo_response(user_input: str) -> str:
    """Génère une réponse de démonstration formatée SunuStat"""
    demo_responses = {
        "population": """**📊 Population du Sénégal (RGPH 2023)**

Selon les dernières données du Recensement Général de la Population et de l'Habitat (RGPH) :

• **Population totale :** 18 275 743 habitants
• **Croissance démographique :** 2,8% par an
• **Densité :** 93 habitants/km²

**🌍 Répartition régionale :**
• Dakar : 4 029 724 habitants (22,0%)
• Thiès : 2 076 809 habitants (11,4%)
• Diourbel : 1 739 748 habitants (9,5%)

*Source : ANSD - RGPH 2023 (données provisoires)*""",

        "pauvreté": """**💰 Indicateurs de Pauvreté au Sénégal**

Selon l'Enquête Harmonisée sur les Conditions de Vie des Ménages (EHCVM) 2018-2019 :

• **Taux de pauvreté national :** 37,8%
• **Pauvreté rurale :** 53,2%
• **Pauvreté urbaine :** 23,7%

**📈 Évolution :**
• 2011 : 46,7%
• 2018-2019 : 37,8%
• Baisse de 8,9 points

*Source : ANSD - EHCVM 2018-2019*""",

        "emploi": """**👔 Situation de l'Emploi au Sénégal**

D'après l'Enquête Nationale sur l'Emploi au Sénégal (ENES) :

• **Taux d'activité :** 49,2%
• **Taux de chômage :** 16,9%
• **Chômage des jeunes (15-34 ans) :** 22,7%

**🏢 Secteurs d'activité :**
• Agriculture : 35,8%
• Services : 38,2%
• Industrie : 26,0%

*Source : ANSD - ENES (dernières données disponibles)*""",

        "education": """**🎓 Statistiques de l'Éducation au Sénégal**

Données du Ministère de l'Éducation nationale :

• **Taux de scolarisation primaire :** 84,3%
• **Taux d'alphabétisation :** 51,9%
• **Parité filles/garçons (primaire) :** 1,04

**📚 Répartition par niveau :**
• Primaire : 2 100 000 élèves
• Moyen/Secondaire : 850 000 élèves
• Supérieur : 180 000 étudiants

*Source : ANSD - Statistiques scolaires*"""
    }
    
    # Recherche de mots-clés pour retourner une réponse appropriée
    user_lower = user_input.lower()
    if any(word in user_lower for word in ["population", "habitants", "rgph", "recensement", "démographie"]):
        return demo_responses["population"]
    elif any(word in user_lower for word in ["pauvreté", "pauvre", "ehcvm", "conditions", "revenus"]):
        return demo_responses["pauvreté"]
    elif any(word in user_lower for word in ["emploi", "chômage", "travail", "enes", "activité"]):
        return demo_responses["emploi"]
    elif any(word in user_lower for word in ["éducation", "école", "alphabétisation", "scolarisation"]):
        return demo_responses["education"]
    else:
        return f"""**📊 SunuStat - ANSD**

Votre question : "{user_input}"

Cette réponse est générée en mode démonstration. Pour des données réelles, assurez-vous que le module Simple RAG est correctement configuré.

**📋 Types de données disponibles :**
• **Démographie** - Population, natalité, mortalité (RGPH)
• **Économie** - Pauvreté, emploi, revenus (EHCVM, ENES)
• **Social** - Éducation, santé, alphabétisation (EDS)
• **Géographie** - Régions, départements, communes

**🎯 Enquêtes ANSD :**
• RGPH (Recensement Population/Habitat)
• EDS (Enquête Démographique et Santé)
• ESPS (Enquête Suivi Pauvreté Sénégal)
• EHCVM (Enquête Conditions Vie Ménages)
• ENES (Enquête Nationale Emploi)

*Posez une question plus spécifique pour obtenir des statistiques détaillées.*"""

def process_special_commands(content: str) -> Optional[str]:
    """Traite les commandes spéciales"""
    if content.lower() == "/help":
        return """**🆘 Aide SunuStat - ANSD**

**📋 Commandes disponibles :**
• `/help` - Afficher cette aide
• `/clear` - Effacer l'historique de conversation
• `/status` - État du système RAG

**📊 Types de données disponibles :**
• **Démographiques** - Population, natalité, mortalité
• **Économiques** - PIB, pauvreté, emploi, croissance
• **Sociales** - Éducation, santé, alphabétisation
• **Géographiques** - Régions, départements, communes

**🎯 Types d'enquêtes ANSD :**
• **RGPH** - Recensement (données population/habitat)
• **EDS** - Enquête Démographique et Santé
• **ESPS** - Enquête Suivi Pauvreté Sénégal
• **EHCVM** - Enquête Conditions Vie Ménages
• **ENES** - Enquête Nationale Emploi Sénégal

**💡 Conseils pour de meilleures réponses :**
• Soyez spécifique dans vos questions
• Mentionnez l'année si important
• Précisez la région si nécessaire
• Demandez des sources précises

**🔧 Backend :** LangGraph + Simple RAG"""
    
    elif content.lower() == "/clear":
        return """🧹 **Historique effacé**

Vous pouvez recommencer une nouvelle conversation."""
    
    elif content.lower() == "/status":
        status_msg = f"""**🔧 État du Système SunuStat**

• **Simple RAG :** {'✅ Actif' if RAG_AVAILABLE else '❌ Indisponible'}
• **Backend :** ✅ Opérationnel
• **Mode :** {'Production' if RAG_AVAILABLE else 'Démonstration'}

**📊 Capacités :**
• Recherche dans documents ANSD
• Génération de réponses contextuelles
• Support des formats markdown
• Streaming de réponses"""
        return status_msg
    
    return None

def format_sources_info(sources: List, visual_info: Dict = None) -> str:
    """Formate les informations des sources avec support visuel"""
    info_parts = []
    
    # Informations sur les sources textuelles
    if sources and len(sources) > 0:
        info_parts.append(f"\n\n📚 **Sources consultées :** {len(sources)} document(s) ANSD")
        
        # Détails des sources principales
        if len(sources) <= 3:
            details = "\n\n📄 **Détails des sources :**\n"
            for i, doc in enumerate(sources, 1):
                if hasattr(doc, 'metadata') and doc.metadata:
                    pdf_name = doc.metadata.get('pdf_name', doc.metadata.get('source', 'Document ANSD'))
                    page_num = doc.metadata.get('page_num', 'N/A')
                    if '/' in pdf_name:
                        pdf_name = pdf_name.split('/')[-1]
                    details += f"• **Source {i}:** {pdf_name}"
                    if page_num != 'N/A':
                        details += f" (page {page_num})"
                    details += "\n"
                else:
                    details += f"• **Source {i}:** Document ANSD\n"
            info_parts.append(details)
    
    # Informations sur le contenu visuel
    if visual_info and visual_info.get("has_visual"):
        info_parts.append(f"\n🖼️ **Contenu visuel :** Graphique/tableau disponible")
        if visual_info.get("image_path"):
            info_parts.append(f"📁 **Fichier :** {visual_info['image_path']}")
    
    return "".join(info_parts)

# Routes de l'API (inchangées mais avec gestion d'erreurs améliorée)

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "service": "SunuStat ANSD - Backend",
        "version": "1.2.0",
        "description": "API pour l'assistant statistique du Sénégal",
        "rag_available": RAG_AVAILABLE,
        "rag_system": "Simple RAG" if RAG_AVAILABLE else "Demo Mode",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "assistants": "/assistants/search",
            "threads": "/threads",
            "stream": "/threads/{thread_id}/runs/stream"
        }
    }

@app.get("/assistants/search")
async def search_assistants(graph_id: str = None):
    """Recherche d'assistants - SunuStat ANSD"""
    return [
        Assistant(
            assistant_id="sunustat-ansd-assistant",
            name="SunuStat - Assistant Statistiques Sénégal",
            graph_id="simple_rag",
            metadata={
                "created_by": "system",
                "description": "Assistant intelligent pour les statistiques officielles du Sénégal (ANSD)",
                "country": "Sénégal",
                "organization": "ANSD",
                "data_sources": ["RGPH", "EDS", "ESPS", "EHCVM", "ENES"],
                "rag_available": RAG_AVAILABLE,
                "version": "1.2.0",
                "capabilities": [
                    "Données démographiques",
                    "Statistiques de pauvreté",
                    "Indicateurs d'emploi",
                    "Données d'éducation",
                    "Support visuel"
                ]
            }
        )
    ]

@app.post("/threads")
async def create_thread():
    """Création d'un nouveau thread SunuStat"""
    import uuid
    from datetime import datetime
    
    thread_id = str(uuid.uuid4())
    print(f"🧵 Nouveau thread créé: {thread_id}")
    
    return Thread(
        thread_id=thread_id,
        created_at=datetime.now().isoformat()
    )

@app.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: RunRequest):
    """Stream d'exécution SunuStat avec formatage ANSD amélioré"""
    
    async def generate_stream():
        try:
            # Extraire la dernière question
            if not request.messages:
                yield {
                    "event": "events",
                    "data": {
                        "event": "on_chat_model_stream",
                        "data": {"chunk": {"content": "❌ Aucun message reçu"}}
                    }
                }
                return

            last_message = request.messages[-1]
            query = last_message.content.strip()
            
            print(f"📝 Thread {thread_id}: Traitement de '{query[:50]}...'")
            
            # Traiter les commandes spéciales
            special_response = process_special_commands(query)
            if special_response:
                # Envoyer la réponse de commande spéciale
                for word in special_response.split():
                    yield {
                        "event": "events",
                        "data": {
                            "event": "on_chat_model_stream",
                            "data": {"chunk": {"content": word + " "}}
                        }
                    }
                    await asyncio.sleep(0.02)
                return
            
            # Émission de l'événement de début avec style SunuStat
            yield {
                "event": "events",
                "data": {
                    "event": "on_retrieval_start",
                    "data": {
                        "input": "🔍 Recherche en cours dans les documents ANSD...",
                        "service": "SunuStat",
                        "query": query,
                        "rag_mode": "Simple RAG" if RAG_AVAILABLE else "Demo"
                    }
                }
            }
            
            # Simulation de la progression avec détails
            progress_steps = [
                "• 📄 Récupération des documents ANSD",
                "• 🔍 Analyse sémantique des données", 
                "• 📊 Traitement des statistiques",
                "• ✍️ Génération de la réponse..."
            ]
            
            for step in progress_steps:
                yield {
                    "event": "events",
                    "data": {
                        "event": "on_progress",
                        "data": {"step": step}
                    }
                }
                await asyncio.sleep(0.4)
            
            # Récupérer l'historique des messages précédents
            chat_history = []
            messages = request.messages[:-1]  # Tous sauf le dernier
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    user_msg = messages[i].content
                    bot_msg = messages[i + 1].content
                    chat_history.append((user_msg, bot_msg))
            
            # Appeler Simple RAG ou générer réponse démo
            rag_result = await call_simple_rag(query, chat_history[-5:])  # Limiter l'historique
            
            # Traiter le résultat selon le format retourné
            if len(rag_result) == 3:
                answer, sources, visual_info = rag_result
            else:
                answer, sources = rag_result
                visual_info = {}
            
            yield {
                "event": "events", 
                "data": {
                    "event": "on_retrieval_end",
                    "data": {
                        "output": f"✅ Analyse terminée - {len(sources) if sources else 0} documents consultés",
                        "documents_found": len(sources) if sources else 0,
                        "has_visual": visual_info.get("has_visual", False) if visual_info else False
                    }
                }
            }
            
            # Formater la réponse finale avec style SunuStat
            formatted_response = f"**📊 SunuStat - ANSD répond :**\n\n{answer}"
            formatted_response += format_sources_info(sources, visual_info)
            
            # Stream de la réponse formatée avec vitesse ajustée
            words = formatted_response.split()
            for i, word in enumerate(words):
                yield {
                    "event": "events",
                    "data": {
                        "event": "on_chat_model_stream",
                        "data": {"chunk": {"content": word + " "}}
                    }
                }
                # Vitesse variable selon le contenu
                if word.startswith("**") or word.startswith("•"):
                    await asyncio.sleep(0.05)  # Plus lent pour les titres
                else:
                    await asyncio.sleep(0.02)  # Plus rapide pour le texte normal
                    
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Erreur dans stream_run: {e}")
            print(f"📋 Détails: {error_details}")
            
            error_msg = f"""❌ **Erreur technique**

Une erreur s'est produite lors du traitement de votre demande :
`{str(e)}`

**🔧 Suggestions :**
• Vérifiez que votre question est claire
• Réessayez avec une formulation différente
• Utilisez `/help` pour voir les commandes disponibles
• Contactez l'administrateur si le problème persiste

**📊 Mode actuel :** {'Simple RAG' if RAG_AVAILABLE else 'Démonstration'}"""

            for word in error_msg.split():
                yield {
                    "event": "events",
                    "data": {
                        "event": "on_chat_model_stream",
                        "data": {"chunk": {"content": word + " "}}
                    }
                }
                await asyncio.sleep(0.03)
    
    return EventSourceResponse(generate_stream())

@app.get("/health")
async def health_check():
    """Health check avec informations SunuStat détaillées"""
    return {
        "status": "healthy",
        "service": "SunuStat ANSD Backend",
        "version": "1.2.0",
        "rag_available": RAG_AVAILABLE,
        "simple_rag_loaded": RAG_AVAILABLE,
        "system_info": {
            "python_version": sys.version.split()[0],
            "current_directory": str(Path.cwd()),
            "src_path_exists": (Path.cwd() / "src").exists()
        },
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }

@app.get("/status")
async def status():
    """Statut détaillé du système"""
    return {
        "service": "SunuStat - ANSD Backend",
        "version": "1.2.0",
        "rag_system": {
            "available": RAG_AVAILABLE,
            "type": "Simple RAG" if RAG_AVAILABLE else "Demo Mode",
            "configuration_loaded": RagConfiguration is not None
        },
        "data_sources": ["RGPH", "EDS", "ESPS", "EHCVM", "ENES"],
        "country": "Sénégal",
        "organization": "ANSD",
        "capabilities": [
            "Questions démographiques",
            "Statistiques de pauvreté", 
            "Données d'emploi",
            "Indicateurs de santé",
            "Statistiques d'éducation",
            "Support de contenu visuel",
            "Streaming de réponses",
            "Commandes spéciales"
        ],
        "endpoints": {
            "total": 6,
            "health_check": "/health",
            "status": "/status", 
            "assistants": "/assistants/search",
            "threads": "/threads",
            "streaming": "/threads/{thread_id}/runs/stream",
            "root": "/"
        }
    }

# Gestionnaire d'événements de démarrage
@app.on_event("startup")
async def startup_event():
    """Événements au démarrage du serveur"""
    print("🚀 SunuStat ANSD Backend - Démarrage terminé")
    print(f"📊 Simple RAG: {'✅ Opérationnel' if RAG_AVAILABLE else '❌ Mode démo'}")
    if RAG_AVAILABLE:
        print("🎯 Prêt à traiter les requêtes avec données ANSD")
    else:
        print("⚠️ Fonctionnement en mode démonstration")

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage lors de l'arrêt"""
    print("👋 SunuStat ANSD Backend - Arrêt en cours...")

if __name__ == "__main__":
    print("🇸🇳 Démarrage SunuStat ANSD Backend v1.2.0")
    print(f"📊 Simple RAG disponible: {RAG_AVAILABLE}")
    
    if RAG_AVAILABLE:
        print("✅ Backend prêt avec Simple RAG intégré")
    else:
        print("⚠️ Backend en mode démonstration")
        print("💡 Vérifiez que le dossier 'src/' est accessible")
    
    print("🌐 Serveur démarrant sur http://localhost:8001")
    
    uvicorn.run(
        "backend_server_sunustat:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )