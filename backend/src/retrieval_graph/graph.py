"""Simple RAG amélioré avec de meilleures capacités de récupération et de réponse."""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from shared import retrieval
from shared.utils import load_chat_model, format_docs
from simple_rag.configuration import RagConfiguration
from simple_rag.state import GraphState, InputState


# Prompt amélioré
SIMPLE_RAG_SYSTEM_PROMPT = """Vous êtes un expert statisticien de l'ANSD (Agence Nationale de la Statistique et de la Démographie du Sénégal), spécialisé dans l'analyse de données démographiques, économiques et sociales du Sénégal.

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
[Développez largement avec toutes les informations complémentaires pertinentes, sans limitation de longueur. Incluez :
- Évolutions temporelles et tendances
- Comparaisons régionales ou démographiques
- Méthodologies détaillées
- Contexte socio-économique
- Implications et analyses
- Données connexes des autres enquêtes ANSD
- Informations contextuelles des rapports officiels ANSD
Organisez en paragraphes clairs et développez chaque aspect important.]

**LIMITATIONS/NOTES :**
[Précautions d'interprétation, changements méthodologiques, définitions spécifiques]

INSTRUCTIONS POUR LES SOURCES :
- Documents fournis : "Document.pdf, page X"
- Connaissances ANSD officielles : "ANSD - [Nom de l'enquête/rapport], [année]"
- Site officiel : "Site officiel ANSD (www.ansd.sn)"
- Distinguez clairement : "Selon les documents fournis..." vs "D'après les publications ANSD..."

Si aucune information n'est disponible (documents + connaissances ANSD) :
"❌ Cette information n'est pas disponible dans les documents fournis ni dans les publications ANSD consultées. 
📞 Pour obtenir cette donnée spécifique, veuillez consulter directement l'ANSD (www.ansd.sn) ou leurs services techniques spécialisés."

DOCUMENTS ANSD DISPONIBLES :
{context}

Analysez maintenant ces documents et répondez à la question de l'utilisateur de manière complète et approfondie."""


def preprocess_query(query: str) -> str:
    """Prétraitement de la requête pour améliorer la recherche."""
    # Ajouter des termes synonymes pour améliorer la recherche
    query_lower = query.lower()
    
    # Dictionnaire de synonymes/termes connexes
    synonyms = {
        "population": ["habitants", "démographie", "recensement", "nombre d'habitants"],
        "économie": ["PIB", "croissance", "économique", "revenus"],
        "pauvreté": ["pauvre", "indigence", "vulnérabilité"],
        "éducation": ["école", "alphabétisation", "scolarisation"],
        "santé": ["mortalité", "morbidité", "espérance de vie"],
        "emploi": ["chômage", "travail", "activité économique"],
    }
    
    # Enrichir la requête avec des synonymes pertinents
    enriched_terms = []
    for key, values in synonyms.items():
        if key in query_lower:
            enriched_terms.extend(values[:2])  # Ajouter 2 synonymes max
    
    if enriched_terms:
        return f"{query} {' '.join(enriched_terms)}"
    
    return query


async def retrieve(state: GraphState, *, config: RagConfiguration) -> dict[str, list | str]: 
    """Récupération améliorée de documents."""
    print("---RETRIEVE AMÉLIORÉ---")
    
    # Extraire et prétraiter la question
    question = " ".join(msg.content for msg in state.messages if isinstance(msg, HumanMessage))
    processed_question = preprocess_query(question)
    
    print(f"Question originale: {question}")
    print(f"Question enrichie: {processed_question}")

    # Configuration de recherche améliorée
    enhanced_config = dict(config)
    if 'configurable' not in enhanced_config:
        enhanced_config['configurable'] = {}
    
    # Paramètres de recherche optimisés
    enhanced_config['configurable']['search_kwargs'] = {
        'k': 15,  # Récupérer plus de documents
        'fetch_k': 50,  # Chercher dans plus de candidats
    }

    # Récupération
    with retrieval.make_retriever(enhanced_config) as retriever:
        documents = retriever.invoke(processed_question)
    
    print(f"Nombre de documents récupérés: {len(documents)}")
    
    # Filtrer et scorer les documents (optionnel)
    scored_documents = []
    for doc in documents:
        content_lower = doc.page_content.lower()
        score = 0
        
        # Simple scoring basé sur les mots-clés de la question
        question_words = question.lower().split()
        for word in question_words:
            if len(word) > 3:  # Ignorer les mots très courts
                score += content_lower.count(word)
        
        scored_documents.append((score, doc))
    
    # Trier par score et garder les meilleurs
    scored_documents.sort(key=lambda x: x[0], reverse=True)
    best_documents = [doc for score, doc in scored_documents[:10]]  # Top 10
    
    print(f"Documents sélectionnés après scoring: {len(best_documents)}")
    
    return {"documents": best_documents, "messages": state.messages}


async def generate(state: GraphState, *, config: RagConfiguration):
    """Génération améliorée de réponse."""
    print("---GENERATE AMÉLIORÉ---")
    
    messages = state.messages
    documents = state.documents

    # Création du prompt amélioré
    prompt = ChatPromptTemplate.from_messages([
        ("system", SIMPLE_RAG_SYSTEM_PROMPT),
        ("placeholder", "{messages}")
    ])
    
    configuration = RagConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    # Formatage amélioré des documents avec métadonnées
    def format_docs_with_metadata(docs):
        if not docs:
            return "Aucun document pertinent trouvé."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            metadata_info = ""
            if doc.metadata:
                source = doc.metadata.get('source', 'Source inconnue')
                metadata_info = f" (Source: {source})"
            
            formatted.append(f"Document {i}{metadata_info}:\n{doc.page_content}\n")
        
        return "\n".join(formatted)

    # Chaîne améliorée
    rag_chain = prompt | model
    
    # Génération avec contexte enrichi
    context = format_docs_with_metadata(documents)
    
    print(f"Contexte généré (premiers 300 caractères): {context[:300]}...")
    
    response = await rag_chain.ainvoke({
        "context": context,
        "messages": messages
    })
    
    print(f"Réponse générée: {response.content[:200]}...")
    
    return {"messages": [response], "documents": documents}


# Configuration du workflow
workflow = StateGraph(GraphState, input=InputState, config_schema=RagConfiguration)

# Définir les nœuds
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Construire le graphe
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compiler
graph = workflow.compile()
graph.name = "ImprovedSimpleRag"