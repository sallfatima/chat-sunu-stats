from typing import Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """État du graphe pour le RAG avec support visuel."""
    
    # Messages de la conversation
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Documents textuels récupérés
    documents: List[Any]
    
    # ✅ AJOUT CRITIQUE : Éléments visuels 
    visual_elements: List[Dict[str, Any]]
    
    # ✅ AJOUT CRITIQUE : Flag pour indiquer la présence de contenu visuel
    has_visual_content: bool

class InputState(TypedDict):
    """État d'entrée pour le graphe."""
    
    # Message initial de l'utilisateur
    messages: Annotated[List[BaseMessage], add_messages]
