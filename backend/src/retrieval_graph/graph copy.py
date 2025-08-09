"""Main entrypoint for the conversational retrieval graph adapted for ANSD.

This module defines the core structure and functionality of the conversational
retrieval graph for ANSD (Agence Nationale de la Statistique et de la Démographie).
"""

from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.state import AgentState, InputState, Router
from shared.utils import format_docs, load_chat_model


async def route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Route the user query to the appropriate handler based on classification.

    Args:
        state (AgentState): The current state including the user's message.
        config (RunnableConfig): Configuration with the model used for routing.

    Returns:
        dict[str, Router]: The routing decision with logic explanation.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Router)
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages
    
    try:
        response = cast(Router, await model.ainvoke(messages))
        print(f"🎯 Classification: {response['type']} - {response['logic']}")
        return {"router": response}
    except Exception as e:
        print(f"❌ Erreur lors du routage: {e}")
        # Routage par défaut vers ANSD
        return {"router": Router(type="ansd", logic="Erreur de classification, routage par défaut vers ANSD")}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to ANSD."""
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.general_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Ask the user for more information when the query is unclear."""
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering an ANSD-related query."""

    class Plan(TypedDict):
        """Generate research plan."""
        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Plan)
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages
    
    try:
        response = cast(Plan, await model.ainvoke(messages))
        steps = response.get("steps", [])
        
        if not steps:
            print("⚠️  Aucune étape générée, création d'une étape par défaut")
            user_question = ""
            for msg in reversed(state.messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    user_question = msg.content
                    break
            
            steps = [f"Rechercher des informations sur: {user_question}"]
        
        print(f"✅ Étapes de recherche créées: {steps}")
        return {"steps": steps, "documents": "delete"}
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du plan de recherche: {e}")
        user_question = ""
        for msg in reversed(state.messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                user_question = msg.content
                break
        
        fallback_steps = [f"Rechercher des informations sur: {user_question}"]
        print(f"🔧 Plan de secours créé: {fallback_steps}")
        return {"steps": fallback_steps, "documents": "delete"}


async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan."""
    print(f"🔍 État actuel des étapes: {state.steps}")
    
    if not state.steps:
        print("❌ 'state.steps' est vide dans 'conduct_research'")
        user_question = ""
        for msg in reversed(state.messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                user_question = msg.content
                break
        
        if user_question:
            default_step = f"Rechercher des informations sur: {user_question}"
            print(f"🔧 Création d'une étape par défaut: {default_step}")
            result = await researcher_graph.ainvoke({"question": default_step})
            return {"documents": result["documents"], "steps": []}
        else:
            print("⚠️  Aucune question utilisateur trouvée, retour de documents vides")
            return {"documents": [], "steps": []}
    
    current_step = state.steps[0]
    print(f"🚀 Exécution de l'étape: {current_step}")
    
    try:
        result = await researcher_graph.ainvoke({"question": current_step})
        remaining_steps = state.steps[1:] if len(state.steps) > 1 else []
        print(f"✅ Recherche terminée. Étapes restantes: {remaining_steps}")
        return {"documents": result["documents"], "steps": remaining_steps}
    except Exception as e:
        print(f"❌ Erreur lors de la recherche: {e}")
        return {"documents": [], "steps": state.steps[1:] if len(state.steps) > 1 else []}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed."""
    remaining_steps = len(state.steps or [])
    print(f"🎯 Vérification de fin: {remaining_steps} étapes restantes")
    
    if remaining_steps > 0:
        print("🔄 Plus d'étapes à exécuter, continuation de la recherche")
        return "conduct_research"
    else:
        print("✅ Recherche terminée, génération de la réponse")
        return "respond"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response based on ANSD documents."""
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    context = format_docs(state.documents)
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    
    print(f"💬 Génération de la réponse avec {len(state.documents)} documents ANSD")
    response = await model.ainvoke(messages)
    return {"messages": [response]}


def route_after_classification(state: AgentState) -> Literal["general_response", "more_info_response", "create_research_plan"]:
    """Route to the appropriate handler based on the classification."""
    router_type = state.router["type"]
    print(f"🔀 Routage vers: {router_type}")
    
    if router_type == "general":
        return "general_response"
    elif router_type == "more-info":
        return "more_info_response"
    elif router_type == "ansd":  # Changé de "langchain" vers "ansd"
        return "create_research_plan"
    else:
        # Par défaut, traiter comme une question ANSD
        return "create_research_plan"


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)

# Ajouter tous les nœuds
builder.add_node("route_query", route_query)
builder.add_node("general_response", respond_to_general_query)
builder.add_node("more_info_response", ask_for_more_info)
builder.add_node("create_research_plan", create_research_plan)
builder.add_node("conduct_research", conduct_research)
builder.add_node("respond", respond)

# Définir le flux
builder.add_edge(START, "route_query")
builder.add_conditional_edges("route_query", route_after_classification)

# Les réponses générales et demandes d'info se terminent directement
builder.add_edge("general_response", END)
builder.add_edge("more_info_response", END)

# Le flux de recherche ANSD
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("respond", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "ANSDRetrievalGraph"