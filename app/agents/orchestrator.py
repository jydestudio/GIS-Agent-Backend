"""
Orchestrator Agent — Multi-tool routing and conversation management.

This agent acts as the main entry point, deciding which tool to call
based on user input or responding directly for general inquiries.
"""

from typing import Annotated, Any, Literal, TypedDict, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.core.logging import get_logger
from app.services.llm import get_llm_service
from app.agents.tools.gis_tools import download_admin_boundary, get_slope_map, get_contour_map
from app.agents.tools.intent_tool import extract_project_intent
from app.agents.tools.map_making import create_cartographic_map
from app.agents.prompts import INTENT_EXTRACTION_SYSTEM_PROMPT
from app.models.schemas import ProjectIntent
from app.agents.tools.gee_tools import search_gee_catalog
from app.agents.tools.analysis_runner import run_gee_analysis

import json

logger = get_logger(__name__)

# ── Tools ────────────────────────────────────────

TOOLS = [
    extract_project_intent,
    download_admin_boundary,
    create_cartographic_map,
    search_gee_catalog,
    run_gee_analysis,
]

tool_node = ToolNode(TOOLS)

# ── Graph State ──────────────────────────────────

class AgentState(TypedDict):
    """Primary state for the GeoAI Orchestrator."""
    messages: Annotated[list[BaseMessage], add_messages]
    plan_approved: bool
    intent: Optional[ProjectIntent]


# ── Helper Functions ────────────────────────────

async def classify_interaction(state: AgentState) -> dict:
    """
    Returns a structured classification of the user's intent.
    """
    messages = state["messages"]
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"is_approval": False, "is_change": False, "is_query": True}
        
    last_user_msg = user_messages[-1].content
    
    prompt = f"""
    Analyze the user's latest message in a GIS mapping conversation.
    User Message: "{last_user_msg}"
    
    Determine the following (True/False):
    1. is_approval: Does the user agree with the current plan or say 'proceed/start'?
    2. is_change: Does the user want to modify the location, data, or criteria?
    3. is_query: Is the user just asking a general question?

    Return ONLY a valid JSON object: {{"is_approval": bool, "is_change": bool, "is_query": bool}}
    """
    

    llm = get_llm_service().llm
    # Tag this internal call so it's hidden from the stream
    response = await llm.ainvoke(prompt, config={"tags": ["hide_from_stream"]})
    content = response.content
    
    # Simple robust parsing for internal check
    import re
    
    # Find JSON structure in content
    match = re.search(r"\{.*\}", str(content), re.DOTALL)
    if not match:
        logger.warning("No JSON found in classification response: %s", content)
        return {"is_approval": False, "is_change": False, "is_query": True}
        
    try:
        # strict=False handles common control character issues in LLM JSON
        return json.loads(match.group(0), strict=False)
    except Exception as e:
        logger.error("Failed to parse classification JSON: %s", str(e))
        return {"is_approval": False, "is_change": False, "is_query": True}


# ── Node Functions ───────────────────────────────

async def call_model(state: AgentState) -> dict[str, Any]:
    """
    Core agent logic: Planning or Execution.
    """
    logger.info("orchestrator — calling model")
    llm_service = get_llm_service()
    messages = state["messages"]

    # 1. READ the persistent state
    # We use .get() to avoid KeyErrors on the first message
    is_approved = state.get("plan_approved", True)
    
    # 2. CLASSIFY the interaction
    # Note: Only classify if this isn't the first message
    # if len(messages) > 1:
    #     intent_data = await classify_interaction(state)

    #     if intent_data["is_change"]:
    #         is_approved = False
    #     elif intent_data["is_approval"]:
    #         is_approved = True
        # If it's a query, is_approved stays as it was

    # 3. PREPARE messages with System Persona
    system_msg = SystemMessage(content=INTENT_EXTRACTION_SYSTEM_PROMPT)
    full_messages = [system_msg] + messages
    last_message = messages[-1]

    # 4. ROUTE based on the corrected 'is_approved' variable
    if not is_approved:
        # --- PLANNING PHASE ---
        response = await llm_service.llm.ainvoke(full_messages)
        
        try:
            # History is everything EXCEPT the very last user message
            intent = await llm_service.extract_intent(last_message.content, history=messages[:-1])
        except Exception as e:
            logger.warning("Failed silent intent extraction: %s", str(e))
            intent = None

        return {
            "messages": [response], 
            "plan_approved": False, # Save state back to graph
            "intent": intent
        }

    else:
        # --- EXECUTION PHASE ---
        llm = llm_service.llm.bind_tools(TOOLS)
        response = await llm.ainvoke(full_messages)
        
        return {
            "messages": [response],
            "plan_approved": True # Keep state as approved
        }



def should_continue(state: AgentState) -> Literal["tools", END]:
    """Route based on whether the last message is a tool call."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ── Graph Builder ────────────────────────────────

def build_orchestrator_graph() -> StateGraph:
    """
    Build the orchestrator graph.
    agent → tools → agent → ... → END
    """
    graph = StateGraph(AgentState)
    
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    
    graph.set_entry_point("agent")
    
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    graph.add_edge("tools", "agent")
    
    return graph


# ── Compiled graph singleton ─────────────────────

_compiled_graph = None
_checkpointer = MemorySaver()

def get_orchestrator_graph():
    """Get or create the orchestrator graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_orchestrator_graph().compile(checkpointer=_checkpointer)
    return _compiled_graph
