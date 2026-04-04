"""
LangGraph-based intent extraction agent.

Day 1 graph: parse_intent → validate_intent → END
Designed to be extended with additional nodes in future phases
(data_shopping, slope_analysis, ahp_matrix, cartography, etc.).
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.core.logging import get_logger
from app.models.schemas import ProjectIntent
from app.services.llm import get_llm_service

logger = get_logger(__name__)


# ── Graph State ──────────────────────────────────


class IntentState(TypedDict):
    """State that flows through the intent extraction graph."""

    prompt: str
    messages: Annotated[list[BaseMessage], add_messages]  # accumulated conversation history
    intent: ProjectIntent | None
    is_valid: bool
    error: str | None


# ── Node Functions ───────────────────────────────


async def parse_intent(state: IntentState) -> dict[str, Any]:
    """
    Node 1: Call the LLM to extract structured intent from the prompt.

    On failure, sets error and returns None intent so the graph
    can route to an error handler in future iterations.
    """
    logger.info("parse_intent — processing prompt")

    try:
        llm_service = get_llm_service()
        history = state.get("messages") or []
        intent = await llm_service.extract_intent(state["prompt"], history=history)
        # Append the new turn to history so the checkpointer persists it
        new_messages = [
            HumanMessage(content=state["prompt"]),
            AIMessage(content=intent.model_dump_json()),
        ]
        return {"intent": intent, "error": None, "messages": new_messages}
    except Exception as e:
        logger.error("Intent extraction failed: %s", str(e))
        return {"intent": None, "error": str(e), "messages": [HumanMessage(content=state["prompt"])]}


async def validate_intent(state: IntentState) -> dict[str, Any]:
    """
    Node 2: Validate the extracted intent for completeness.

    Checks that critical fields are populated and adjusts
    confidence if the intent looks incomplete. This node exists
    as a hook for more sophisticated validation in later phases.
    """
    intent = state.get("intent")

    if intent is None:
        logger.warning("validate_intent — no intent to validate")
        return {"is_valid": False}

    issues: list[str] = []

    # Check study area completeness
    if not intent.study_area.name or len(intent.study_area.name.strip()) < 2:
        issues.append("Study area name is missing or too short")

    # Check goal quality
    if not intent.goal or len(intent.goal.strip()) < 10:
        issues.append("Goal description is too vague")

    # Check criteria selection
    if intent.analysis_mode == "ahp" and len(intent.suggested_criteria) < 2:
        issues.append("AHP mode requires at least 2 criteria")

    if issues:
        logger.warning("Intent validation issues: %s", "; ".join(issues))
        # Penalise confidence for each issue
        adjusted_confidence = max(0.1, intent.confidence - (0.1 * len(issues)))
        intent.confidence = round(adjusted_confidence, 2)
        intent.reasoning += f" [Validation adjusted: {'; '.join(issues)}]"

    is_valid = len(issues) == 0
    logger.info("validate_intent — valid=%s, confidence=%.2f", is_valid, intent.confidence)

    return {"intent": intent, "is_valid": is_valid}


# ── Graph Builder ────────────────────────────────


def build_intent_graph() -> StateGraph:
    """
    Construct the Day 1 intent extraction graph.

    Graph structure:
        START → parse_intent → validate_intent → END

    Future phases will insert nodes between validate_intent and END:
        ... → data_shopping → slope_analysis → ahp_matrix → ...
    """
    graph = StateGraph(IntentState)

    # Add nodes
    graph.add_node("parse_intent", parse_intent)
    # graph.add_node("validate_intent", validate_intent)

    # Wire edges
    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", END)
    # graph.add_edge("parse_intent", "validate_intent")
    # graph.add_edge("validate_intent", END)

    return graph


# ── Compiled graph (reusable) ────────────────────

_compiled_graph = None
_checkpointer = MemorySaver()


def get_intent_graph():
    """Get or create the compiled intent graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_intent_graph().compile(checkpointer=_checkpointer)
    return _compiled_graph
