from typing import Annotated
from langchain_core.tools import tool
from app.services.llm import get_llm_service
from app.models.schemas import ProjectIntent
import json

@tool
async def extract_project_intent(
    prompt: Annotated[str, "The user's original natural language prompt describing their geospatial project."]
) -> str:
    """
    Analyze a user's request and generate a structured geospatial analysis plan.
    Use this when the user describes a new project or asks for a methodology.
    Returns a JSON string containing the 'workflow' and other project details.
    """
    llm_service = get_llm_service()
    # history=[] for now, as the orchestrator handles history
    intent: ProjectIntent = await llm_service.extract_intent(prompt, history=[])
    return intent.model_dump_json()
