"""
API v1 router — agent endpoints.

All versioned API routes are defined here. Each route is thin:
it validates input, delegates to the agent/service layer, and
formats the response.
"""

import json
import time
import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.intent_agent import get_intent_graph
from app.agents.orchestrator import get_orchestrator_graph
from app.core.database import async_session_factory
from app.core.logging import get_logger
from app.models.database import AnalysisSession
from app.models.schemas import AgentResponse, PromptRequest
from app.services.llm import get_llm_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


async def _persist_session(
    session_id: uuid.UUID,
    prompt: str,
    intent_dict: dict | None,
    status: str,
    model_used: str,
    latency_ms: int,
) -> None:
    """Write the analysis session to the DB. Runs in the background after the response is sent."""
    try:
        async with async_session_factory() as db:
            session = AnalysisSession(
                id=session_id,
                prompt=prompt,
                intent=intent_dict,
                status=status,
                model_used=model_used,
                latency_ms=latency_ms,
            )
            db.add(session)
            await db.commit()
    except Exception as e:
        logger.error("Background DB persist failed for session %s: %s", session_id, str(e))



@router.post(
    "/chat",
    summary="General chat with the multi-tool GeoAI Agent",
    description="Supports general conversation and automatic tool routing for GIS tasks.",
)
async def chat(
    request: PromptRequest,
    background_tasks: BackgroundTasks,
):
    """
    New orchestrator endpoint. 
    Handles general chat and multi-tool orchestration.
    """
    start_time = time.perf_counter()
    session_id = uuid.uuid4()
    
    graph = get_orchestrator_graph()
    graph_config = {"configurable": {"thread_id": request.thread_id}}
    llm_service = get_llm_service()

    if request.stream:
        async def chat_stream_generator():
            try:
                final_intent = None
                accumulated_text = ""
                # We use astream_events to capture both tool calls and final content
                async for event in graph.astream_events(
                    {"messages": [HumanMessage(content=request.prompt)]},
                    config=graph_config,
                    version="v2"
                ):
                    kind = event["event"]
                    langgraph_node = event["metadata"].get("langgraph_node")
                    
                    # Capture the final state when the graph finishes
                    if kind == "on_chain_end" and event["name"] == "LangGraph":
                        output = event["data"]["output"]
                        if isinstance(output, dict) and "intent" in output:
                            final_intent = output["intent"]

                    # Stream tokens from the 'agent' node (the LLM)
                    if kind == "on_chat_model_stream" and langgraph_node == "agent":
                        tags = event.get("tags", [])
                        if "hide_from_stream" in tags:
                            continue
                            
                        content = event["data"]["chunk"].content
                        if content:
                            accumulated_text += content
                            # The frontend expects 'chunk' to be an object with 'workflow'
                            # and it expects the FULL accumulated text so far in every chunk.
                            yield f"data: {json.dumps({'chunk': {'workflow': accumulated_text}})}\n\n"
                    
                    # Notify about tool starts
                    elif kind == "on_tool_start":
                        tool_name = event["name"]
                        yield f"data: {json.dumps({'status': 'tool_start', 'tool': tool_name})}\n\n"
                    
                    # Tool end
                    elif kind == "on_tool_end":
                        tool_name = event["name"]
                        output = event["data"]["output"]
                        
                        # Extract content if it's a ToolMessage
                        if hasattr(output, "content"):
                            output = output.content
                            
                        # If the output is a JSON string (from some tools), parse it so we send a clean object
                        if isinstance(output, str):
                            try:
                                output = json.loads(output)
                            except:
                                pass
                                
                        yield f"data: {json.dumps({'status': 'tool_end', 'tool': tool_name, 'output': output})}\n\n"
                
                # Yield a signal that text is finished (for UI unlocking)
                yield f"data: {json.dumps({'status': 'text_finished'})}\n\n"

                # If we didn't get a structured intent (e.g. casual chat),
                # build a dummy one with the accumulated text as workflow.
                if not final_intent:
                    final_intent = {
                        "workflow": accumulated_text,
                        "goal": "General inquiry",
                        "study_area": {"name": "Unspecified"}
                    }
                else:
                    # If it's a Pydantic model, dump it
                    if hasattr(final_intent, "model_dump"):
                        final_intent = final_intent.model_dump()

                latency_ms = int((time.perf_counter() - start_time) * 1000)
                # Ensure the final state has the full content for 'lastIntent' persistence in the frontend
                yield f"data: {json.dumps({
                    'status': 'completed', 
                    'session_id': str(session_id), 
                    'thread_id': request.thread_id,
                    'intent': final_intent,
                    'latency_ms': latency_ms,
                    'model_used': llm_service.model_name,
                    'created_at': datetime.now(timezone.utc).isoformat()
                })}\n\n"
                
            except Exception as e:
                logger.error("Chat streaming failed: %s", str(e))
                yield f"data: {json.dumps({'status': 'error', 'detail': str(e)})}\n\n"

        return StreamingResponse(chat_stream_generator(), media_type="text/event-stream")

    # Standard invocation
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=request.prompt)]},
            config=graph_config
        )
        last_message = result["messages"][-1]
        
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "session_id": session_id,
            "thread_id": request.thread_id,
            "response": last_message.content,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        logger.error("Chat failed: %s", str(e))
        raise HTTPException(status_code=502, detail=f"Agent failed: {str(e)}")
