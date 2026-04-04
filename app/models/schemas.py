"""
Pydantic schemas for API request/response validation.

These are the contracts between the frontend and the agent backend.
Every field is explicitly typed and documented.
"""

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────


class PromptRequest(BaseModel):
    """Incoming user prompt for analysis."""

    prompt: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural language description of the geospatial analysis goal.",
        examples=["Find suitable hospital sites in Enugu State"],
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response as it is generated.",
    )
    thread_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Conversation thread ID. Pass the same ID across requests to maintain history.",
    )


# ── Intent Models ────────────────────────────────


class StudyArea(BaseModel):
    """Extracted geographic study area from the user prompt."""

    name: Optional[str] = Field(None, description="Name of the study area as understood from the prompt.")
    country: Optional[str] = Field(None, description="Country the study area belongs to.")
    admin_level: Optional[str] = Field(
        None,
        description="Administrative level (e.g., 'state', 'lga', 'city', 'district').",
    )




class ProjectIntent(BaseModel):
    """
    Structured interpretation of a user's geospatial analysis request.
    """

    workflow: Optional[str] = Field(
        default="I'm ready to help with your geospatial project. Please describe what you'd like to analyze!",
        description="Detailed markdown analysis plan.",
    )
    study_area: StudyArea = Field(
        default_factory=lambda: StudyArea(name="Unspecified", country=None, admin_level=None),
        description="Where the analysis should be performed."
    )
    goal: Optional[str] = Field(
        default="General inquiry",
        description="Clear, normalised sentence describing the objective.",
    )
    analysis_mode: Optional[Literal["standalone", "ahp"]] = Field(
        default="standalone",
        description="Analysis mode (ahp or standalone).",
    )
    suggested_criteria: Optional[list[str]] = Field(
        default_factory=list,
        description="Criteria recommendations.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0–1).",
    )
    reasoning: Optional[str] = Field(
        default="Waiting for more details...",
        description="Brief explanation of logic.",
    )


# ── Response ─────────────────────────────────────


class AgentResponse(BaseModel):
    """Wrapper for Day 1 agent output with execution metadata."""

    session_id: UUID = Field(..., description="Unique identifier for this analysis session.")
    thread_id: str = Field(..., description="Conversation thread ID. Pass this back on follow-up requests.")
    intent: ProjectIntent = Field(..., description="The agent's structured interpretation.")
    model_used: str = Field(..., description="LLM model identifier used for extraction.")
    latency_ms: int = Field(..., description="Time taken to process the request in milliseconds.")
    created_at: datetime = Field(..., description="Timestamp of the analysis.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    app_name: str
    version: str
    environment: str
