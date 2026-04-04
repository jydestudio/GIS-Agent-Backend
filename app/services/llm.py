"""
LLM service for interacting with language models via OpenRouter.

Encapsulates all LLM communication. Uses langchain-openai's ChatOpenAI
pointed at OpenRouter's API. Provides structured output extraction
using Pydantic model binding.
"""

from collections.abc import AsyncGenerator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from app.agents.prompts import INTENT_EXTRACTION_SYSTEM_PROMPT
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import ProjectIntent

logger = get_logger(__name__)


class LLMService:
    """
    Manages LLM interactions via OpenRouter.

    Uses structured output (function calling) to guarantee
    responses conform to our Pydantic schemas — no string parsing.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.default_model

        self._llm = ChatOpenAI(
            model=self._model_name,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=0,
            max_tokens=8192,
            streaming=True,
        )

        logger.info("LLM service initialised — model=%s", self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def llm(self) -> ChatOpenAI:
        """Expose the underlying ChatOpenAI instance for tool binding."""
        return self._llm

    async def extract_intent(self, prompt: str, history: list[BaseMessage] | None = None) -> ProjectIntent:
        """
        Extract a structured ProjectIntent from a natural language prompt.

        Args:
            prompt: The user's raw natural language input.

        Returns:
            A validated ProjectIntent object.
        """
        logger.debug("Extracting intent from prompt: %s", prompt[:100])

        json_instruction = (
            "\n\nYou MUST respond with ONLY a valid JSON object. Do not include markdown code fences or explanations. "
            "Ensure all fields are populated based on the user's prompt. "
            "The JSON must follow this structure exactly:\n"
            "{\n"
            '  "workflow": "A detailed markdown analysis plan",\n'
            '  "study_area": {\n'
            '    "name": "Specific name of the area (e.g. Enugu State)",\n'
            '    "country": "Inferred country name",\n'
            '    "admin_level": "state | lga | city | country"\n'
            '  },\n'
            '  "goal": "Clear, normalized sentence describing the objective",\n'
            '  "analysis_mode": "ahp" or "standalone",\n'
            '  "suggested_criteria": ["list", "of", "criteria"],\n'
            '  "confidence": 0.95,\n'
            '  "reasoning": "Quick explanation of your logic"\n'
            "}"
        )

        messages_with_json = [
            SystemMessage(content=INTENT_EXTRACTION_SYSTEM_PROMPT + json_instruction),
            *(history or []),
            HumanMessage(content=prompt),
        ]

        # Use ainvoke for internal logic to ensure we get the full response
        # Tag it so the router can ignore the events
        config = {"tags": ["hide_from_stream"]}
        response = await self._llm.ainvoke(messages_with_json, config=config)
        raw_text = response.content
        
        if not raw_text:
            logger.error("LLM returned empty response during intent extraction!")
            # Provide a fallback valid JSON if the model is silent
            raw_text = '{"workflow": "Could you please elaborate on your request?", "reasoning": "Empty response"}'

        logger.debug("Raw LLM response (first 100 chars): %s", str(raw_text)[:100])

        return self._parse_json_response(str(raw_text))

    async def stream_intent(self, prompt: str, history: list[BaseMessage] | None = None) -> AsyncGenerator[dict, None]:
        """
        Stream a structured ProjectIntent from a natural language prompt.
        """
        json_instruction = (
            "\n\nYou MUST respond with ONLY a valid JSON object. Do not include markdown code fences or explanations. "
            "The JSON must follow this structure exactly:\n"
            "{\n"
            '  "workflow": "A detailed markdown analysis plan",\n'
            '  "study_area": {"name": "...", "country": "...", "admin_level": "..."},\n'
            '  "goal": "...",\n'
            '  "analysis_mode": "ahp | standalone",\n'
            '  "suggested_criteria": ["criterion1", "criterion2"],\n'
            '  "confidence": 0.95,\n'
            '  "reasoning": "..."\n'
            "}"
        )

        messages = [
            SystemMessage(content=INTENT_EXTRACTION_SYSTEM_PROMPT + json_instruction),
            *(history or []),
            HumanMessage(content=prompt),
        ]

        logger.debug("Streaming intent from prompt: %s", prompt[:100])

        chain = self._llm | JsonOutputParser()
        async for chunk in chain.astream(messages):
            yield chunk

    @staticmethod
    def _parse_json_response(text: str) -> ProjectIntent:
        """
        Parse a ProjectIntent from raw LLM text.

        Handles common LLM response quirks:
        - Markdown code fences (```json ... ```)
        - Control characters / invalid escapes (using strict=False)
        - Leading/trailing whitespace or explanation text
        """
        import json
        import re

        # Strip markdown code fences
        fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        match = re.search(fence_pattern, text, re.DOTALL)
        if match:
            text = match.group(1)

        # Try to find the first { and last } to isolate the JSON object
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        try:
            # strict=False is crucial for handling unescaped control chars in markdown strings
            data = json.loads(text, strict=False)
            return ProjectIntent.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("JSON Parse Error: %s in text: %s", str(e), text[:500])
            raise ValueError(f"Failed to parse LLM response into ProjectIntent: {e}") from e


# ── Singleton accessor ───────────────────────────

_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """
    Get or create the LLM service singleton.
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
