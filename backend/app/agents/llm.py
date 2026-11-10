"""LLM abstraction.

The agents never import a concrete provider directly — they call
:func:`get_llm` / :func:`structured_invoke` / :func:`text_invoke`, which read the
provider and model from configuration. When no API key is configured (or a call
fails), the helpers return ``None`` and the agents fall back to deterministic
heuristics that are clearly labelled as such — never fabricated LLM output.
"""
from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_WARNED_NO_KEY = False


def get_llm(temperature: float | None = None):
    """Instantiate the configured chat model, or ``None`` if unavailable."""
    global _WARNED_NO_KEY
    settings = get_settings()
    api_key = settings.active_api_key
    if not api_key:
        if not _WARNED_NO_KEY:
            logger.warning(
                "No LLM API key configured (provider=%s). Agents will use "
                "deterministic heuristic fallbacks.",
                settings.llm_provider,
            )
            _WARNED_NO_KEY = True
        return None

    temp = settings.llm_temperature if temperature is None else temperature
    provider = settings.llm_provider.lower()
    try:
        if provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=settings.llm_model,
                api_key=api_key,
                temperature=temp,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout,
                max_retries=2,
            )
        if provider == "openai":
            from langchain_openai import ChatOpenAI  # optional dependency

            return ChatOpenAI(
                model=settings.llm_model,
                api_key=api_key,
                temperature=temp,
                timeout=settings.llm_timeout,
                max_retries=2,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not initialise LLM (provider=%s): %s", provider, exc)
        return None

    logger.warning("Unknown LLM provider '%s'.", provider)
    return None


def llm_available() -> bool:
    return get_llm() is not None


def structured_invoke(
    output_model: type[T],
    system: str,
    human: str,
    temperature: float | None = None,
) -> T | None:
    """Invoke the LLM and coerce the reply into ``output_model``.

    Returns ``None`` on any failure so callers can fall back to heuristics.
    """
    llm = get_llm(temperature)
    if llm is None:
        return None
    try:
        structured = llm.with_structured_output(output_model)
        result = structured.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        if isinstance(result, output_model):
            return result
        if isinstance(result, dict):
            return output_model.model_validate(result)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Structured LLM call failed (%s): %s", output_model.__name__, exc)
        return None


def text_invoke(system: str, human: str, temperature: float | None = None) -> str | None:
    """Invoke the LLM for a plain-text reply, or ``None`` on failure."""
    llm = get_llm(temperature)
    if llm is None:
        return None
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        content = response.content
        return content if isinstance(content, str) else str(content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Text LLM call failed: %s", exc)
        return None
