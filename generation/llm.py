"""
Ollama LLM client for StudyRAG.

Thin wrapper around the Ollama Python client. Sends a system + user
message pair and returns the raw response text.
"""

from __future__ import annotations

import logging
import time

import ollama

from config import settings

logger = logging.getLogger(__name__)


def generate(system_prompt: str, user_prompt: str) -> str:
    """
    Send a chat completion request to Ollama.

    Args:
        system_prompt: instructions for the model (citation policy etc.)
        user_prompt: the context + question

    Returns:
        The model's response text.

    Raises:
        ConnectionError: if Ollama is unreachable.
    """
    client = ollama.Client(host=settings.ollama_base_url)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.info("Sending query to '%s' ...", settings.ollama_model)
    start = time.time()

    try:
        response = client.chat(
            model=settings.ollama_model,
            messages=messages,
        )
    except Exception as e:
        logger.error("Ollama request failed: %s", e)
        raise ConnectionError(
            f"Could not reach Ollama at {settings.ollama_base_url}. "
            f"Is it running? ('ollama serve')"
        ) from e

    elapsed = time.time() - start
    reply = response["message"]["content"]

    logger.info("Got response in %.1fs (%d chars)", elapsed, len(reply))
    return reply
