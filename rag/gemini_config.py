"""
Shared Gemini setup for every RAG demo in this folder.

.env keys used:
  GEMINI_API_KEY          — required (https://aistudio.google.com/apikey)
  GEMINI_MODEL            — chat model,      default gemini-3.5-flash
  GEMINI_EMBEDDING_MODEL  — embedding model, default gemini-embedding-001
  GEMINI_RPM              — optional client-side requests/minute cap (free tier ≈ 10)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_RPM = float(os.getenv("GEMINI_RPM", "0") or 0)


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Add it to Practice/.env "
            "(get one at https://aistudio.google.com/apikey)."
        )
    return key


# One limiter shared by every chat model in the process, so nested agents
# (router + outer agent, HyDE generator + QA agent) stay under the same cap.
_rate_limiter = (
    InMemoryRateLimiter(requests_per_second=GEMINI_RPM / 60, check_every_n_seconds=0.1, max_bucket_size=1)
    if GEMINI_RPM > 0
    else None
)


def get_llm(temperature: float | None = None) -> ChatGoogleGenerativeAI:
    """Drop-in replacement for ChatOpenAI(model="gpt-4o-mini", temperature=0).

    temperature is left unset by default: Gemini 3+ models are tuned for their
    default sampling, and some (e.g. gemini-3.6-flash) reject a custom value.
    """
    kwargs = {"temperature": temperature} if temperature is not None else {}
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=_api_key(),
        rate_limiter=_rate_limiter,
        max_retries=2,  # backs off on 429 RESOURCE_EXHAUSTED from the free tier
        **kwargs,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Drop-in replacement for OpenAIEmbeddings(model="text-embedding-3-small")."""
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=_api_key(),
    )
