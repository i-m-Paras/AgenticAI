"""
Shared RAGAS setup for every eval in this folder (Gemini as the judge).

The course eval files each repeat the same ~60 lines: the ragas import shim,
the GPT-4o judge, evaluate() and the scorecard printout. Here they live once,
so each N_*_eval.py only contains its RAG strategy + `score(samples, title)`.

Import this module BEFORE anything from ragas — it installs the shim below.
"""

import sys
import types
from pathlib import Path

# Practice/rag holds gemini_config.py (get_llm / get_embeddings, loads Practice/.env)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))
from gemini_config import get_llm, get_embeddings  # noqa: E402

# ragas 0.4.x imports ChatVertexAI from langchain_community.chat_models.vertexai,
# which was removed in langchain-community 0.4. ragas only uses it in an
# isinstance() check, so a stub class is enough — the course doc installs
# langchain-google-vertexai for this instead.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

from ragas import EvaluationDataset, SingleTurnSample, RunConfig, evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics._context_precision import LLMContextPrecisionWithReference  # noqa: E402
from ragas.metrics._context_recall import LLMContextRecall  # noqa: E402
from ragas.metrics._faithfulness import Faithfulness  # noqa: E402
from ragas.metrics._answer_relevance import AnswerRelevancy  # noqa: E402
from ragas.metrics._factual_correctness import FactualCorrectness  # noqa: E402

from golden_dataset import TEST_CASES  # noqa: E402

__all__ = ["get_llm", "get_embeddings", "SingleTurnSample", "TEST_CASES", "test_cases", "score"]

METRIC_COLS = [
    "llm_context_precision_with_reference",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "factual_correctness(mode=f1)",
]


def test_cases(limit: int | None) -> list[dict]:
    """The golden set, optionally cut to the first `limit` questions (quick, cheap runs)."""
    return TEST_CASES[:limit] if limit else TEST_CASES


def _icon(val: float) -> str:
    return "✅" if val >= 0.7 else ("⚠️ " if val >= 0.5 else "❌")


def score(samples: list[SingleTurnSample], title: str) -> None:
    """Run RAGAS with Gemini as judge and print the per-question + aggregate scorecard."""
    print(f"\nRunning RAGAS evaluation (Gemini as judge) on {len(samples)} samples...")
    print("  (free tier is ~10 requests/min, so this takes a while — every metric is 1-3 LLM calls per question)")

    # bypass_temperature: keep Gemini's default sampling (see gemini_config.get_llm).
    # bypass_n: AnswerRelevancy asks for 3 completions; send 3 prompts instead of
    # relying on Gemini's candidate_count.
    evaluator_llm = LangchainLLMWrapper(get_llm(), bypass_temperature=True, bypass_n=True)
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            AnswerRelevancy(embeddings=evaluator_embeddings),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
        # Few workers + long timeout: the shared rate limiter in gemini_config makes
        # jobs queue, and ragas' default 180s timeout would turn the waits into NaN.
        run_config=RunConfig(max_workers=4, timeout=900, max_retries=10, max_wait=90),
    )

    df = results.to_pandas()

    print(f"\nRAGAS SCORECARD — {title} on sample_project")
    print("\nPer-question breakdown:")
    for _, row in df.iterrows():
        print(f"\n  Q: {row['user_input'][:65]}")
        for col in METRIC_COLS:
            if col in row:
                val = row[col]
                print(f"    {_icon(val)} {col:<42}: {val:.3f}")

    print("\nAggregate averages:")
    for col in METRIC_COLS:
        if col in df.columns:
            avg = df[col].mean()
            print(f"  {_icon(avg)} {col:<42}: {avg:.3f}")
    print()
