"""
Hybrid (semantic + BM25 via RRF) RAG + RAGAS eval.

Doc runs to reproduce (see README):
  python 5_hybridrag_eval.py                                       # HF + BM25,     chunk=256
  python 5_hybridrag_eval.py --embeddings gemini                   # Gemini + BM25, chunk=256
  python 5_hybridrag_eval.py --embeddings gemini --chunk-size 1500 # Gemini + BM25, chunk=1500 (doc's best)
  python 5_hybridrag_eval.py --weights 0.7 0.3                     # lean semantic
"""

import argparse
from pathlib import Path

from ragas_eval import SingleTurnSample, get_embeddings, get_llm, score, test_cases  # loads Practice/.env

from pydantic import ConfigDict
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def load_codebase(repo_path: str) -> list:
    docs = []
    for path in Path(repo_path).rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs


# ---------------------------------------------------------------------------
# 2. CHUNK — character-based splits shared by both retrievers
# ---------------------------------------------------------------------------

def chunk_code(docs: list, chunk_size: int) -> list:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=32,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# 3a. SEMANTIC — dense vector retriever
# ---------------------------------------------------------------------------

def build_vector_retriever(chunks: list, embeddings: str, k: int):
    # "gemini" stands in for the doc's OpenAI text-embedding-3-small; "hf" is the free local model.
    model = get_embeddings() if embeddings == "gemini" else HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embedding=model)
    return vector_store.as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# 3b. LEXICAL — sparse BM25 retriever
# ---------------------------------------------------------------------------

class BM25Retriever(BaseRetriever):
    """Thin LangChain-compatible retriever backed by rank_bm25."""
    docs: list
    bm25: object
    k: int = 4

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [self.docs[i] for i in top_indices]


def build_bm25_retriever(chunks: list, k: int) -> BM25Retriever:
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)
    return BM25Retriever(docs=chunks, bm25=bm25, k=k)


# ---------------------------------------------------------------------------
# 3c. HYBRID — fuse both via Reciprocal Rank Fusion (weights sum to 1.0)
# ---------------------------------------------------------------------------

def build_hybrid_retriever(chunks: list, weights: list, embeddings: str, k: int):
    vector_retriever = build_vector_retriever(chunks, embeddings, k)
    bm25_retriever = build_bm25_retriever(chunks, k)
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=weights,
    )


# ---------------------------------------------------------------------------
# 4. AGENT  (same as rag/6_hybridrag.py)
# ---------------------------------------------------------------------------

def build_agent(retriever):
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic.",
    )
    return create_agent(
        get_llm(), tools=[retriever_tool],
        system_prompt=(
            "You are a senior engineer. Always use search_codebase before answering. "
            "Reference specific file and function names. "
            "If not found say 'I could not find that in the codebase'."
        ),
        middleware=[
            ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=2, exit_behavior="end"),
        ],
    )


def run_agent(agent, question: str) -> tuple[str, list[str]]:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # .text, not .content: Gemini 3 returns content as a list of parts
    answer = result["messages"][-1].text

    contexts = []
    for msg in result["messages"]:
        if msg.type == "tool" and isinstance(msg.content, str):
            contexts.append(msg.content)

    return answer, contexts


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).parent.parent / "sample_project"))
    parser.add_argument(
        "--weights", nargs=2, type=float, default=[0.5, 0.5],
        metavar=("SEMANTIC", "LEXICAL"),
        help="Retriever blend weights (must sum to 1.0). Default: 0.5 0.5",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=4, help="Chunks each retriever returns before fusion (default: 4)")
    parser.add_argument("--embeddings", choices=["hf", "gemini"], default="hf")
    parser.add_argument("--limit", type=int, help="Only run the first N golden questions")
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    docs = load_codebase(repo_path)
    chunks = chunk_code(docs, args.chunk_size)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={args.chunk_size}, k={args.k}, embeddings={args.embeddings})")
    print(f"Hybrid weights — semantic: {args.weights[0]}, lexical: {args.weights[1]}")

    retriever = build_hybrid_retriever(chunks, args.weights, args.embeddings, args.k)
    agent = build_agent(retriever)

    # Collect RAG outputs
    print("\nRunning agent on test cases...")
    samples = []
    for tc in test_cases(args.limit):
        answer, contexts = run_agent(agent, tc["question"])
        samples.append(SingleTurnSample(
            user_input=tc["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=tc["reference"],
        ))

        print(f"\nQ: {tc['question']}")
        print(f"Contexts retrieved ({len(contexts)}):")
        for i, ctx in enumerate(contexts, 1):
            print(f"  [{i}] {ctx[:200]}{'...' if len(ctx) > 200 else ''}")
        print(f"Answer: {answer}")

    w = args.weights
    score(samples, f"Hybrid RAG ({args.embeddings} + BM25, chunk={args.chunk_size}, w={w[0]}/{w[1]})")
