"""
Semantic RAG + RAGAS eval.

Doc runs to reproduce (see README):
  python 1_semanticrag_eval.py                          # chunk=256,  k=4
  python 1_semanticrag_eval.py --chunk-size 1500        # chunk=1500, k=4
  python 1_semanticrag_eval.py --chunk-size 1500 --k 1  # chunk=1500, k=1
"""

import argparse
from pathlib import Path

from ragas_eval import SingleTurnSample, get_embeddings, get_llm, score, test_cases  # loads Practice/.env

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
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
# 2. CHUNK
# ---------------------------------------------------------------------------

def chunk_code(docs: list, chunk_size: int) -> list:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=32,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# 3. EMBED & STORE
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list, embeddings: str) -> Chroma:
    # "gemini" stands in for the doc's OpenAI text-embedding-3-small; "hf" is the free local model.
    model = get_embeddings() if embeddings == "gemini" else HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embedding=model)


# ---------------------------------------------------------------------------
# 4. RETRIEVER + AGENT  (same as rag/1_semanticrag.py)
# ---------------------------------------------------------------------------

def build_agent(vector_store: Chroma, k: int):
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": k}),
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
    """Invoke the agent and extract (answer, contexts) from the message trace."""
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
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=4, help="Chunks retrieved per search (default: 4)")
    parser.add_argument("--embeddings", choices=["gemini", "hf"], default="gemini")
    parser.add_argument("--limit", type=int, help="Only run the first N golden questions")
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    docs = load_codebase(repo_path)
    chunks = chunk_code(docs, args.chunk_size)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={args.chunk_size}, k={args.k}, embeddings={args.embeddings})")

    vector_store = build_vector_store(chunks, args.embeddings)
    agent = build_agent(vector_store, args.k)

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

    score(samples, f"Semantic RAG (chunk={args.chunk_size}, k={args.k}, {args.embeddings} embeddings)")
