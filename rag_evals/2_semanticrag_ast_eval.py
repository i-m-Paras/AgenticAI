"""
AST-chunked Semantic RAG + RAGAS eval (doc run: k=1).

  python 2_semanticrag_ast_eval.py          # AST chunks, k=1
  python 2_semanticrag_ast_eval.py --k 4    # try more chunks per search
"""

import ast
import argparse
from pathlib import Path

from ragas_eval import SingleTurnSample, get_embeddings, get_llm, score, test_cases  # loads Practice/.env

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent


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
# 2. CHUNK — AST-based (one chunk per class or module-level function)
# ---------------------------------------------------------------------------

def chunk_code(docs: list) -> list:
    chunks = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        code = doc.page_content

        try:
            tree = ast.parse(code)
        except SyntaxError:
            chunks.append(doc)
            continue

        # Add parent references — ast.walk doesn't expose parents by default
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        for node in ast.walk(tree):
            parent = getattr(node, "parent", None)

            # Top-level class → grab whole class including all methods
            if isinstance(node, ast.ClassDef) and isinstance(parent, ast.Module):
                text = ast.get_source_segment(code, node)
                if text:
                    chunks.append(Document(
                        page_content=text,
                        metadata={"source": source, "type": "class", "name": node.name},
                    ))

            # Module-level function — not a method inside a class
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(parent, ast.Module):
                text = ast.get_source_segment(code, node)
                if text:
                    chunks.append(Document(
                        page_content=text,
                        metadata={"source": source, "type": "function", "name": node.name},
                    ))

    return chunks


# ---------------------------------------------------------------------------
# 3. EMBED & STORE
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list, embeddings: str) -> Chroma:
    # "gemini" stands in for the doc's OpenAI text-embedding-3-small; "hf" is the free local model.
    model = get_embeddings() if embeddings == "gemini" else HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embedding=model)


# ---------------------------------------------------------------------------
# 4. RETRIEVER + AGENT  (same as rag/3_semanticrag_ast.py)
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
    parser.add_argument("--k", type=int, default=1, help="Chunks retrieved per search (default: 1)")
    parser.add_argument("--embeddings", choices=["gemini", "hf"], default="gemini")
    parser.add_argument("--limit", type=int, help="Only run the first N golden questions")
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    docs = load_codebase(repo_path)
    chunks = chunk_code(docs)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (AST-based, k={args.k}, embeddings={args.embeddings})")

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

    score(samples, f"AST-chunked Semantic RAG (k={args.k}, {args.embeddings} embeddings)")
