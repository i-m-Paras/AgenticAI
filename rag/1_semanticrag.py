import argparse
from pathlib import Path

from gemini_config import get_llm  # loads Practice/.env

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent

from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware


CHUNK_SIZE = 256 


def load_codebase(repo_path: str) -> list:
    docs = []
    for path in Path(repo_path).rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs


def chunk_code(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=32,
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks: list) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embedding=embeddings)


def build_agent(vector_store: Chroma):
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": 4}),
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic.",
    )
    llm = get_llm()
    return create_agent(
        llm, tools=[retriever_tool],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).parent.parent / "sample_project"))
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    docs = load_codebase(repo_path)
    chunks = chunk_code(docs)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")

    vector_store = build_vector_store(chunks)
    agent = build_agent(vector_store)

    print("Ready. Ask your question. Type 'exit' to quit")
    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in ("exit", "quit"):
            break

        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            last_msg = step["messages"][-1]
            if last_msg.type == "tool":
                print(f"  [{last_msg.name}] retrieved {len(last_msg.text)} chars")
            elif last_msg.type == "ai" and not last_msg.tool_calls:
                print(f"Agent: {last_msg.text}")
