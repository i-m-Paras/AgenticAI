"""
GraphRAG + RAGAS eval (doc run: depth=2).

Building the graph costs one LLM call per source file before any question runs.

  python 4_graphrag_eval.py
  python 4_graphrag_eval.py --depth 1
"""

import argparse
from pathlib import Path

from ragas_eval import SingleTurnSample, get_llm, score, test_cases  # loads Practice/.env

import networkx as nx
from langchain.agents import create_agent
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODEBASE = PROJECT_ROOT / "sample_project"

CODE_EXTENSIONS = {".py", ".ts", ".js", ".java", ".go", ".rs", ".md"}
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".mypy_cache", ".ruff_cache"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CodeRelationship(BaseModel):
    subject: str = Field(description="Class, function, module, or file path")
    predicate: str = Field(
        description="Relationship type (e.g., DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, INHERITS_FROM)"
    )
    obj: str = Field(description="Target class, function, module, or file path")


class GraphDocument(BaseModel):
    relationships: list[CodeRelationship] = Field(description="All code relationships extracted from the source file")


class Entities(BaseModel):
    names: list[str] = Field(
        description="Code entities in the query: class names, function names, modules, file paths"
    )


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def load_codebase(root: Path) -> list[tuple[str, str]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase path does not exist: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))
    return files


# ---------------------------------------------------------------------------
# 2. EXTRACT RELATIONSHIPS & BUILD GRAPH
# ---------------------------------------------------------------------------

def extract_relationships(relationship_extractor, files: list[tuple[str, str]]) -> list[CodeRelationship]:
    relationships = []
    for rel_path, content in files:
        print(f"  Extracting {rel_path} ...")
        result = relationship_extractor.invoke(
            {"messages": [{"role": "user", "content": f"File: {rel_path}\n\n{content}"}]}
        )
        relationships.extend(result["structured_response"].relationships)
    return relationships


def build_graph(relationships: list[CodeRelationship]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for r in relationships:
        graph.add_edge(r.subject.strip(), r.obj.strip(), relation=r.predicate)
    return graph


# ---------------------------------------------------------------------------
# 3. RETRIEVE
# ---------------------------------------------------------------------------

def match_nodes(graph: nx.DiGraph, entity: str) -> list[str]:
    needle = entity.strip().lower()
    return [node for node in graph.nodes() if needle in node.lower() or node.lower() in needle]


def graph_retrieve(graph: nx.DiGraph, entity_extractor, query: str, depth: int = 2) -> str:
    result = entity_extractor.invoke({"messages": [{"role": "user", "content": query}]})
    entities = result["structured_response"].names

    relationships = []
    for entity in entities:
        for node in match_nodes(graph, entity):
            neighbourhood = nx.ego_graph(graph, node, radius=depth, undirected=True)
            for source, target, data in neighbourhood.edges(data=True):
                relationships.append(f"{source} -[{data['relation']}]-> {target}")

    if not relationships:
        return "No relevant graph data found."
    return "Knowledge Graph context:\n" + "\n".join(sorted(set(relationships)))


# ---------------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------------

def run_query(qa_agent, entity_extractor, graph: nx.DiGraph, question: str, depth: int = 2) -> tuple[str, list[str]]:
    context = graph_retrieve(graph, entity_extractor, question, depth=depth)
    result = qa_agent.invoke(
        {"messages": [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]}
    )
    # .text, not .content: Gemini 3 returns content as a list of parts
    answer = result["messages"][-1].text
    return answer, [context]


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_CODEBASE)
    parser.add_argument("--depth", type=int, default=2, help="Graph neighbourhood radius (default: 2)")
    parser.add_argument("--limit", type=int, help="Only run the first N golden questions")
    args = parser.parse_args()

    files = load_codebase(args.repo.resolve())
    if not files:
        raise SystemExit(f"No source files found under {args.repo}")

    llm = get_llm()

    relationship_extractor = create_agent(
        model=llm,
        tools=[],
        response_format=GraphDocument,
        system_prompt=(
            "Extract code relationships from the source file as (subject, predicate, object) facts.\n"
            "Use ALL_CAPS predicates such as: DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, "
            "INHERITS_FROM, IMPLEMENTS, VALIDATES, SENDS_TO, CONFIGURES.\n"
            "Subjects and objects should be class names, function names, module paths, or file paths.\n"
            "Capture imports, constructor dependencies, method calls, and cross-module relationships.\n"
            "Be consistent: use the same name for the same class or module across relationships."
        ),
    )
    entity_extractor = create_agent(
        model=llm,
        tools=[],
        response_format=Entities,
        system_prompt="Extract code-related entities from the user message: class names, function names, module names, and file paths.",
    )
    qa_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=(
            "You are a codebase assistant. Answer ONLY from the context provided in the user message. "
            "Reference specific classes, files, and relationships when possible. "
            "If you cannot answer from context, say so."
        ),
    )

    print(f"Loaded {len(files)} files — extracting knowledge graph...")
    relationships = extract_relationships(relationship_extractor, files)
    graph = build_graph(relationships)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Collect GraphRAG outputs
    print("\nRunning agent on test cases...")
    samples = []
    for tc in test_cases(args.limit):
        answer, contexts = run_query(qa_agent, entity_extractor, graph, tc["question"], depth=args.depth)
        samples.append(SingleTurnSample(
            user_input=tc["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=tc["reference"],
        ))

        print(f"\nQ: {tc['question']}")
        print("Context retrieved:")
        print(f"  {contexts[0][:200]}{'...' if len(contexts[0]) > 200 else ''}")
        print(f"Answer: {answer}")

    score(samples, f"GraphRAG (depth={args.depth})")
