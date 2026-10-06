# RAG Practice (Gemini)

The course RAG code from *RAG - Agentic AI System Design Code.docx*, ported from OpenAI to Gemini.

## Setup (once)

```bash
cd Practice
source env/bin/activate          # always activate before running anything
pip install -r requirements.txt  # already done; re-run after editing requirements.txt
```

Put your key in `Practice/.env`:

```
GEMINI_API_KEY=your-key-here
```

The venv was created with Python 3.12 (`python3.12 -m venv env`). The system `python3` is 3.9,
which is too old for LangChain 1.x (`create_agent`, middleware). If you ever need to recreate it:

```bash
rm -rf env
~/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 -m venv env
source env/bin/activate && pip install -r requirements.txt
```

## Run

```bash
source env/bin/activate
cd rag
python 0_check_gemini.py      # verify the key works (chat + embeddings)
python 1_semanticrag.py       # then any demo
```

| Script | What it shows | Gemini usage |
|---|---|---|
| `0_check_gemini.py` | Key / model sanity check | chat + embeddings |
| `1_semanticrag.py` | Semantic RAG, recursive splitter, call-limit middleware | chat (local MiniLM embeddings) |
| `2_astdemo.py` | AST visualisation | none |
| `3_semanticrag_ast.py` | AST-based chunking | chat (local embeddings) |
| `4_bm25_demo.py` | BM25 scoring step by step | none |
| `5_lexicalrag.py` | Lexical (BM25) RAG | chat |
| `6_hybridrag.py` | Hybrid semantic + BM25 via RRF (`--weights 0.7 0.3`) | chat (local embeddings) |
| `rrf_demo.py` | Reciprocal Rank Fusion step by step | none |
| `7_rag_metrics.py` | Recall / Precision / MRR / Hit / nDCG | embeddings |
| `8_graphrag.py` | Knowledge-graph RAG (one LLM call per file at startup) | chat |
| `9_agenticrag.py` | Router agent picks semantic vs lexical | chat + embeddings |
| `10_hyderag.py` | HyDE retrieval | chat + embeddings |
| `11_vectorlessrag.py` | PageIndex tree navigation, no vectors | chat |
| `12_raptorrag.py` | RAPTOR tree of cluster summaries | chat + embeddings |

## Differences from the doc

- `ChatOpenAI` / `OpenAIEmbeddings` → `get_llm()` / `get_embeddings()` in `rag/gemini_config.py`
  (model set by `GEMINI_MODEL`, default `gemini-3.5-flash`; embeddings `gemini-embedding-001`).
- `temperature=0` isn't forced: Gemini 3+ models are tuned for their default sampling, and some reject a custom value.
- `GEMINI_RPM=10` in `.env` throttles requests for the free tier; remove it on a paid key.
- The chat loop prints only the final AI answer (plus a one-line note per retrieval), instead of echoing your question and raw tool output as `Agent:`.
- `ToolCallLimitMiddleware(tool_name="retriever_tool")` in the doc's hybrid/agentic code never matched a real tool; it is now `search_codebase` / `retrieval_tool`.
- `12_raptorrag.py` imported a missing `agentic_ai.config`; it now uses `gemini_config`.

## RAG evaluations (`rag_evals/`)

The RAGAS evals from *RAG Evaluation Examples.docx*. Each script runs one RAG strategy over the 18
golden questions in `golden_dataset.py`, then scores the answers with Gemini as the judge
(context precision, context recall, faithfulness, answer relevancy, factual correctness).

```bash
source env/bin/activate
cd rag_evals
python 1_semanticrag_eval.py --limit 2   # quick smoke test on 2 questions first
python 1_semanticrag_eval.py             # full run, 18 questions
```

Rough cost per full run: ~50 agent calls + ~200 judge calls. At `GEMINI_RPM=10` that's ~30 min,
and it can use up a free-tier daily quota. Use `--limit N` while experimenting.

| Doc config | Command |
|---|---|
| Semantic, chunk=256, k=4 | `python 1_semanticrag_eval.py` |
| Semantic, chunk=1500, k=4 | `python 1_semanticrag_eval.py --chunk-size 1500` |
| Semantic, chunk=1500, k=1 | `python 1_semanticrag_eval.py --chunk-size 1500 --k 1` |
| AST chunks, k=1 | `python 2_semanticrag_ast_eval.py` |
| BM25, chunk=256, k=4 | `python 3_lexicalrag_eval.py` |
| GraphRAG, depth=2 | `python 4_graphrag_eval.py` |
| Hybrid (HuggingFace + BM25), chunk=256 | `python 5_hybridrag_eval.py` |
| Hybrid (Gemini + BM25), chunk=256 | `python 5_hybridrag_eval.py --embeddings gemini` |
| Hybrid (Gemini + BM25), chunk=1500 | `python 5_hybridrag_eval.py --embeddings gemini --chunk-size 1500` |

Every script takes `--limit`. The semantic and hybrid scripts also take `--embeddings gemini|hf`
(Gemini replaces the doc's OpenAI embeddings; `hf` is the free local MiniLM).

Differences from the doc's eval code:

- Shared setup lives in `rag_evals/ragas_eval.py` (judge, `evaluate()`, scorecard) instead of being repeated in every file.
- Judge: `LangchainLLMWrapper(get_llm())` + `LangchainEmbeddingsWrapper(get_embeddings())` replace
  `llm_factory("gpt-4o")` and OpenAI embeddings. The judge uses the same rate limiter as the agents.
- The `ChatVertexAI` shim uses a stub class, so `langchain-google-vertexai` isn't installed.
- Chunk size, k and embeddings are CLI flags instead of constants you edit.
- The doc's scorecard looked up a `factual_correctness` column, but ragas names it
  `factual_correctness(mode=f1)`, so that metric never printed. It prints now.
- Answers are read with `.text`, not `.content`, because Gemini 3 returns content as a list of parts.
