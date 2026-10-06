"""Quick check that GEMINI_API_KEY works for both chat and embeddings."""

from gemini_config import GEMINI_MODEL, GEMINI_EMBEDDING_MODEL, get_llm, get_embeddings

reply = get_llm().invoke("Reply with exactly: Gemini is ready")
print(f"Chat      ({GEMINI_MODEL}): {reply.text.strip()}")

vector = get_embeddings().embed_query("hello world")
print(f"Embedding ({GEMINI_EMBEDDING_MODEL}): {len(vector)} dimensions")
