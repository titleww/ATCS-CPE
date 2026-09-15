# -*- coding: utf-8 -*-
# Problem 08: Analyzing Configuration of the Astronomy RAG Pipeline
from data_loader import load_qa

CONFIG = {
    "KB_SOURCE": "astronomy_knowledge_base.txt",
    "USE_HYBRID": True,
    "USE_RERANK": False,
    "USE_MEMORY": True,
    "USE_LLM": True,
    "SHOW_SOURCES": True,
}


def run():
    data = load_qa()
    print("CONFIG =", CONFIG)
    print(f"Loaded Knowledge Base from {CONFIG['KB_SOURCE']}: {len(data)} Q&A entries")
    print("\nActive Pipeline Steps:")

    if CONFIG["USE_MEMORY"]:
        print("1. [Active] Conversation Memory (tracks multi-turn context)")
    else:
        print("1. [Disabled] Stateless Querying")

    if CONFIG["USE_HYBRID"]:
        print(f"2. [Active] Hybrid Search (FAISS Dense + BM25 Sparse with RRF) on {len(data)} documents")
    else:
        print(f"2. [Active] Pure Dense Search on {len(data)} documents")

    if CONFIG["USE_RERANK"]:
        print("3. [Active] Cross-Encoder Re-ranking on Top Candidates")
    else:
        print("3. [Bypassed] Sending Top-k directly to LLM")

    if CONFIG["USE_LLM"]:
        print("4. [Active] LLM Answer Generation with Strict Grounding Prompt")
    else:
        print("4. [NoLLM Mode] Extracting raw text directly from top chunk")

    if CONFIG["SHOW_SOURCES"]:
        print("5. [Active] Rendering Source Citations at the end of response")
    else:
        print("5. [Disabled] Hiding Source Citations")

    print("\nConclusion: The pipeline behavior is dynamically controlled via config toggles.")


if __name__ == "__main__":
    run()
