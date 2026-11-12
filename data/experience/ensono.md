# AI Data Scientist Intern (Full-Time) — Ensono
May 2026 – October 2026 | Remote

## LLM inference optimization
- Engineered a two-tier KV-cache and prompt-prefilling architecture with conversation caching, cutting LLM inference latency by 94% (225s to 14s, a 16x speedup).
- Implemented persistent system-prompt caching, TTL-based memory management with eviction (reducing peak RAM by 15 GB), and CPU inference optimizations.

## Prompt engineering for structured queries
- Implemented and refined system prompts that incorporated infrastructure JSON schema context with multiple fields, guiding the internal LLM (Qwen2.5-1.5B) and raising JMESPath query execution success from 60% to 80%.

## LLM evaluation and benchmarking
- Took ownership of evaluating and improving internal LLMs (Qwen2.5-7B and Qwen2.5-14B).
- Designed benchmarking pipelines with 20 JMESPath queries, comparing accuracy and latency against 5 cloud-based LLMs.

## MCP-based agentic system
- Integrated Portkey AI into an MCP-based agentic system supporting enterprise LLMs across multiple providers, including OpenAI, Anthropic Claude, Google Gemini, Meta Llama, and DeepSeek.
- Implemented MCP JSON-RPC initialization, tool discovery and execution, and a multi-step agentic loop that routes LLM tool calls to JMESPath/JSONata tools and returns results to the model.

## Full-stack on-premise AI application
- Collaborated with the team to build a full-stack, on-premise AI data-query application using Python, Django, Hugging Face, internal LLMs, JMESPath, REST APIs, and SSE streaming.
- Enabled natural-language and structured querying of live enterprise infrastructure data.

## RAG evaluation
- Evaluated RAG-based retrieval against full-schema context to identify limitations in retrieving relevant JSON fields for accurate JMESPath query generation.

Tools: Python, Django, Hugging Face, Qwen2.5, Portkey AI, MCP, JSON-RPC, JMESPath, JSONata, REST APIs, SSE, RAG
