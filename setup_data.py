"""Creates one markdown file per experience item in data/experience/.
Run from the resume-tailor folder:  python setup_data.py
"""
from pathlib import Path

FILES = {
    "ensono.md": """# AI Data Scientist Intern (Full-Time) — Ensono
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
""",
    "sports-media.md": """# Data Science Intern (Part-Time) — Sports Media, Inc
January 2026 – May 2026 | Remote

- Built an end-to-end NLP analytics pipeline using TensorFlow and XGBoost with 74% accuracy.
- Containerized the application with Docker and deployed it to Hugging Face Spaces, delivering real-time prediction endpoints.
- Built an interactive dashboard for leadership and advertising partners to track fan sentiment and target campaigns.
- Presented deployment architecture, model trade-offs, and scalability considerations to team leadership.

Tools: Python, TensorFlow, XGBoost, NLP, Docker, Hugging Face Spaces
""",
    "bytes-and-bots.md": """# Mentor / Programmer (Part-Time) — Bytes and Bots
September 2025 – May 2026 | Irvine, CA

- Mentored a student in building AI-driven automation workflows using Python and n8n.
- Workflows extracted structured data from YouTube videos to support content analysis and project organization.

Tools: Python, n8n, workflow automation, mentoring
""",
    "pneumonia-classifier.md": """# Project: Pneumonia Medical Image Classification
December 2025 – January 2026

- Built a CNN pipeline to classify chest X-rays for pneumonia, improving diagnostic signal detection and reducing the risk of false negatives in healthcare scenarios.
- Achieved 0.96 ROC-AUC, 92% recall, and 90% accuracy on a 624-image held-out test set.

Tools: Python, TensorFlow, CNN, computer vision, medical imaging
""",
    "skills.md": """# Technical Skills

- Languages: Python, R, SQL, C++, Bash, PowerShell
- ML/AI: Scikit-learn, TensorFlow, Transformers, NLP, CNN, RNN, Random Forest, XGBoost
- Generative AI: LLMs, RAG, MCP, prompt engineering, agentic workflows, LLM evaluation, Hugging Face
- Developer Tools: Jupyter, Databricks, Git, BigQuery, FastAPI, Django, SSE, VS Code, Docker
- Libraries: Pandas, NumPy, Tidyverse, dplyr, tidyr, stats, Matplotlib
""",
    "education.md": """# Education

California State University, San Bernardino — San Bernardino, CA
Bachelor of Science in Bioinformatics, Department of Computer Science and Engineering
Graduated May 2024
""",
    "certifications.md": """# Certifications

- Deep Learning Specialization — Andrew Ng (DeepLearning.AI), 2026
- Data Science Specialization — Johns Hopkins University, 2025
- Building AI Agents and Agentic Workflows — IBM, in progress
""",
}

out_dir = Path("data/experience")
out_dir.mkdir(parents=True, exist_ok=True)
for name, text in FILES.items():
    (out_dir / name).write_text(text, encoding="utf-8")
    print(f"wrote {out_dir / name}")