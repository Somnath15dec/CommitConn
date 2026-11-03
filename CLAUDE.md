# Project: language-steerable music recommender (hackathon, 18h)
Goal: NL query -> Semantic IDs (3x256 residual k-means + dedupe digit) -> constrained beam search -> explanation.
Switch: familiar vs non-habitual = mask on level-1 SID token using user history.
Stack: Python, Qwen3-0.6B + LoRA (peft), sentence-transformers, faiss-cpu, FastAPI.
Hardware: RTX 4060 8GB.
Rules:
- One milestone per task; conventional commit after each.
- Every module gets a pytest test.
- Never silently change data thresholds or templates; ask first.
- Never print whole data files: shape, columns, head() only. Trim logs to the last 50 lines.
- Train and eval query templates must never overlap.
