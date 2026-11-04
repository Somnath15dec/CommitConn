# Shelf — a language-steerable music recommender

Natural-language query → Semantic IDs (3×256 residual k-means + dedupe digit) → trie-constrained beam search with a
Qwen3-0.6B + LoRA generator → grounded one-line explanation. A "Familiar ⟷ Discover" dial biases the first SID
token toward or away from the listener's history clusters. See `docs/PHASE1.md` and `eval/PLAN.md` for data and
evaluation details, and `eval/results_*.md` for results.

## Run the demo locally

Requirements: the Python venv (`requirements.txt`), the trained adapter in `checkpoints/run1/epoch3`, Ollama with
`qwen3:4b-instruct` pulled (`ollama pull qwen3:4b-instruct`), and Node 18+.

**1. API** (from the repo root; loads the SID model, FAISS and the explainer, then warms up — about 30–60 s):

```powershell
.venv\Scripts\python -m uvicorn src.api:app --port 8000
```

Check it: `http://localhost:8000/health` · interactive docs: `http://localhost:8000/docs`.

**2. UI** (in a second terminal):

```powershell
cd ui
npm install        # first time only
npm run dev        # http://localhost:5173
```

The UI talks to `http://localhost:8000` by default; set `VITE_API_URL` to point elsewhere. CORS is enabled for
`localhost:5173`.

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/recommend` | `{query, user_id?, dial ∈ [-1, 1], k, explain: "template" \| "llm"}` → results + `meta` (system, guard, latency) |
| POST | `/explain` | `{query (cleaned), track_ids}` → grounded LLM explanations, template fallback per item |
| GET | `/users` | 20 demo listeners with their top 3 clusters |
| POST | `/feedback` | `{user_id, query, track_id, action: skip \| save \| replay, dial, timestamp}` → `logs/feedback.jsonl` |
| GET | `/health` | SID model, FAISS and explainer status |

The UI calls `/recommend` with template explanations (fast) and then `/explain` to replace them with AI text.
