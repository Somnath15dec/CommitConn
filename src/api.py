"""Step K: FastAPI app — guarded, language-steerable recommendations with explanations.

    uvicorn src.api:app --port 8000

POST /recommend {query, user_id?, dial in [-1, 1], k, explain}: guards (src/guards.py) -> SID model
(trie-constrained beam, discovery dial lambda = -dial on the user's history level-1 codes, history tracks excluded)
with a 2 s budget, else FAISS ("faiss_fallback"). explain = "template" (default, fast) returns template
explanations; "llm" runs the batch explainer on the CLEANED query (template fallback per item).
Two-phase UI flow: /recommend (templates, ~0.4 s) then POST /explain {query (cleaned), track_ids} for LLM text.
GET /users, POST /feedback (-> logs/feedback.jsonl), GET /health. CORS for the local UI dev server.
Startup warm-up loads everything and runs one dummy explanation.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import explain as X
from src import guards as G
from src.analyze_readiness import content_desc
from src.cluster_names import CLUSTER_NAMES
from src.paths import CATALOG, REAL_USERS, ROOT
from src.train_data import SIDS

SID_BUDGET_S = 2.0
BEAMS = 50
MAX_K = 50
DEMO_USERS, DEMO_SEED, DEMO_MIN_HISTORY = 20, 24, 30
FEEDBACK_LOG = ROOT / "logs" / "feedback.jsonl"
UI_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"]


# ---------------------------------------------------------------- schemas
class RecommendIn(BaseModel):
    query: str = Field("", max_length=20_000)
    user_id: str | None = None
    dial: float = Field(0.0, ge=-1.0, le=1.0, description="-1 familiar ... +1 discovery (lambda = -dial)")
    k: int = Field(10, ge=1, le=MAX_K)
    explain: Literal["template", "llm"] = "template"


class ExplainIn(BaseModel):
    query: str = Field(..., max_length=2_000)
    track_ids: list[str] = Field(..., min_length=1, max_length=MAX_K)


class FeedbackIn(BaseModel):
    user_id: str
    query: str = Field(..., max_length=2_000)
    track_id: str
    action: Literal["skip", "save", "replay"]
    dial: float = Field(0.0, ge=-1.0, le=1.0)
    timestamp: str | float | None = None


# ---------------------------------------------------------------- services
class Services:
    """Data always; models only when load_models=True (tests inject fakes for sid / faiss / explain)."""

    def __init__(self, load_models: bool = True, words=None, classifier=None):
        cat = pd.read_parquet(CATALOG)
        cat["content_desc"] = cat.apply(content_desc, axis=1)
        sids = pd.read_parquet(SIDS)
        names = json.loads(CLUSTER_NAMES.read_text(encoding="utf-8"))
        self.sids = sids
        self.level1 = dict(zip(sids.track_id, sids.a.astype(int)))
        sid_str = dict(zip(sids.track_id, sids[["a", "b", "c", "d"]].astype(int).astype(str).agg("-".join, axis=1)))
        self.meta = {r.track_id: {"track_id": r.track_id, "track_name": r.track_name,
                                  "artists": r.artists.replace(";", ", "), "sid": sid_str[r.track_id],
                                  "cluster_name": names[str(self.level1[r.track_id])]["name"],
                                  "content_desc": r.content_desc, "tags": list(r.tags), "context": list(r.context)}
                     for r in cat.itertuples(index=False)}
        u = pd.read_parquet(REAL_USERS)
        self.history = u[u.split == "history"].groupby("user_id").track_id.agg(list).to_dict()
        self.cluster_names = names
        self.sid_ready = self.faiss_ready = False
        self.explainer_warm = None
        self._dec_cache: dict[str, object] = {}
        self._pool = cf.ThreadPoolExecutor(max_workers=1)          # GPU calls are serialized
        if words is None or classifier is None:
            words, classifier, self._st = G.default_components("cpu")
        self.words, self.classifier = words, classifier
        if load_models:
            self._load_models(cat)

    def _load_models(self, cat: pd.DataFrame) -> None:
        from src.baseline import FaissBaseline, load_catalog
        from src.decode import SidDecoder
        from src.evaluate_final import load_sid_model
        self.fb = FaissBaseline(load_catalog())
        self.faiss_ready = True
        self.model, self.tok = load_sid_model()
        self._SidDecoder = SidDecoder
        self._dec_cache["*"] = SidDecoder(self.model, self.tok, self.sids)
        self.sid_ready = True

    # -- retrieval (overridable)
    def _decoder(self, user_id: str | None):
        if not user_id:
            return self._dec_cache["*"]
        if user_id not in self._dec_cache:
            if len(self._dec_cache) > 64:
                self._dec_cache = {"*": self._dec_cache["*"]}
            self._dec_cache[user_id] = self._SidDecoder(self.model, self.tok, self.sids,
                                                        exclude=set(self.history[user_id]))
        return self._dec_cache[user_id]

    def sid(self, query: str, k: int, user_id: str | None, lam: float) -> list[str]:
        from src.dial import dial_fn
        dec = self._decoder(user_id)
        fn = None
        if user_id and lam != 0:
            fn = dial_fn(self.tok, {self.level1[t] for t in self.history[user_id]}, lam)
        res = dec.search([query], n=k, num_beams=max(BEAMS, k), batch_size=1, logits_processor_fn=fn)[0]
        return [t for t, _ in res]

    def faiss(self, query: str, k: int, user_id: str | None) -> list[str]:
        excl = set(self.history.get(user_id, [])) if user_id else set()
        got = self.fb.search([query], k + len(excl))[0]
        return [t for t in got if t not in excl][:k]

    def explain(self, query: str, tracks: list[dict]) -> list[dict]:
        return X.explain_batch(query, tracks)

    def sid_with_budget(self, query, k, user_id, lam, budget: float | None = None) -> list[str]:
        """Raises TimeoutError / any SID error; the caller falls back to FAISS. Budget read at call time."""
        return self._pool.submit(self.sid, query, k, user_id, lam).result(
            timeout=SID_BUDGET_S if budget is None else budget)

    def explainer_status(self) -> dict:
        try:
            r = requests.get(X.OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=1.0)
            models = [m["name"] for m in r.json().get("models", [])]
            return {"reachable": True, "model": X.EXPLAIN_MODEL, "model_present": X.EXPLAIN_MODEL in models,
                    "warm_up": self.explainer_warm}
        except (requests.RequestException, ValueError):
            return {"reachable": False, "model": X.EXPLAIN_MODEL, "model_present": False, "warm_up": self.explainer_warm}

    def warm_up(self) -> None:
        t = next(iter(self.meta.values()))
        r = X.explain_batch("warm-up", [t], timeout=120)
        self.explainer_warm = r[0]["source"]
        if self.sid_ready:
            self.sid("music for a quiet evening", 10, None, 0.0)

    def demo_users(self, n: int = DEMO_USERS, seed: int = DEMO_SEED) -> list[dict]:
        elig = sorted(u for u, h in self.history.items() if len(h) >= DEMO_MIN_HISTORY)
        out = []
        for u in sorted(np.random.default_rng(seed).choice(elig, n, replace=False)):
            top = Counter(self.level1[t] for t in self.history[u]).most_common(3)
            out.append({"user_id": u, "n_history": len(self.history[u]),
                        "top_clusters": [self.cluster_names[str(a)]["name"] for a, _ in top]})
        return out


# ---------------------------------------------------------------- recommend logic
def recommend(svc: Services, req: RecommendIn) -> dict:
    t0 = time.perf_counter()
    if req.user_id and req.user_id not in svc.history:
        raise HTTPException(404, f"unknown user_id {req.user_id!r}")
    g = G.check(req.query, svc.words, svc.classifier)
    meta = {"system": None, "guard": g.guard, "notes": g.notes, "dial": req.dial, "lambda": -req.dial}
    if g.guard:
        meta["latency_ms"] = (time.perf_counter() - t0) * 1000
        return {"query": g.query, "message": g.message, "results": [], "meta": meta}
    lam = -req.dial
    try:
        ids = svc.sid_with_budget(g.query, req.k, req.user_id, lam)
        meta["system"] = "sid"
        if not ids:
            raise ValueError("SID model returned no tracks")
    except Exception as e:                                         # timeout, model error, empty -> FAISS
        meta["system"], meta["fallback_reason"] = "faiss_fallback", f"{type(e).__name__}: {e}"[:200]
        ids = svc.faiss(g.query, req.k, req.user_id)
    tracks = [svc.meta[t] for t in ids]
    exps = explanations(svc, g.query, tracks, req.explain)
    H = {svc.level1[t] for t in svc.history[req.user_id]} if req.user_id else None
    results = [card(svc, t, e, H) for t, e in zip(tracks, exps)]
    meta["latency_ms"] = (time.perf_counter() - t0) * 1000
    return {"query": g.query, "message": None, "results": results, "meta": meta}


def explanations(svc: Services, query: str, tracks: list[dict], mode: str) -> list[dict]:
    """mode "template": instant templates; "llm": batch explainer (cleaned query only), template fallback per item."""
    if mode == "template":
        return [{"text": X.fallback(query, t), "source": "template", "reason": None} for t in tracks]
    try:
        return svc.explain(query, tracks)
    except Exception as e:
        return [{"text": X.fallback(query, t), "source": "fallback", "reason": f"explainer error: {type(e).__name__}"}
                for t in tracks]


def card(svc: Services, t: dict, e: dict, H: set[int] | None) -> dict:
    return {**{k: t[k] for k in ("track_id", "track_name", "artists", "sid", "cluster_name")},
            "level1": svc.level1[t["track_id"]],
            "new_to_user": None if H is None else svc.level1[t["track_id"]] not in H,
            "explanation": e["text"], "explanation_source": e["source"],
            "explanation_fallback_reason": None if e["source"] in ("llm", "template") else e.get("reason")}


def explain_tracks(svc: Services, req: ExplainIn) -> dict:
    """Phase 2: LLM explanations for tracks already shown. The query is re-guarded; a guarded query never reaches
    the explainer (templates only)."""
    t0 = time.perf_counter()
    unknown = [t for t in req.track_ids if t not in svc.meta]
    if unknown:
        raise HTTPException(404, f"unknown track_id(s): {unknown[:3]}")
    g = G.check(req.query, svc.words, svc.classifier)
    tracks = [svc.meta[t] for t in req.track_ids]
    if g.guard:
        exps = [{"text": X.fallback(G.clean(req.query), t), "source": "fallback", "reason": f"guard: {g.guard}"}
                for t in tracks]
    else:
        exps = explanations(svc, g.query, tracks, "llm")
    items = [{"track_id": t["track_id"], "explanation": e["text"], "explanation_source": e["source"],
              "explanation_fallback_reason": None if e["source"] == "llm" else e.get("reason")}
             for t, e in zip(tracks, exps)]
    return {"query": g.query, "items": items, "meta": {"guard": g.guard, "latency_ms": (time.perf_counter() - t0) * 1000}}


# ---------------------------------------------------------------- app
def create_app(services: Services | None = None, warm: bool = True) -> FastAPI:
    state = {"svc": services}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["svc"] is None:
            state["svc"] = Services(load_models=True)
        if warm:
            state["svc"].warm_up()
        yield

    app = FastAPI(title="Shelf — language-steerable music recommender", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=UI_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.post("/recommend")
    def _recommend(req: RecommendIn):
        return recommend(state["svc"], req)

    @app.post("/explain")
    def _explain(req: ExplainIn):
        return explain_tracks(state["svc"], req)

    @app.get("/users")
    def _users():
        return {"users": state["svc"].demo_users()}

    @app.post("/feedback")
    def _feedback(fb: FeedbackIn):
        svc = state["svc"]
        if fb.track_id not in svc.meta:
            raise HTTPException(404, f"unknown track_id {fb.track_id!r}")
        rec = {**fb.model_dump(), "received_at": datetime.now(timezone.utc).isoformat()}
        path = getattr(svc, "feedback_log", FEEDBACK_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True}

    @app.get("/health")
    def _health():
        svc = state["svc"]
        return {"sid_model": {"loaded": svc.sid_ready, "budget_s": SID_BUDGET_S},
                "faiss": {"loaded": svc.faiss_ready, "size": getattr(getattr(svc, "fb", None), "ids", np.array([])).size},
                "explainer": svc.explainer_status(),
                "guards": {"classifier_threshold": svc.classifier.threshold,
                           "classifier_loo_accuracy": svc.classifier.loo_accuracy}}

    return app


app = create_app()
