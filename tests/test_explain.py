import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from src import explain as X

TRACK = {"content_desc": "Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study.",
         "tags": ["calm", "melancholic", "instrumental"], "context": ["study"], "cluster_name": "Calm Ambient Study"}


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        texts = self.text if isinstance(self.text, list) else [self.text]
        return {"message": {"content": json.dumps({"items": [{"id": i, "explanation": t}
                                                             for i, t in enumerate(texts, 1)]})}}


def fake_llm(text):
    return lambda url, timeout, json: _Resp(text)


def test_grounded_llm_text_is_kept():
    r = X.explain("something calm for studying", TRACK, post=fake_llm("A calm, melancholic instrumental piece for focused study."))
    assert r["source"] == "llm" and r["text"].startswith("A calm")


def test_grounding_rejects_invented_tag():
    r = X.explain("something calm for studying", TRACK, post=fake_llm("An upbeat, happy track perfect for a party."))
    assert r["source"] == "fallback" and r["reason"].startswith("ungrounded")
    assert {"upbeat", "happy", "party"} <= set(X.ungrounded("An upbeat, happy track perfect for a party.", TRACK))
    assert X.ungrounded("Soft and calm, great for late night focus.", TRACK) == []      # synonyms map to study


def test_fallback_template():
    r = X.explain("calm music for studying", TRACK, post=fake_llm(" ".join(["word"] * 30)))
    assert r["source"] == "fallback" and r["reason"] == "too long"
    assert r["text"] == "calm, melancholic, instrumental — matches 'calm studying'"
    assert X.fallback("pump me up", TRACK) == "calm, melancholic, instrumental — closest match for 'pump me up'"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_fallback_when_ollama_is_down():
    url = f"http://127.0.0.1:{_free_port()}/api/chat"              # nothing listening
    r = X.explain("calm music for studying", TRACK, url=url)
    assert r["source"] == "fallback" and r["reason"].startswith("ollama unavailable")


def test_fallback_on_timeout():
    class Slow(BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(1.5)
            self.send_response(200); self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t0 = time.perf_counter()
        r = X.explain("calm music", TRACK, timeout=0.3, url=f"http://127.0.0.1:{srv.server_port}/api/chat")
        assert r["source"] == "fallback" and r["reason"] == "timeout" and time.perf_counter() - t0 < 1.4
    finally:
        srv.shutdown()


def test_bad_json_falls_back():
    class Bad(_Resp):
        def json(self):
            return {"message": {"content": "not json"}}
    r = X.explain("calm", TRACK, post=lambda url, timeout, json: Bad(""))
    assert r["source"] == "fallback" and r["reason"].startswith("bad response")


TRACK2 = {"content_desc": "Genre: punk. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.",
          "tags": ["fast", "energetic", "neutral mood"], "context": ["workout", "running", "gym"], "cluster_name": "Punk"}


def test_batch_grounding_and_fallback_per_item():
    out = X.explain_batch("calm music for studying", [TRACK, TRACK2],
                          post=fake_llm(["A calm instrumental piece for study.", "A sad, sleepy track."]))
    assert [o["source"] for o in out] == ["llm", "fallback"] and out[1]["reason"].startswith("ungrounded")
    one = X.explain_batch("calm", [TRACK, TRACK2], post=fake_llm(["Calm and instrumental."]))   # item 2 missing
    assert [o["source"] for o in one] == ["llm", "fallback"] and one[1]["reason"] == "missing item"
    assert X.explain_batch("x", []) == []


def test_prompt_view_is_cleaned_and_lists_matches():
    f = X.facts("fast songs for the gym", TRACK2)
    assert "neutral mood" not in f and "moderate energy" not in f
    assert "Matches the request: fast, gym, workout." in f
    assert "Matches the request: nothing directly." in X.facts("something romantic", TRACK2)
    assert X.timeout_for(1) == X.TIMEOUT_S and X.timeout_for(10) > X.TIMEOUT_S


def test_negative_explanations_fall_back():
    for bad in ["No direct match for a long drive at night.", "This doesn't really fit the request.",
                "Not a great match, but calm."]:
        r = X.explain("songs for a long drive at night", TRACK, post=fake_llm(bad))
        assert r["source"] == "fallback" and r["reason"] == "says it does not match", bad
    assert X.explain("calm study", TRACK, post=fake_llm("A calm instrumental match for study."))["source"] == "llm"
