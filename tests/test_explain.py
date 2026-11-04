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
        return {"message": {"content": json.dumps({"explanation": self.text})}}


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
    assert r["source"] == "fallback" and r["reason"] == "empty or too long"
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
