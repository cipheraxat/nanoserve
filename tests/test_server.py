from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from nanoserve.server import create_app


@pytest.fixture
def client(make_engine):
    with TestClient(create_app(make_engine())) as c:
        yield c


def test_completion(client):
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 5, "temperature": 0, "ignore_eos": True})
    assert r.status_code == 200
    body = r.json()
    assert body["usage"]["completion_tokens"] == 5
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["text"].count("t") == 5


def test_streaming_matches_non_streaming(client):
    req = {"prompt": [5, 6, 7, 8], "max_tokens": 6, "temperature": 0, "ignore_eos": True}
    full = client.post("/v1/completions", json=req).json()["choices"][0]["text"]
    with client.stream("POST", "/v1/completions", json={**req, "stream": True}) as r:
        events = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert "".join(c["choices"][0]["text"] for c in chunks) == full
    assert chunks[-1]["usage"]["completion_tokens"] == 6


def test_chat_completion_and_stream(client):
    req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0, "ignore_eos": True}
    body = client.post("/v1/chat/completions", json=req).json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    with client.stream("POST", "/v1/chat/completions", json={**req, "stream": True}) as r:
        events = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    first = json.loads(events[0])
    assert first["choices"][0]["delta"]["role"] == "assistant"


def test_rejects_bad_requests(client):
    assert client.post("/v1/completions", json={"prompt": "x", "max_tokens": 10_000}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": "x", "top_p": 0}).status_code == 400


def test_concurrent_requests_and_metrics(client):
    import concurrent.futures as cf

    def call(i):
        return client.post(
            "/v1/completions", json={"prompt": [i + 1] * 10, "max_tokens": 4, "ignore_eos": True}
        ).status_code

    with cf.ThreadPoolExecutor(8) as pool:
        assert list(pool.map(call, range(8))) == [200] * 8
    text = client.get("/metrics").text
    assert "nanoserve_generation_tokens_total 32" in text
    assert "nanoserve_requests_total 8" in text
