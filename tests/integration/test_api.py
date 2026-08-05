"""API surface tests.

These exercise the real FastAPI application with the offline provider and an
empty index. An empty corpus is the interesting case: it is exactly the state a
fresh deployment is in, and the service must degrade to an honest refusal
rather than erroring or inventing an answer.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from secrag.core.config import get_settings


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    """A fresh app against an isolated, empty index.

    Configuration is injected through environment variables rather than by
    monkeypatching `get_settings`. Patching the function object does not reach
    modules that imported it by name, which silently left the app pointed at
    the developer's real index: the tests then passed for the wrong reason, and
    fought the running service for the index lock.
    """
    monkeypatch.setenv("SECRAG_DATA_DIR", str(tmp_path / "data"))
    # The index must be isolated per test, but the ONNX weights must not be.
    # Leaving the model cache under tmp_path re-downloads roughly 150 MB for
    # every test, which turned an 8 second suite into 89 seconds and would hit
    # the Hugging Face rate limit in CI.
    monkeypatch.setenv(
        "SECRAG_MODEL_CACHE_DIR",
        str(pathlib.Path(__file__).resolve().parents[2] / "data" / "models"),
    )
    monkeypatch.setenv("SECRAG_LLM_PROVIDERS", "echo")
    monkeypatch.setenv("SECRAG_ENABLE_SPLADE", "false")
    monkeypatch.setenv("SECRAG_GROQ_API_KEY", "")
    monkeypatch.setenv("SECRAG_GEMINI_API_KEY", "")
    get_settings.cache_clear()

    from secrag.api.app import create_app

    settings = get_settings()
    settings.ensure_dirs()
    assert settings.data_dir == tmp_path / "data", "test must not touch the real index"

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()


def test_health_answers_before_warmup_completes(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_corpus_size(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert "corpus_chunks" in body


def test_stats_exposes_configuration(client: TestClient) -> None:
    body = client.get("/v1/stats").json()
    assert "corpus_chunks" in body
    assert "cache" in body
    assert body["settings"]["dense_model"]


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/query" in schema["paths"]
    assert "/v1/query/stream" in schema["paths"]


def test_metrics_are_prometheus_formatted(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_empty_corpus_refuses_rather_than_inventing(client: TestClient) -> None:
    """A fresh deployment must say it does not know, not answer from nothing."""
    response = client.post("/v1/query", json={"question": "What are the risk factors?"})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"]["status"] == "refused_no_context"
    assert body["answer"]["citations"] == []
    assert body["answer"]["groundedness"] == 0.0


def test_query_validation_rejects_a_too_short_question(client: TestClient) -> None:
    assert client.post("/v1/query", json={"question": "x"}).status_code == 422


def test_query_validation_rejects_an_out_of_range_top_k(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "What are the risks?", "top_k": 500})
    assert response.status_code == 422


def test_stream_emits_meta_then_tokens_then_done(client: TestClient) -> None:
    with client.stream(
        "POST", "/v1/query/stream", json={"question": "What are the risk factors?"}
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = "".join(response.iter_text())

    events = [
        line.removeprefix("event: ").strip()
        for line in body.splitlines()
        if line.startswith("event: ")
    ]
    assert events[0] == "meta", "sources must arrive before the first token"
    assert events[-1] == "done"

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert "trace" in payloads[-1]
    assert "groundedness" in payloads[-1]


def test_stream_reports_cache_state(client: TestClient) -> None:
    """The streaming path must consult the cache, not just the JSON endpoint.

    The web UI streams, so a cache that only the non-streaming endpoint checked
    was unreachable in the one path users actually take. It showed up as a
    permanently empty hit counter.
    """
    with client.stream(
        "POST", "/v1/query/stream", json={"question": "What are the risk factors?"}
    ) as response:
        body = "".join(response.iter_text())

    done = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ][-1]
    assert "cached" in done, "the client cannot tell a cached answer from a fresh one"
    assert done["cached"] is False
