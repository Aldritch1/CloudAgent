from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cloudagent.metrics import get_metrics, llm_calls_total, record_cache_hit, record_llm_call


class TestMetricsMiddleware:
    @patch("cloudagent.metrics.http_requests_total")
    @patch("cloudagent.metrics.http_request_duration_seconds")
    def test_middleware_records_request(self, mock_duration, mock_counter):
        from cloudagent.metrics import MetricsMiddleware
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        mock_counter.labels.assert_called_once()
        mock_duration.labels.assert_called_once()


def test_llm_call_counter():
    record_llm_call("chat", "success")
    output = get_metrics().decode()
    assert 'llm_calls_total{agent_type="chat",status="success"}' in output


def test_cache_hit_counter():
    record_cache_hit("l1", True)
    output = get_metrics().decode()
    assert 'cache_hits_total{hit="True",tier="l1"}' in output


def test_metrics_output_contains_expected_names():
    output = get_metrics().decode()
    assert "llm_calls_total" in output
    assert "cache_hits_total" in output
