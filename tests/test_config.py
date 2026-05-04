import importlib

import pytest


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://test:6379/0")
    monkeypatch.setenv("MODEL_NAME", "gpt-4")
    monkeypatch.setenv("MILVUS_URI", "http://test:19530")
    monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@test:5432/db")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120")
    monkeypatch.setenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "3")
    monkeypatch.setenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30")
    monkeypatch.setenv("ENABLE_METRICS", "false")
    monkeypatch.setenv("DEFAULT_TENANT_ID", "acme")
    monkeypatch.setenv("MCP_SERVERS", "order,sms,ticket")
    monkeypatch.setenv("ORDER_SERVICE_URL", "")
    monkeypatch.setenv("SMS_SERVICE_URL", "")
    monkeypatch.setenv("TICKET_SERVICE_URL", "")
    monkeypatch.setenv("ENABLE_SSE", "false")
    monkeypatch.setenv("CORS_ORIGINS", "https://example.com")


def test_settings_loads_from_env(patch_env):
    # Import here AFTER env vars are patched; reload to clear singleton cache
    import cloudagent.config
    importlib.reload(cloudagent.config)
    from cloudagent.config import settings

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert str(settings.redis_url) == "redis://test:6379/0"
    assert settings.model_name == "gpt-4"
    assert settings.milvus_uri == "http://test:19530"
    assert settings.neo4j_uri == "bolt://test:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password.get_secret_value() == "secret"
    assert settings.database_url == "postgresql://u:p@test:5432/db"
    assert settings.rate_limit_requests_per_minute == 120
    assert settings.circuit_breaker_failure_threshold == 3
    assert settings.circuit_breaker_recovery_timeout == 30
    assert settings.enable_metrics is False
    assert settings.default_tenant_id == "acme"
    assert settings.mcp_servers == "order,sms,ticket"
    assert settings.order_service_url == ""
    assert settings.enable_sse is False
    assert settings.cors_origins == "https://example.com"


def test_settings_class_instantiation(patch_env):
    from cloudagent.config import Settings

    s = Settings()
    assert s.openai_api_key.get_secret_value() == "test-key"
    assert str(s.redis_url) == "redis://test:6379/0"
    assert s.model_name == "gpt-4"
    assert s.milvus_uri == "http://test:19530"
    assert s.neo4j_uri == "bolt://test:7687"
    assert s.neo4j_user == "neo4j"
    assert s.neo4j_password.get_secret_value() == "secret"
    assert s.database_url == "postgresql://u:p@test:5432/db"
    assert s.rate_limit_requests_per_minute == 120
    assert s.circuit_breaker_failure_threshold == 3
    assert s.circuit_breaker_recovery_timeout == 30
    assert s.enable_metrics is False
    assert s.default_tenant_id == "acme"
    assert s.mcp_servers == "order,sms,ticket"
    assert s.order_service_url == ""
    assert s.enable_sse is False
    assert s.cors_origins == "https://example.com"
