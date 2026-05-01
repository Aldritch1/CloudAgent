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


def test_settings_loads_from_env(patch_env):
    # Import here AFTER env vars are patched
    from cloudagent.config import settings

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert str(settings.redis_url) == "redis://test:6379/0"
    assert settings.model_name == "gpt-4"
    assert settings.milvus_uri == "http://test:19530"
    assert settings.neo4j_uri == "bolt://test:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password.get_secret_value() == "secret"
    assert settings.database_url == "postgresql://u:p@test:5432/db"


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
