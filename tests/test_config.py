import importlib

import pytest


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://test:6379/0")
    monkeypatch.setenv("MODEL_NAME", "gpt-4")


def test_settings_loads_from_env(patch_env):
    # Import here AFTER env vars are patched
    from cloudagent.config import settings

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert str(settings.redis_url) == "redis://test:6379/0"
    assert settings.model_name == "gpt-4"


def test_settings_class_instantiation(patch_env):
    from cloudagent.config import Settings

    s = Settings()
    assert s.openai_api_key.get_secret_value() == "test-key"
    assert str(s.redis_url) == "redis://test:6379/0"
    assert s.model_name == "gpt-4"
