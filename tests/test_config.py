import os

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["REDIS_URL"] = "redis://test:6379/0"
os.environ["MODEL_NAME"] = "gpt-4"

from cloudagent.config import Settings, settings


def test_settings_loads_from_env():
    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert str(settings.redis_url) == "redis://test:6379/0"
    assert settings.model_name == "gpt-4"


def test_settings_class_instantiation():
    s = Settings()
    assert s.openai_api_key.get_secret_value() == "test-key"
    assert str(s.redis_url) == "redis://test:6379/0"
    assert s.model_name == "gpt-4"
