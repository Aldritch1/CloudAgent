from pydantic import RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: SecretStr
    redis_url: RedisDsn = "redis://localhost:6379/0"
    model_name: str = "gpt-3.5-turbo"
    milvus_uri: str = "http://localhost:19530"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("password")
    database_url: str = "postgresql://cloudagent:cloudagent@localhost:5432/cloudagent"


settings = Settings()
