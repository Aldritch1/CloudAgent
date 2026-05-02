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
    jwt_secret: SecretStr = SecretStr("")
    jwt_algorithm: str = "HS256"
    jwt_disabled: bool = False

    rate_limit_requests_per_minute: int = 60
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    enable_metrics: bool = True
    default_tenant_id: str = "default"

    mcp_servers: str = "order,sms,ticket"
    order_service_url: str = ""
    sms_service_url: str = ""
    ticket_service_url: str = ""


settings = Settings()
