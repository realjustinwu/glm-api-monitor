from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    glm_upstream_url: str = "https://open.bigmodel.cn"
    database_url: str = "postgresql://glm:glm@timescaledb:5432/glm_monitor"
    proxy_port: int = 8000
    request_timeout: int = 120

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
