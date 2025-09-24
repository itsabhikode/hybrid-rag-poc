from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Airflow configuration
    airflow_base_url: str = "http://airflow:8080"
    airflow_username: str = "admin"
    airflow_password: str = "MvNAADknzGWgABwF"
    airflow_api_version: str = "api/v2"
    airflow_use_mock: bool = True  # Set to False when real Airflow auth is working
    
    # Application configuration
    output_base_dir: str = "output"

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables


@lru_cache()
def get_settings():
    return Settings()

settings: Settings = get_settings()
