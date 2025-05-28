from pydantic_settings import BaseSettings, SettingsConfigDict


# DATABSE_URL wird aus .env Datei gezogen!
class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET: str
    JWT_ALGORITHM: str
    REDIS_HOST: str
    REDIS_PORT: int

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# aufrufen um importieren zu können
Config = Settings()
