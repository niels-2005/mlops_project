from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@postgres:5432/mydb"
    JWT_SECRET: str = "e698218fbf1d9d46b06a6c1aa41b3124"
    JWT_ALGORITHM: str = "HS256"


Config = Settings()
