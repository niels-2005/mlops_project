from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.auth.models import User
from backend.config import Config
from backend.prediction.models import Predictions

async_engine = create_async_engine(
    url=Config.DATABASE_URL,
    echo=True,
)


async def init_db() -> None:
    """
    Initialize the database by creating all tables defined in SQLModel metadata.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session():
    """
    Async generator that yields a SQLAlchemy AsyncSession for DB operations.
    """
    Session = sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with Session() as session:
        yield session
