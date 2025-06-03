import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.mlops.main import run_mlops_pipeline

from .auth.routes import auth_router
from .db.main import init_db
from .middleware import register_middleware
from .prediction.routes import prediction_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_mlops_pipeline()
    await init_db()
    yield


version = "v1"

description = """
A REST API for a book review web service.

This REST API is able to;
- Create Read Update And delete books
- Add reviews to books
- Add tags to Books e.t.c.
    """

version_prefix = f"/api/{version}"

app = FastAPI(
    title="MLOps",
    description=description,
    version=version,
    license_info={"name": "MIT License", "url": "https://opensource.org/license/mit"},
    contact={
        "name": "Niels Scholz",
        "url": "https://github.com/niels-2005",
    },
    terms_of_service="https://example.com/tos",
    openapi_url=f"{version_prefix}/openapi.json",
    docs_url=f"{version_prefix}/docs",
    redoc_url=f"{version_prefix}/redoc",
    debug=True,
    lifespan=lifespan,
)

register_middleware(app)

app.include_router(
    prediction_router, prefix=f"{version_prefix}/predict", tags=["predict"]
)
app.include_router(auth_router, prefix=f"{version_prefix}/auth", tags=["auth"])
