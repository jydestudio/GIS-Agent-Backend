"""
Async database engine and session factory for Neon PostgreSQL.

Uses SQLAlchemy 2.0 async API with asyncpg driver.
Connection pooling is configured for serverless-friendly behaviour.
"""

import ssl as ssl_module
from collections.abc import AsyncGenerator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

settings = get_settings()


def _prepare_neon_url(url: str) -> tuple[str, dict]:
    """
    Prepare a Neon DATABASE_URL for asyncpg.

    Handles two common issues:
    1. Neon gives URLs with postgresql:// or postgres:// — asyncpg needs postgresql+asyncpg://
    2. asyncpg doesn't accept `sslmode` as a query param — needs `ssl` via connect_args
    """
    # Fix scheme: Neon gives postgresql:// but SQLAlchemy+asyncpg needs postgresql+asyncpg://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)

    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    connect_args: dict = {}
    if "sslmode" in params:
        mode = params.pop("sslmode")[0]
        if mode in ("require", "verify-ca", "verify-full"):
            # Create a permissive SSL context for Neon
            ctx = ssl_module.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl_module.CERT_NONE
            connect_args["ssl"] = ctx

    # Rebuild URL without sslmode
    clean_query = urlencode(params, doseq=True)
    clean_url = urlunparse(parsed._replace(query=clean_query))

    return clean_url, connect_args


_db_url, _connect_args = _prepare_neon_url(settings.database_url)

# ── Engine ───────────────────────────────────────
# Neon uses connection pooling on their side, so we keep local pool small.
# pool_pre_ping handles Neon's cold-start connection drops gracefully.
engine = create_async_engine(
    _db_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args=_connect_args,
)

# ── Session factory ──────────────────────────────
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session.

    Usage in routes:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
