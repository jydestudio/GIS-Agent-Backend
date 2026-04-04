"""
Structured logging configuration.

Provides consistent, readable log output across the entire application.
Uses stdlib logging — no external dependencies.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level override. Defaults to DEBUG in development, INFO otherwise.
    """
    from app.core.config import get_settings

    settings = get_settings()
    log_level = level or "INFO"

    # ── Format ───────────────────────────────────
    fmt = (
        "%(asctime)s │ %(levelname)-8s │ %(name)-28s │ %(message)s"
        if settings.is_development
        else "%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Quiet down noisy third-party loggers
    for noisy in (
        "httpx", "httpcore", "openai", "urllib3",
        "sqlalchemy", "sqlalchemy.engine", "sqlalchemy.pool",
        "langchain", "langchain_core", "langchain_openai",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a module."""
    return logging.getLogger(name)
