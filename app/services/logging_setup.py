import logging
import os


def setup_logging(default_level: str = "INFO") -> None:
    """Configure root logging from ``LOG_LEVEL``/``LOG_FORMAT``/``LOG_DATE_FORMAT`` env vars."""
    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    log_format = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    date_format = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        force=True,
    )
