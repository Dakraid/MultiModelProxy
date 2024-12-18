"""
Taken from https://github.com/theroyallab/tabbyAPI/blob/main/common/logger.py
Internal logging utility.
"""

import logging
import os
from typing import Optional, TypeVar

from loguru import logger
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
)

RICH_CONSOLE = Console()
LOG_LEVEL = os.getenv("MMP_LOG_LEVEL", "DEBUG")
T = TypeVar("T")


def unwrap(wrapped: Optional[T], default: T = None) -> T:
    """Unwrap function for Optionals."""
    if wrapped is None:
        return default

    return wrapped


def get_progress_bar():
    return Progress(console=RICH_CONSOLE)


def get_loading_progress_bar():
    """Gets a pre-made progress bar for loading tasks."""

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=RICH_CONSOLE,
    )


def _log_formatter(record: dict):
    """Log message formatter."""

    color_map = {
        "TRACE": "dim blue",
        "DEBUG": "cyan",
        "INFO": "green",
        "SUCCESS": "bold green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold white on red",
    }
    level = record.get("level")
    level_color = color_map.get(level.name, "cyan")
    colored_level = f"[{level_color}]{level.name}[/{level_color}]:"

    separator = " " * (9 - len(level.name))

    message = unwrap(record.get("message"), "")

    # Replace once loguru allows for turning off str.format
    message = message.replace("{", "{{").replace("}", "}}").replace("<", "\<")

    # Escape markup tags from Rich
    message = escape(message)
    lines = message.splitlines()

    fmt = ""
    if len(lines) > 1:
        fmt = "\n".join([f"{colored_level}{separator}{line}" for line in lines])
    else:
        fmt = f"{colored_level}{separator}{message}"

    return fmt


# Uvicorn log handler
# Uvicorn log portions inspired from https://github.com/encode/uvicorn/discussions/2027#discussioncomment-6432362
class UvicornLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger.opt(exception=record.exc_info).log(record.levelname, self.format(record).rstrip())


# Uvicorn config for logging. Passed into run when creating all loggers in server
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "uvicorn": {
            "class": f"{UvicornLoggingHandler.__module__}.{UvicornLoggingHandler.__qualname__}",  # noqa
        },
    },
    "root": {"handlers": ["uvicorn"], "propagate": False, "level": LOG_LEVEL},
}


def setup_logger():
    """Bootstrap the logger."""

    logger.remove()

    logger.add(
        RICH_CONSOLE.print,
        level=LOG_LEVEL,
        format=_log_formatter,
        colorize=True,
    )
