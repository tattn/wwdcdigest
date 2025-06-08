"""Logging utilities for wwdcdigest."""

import logging
import sys
from typing import TextIO

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "wwdcdigest",
    level: int = logging.INFO,
    log_file: str | None = None,
    file_stream: TextIO | None = None,
) -> logging.Logger:
    """Set up and configure a logger.

    Args:
        name: The name of the logger.
        level: The logging level.
        log_file: Optional file path to write logs to.
        file_stream: Optional file stream to write logs to.

    Returns:
        A configured logger instance.
    """
    # Configure rich handler for console output
    console = Console(file=file_stream or sys.stderr)
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    # Configure basic logging
    handlers = [rich_handler]

    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        # Cast to avoid type error with different handler types
        handlers.append(file_handler)  # type: ignore

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    return logging.getLogger(name)
