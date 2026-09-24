"""Logging configuration for library and experiment code.

Library modules use module-level loggers (never ``print``); experiment scripts call
:func:`configure_logging` once at startup.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once, writing to stderr with a concise format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring a default handler exists."""
    if not _CONFIGURED:
        # Attach a NullHandler so library import never emits noise; scripts call
        # configure_logging() to actually surface messages.
        logging.getLogger().addHandler(logging.NullHandler())
    return logging.getLogger(name)
