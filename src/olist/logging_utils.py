"""Structured logging: one line per event, ``key=value`` fields after the message.

Usage::

    log = get_logger(__name__)
    log.info("layer built", extra={"fields": {"layer": "staging", "seconds": 1.2}})

The pipeline never logs row-level data (customer ids, reviews); only counts and timings.
"""

from __future__ import annotations

import logging
import os
import sys


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name}: "
        base += record.getMessage()
        fields = getattr(record, "fields", None)
        if fields:
            base += " " + " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return f'"{s}"' if " " in s else s


def get_logger(name: str) -> logging.Logger:
    """A logger that writes key=value lines to stderr (level from OLIST_LOG_LEVEL)."""
    root = logging.getLogger("olist")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(KeyValueFormatter())
        root.addHandler(handler)
        root.setLevel(os.environ.get("OLIST_LOG_LEVEL", "INFO"))
        root.propagate = False
    return logging.getLogger(name if name.startswith("olist") else f"olist.{name}")
