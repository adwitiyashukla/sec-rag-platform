"""Structured logging.

Human-readable colourised output locally, single-line JSON in containers, and
the active trace id injected into every record so a log line can always be tied
back to the request that produced it.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

_CONFIGURED = False


def _add_trace_id(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    """Attach the current trace id, if a trace is active."""
    from secrag.observability.tracing import current_trace_id

    if (trace_id := current_trace_id()) is not None:
        event_dict.setdefault("trace_id", trace_id)
    return event_dict


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Configure structlog and the stdlib root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _add_trace_id,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level.upper())
    # These libraries are chatty at INFO and drown out anything useful.
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use.

    The module name is bound explicitly rather than via stdlib.add_logger_name,
    because this stack renders through PrintLogger, which has no .name to read.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger().bind(logger=name)  # type: ignore[no-any-return]
