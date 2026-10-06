"""
Structured logging module with sensitive information masking.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SENSITIVE_KEYS = {
    "api_key",
    "password",
    "token",
    "secret",
    "authorization",
    "llm_api_key",
    "qdrant_api_key",
}


def mask_sensitive_data(data: Any) -> Any:
    """Recursively mask sensitive values in dicts/lists for logging."""
    if isinstance(data, dict):
        masked: Dict[str, Any] = {}
        for k, v in data.items():
            if any(sensitive in k.lower() for sensitive in SENSITIVE_KEYS):
                masked[k] = "***REDACTED***" if v else None
            else:
                masked[k] = mask_sensitive_data(v)
        return masked
    elif isinstance(data, list):
        return [mask_sensitive_data(item) for item in data]
    return data


class StructuredJsonFormatter(logging.Formatter):
    """Custom logging formatter that outputs JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_obj.update(mask_sensitive_data(record.extra))

        for key, value in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "extra",
            }:
                if isinstance(value, (dict, list)):
                    log_obj[key] = mask_sensitive_data(value)
                else:
                    log_obj[key] = value

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def setup_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logger with stdout handler."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Helper to get a named logger."""
    return logging.getLogger(name)
