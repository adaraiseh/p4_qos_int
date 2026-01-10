"""Unified logging configuration for P4 QoS INT project.

Provides consistent logging across all modules with:
- Console output: Always INFO level
- File output: Configurable level (DEBUG by default)
- Separate timestamped log files per module in log/ directory
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

LOG_DIR = "log"
_log_files: dict = {}  # module_name -> log_file_path
_console_initialized = False


class BatchFlushHandler(logging.StreamHandler):
    """Handler that batches flushes for CPU efficiency."""

    def __init__(self, stream=None, flush_interval: int = 10):
        super().__init__(stream)
        self.flush_interval = flush_interval
        self.message_count = 0

    def emit(self, record):
        super().emit(record)
        self.message_count += 1
        if self.message_count >= self.flush_interval:
            self.flush()
            self.message_count = 0


def setup_unified_logging(module_name: str = "rl_agent", log_level: str = "debug") -> logging.Logger:
    """Configure unified logging with console and file handlers.

    Args:
        module_name: Name for the log file (e.g., "rl_agent", "network", "collector").
                     Log file will be named: <module_name>_<timestamp>.log
        log_level: File log level ("debug", "info", "warning", "error").
                   Console always shows INFO level.

    Returns:
        Configured root logger instance.
    """
    global _log_files, _console_initialized

    # Parse log level string for file handler
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    file_level = level_map.get(log_level.lower(), logging.DEBUG)

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create timestamped log file for this module (once per module per session)
    if module_name not in _log_files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_files[module_name] = os.path.join(LOG_DIR, f"{module_name}_{timestamp}.log")

    log_file_path = _log_files[module_name]

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Check if this specific file handler already exists
    has_this_file_handler = any(
        isinstance(h, logging.FileHandler) and
        getattr(h, 'baseFilename', '') == os.path.abspath(log_file_path)
        for h in logger.handlers
    )

    if not has_this_file_handler:
        # File handler - configurable level with full timestamps
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Add console handler only once
    if not _console_initialized:
        has_console_handler = any(
            isinstance(h, (logging.StreamHandler, BatchFlushHandler)) and
            not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )

        if not has_console_handler:
            # Console handler - always INFO level
            console_handler = BatchFlushHandler(sys.stdout, flush_interval=10)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        _console_initialized = True

    return logger


def get_log_file_path(module_name: str = "rl_agent") -> Optional[str]:
    """Get path to log file for a specific module.

    Args:
        module_name: Module name to get log file path for.

    Returns:
        Path to log file, or None if logging not yet initialized for this module.
    """
    return _log_files.get(module_name)


def set_console_level(level: int):
    """Change console handler log level for all loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, (logging.StreamHandler, BatchFlushHandler)):
            if not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
