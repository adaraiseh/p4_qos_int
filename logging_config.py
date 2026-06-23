"""Unified logging configuration for P4 QoS INT project.

Provides consistent logging across all modules with:
- Console output: Always INFO level
- File output: Configurable level (DEBUG by default)
- Separate timestamped log files per module in log/ directory
"""

import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_DIR = "log"
_log_files: dict = {}  # module_name -> log_file_path
_console_initialized = False
_artifact_owner: Optional[tuple[int, int]] = None


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


def _resolve_artifact_owner() -> Optional[tuple[int, int]]:
    """Return the real workspace user for files created from sudo-run processes."""
    global _artifact_owner
    if _artifact_owner is not None:
        return None if _artifact_owner == (-1, -1) else _artifact_owner

    uid = os.environ.get("P4_QOS_ARTIFACT_UID") or os.environ.get("SUDO_UID")
    gid = os.environ.get("P4_QOS_ARTIFACT_GID") or os.environ.get("SUDO_GID")
    if uid and gid:
        try:
            _artifact_owner = (int(uid), int(gid))
            return _artifact_owner
        except ValueError:
            pass

    # When running directly as root from the repo, prefer the workspace owner.
    # This keeps artifacts editable by the normal login user after the run.
    try:
        cwd_stat = Path.cwd().stat()
        if os.geteuid() == 0 and cwd_stat.st_uid != 0:
            _artifact_owner = (cwd_stat.st_uid, cwd_stat.st_gid)
            return _artifact_owner
    except OSError:
        pass

    _artifact_owner = (-1, -1)
    return None


def normalize_artifact_permissions(
    path: str | os.PathLike,
    *,
    file_mode: Optional[int] = None,
    dir_mode: Optional[int] = None,
) -> None:
    """Best-effort owner/mode fix for generated logs and training artifacts."""
    artifact_path = Path(path)
    try:
        owner = _resolve_artifact_owner()
        if owner is not None:
            uid, gid = owner
            try:
                os.chown(artifact_path, uid, gid)
            except PermissionError:
                pass
        mode = dir_mode if artifact_path.is_dir() else file_mode
        if mode is not None:
            try:
                os.chmod(artifact_path, mode)
            except PermissionError:
                pass
    except OSError:
        pass


def normalize_artifact_tree(
    path: str | os.PathLike,
    *,
    file_mode: int = 0o664,
    dir_mode: int = 0o775,
) -> None:
    """Normalize a directory tree without failing the caller on permission errors."""
    root = Path(path)
    normalize_artifact_permissions(root, dir_mode=dir_mode)
    if not root.is_dir():
        return
    for item in root.rglob("*"):
        normalize_artifact_permissions(item, file_mode=file_mode, dir_mode=dir_mode)


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
    normalize_artifact_permissions(LOG_DIR, dir_mode=0o775)

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
        # 50 MB limit, no backup files (older data deleted when limit reached)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=50 * 1024 * 1024,  # 50 MB
            backupCount=0,
        )
        normalize_artifact_permissions(log_file_path, file_mode=0o664)
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
