"""
logger.py
Centralized logging configuration for the project.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

# Ensure logs directory exists relative to project root
_project_root = Path(__file__).resolve().parent.parent
_log_dir = _project_root / "app_logs"
_log_dir.mkdir(exist_ok=True)

# Common formatter
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Info logger
info_handler = RotatingFileHandler(
    str(_log_dir / "app_info.log"), maxBytes=500 * 1024 * 1024, backupCount=5
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

# Error logger
error_handler = RotatingFileHandler(
    str(_log_dir / "app_error.log"), maxBytes=500 * 1024 * 1024, backupCount=5
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Create main logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
