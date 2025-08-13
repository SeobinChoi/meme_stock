"""
Common utilities package for meme stock data collection pipeline.

This package provides unified utilities for:
- Path management and directory structure
- Time handling and UTC conversions  
- Data validation and schema enforcement
- Atomic file I/O operations
- Metadata generation and versioning
- Centralized logging configuration
"""

__version__ = "1.0.0"

# Import key utilities for easier access
from .paths import DATA_ROOT, ensure_dirs_exist
from .time_utils import now_utc_iso, to_utc_iso
from .logging_utils import get_logger
from .metadata import build_metadata, write_metadata

__all__ = [
    "DATA_ROOT",
    "ensure_dirs_exist", 
    "now_utc_iso",
    "to_utc_iso",
    "get_logger",
    "build_metadata",
    "write_metadata"
]