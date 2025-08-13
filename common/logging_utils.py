"""
Centralized logging configuration and utilities.

Provides consistent logging setup across all data collection
and processing components with file rotation and structured output.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
from .paths import dir_logs

# Global logger registry to avoid duplicate handlers
_LOGGERS = {}

def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Get configured logger with file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        
    Returns:
        Configured logger instance
    """
    # Return existing logger if already configured
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    # Create new logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        _LOGGERS[name] = logger
        return logger
    
    # Create formatter for structured logging
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with date-based naming
    log_dir = dir_logs()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{today}.log"
    
    # Use rotating file handler to prevent huge log files
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    _LOGGERS[name] = logger
    return logger

def log_run_start(logger: logging.Logger, collector: str, **kwargs) -> None:
    """
    Log start of data collection run.
    
    Args:
        logger: Logger instance
        collector: Collector type ("stocks", "crypto", "reddit")
        **kwargs: Additional run parameters to log
    """
    params = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[RUN] collector={collector} {params}")

def log_validation(logger: logging.Logger, symbol: str, **results) -> None:
    """
    Log validation results.
    
    Args:
        logger: Logger instance  
        symbol: Asset symbol
        **results: Validation results to log
    """
    results_str = " ".join(f"{k}={v}" for k, v in results.items())
    logger.info(f"[VALIDATION] symbol={symbol} {results_str}")

def log_write(logger: logging.Logger, path: Path, rows: int, checksum: str, version: str) -> None:
    """
    Log successful file write.
    
    Args:
        logger: Logger instance
        path: File path that was written
        rows: Number of rows written
        checksum: SHA256 checksum
        version: Version timestamp
    """
    logger.info(f"[WRITE] path={path} rows={rows} checksum={checksum[:8]}... version={version}")

def log_error(logger: logging.Logger, message: str, exc_info: bool = True) -> None:
    """
    Log error with optional exception traceback.
    
    Args:
        logger: Logger instance
        message: Error message
        exc_info: Whether to include exception traceback
    """
    logger.error(f"[ERROR] {message}", exc_info=exc_info)

def log_collection_summary(logger: logging.Logger, collector: str, success_count: int, 
                          total_count: int, failed: Optional[list] = None) -> None:
    """
    Log summary of collection operation.
    
    Args:
        logger: Logger instance
        collector: Collector type
        success_count: Number of successful collections
        total_count: Total number attempted
        failed: List of failed items (optional)
    """
    logger.info(f"[SUMMARY] collector={collector} success={success_count}/{total_count}")
    
    if failed:
        logger.warning(f"[SUMMARY] failed_items={', '.join(failed)}")

def setup_file_logger(log_file: Path, logger_name: str = "", level: str = "INFO") -> logging.Logger:
    """
    Set up a logger that only writes to a specific file.
    
    Args:
        log_file: Path to log file
        logger_name: Name for the logger (defaults to filename)
        level: Logging level
        
    Returns:
        File-only logger instance
    """
    if not logger_name:
        logger_name = log_file.stem
    
    # Create unique logger name to avoid conflicts
    unique_name = f"file_{logger_name}_{id(log_file)}"
    
    if unique_name in _LOGGERS:
        return _LOGGERS[unique_name]
    
    logger = logging.getLogger(unique_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Ensure parent directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler only
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate console output
    logger.propagate = False
    
    _LOGGERS[unique_name] = logger
    return logger

def get_log_files(days_back: int = 7) -> list[Path]:
    """
    Get list of log files from recent days.
    
    Args:
        days_back: Number of days back to search
        
    Returns:
        List of log file paths sorted by date
    """
    log_dir = dir_logs()
    
    if not log_dir.exists():
        return []
    
    log_files = []
    
    # Look for daily log files
    from datetime import timedelta
    today = datetime.now()
    
    for i in range(days_back + 1):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y%m%d")
        log_file = log_dir / f"{date_str}.log"
        
        if log_file.exists():
            log_files.append(log_file)
    
    # Also include any other .log files
    for log_file in log_dir.glob("*.log"):
        if log_file not in log_files:
            log_files.append(log_file)
    
    return sorted(log_files, reverse=True)  # Most recent first

def tail_log(log_file: Path, lines: int = 50) -> list[str]:
    """
    Get the last N lines from a log file.
    
    Args:
        log_file: Path to log file
        lines: Number of lines to return
        
    Returns:
        List of log lines (most recent last)
    """
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            return all_lines[-lines:] if len(all_lines) > lines else all_lines
    except (OSError, UnicodeDecodeError):
        return []

def search_logs(pattern: str, log_files: Optional[list[Path]] = None) -> list[str]:
    """
    Search for pattern in log files.
    
    Args:
        pattern: Pattern to search for
        log_files: List of log files to search (defaults to recent files)
        
    Returns:
        List of matching log lines
    """
    if log_files is None:
        log_files = get_log_files()
    
    matches = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.lower() in line.lower():
                        matches.append(f"{log_file.name}:{line_num}: {line.strip()}")
        except (OSError, UnicodeDecodeError):
            continue
    
    return matches

def cleanup_old_logs(days_to_keep: int = 30) -> int:
    """
    Clean up log files older than specified days.
    
    Args:
        days_to_keep: Number of days of logs to keep
        
    Returns:
        Number of files deleted
    """
    log_dir = dir_logs()
    
    if not log_dir.exists():
        return 0
    
    from datetime import timedelta
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    deleted_count = 0
    
    for log_file in log_dir.glob("*.log*"):  # Include .log.1, .log.2, etc.
        try:
            # Get file modification time
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            if mtime < cutoff_date:
                log_file.unlink()
                deleted_count += 1
                
        except (OSError, FileNotFoundError):
            continue
    
    return deleted_count