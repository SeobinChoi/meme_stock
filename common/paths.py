"""
Path management and directory structure utilities.

Provides centralized path management for the data collection pipeline,
ensuring consistent directory structure across all collectors.
"""

from pathlib import Path
from typing import Optional

# Project data root directory
DATA_ROOT = Path("data")

def dir_raw_stocks() -> Path:
    """Get raw stocks data directory."""
    return DATA_ROOT / "raw" / "stocks"

def dir_raw_crypto() -> Path:
    """Get raw crypto data directory."""
    return DATA_ROOT / "raw" / "crypto"

def dir_raw_reddit() -> Path:
    """Get raw reddit data directory.""" 
    return DATA_ROOT / "raw" / "reddit"

def dir_processed() -> Path:
    """Get processed data directory."""
    return DATA_ROOT / "processed"

def dir_features() -> Path:
    """Get features data directory."""
    return DATA_ROOT / "features"

def dir_models() -> Path:
    """Get models directory."""
    return DATA_ROOT / "models"

def dir_logs() -> Path:
    """Get logs directory."""
    return Path("logs")

def ensure_dirs_exist() -> None:
    """Create all necessary directories if they don't exist."""
    directories = [
        dir_raw_stocks(),
        dir_raw_crypto(), 
        dir_raw_reddit(),
        dir_processed(),
        dir_features(),
        dir_models(),
        dir_logs()
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def build_raw_path(asset_type: str, stem: str) -> Path:
    """
    Build path to raw data file.
    
    Args:
        asset_type: Type of asset ("stock", "crypto", "reddit")
        stem: File stem (e.g., "GME_stock_data")
        
    Returns:
        Path to the raw data file
        
    Examples:
        >>> build_raw_path("stock", "GME_stock_data")
        Path("data/raw/stocks/GME_stock_data.csv")
        >>> build_raw_path("crypto", "BTC_crypto_data")  
        Path("data/raw/crypto/BTC_crypto_data.csv")
    """
    if asset_type == "stock":
        base_dir = dir_raw_stocks()
    elif asset_type == "crypto":
        base_dir = dir_raw_crypto()
    elif asset_type == "reddit":
        base_dir = dir_raw_reddit()
    else:
        raise ValueError(f"Unknown asset_type: {asset_type}")
    
    return base_dir / f"{stem}.csv"

def versioned_filename(base: Path, ts: str) -> Path:
    """
    Generate versioned filename by inserting timestamp before extension.
    
    Args:
        base: Base file path
        ts: Timestamp string (format: YYYYMMDDHHMMSS)
        
    Returns:
        Versioned path with timestamp inserted before extension
        
    Examples:
        >>> versioned_filename(Path("GME_stock_data.csv"), "20250812153000")
        Path("GME_stock_data_v20250812153000.csv")
        >>> versioned_filename(Path("data/raw/BTC.csv"), "20250812153000")
        Path("data/raw/BTC_v20250812153000.csv")
    """
    # Split path into parts
    stem = base.stem  # filename without extension
    suffix = base.suffix  # file extension
    parent = base.parent
    
    # Insert version timestamp before extension
    versioned_stem = f"{stem}_v{ts}"
    return parent / f"{versioned_stem}{suffix}"

def get_data_index_path() -> Path:
    """Get path to the dataset index file."""
    return DATA_ROOT / "INDEX.jsonl"

def infer_asset_type_from_path(path: Path) -> str:
    """
    Infer asset type from file path.
    
    Args:
        path: File path to analyze
        
    Returns:
        Asset type string ("stock", "crypto", "reddit", "unknown")
    """
    path_str = str(path).lower()
    
    if "/stocks/" in path_str or "_stock_" in path_str:
        return "stock"
    elif "/crypto/" in path_str or "_crypto_" in path_str:
        return "crypto"
    elif "/reddit/" in path_str or "reddit_" in path_str:
        return "reddit"
    else:
        return "unknown"

def extract_symbol_from_filename(filename: str) -> Optional[str]:
    """
    Extract symbol from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        Symbol if found, None otherwise
        
    Examples:
        >>> extract_symbol_from_filename("GME_stock_data.csv")
        "GME"
        >>> extract_symbol_from_filename("BTC_crypto_data_v20250812.csv")
        "BTC"
    """
    # Remove version suffix if present
    name = filename.replace('.csv', '')
    if '_v2025' in name or '_v2024' in name:  # Remove version
        name = name.split('_v')[0]
    
    # Extract first part before underscore
    parts = name.split('_')
    if parts:
        return parts[0].upper()
    
    return None