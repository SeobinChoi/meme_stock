"""
Unified metadata generation and management utilities.

Provides consistent metadata schema and operations across
all data collection and processing pipelines.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from .time_utils import now_utc_iso, parse_any_ts, count_missing_business_days, count_missing_calendar_days
from .paths import infer_asset_type_from_path, extract_symbol_from_filename
from .io_utils import atomic_write_json, append_to_index, compute_sha256

def build_metadata(
    symbol: str,
    asset_type: str,
    source: str,
    df: pd.DataFrame,
    date_col: str = "date",
    date_range: Optional[str] = None,
    notes: Optional[str] = None,
    version: Optional[str] = None,
    csv_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Build unified metadata dictionary.
    
    Args:
        symbol: Asset symbol (e.g., "GME", "BTC", "WSB")
        asset_type: Type of asset ("stock", "crypto", "reddit")
        source: Data source ("yfinance", "polygon", "coingecko", etc.)
        df: DataFrame containing the data
        date_col: Name of the date column
        date_range: Date range string (optional, will be computed if None)
        notes: Additional notes
        version: Version timestamp (optional, will be generated if None)
        csv_path: Path to CSV file for checksum (optional)
        
    Returns:
        Metadata dictionary following unified schema
    """
    from .time_utils import generate_version_timestamp
    
    # Basic metadata
    metadata = {
        "symbol": symbol,
        "asset_type": asset_type,
        "source": source,
        "total_records": len(df),
        "collection_timestamp": now_utc_iso(),
        "notes": notes or "",
        "version": version or generate_version_timestamp()
    }
    
    # Compute date range and missing dates if data present
    if not df.empty and date_col in df.columns:
        # Parse dates
        dates = pd.to_datetime(df[date_col])
        min_date = dates.min()
        max_date = dates.max()
        
        # Format date range
        if date_range is None:
            metadata["date_range"] = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        else:
            metadata["date_range"] = date_range
        
        # Compute missing dates based on asset type
        date_list = [parse_any_ts(str(d)) for d in dates]
        start_dt = parse_any_ts(str(min_date))
        end_dt = parse_any_ts(str(max_date))
        
        if asset_type == "stock":
            missing_dates = count_missing_business_days(date_list, start_dt, end_dt)
        else:  # crypto, reddit use calendar days
            missing_dates = count_missing_calendar_days(date_list, start_dt, end_dt)
        
        metadata["missing_dates"] = missing_dates
    else:
        metadata["date_range"] = "empty"
        metadata["missing_dates"] = 0
    
    # Add checksum if CSV path provided
    if csv_path and csv_path.exists():
        metadata["checksum_sha256"] = compute_sha256(csv_path)
    
    return metadata

def build_metadata_from_file(csv_path: Path, source: Optional[str] = None) -> Dict[str, Any]:
    """
    Build metadata by analyzing an existing CSV file.
    
    Args:
        csv_path: Path to CSV file
        source: Data source (will be inferred if None)
        
    Returns:
        Metadata dictionary
    """
    # Infer basic info from path
    asset_type = infer_asset_type_from_path(csv_path)
    symbol = extract_symbol_from_filename(csv_path.name) or "UNKNOWN"
    
    # Infer source if not provided
    if source is None:
        if asset_type == "stock" and "yfinance" not in csv_path.name.lower():
            source = "yfinance"  # Default assumption
        elif asset_type == "crypto":
            source = "coingecko"  # Default assumption
        elif asset_type == "reddit":
            source = "praw"  # Default assumption
        else:
            source = "unknown"
    
    # Read CSV to analyze
    try:
        df = pd.read_csv(csv_path)
        
        # Try different date column names
        date_col = None
        for col in ["date", "Date", "datetime", "timestamp"]:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # Create minimal metadata for files without date column
            return {
                "symbol": symbol,
                "asset_type": asset_type,
                "source": source,
                "date_range": "unknown", 
                "total_records": len(df),
                "missing_dates": 0,
                "collection_timestamp": now_utc_iso(),
                "notes": "backfilled_no_date_column",
                "version": generate_version_timestamp(),
                "checksum_sha256": compute_sha256(csv_path)
            }
        
        # Build metadata using the DataFrame
        metadata = build_metadata(
            symbol=symbol,
            asset_type=asset_type, 
            source=source,
            df=df,
            date_col=date_col,
            notes="backfilled",
            csv_path=csv_path
        )
        
        return metadata
        
    except Exception as e:
        # Fallback metadata for files that can't be read
        return {
            "symbol": symbol,
            "asset_type": asset_type,
            "source": source,
            "date_range": "error",
            "total_records": 0,
            "missing_dates": 0,
            "collection_timestamp": now_utc_iso(),
            "notes": f"backfilled_read_error: {str(e)}",
            "version": generate_version_timestamp(),
            "checksum_sha256": ""
        }

def write_metadata(metadata: Dict[str, Any], csv_path: Path) -> Path:
    """
    Write metadata to sidecar JSON file and update index.
    
    Args:
        metadata: Metadata dictionary
        csv_path: Path to associated CSV file
        
    Returns:
        Path to metadata file
    """
    from .io_utils import write_metadata_sidecar
    
    # Write sidecar metadata file
    meta_path = write_metadata_sidecar(csv_path, metadata)
    
    # Update dataset index
    append_to_index(metadata, csv_path)
    
    return meta_path

def update_index(metadata: Dict[str, Any], csv_path: Path) -> None:
    """
    Update dataset index with new entry.
    
    Args:
        metadata: Metadata dictionary
        csv_path: Path to CSV file
    """
    append_to_index(metadata, csv_path)

def load_metadata(csv_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load metadata from sidecar JSON file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Metadata dictionary or None if not found
    """
    # Determine metadata file path
    if "_v" in csv_path.stem:
        meta_path = csv_path.with_suffix('.meta.json')
    else:
        meta_path = csv_path.with_suffix('.meta.json')
    
    if not meta_path.exists():
        return None
    
    try:
        import json
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def validate_metadata_schema(metadata: Dict[str, Any]) -> List[str]:
    """
    Validate metadata against expected schema.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields
    required_fields = [
        "symbol", "asset_type", "source", "date_range",
        "total_records", "missing_dates", "collection_timestamp", "notes"
    ]
    
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
    
    # Type validation
    if "total_records" in metadata:
        if not isinstance(metadata["total_records"], int) or metadata["total_records"] < 0:
            errors.append("total_records must be non-negative integer")
    
    if "missing_dates" in metadata:
        if not isinstance(metadata["missing_dates"], int) or metadata["missing_dates"] < 0:
            errors.append("missing_dates must be non-negative integer")
    
    if "asset_type" in metadata:
        valid_types = ["stock", "crypto", "reddit", "features", "processed"]
        if metadata["asset_type"] not in valid_types:
            errors.append(f"asset_type must be one of: {valid_types}")
    
    # Date format validation (basic check)
    if "collection_timestamp" in metadata:
        try:
            parse_any_ts(metadata["collection_timestamp"])
        except:
            errors.append("collection_timestamp must be valid ISO8601 format")
    
    return errors

def get_metadata_summary(metadata: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of metadata.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Summary string
    """
    symbol = metadata.get("symbol", "UNKNOWN")
    asset_type = metadata.get("asset_type", "unknown")
    source = metadata.get("source", "unknown")
    records = metadata.get("total_records", 0)
    missing = metadata.get("missing_dates", 0)
    date_range = metadata.get("date_range", "unknown")
    
    return (
        f"{symbol} ({asset_type}) from {source}: "
        f"{records:,} records, {missing} missing days, "
        f"range: {date_range}"
    )

def merge_metadata_notes(existing_notes: str, new_notes: str) -> str:
    """
    Merge notes from existing and new metadata.
    
    Args:
        existing_notes: Existing notes string
        new_notes: New notes to append
        
    Returns:
        Merged notes string
    """
    if not existing_notes:
        return new_notes
    if not new_notes:
        return existing_notes
    
    # Avoid duplicating the same note
    if new_notes in existing_notes:
        return existing_notes
    
    return f"{existing_notes}; {new_notes}"