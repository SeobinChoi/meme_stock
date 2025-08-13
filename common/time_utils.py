"""
Time handling and UTC conversion utilities.

Provides consistent datetime handling across the pipeline,
with all times normalized to UTC ISO8601 format.
"""

import re
from datetime import datetime, timezone, timedelta
from typing import List, Union, Any
import pandas as pd

def now_utc_iso() -> str:
    """
    Get current timestamp in UTC ISO8601 format.
    
    Returns:
        Current time as "YYYY-MM-DDTHH:MM:SS+00:00"
    """
    return datetime.now(timezone.utc).isoformat()

def to_utc_iso(dt_like: Any) -> str:
    """
    Convert datetime-like object to UTC ISO8601 string.
    
    Args:
        dt_like: datetime, pandas Timestamp, or parseable string
        
    Returns:
        UTC ISO8601 formatted string
        
    Examples:
        >>> to_utc_iso(datetime(2023, 1, 1, 12, 0))
        "2023-01-01T12:00:00+00:00"
    """
    if isinstance(dt_like, str):
        dt = parse_any_ts(dt_like)
    elif isinstance(dt_like, pd.Timestamp):
        dt = dt_like.to_pydatetime()
    elif isinstance(dt_like, datetime):
        dt = dt_like
    else:
        # Try pandas conversion
        dt = pd.to_datetime(dt_like).to_pydatetime()
    
    # Ensure timezone aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    return dt.isoformat()

def parse_any_ts(ts: str) -> datetime:
    """
    Robust parser for various timestamp formats.
    
    Args:
        ts: Timestamp string in various formats
        
    Returns:
        datetime object in UTC
        
    Handles formats:
        - "YYYY-MM-DD"
        - "YYYY-MM-DD HH:MM:SS" 
        - "YYYY-MM-DD HH:MM:SS±HH:MM"
        - "YYYY-MM-DDTHH:MM:SS+00:00"
        - Unix timestamps
    """
    # Strip whitespace
    ts = ts.strip()
    
    # Try ISO format first
    iso_patterns = [
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$',  # Full ISO
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$',                # Z suffix  
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$',                 # ISO without TZ
    ]
    
    for pattern in iso_patterns:
        if re.match(pattern, ts):
            if ts.endswith('Z'):
                ts = ts.replace('Z', '+00:00')
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
    
    # Try pandas parsing (handles many formats)
    try:
        dt = pd.to_datetime(ts, utc=True).to_pydatetime()
        return dt
    except:
        pass
    
    # Try simple date format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', ts):
        dt = datetime.strptime(ts, '%Y-%m-%d')
        return dt.replace(tzinfo=timezone.utc)
    
    # Try datetime formats without timezone
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    # Try unix timestamp
    try:
        # Check if it looks like unix timestamp
        if ts.replace('.', '').isdigit():
            timestamp = float(ts)
            # Handle both seconds and milliseconds
            if timestamp > 1e10:  # Looks like milliseconds
                timestamp = timestamp / 1000
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt
    except:
        pass
    
    raise ValueError(f"Could not parse timestamp: {ts}")

def floor_utc_day(dt: datetime) -> datetime:
    """
    Floor datetime to start of UTC day (00:00:00).
    
    Args:
        dt: datetime object
        
    Returns:
        datetime floored to start of day in UTC
    """
    # Convert to UTC first
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    # Floor to start of day
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def count_missing_business_days(dates: List[datetime], start: datetime, end: datetime) -> int:
    """
    Count missing business days in date range.
    
    Args:
        dates: List of datetime objects (actual dates present)
        start: Range start datetime
        end: Range end datetime
        
    Returns:
        Number of missing business days
    """
    # Generate expected business days
    expected_dates = pd.bdate_range(start=start.date(), end=end.date(), freq='D')
    
    # Convert actual dates to date objects and create set for fast lookup
    actual_dates = set(dt.date() for dt in dates)
    
    # Count missing
    missing = 0
    for expected_date in expected_dates:
        if expected_date.date() not in actual_dates:
            missing += 1
    
    return missing

def count_missing_calendar_days(dates: List[datetime], start: datetime, end: datetime) -> int:
    """
    Count missing calendar days in date range.
    
    Args:
        dates: List of datetime objects (actual dates present)  
        start: Range start datetime
        end: Range end datetime
        
    Returns:
        Number of missing calendar days
    """
    # Generate expected calendar days
    expected_dates = pd.date_range(start=start.date(), end=end.date(), freq='D')
    
    # Convert actual dates to date objects
    actual_dates = set(dt.date() for dt in dates)
    
    # Count missing
    missing = 0
    for expected_date in expected_dates:
        if expected_date.date() not in actual_dates:
            missing += 1
    
    return missing

def generate_version_timestamp() -> str:
    """
    Generate version timestamp string.
    
    Returns:
        Version timestamp in format YYYYMMDDHHMMSS
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

def parse_version_from_filename(filename: str) -> str:
    """
    Extract version timestamp from versioned filename.
    
    Args:
        filename: Filename containing version
        
    Returns:
        Version timestamp or empty string if not found
        
    Examples:
        >>> parse_version_from_filename("GME_stock_data_v20250812153000.csv")
        "20250812153000"
    """
    match = re.search(r'_v(\d{14})', filename)
    return match.group(1) if match else ""

def is_weekend(dt: datetime) -> bool:
    """
    Check if datetime falls on weekend.
    
    Args:
        dt: datetime object
        
    Returns:
        True if Saturday or Sunday
    """
    return dt.weekday() >= 5  # 5=Saturday, 6=Sunday