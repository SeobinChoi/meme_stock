"""
Data validation and schema enforcement utilities.

Centralized validation gates that ensure data quality and 
consistency across all collected datasets.
"""

from datetime import datetime
from typing import Sequence, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .time_utils import count_missing_business_days, count_missing_calendar_days

class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

def assert_no_duplicates(dates: Sequence[datetime], context: str = "") -> None:
    """
    Assert that there are no duplicate dates.
    
    Args:
        dates: Sequence of datetime objects
        context: Additional context for error message
        
    Raises:
        ValidationError: If duplicates are found
    """
    date_list = list(dates)
    unique_dates = set(date_list)
    
    if len(date_list) != len(unique_dates):
        duplicates = []
        seen = set()
        for date in date_list:
            if date in seen:
                duplicates.append(date)
            seen.add(date)
        
        ctx = f" in {context}" if context else ""
        raise ValidationError(f"Duplicate dates found{ctx}: {duplicates}")

def assert_monotonic_ascending(dates: Sequence[datetime], context: str = "") -> None:
    """
    Assert that dates are in strictly ascending order.
    
    Args:
        dates: Sequence of datetime objects
        context: Additional context for error message
        
    Raises:
        ValidationError: If dates are not monotonic ascending
    """
    date_list = list(dates)
    
    for i in range(1, len(date_list)):
        if date_list[i] <= date_list[i-1]:
            ctx = f" in {context}" if context else ""
            raise ValidationError(
                f"Dates not in ascending order{ctx}: "
                f"{date_list[i-1]} >= {date_list[i]} at index {i}"
            )

def assert_bounds_ohlc(df: pd.DataFrame, low: str = "low", high: str = "high", 
                      open_: str = "open", close: str = "close", context: str = "") -> None:
    """
    Assert OHLC price bounds are valid.
    
    Args:
        df: DataFrame with OHLC data
        low: Column name for low prices  
        high: Column name for high prices
        open_: Column name for open prices
        close: Column name for close prices
        context: Additional context for error message
        
    Raises:
        ValidationError: If price bounds are invalid
    """
    # Check that all required columns exist
    required_cols = [low, high, open_, close]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        ctx = f" in {context}" if context else ""
        raise ValidationError(f"Missing OHLC columns{ctx}: {missing_cols}")
    
    # Check Low <= High
    low_high_violations = df[df[low] > df[high]]
    if not low_high_violations.empty:
        ctx = f" in {context}" if context else ""
        raise ValidationError(
            f"Low > High violations{ctx}: {len(low_high_violations)} rows"
        )
    
    # Check Low <= Open <= High
    open_bound_violations = df[(df[open_] < df[low]) | (df[open_] > df[high])]
    if not open_bound_violations.empty:
        ctx = f" in {context}" if context else ""
        raise ValidationError(
            f"Open price outside Low/High bounds{ctx}: {len(open_bound_violations)} rows"
        )
    
    # Check Low <= Close <= High  
    close_bound_violations = df[(df[close] < df[low]) | (df[close] > df[high])]
    if not close_bound_violations.empty:
        ctx = f" in {context}" if context else ""
        raise ValidationError(
            f"Close price outside Low/High bounds{ctx}: {len(close_bound_violations)} rows"
        )

def compute_missing_days_stock(dates: List[datetime], start: datetime, end: datetime) -> int:
    """
    Compute missing business days for stock data.
    
    Args:
        dates: List of actual dates present
        start: Range start
        end: Range end
        
    Returns:
        Number of missing business days
    """
    return count_missing_business_days(dates, start, end)

def compute_missing_days_crypto(dates: List[datetime], start: datetime, end: datetime) -> int:
    """
    Compute missing calendar days for crypto data.
    
    Args:
        dates: List of actual dates present  
        start: Range start
        end: Range end
        
    Returns:
        Number of missing calendar days
    """
    return count_missing_calendar_days(dates, start, end)

def assert_non_negative(series: pd.Series, name: str, context: str = "") -> None:
    """
    Assert that all values in series are non-negative.
    
    Args:
        series: Pandas Series to check
        name: Name of the series for error message
        context: Additional context for error message
        
    Raises:
        ValidationError: If negative values are found
    """
    negative_values = series[series < 0]
    if not negative_values.empty:
        ctx = f" in {context}" if context else ""
        raise ValidationError(
            f"Negative values in {name}{ctx}: {len(negative_values)} occurrences"
        )

def summary_report(df: pd.DataFrame, date_col: str = "date") -> Dict[str, Any]:
    """
    Generate summary report for dataset.
    
    Args:
        df: DataFrame to analyze
        date_col: Name of the date column
        
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {
            "total_records": 0,
            "date_range": "empty",
            "missing_values": {},
            "data_types": {},
            "gaps": []
        }
    
    # Convert date column to datetime if needed
    if date_col in df.columns:
        if df[date_col].dtype == 'object':
            dates = pd.to_datetime(df[date_col])
        else:
            dates = df[date_col]
        
        min_date = dates.min()
        max_date = dates.max()
        date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        
        # Find gaps (dates with >1 day difference)
        gaps = []
        sorted_dates = dates.sort_values()
        for i in range(1, len(sorted_dates)):
            prev_date = sorted_dates.iloc[i-1]
            curr_date = sorted_dates.iloc[i]
            diff_days = (curr_date - prev_date).days
            if diff_days > 1:
                gaps.append({
                    "start": prev_date.strftime('%Y-%m-%d'),
                    "end": curr_date.strftime('%Y-%m-%d'),
                    "days": diff_days - 1
                })
    else:
        date_range = "no_date_column"
        gaps = []
    
    # Missing values summary
    missing_values = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            missing_values[col] = {
                "count": int(null_count),
                "percentage": float(null_count / len(df) * 100)
            }
    
    # Data types
    data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    return {
        "total_records": len(df),
        "date_range": date_range,
        "missing_values": missing_values,
        "data_types": data_types,
        "gaps": gaps
    }

def validate_stock_data(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Comprehensive validation for stock data.
    
    Args:
        df: Stock data DataFrame
        symbol: Stock symbol for context
        
    Returns:
        Validation report dictionary
        
    Raises:
        ValidationError: If critical validation fails
    """
    context = f"stock data for {symbol}"
    report = {"symbol": symbol, "asset_type": "stock", "validations": {}}
    
    try:
        # Required columns check
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns in {context}: {missing_cols}")
        
        # Convert dates for validation
        dates = pd.to_datetime(df["date"]).tolist()
        
        # Run validations
        assert_no_duplicates(dates, context)
        report["validations"]["no_duplicates"] = True
        
        assert_monotonic_ascending(dates, context)  
        report["validations"]["monotonic_dates"] = True
        
        assert_bounds_ohlc(df, context=context)
        report["validations"]["ohlc_bounds"] = True
        
        assert_non_negative(df["volume"], "volume", context)
        report["validations"]["non_negative_volume"] = True
        
        # Summary stats
        report["summary"] = summary_report(df)
        report["status"] = "PASS"
        
    except ValidationError as e:
        report["status"] = "FAIL"
        report["error"] = str(e)
        raise
    
    return report

def validate_crypto_data(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Comprehensive validation for crypto data.
    
    Args:
        df: Crypto data DataFrame
        symbol: Crypto symbol for context
        
    Returns:
        Validation report dictionary
        
    Raises:
        ValidationError: If critical validation fails
    """
    context = f"crypto data for {symbol}"
    report = {"symbol": symbol, "asset_type": "crypto", "validations": {}}
    
    try:
        # Required columns check
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns in {context}: {missing_cols}")
        
        # Convert dates for validation
        dates = pd.to_datetime(df["date"]).tolist()
        
        # Run validations (same as stock)
        assert_no_duplicates(dates, context)
        report["validations"]["no_duplicates"] = True
        
        assert_monotonic_ascending(dates, context)
        report["validations"]["monotonic_dates"] = True
        
        assert_bounds_ohlc(df, context=context)
        report["validations"]["ohlc_bounds"] = True
        
        assert_non_negative(df["volume"], "volume", context)
        report["validations"]["non_negative_volume"] = True
        
        # Summary stats
        report["summary"] = summary_report(df)
        report["status"] = "PASS"
        
    except ValidationError as e:
        report["status"] = "FAIL"
        report["error"] = str(e)
        raise
    
    return report

def validate_reddit_data(df: pd.DataFrame, symbol: str = "WSB") -> Dict[str, Any]:
    """
    Comprehensive validation for Reddit data.
    
    Args:
        df: Reddit data DataFrame
        symbol: Identifier for context
        
    Returns:
        Validation report dictionary
        
    Raises:
        ValidationError: If critical validation fails
    """
    context = f"Reddit data for {symbol}"
    report = {"symbol": symbol, "asset_type": "reddit", "validations": {}}
    
    try:
        # Required columns check (flexible based on what's available)
        required_cols = ["date"]
        optional_cols = ["score", "comments", "posts", "total_engagement", "is_weekend",
                        "num_comments", "title_length", "word_count"]
        
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise ValidationError(f"Missing required columns in {context}: {missing_required}")
        
        # Convert dates for validation
        dates = pd.to_datetime(df["date"]).tolist()
        
        # Run validations
        assert_no_duplicates(dates, context)
        report["validations"]["no_duplicates"] = True
        
        assert_monotonic_ascending(dates, context)
        report["validations"]["monotonic_dates"] = True
        
        # Validate non-negative counts if columns exist
        count_cols = [col for col in optional_cols if col in df.columns and col != "is_weekend"]
        for col in count_cols:
            assert_non_negative(df[col], col, context)
        
        report["validations"]["non_negative_counts"] = True
        
        # Summary stats
        report["summary"] = summary_report(df)
        report["status"] = "PASS"
        
    except ValidationError as e:
        report["status"] = "FAIL"  
        report["error"] = str(e)
        raise
    
    return report