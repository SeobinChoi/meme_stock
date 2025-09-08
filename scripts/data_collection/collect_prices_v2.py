#!/usr/bin/env python3
"""
Stock Price Data Collector (Version 2)
Refactored to use common utilities and unified metadata system.

Usage:
    python collect_prices_v2.py stocks --tickers GME AMC BB KOSS BBBY --start 2020-12-01 --end 2023-12-31
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import warnings
import pandas as pd
import yfinance as yf
import requests
import time

# Add common package to path
sys.path.append(str(Path(__file__).parent))

from common.logging_utils import get_logger, log_run_start, log_validation, log_write, log_collection_summary
from common.paths import ensure_dirs_exist, dir_raw_stocks, build_raw_path
from common.time_utils import now_utc_iso, to_utc_iso, parse_any_ts
from common.validation import validate_stock_data, ValidationError
from common.metadata import build_metadata, write_metadata
from common.io_utils import safe_write_versioned

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class StockPriceCollectorV2:
    """Enhanced stock price collector using unified common utilities."""
    
    def __init__(self, use_polygon_fallback: bool = True, log_level: str = "INFO"):
        self.logger = get_logger(__name__, log_level)
        self.use_polygon_fallback = use_polygon_fallback
        
        # Ensure directories exist
        ensure_dirs_exist()
        
        # Polygon.io configuration
        import os
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not self.polygon_api_key and use_polygon_fallback:
            self.logger.warning("POLYGON_API_KEY not found in environment. Polygon fallback disabled.")
            self.use_polygon_fallback = False
        elif self.polygon_api_key:
            self.logger.info("Polygon.io fallback enabled")
        
        self.polygon_base_url = "https://api.polygon.io/v2/aggs/ticker"
    
    def is_yfinance_data_insufficient(self, data: pd.DataFrame, ticker: str, start_date: str, end_date: str, stock: yf.Ticker) -> bool:
        """Detect if yfinance data is insufficient (likely delisted)."""
        if data.empty:
            self.logger.warning(f"{ticker}: yfinance returned empty data")
            return True
        
        if len(data) < 10:
            self.logger.warning(f"{ticker}: yfinance returned only {len(data)} rows")
            return True
        
        # Calculate expected business days
        expected_days = len(pd.bdate_range(start_date, end_date))
        missing_ratio = (expected_days - len(data)) / expected_days if expected_days > 0 else 1
        
        if missing_ratio > 0.3:
            self.logger.warning(f"{ticker}: Missing {missing_ratio:.1%} of expected business days")
            return True
        
        # Check for excessive NaN values in Close
        nan_ratio = data['Close'].isna().sum() / len(data) if len(data) > 0 else 1
        if nan_ratio > 0.1:
            self.logger.warning(f"{ticker}: {nan_ratio:.1%} of Close values are NaN")
            return True
        
        # Check ticker info for delisting indicators
        try:
            info = stock.info
            if info.get('quoteType') is None or info.get('regularMarketPrice') is None:
                self.logger.warning(f"{ticker}: Missing market data in ticker info (likely delisted)")
                return True
        except Exception as e:
            self.logger.warning(f"{ticker}: Could not fetch ticker info: {e}")
        
        return False
    
    def fetch_from_polygon(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch stock data from Polygon.io as fallback."""
        if not self.polygon_api_key:
            self.logger.error(f"{ticker}: Polygon API key not available")
            return None
        
        url = f"{self.polygon_base_url}/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'apikey': self.polygon_api_key,
            'adjusted': 'false'
        }
        
        max_retries = 5
        base_delay = 0.25
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"{ticker}: Fetching from Polygon.io (attempt {attempt + 1}/{max_retries})")
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:  # Rate limit
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"{ticker}: Rate limited, waiting {delay}s")
                    time.sleep(delay)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') != 'OK':
                    self.logger.error(f"{ticker}: Polygon API error: {data.get('status', 'unknown')}")
                    return None
                
                results = data.get('results', [])
                if not results:
                    self.logger.warning(f"{ticker}: No results from Polygon.io")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                
                # Map Polygon fields to our schema
                df = df.rename(columns={
                    't': 'timestamp',
                    'o': 'open', 
                    'h': 'high',
                    'l': 'low', 
                    'c': 'close',
                    'v': 'volume'
                })
                
                # Convert Unix milliseconds to UTC datetime
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df['date'] = df['date'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
                
                # Select and order columns
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                
                # Ensure correct data types
                df['open'] = df['open'].astype('float64')
                df['high'] = df['high'].astype('float64')
                df['low'] = df['low'].astype('float64')
                df['close'] = df['close'].astype('float64')
                df['volume'] = df['volume'].astype('int64')
                
                # Sort by date
                df = df.sort_values('date').reset_index(drop=True)
                
                self.logger.info(f"{ticker}: Successfully fetched {len(df)} records from Polygon.io")
                return df
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"{ticker}: Polygon API request failed: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(base_delay * (2 ** attempt))
            
            except Exception as e:
                self.logger.error(f"{ticker}: Unexpected error with Polygon.io: {e}")
                return None
        
        return None
    
    def normalize_yfinance_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Convert yfinance data to schema contract format."""
        # Reset index to make Date a column
        df = data.reset_index()
        
        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure date column is properly named
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'date'})
        
        # Convert date to UTC ISO8601 format
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert('UTC')
        df['date'] = df['date'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        
        # Select only required columns in correct order
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        # Ensure correct data types
        df['open'] = df['open'].astype('float64')
        df['high'] = df['high'].astype('float64') 
        df['low'] = df['low'].astype('float64')
        df['close'] = df['close'].astype('float64')
        df['volume'] = df['volume'].astype('int64')
        
        # Sort by date ascending
        df = df.sort_values('date').reset_index(drop=True)
        
        self.logger.debug(f"Normalized data format for {ticker}: {df.shape}")
        return df
    
    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str, Dict[str, int]]:
        """
        Fetch stock data with Polygon.io fallback for delisted tickers.
        
        Returns:
            Tuple of (DataFrame, source, stats_dict)
        """
        yf_data = None
        polygon_data = None
        fallback_used = False
        yf_rows = 0
        polygon_rows = 0
        source = "yfinance"
        
        # Try Yahoo Finance first
        try:
            stock = yf.Ticker(ticker)
            yf_raw_data = stock.history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
                back_adjust=False
            )
            
            yf_rows = len(yf_raw_data)
            self.logger.info(f"{ticker}: yfinance returned {yf_rows} rows")
            
            # Check if yfinance data is insufficient
            if self.is_yfinance_data_insufficient(yf_raw_data, ticker, start_date, end_date, stock):
                if self.use_polygon_fallback:
                    self.logger.info(f"{ticker}: yfinance data insufficient, trying Polygon.io fallback")
                    polygon_data = self.fetch_from_polygon(ticker, start_date, end_date)
                    if polygon_data is not None:
                        polygon_rows = len(polygon_data)
                        fallback_used = True
                        source = "polygon"
                        self.logger.info(f"{ticker}: Using Polygon.io data ({polygon_rows} rows)")
                    else:
                        self.logger.error(f"{ticker}: Polygon.io fallback failed")
                        raise ValueError(f"No sufficient data available for {ticker}")
                else:
                    raise ValueError(f"yfinance data insufficient for {ticker} and fallback disabled")
            else:
                # yfinance data is sufficient
                yf_data = self.normalize_yfinance_data(yf_raw_data, ticker)
                
        except Exception as e:
            if self.use_polygon_fallback and not fallback_used:
                self.logger.warning(f"{ticker}: yfinance failed ({e}), trying Polygon.io fallback")
                polygon_data = self.fetch_from_polygon(ticker, start_date, end_date)
                if polygon_data is not None:
                    polygon_rows = len(polygon_data)
                    fallback_used = True
                    source = "polygon"
                    self.logger.info(f"{ticker}: Using Polygon.io data ({polygon_rows} rows)")
                else:
                    self.logger.error(f"{ticker}: Both yfinance and Polygon.io failed")
                    raise ValueError(f"No data available for {ticker}")
            else:
                raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
        
        # Use the successful data source
        if fallback_used and polygon_data is not None:
            df = polygon_data
        elif yf_data is not None:
            df = yf_data
        else:
            raise ValueError(f"No valid data obtained for {ticker}")
        
        stats = {
            "yf_rows": yf_rows,
            "polygon_rows": polygon_rows,
            "fallback_used": fallback_used
        }
        
        return df, source, stats
    
    def collect_single_ticker(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Collect data for a single ticker using unified pipeline.
        
        Returns:
            Result dictionary with status and details
        """
        result = {
            "ticker": ticker,
            "status": "unknown",
            "records": 0,
            "source": "",
            "path": "",
            "error": None
        }
        
        try:
            # Fetch data
            df, source, stats = self.fetch_stock_data(ticker, start_date, end_date)
            
            # Validate data using common validation
            validation_report = validate_stock_data(df, ticker)
            log_validation(self.logger, ticker, **validation_report["validations"])
            
            # Build metadata using common utilities
            notes_parts = [
                "delisted_handling=true",
                f"fallback_used={str(stats['fallback_used']).lower()}",
                f"yf_rows={stats['yf_rows']}",
                f"polygon_rows={stats['polygon_rows']}"
            ]
            notes = "; ".join(notes_parts)
            
            metadata = build_metadata(
                symbol=ticker,
                asset_type="stock",
                source=source,
                df=df,
                date_range=f"{start_date} to {end_date}",
                notes=notes
            )
            
            # Write data using atomic versioned write
            base_path = build_raw_path("stock", f"{ticker}_stock_data")
            csv_path, updated_metadata = safe_write_versioned(df, base_path, metadata)
            
            # Write metadata and update index
            meta_path = write_metadata(updated_metadata, csv_path)
            
            log_write(
                self.logger,
                csv_path,
                len(df),
                updated_metadata.get("checksum_sha256", "")[:8],
                updated_metadata.get("version", "")
            )
            
            result.update({
                "status": "success",
                "records": len(df),
                "source": source,
                "path": str(csv_path),
                "meta_path": str(meta_path)
            })
            
        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e)
            })
            self.logger.error(f"Failed to collect {ticker}: {str(e)}")
        
        return result
    
    def collect_stocks(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Collect stock data for multiple tickers.
        
        Returns:
            Summary dictionary with results
        """
        log_run_start(
            self.logger, 
            "stocks",
            tickers=",".join(tickers),
            start=start_date,
            end=end_date,
            fallback=self.use_polygon_fallback
        )
        
        results = []
        for ticker in tickers:
            result = self.collect_single_ticker(ticker, start_date, end_date)
            results.append(result)
        
        # Summary
        success_results = [r for r in results if r["status"] == "success"]
        failed_results = [r for r in results if r["status"] == "error"]
        
        log_collection_summary(
            self.logger,
            "stocks",
            len(success_results),
            len(tickers),
            [r["ticker"] for r in failed_results]
        )
        
        return {
            "total": len(tickers),
            "success": len(success_results),
            "failed": len(failed_results),
            "results": results
        }

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Price Data Collector V2 (Unified Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_prices_v2.py stocks --tickers GME AMC BB --start 2020-12-01 --end 2023-12-31
  python collect_prices_v2.py stocks --tickers BBBY --start 2020-12-01 --end 2023-12-31 --use-polygon-fallback true
  python collect_prices_v2.py stocks --tickers KOSS --start 2021-01-01 --end 2023-06-30 --use-polygon-fallback false
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Collection mode')
    
    # Stocks subcommand
    stocks_parser = subparsers.add_parser('stocks', help='Collect stock price data')
    stocks_parser.add_argument(
        '--tickers', 
        nargs='+', 
        required=True,
        help='Stock tickers to collect (e.g., GME AMC BB)'
    )
    stocks_parser.add_argument(
        '--start',
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    stocks_parser.add_argument(
        '--end',
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    stocks_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    stocks_parser.add_argument(
        '--use-polygon-fallback',
        type=str,
        choices=['true', 'false'],
        default='true',
        help='Enable Polygon.io fallback for delisted tickers (default: true)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'stocks':
        use_fallback = args.use_polygon_fallback.lower() == 'true'
        collector = StockPriceCollectorV2(
            use_polygon_fallback=use_fallback,
            log_level=args.log_level
        )
        
        summary = collector.collect_stocks(args.tickers, args.start, args.end)
        
        # Exit code based on results
        if summary["failed"] > 0:
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()