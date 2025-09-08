#!/usr/bin/env python3
"""
Stock Price Data Collector
Follows schema_contract.yaml specification for data collection

Usage:
    python collect_prices.py stocks --tickers GME AMC BB KOSS BBBY --start 2020-12-01 --end 2023-12-31
"""

import argparse
import json
import logging
import os
import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import warnings

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class StockPriceCollector:
    """Collects stock price data following schema contract specifications"""
    
    def __init__(self, log_level: str = "INFO", use_polygon_fallback: bool = True):
        self.setup_logging(log_level)
        self.output_dir = Path("data/raw/stocks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_polygon_fallback = use_polygon_fallback
        
        # Required columns per schema contract
        self.required_columns = ["date", "open", "high", "low", "close", "volume"]
        
        # Polygon.io configuration
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not self.polygon_api_key and use_polygon_fallback:
            self.logger.warning("POLYGON_API_KEY not found in environment. Polygon fallback disabled.")
            self.use_polygon_fallback = False
        elif self.polygon_api_key:
            self.logger.info("Polygon.io fallback enabled")
        
        self.polygon_base_url = "https://api.polygon.io/v2/aggs/ticker"
        
    def setup_logging(self, log_level: str):
        """Set up logging to file and console"""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = Path(f"logs/collect_prices_{timestamp}.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Stock Price Collector initialized - Logging to {log_file}")
    
    def is_yfinance_data_insufficient(self, data: pd.DataFrame, ticker: str, start_date: str, end_date: str, stock: yf.Ticker) -> bool:
        """
        Detect if yfinance data is insufficient (likely delisted)
        
        Args:
            data: Raw yfinance DataFrame
            ticker: Stock symbol
            start_date: Start date string
            end_date: End date string  
            stock: yfinance Ticker object
            
        Returns:
            True if data is insufficient, False otherwise
        """
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
        """
        Fetch stock data from Polygon.io as fallback
        
        Args:
            ticker: Stock symbol
            start_date: Start date in YYYY-MM-DD format  
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
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

    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch stock data with Polygon.io fallback for delisted tickers
        
        Args:
            ticker: Stock symbol (e.g., 'GME')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Tuple of (DataFrame with OHLCV data, metadata dict)
        """
        self.logger.info(f"Fetching {ticker} data from {start_date} to {end_date}")
        
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
                yf_data = self._normalize_data_format(yf_raw_data, ticker)
                
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
                self.logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
                raise
        
        # Use the successful data source
        if fallback_used and polygon_data is not None:
            df = polygon_data
        elif yf_data is not None:
            df = yf_data
        else:
            raise ValueError(f"No valid data obtained for {ticker}")
        
        # Create enhanced metadata
        metadata = self._create_enhanced_metadata(
            ticker, start_date, end_date, df, source, fallback_used, yf_rows, polygon_rows
        )
        
        self.logger.info(f"Successfully processed {len(df)} records for {ticker} (source: {source})")
        return df, metadata
    
    def _normalize_data_format(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Convert yfinance data to schema contract format
        
        Args:
            data: Raw yfinance DataFrame
            ticker: Stock symbol
            
        Returns:
            Normalized DataFrame following schema contract
        """
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
    
    def _create_metadata(self, ticker: str, start_date: str, end_date: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Create metadata JSON following schema contract"""
        
        # Count missing business days (rough estimate)
        total_business_days = pd.bdate_range(start_date, end_date)
        missing_dates = len(total_business_days) - len(df)
        
        metadata = {
            "symbol": ticker,
            "asset_type": "stock",
            "source": "yfinance",
            "date_range": f"{start_date} to {end_date}",
            "total_records": len(df),
            "missing_dates": missing_dates,
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": "delisted_handling=false"
        }
        
        return metadata
    
    def _create_enhanced_metadata(self, ticker: str, start_date: str, end_date: str, df: pd.DataFrame, 
                                source: str, fallback_used: bool, yf_rows: int, polygon_rows: int) -> Dict[str, Any]:
        """Create enhanced metadata JSON with fallback tracking"""
        
        # Count missing business days
        total_business_days = pd.bdate_range(start_date, end_date)
        missing_dates = len(total_business_days) - len(df)
        
        # Create notes with detailed tracking
        notes_parts = [
            "delisted_handling=true",
            f"fallback_used={str(fallback_used).lower()}",
            f"yf_rows={yf_rows}",
            f"polygon_rows={polygon_rows}"
        ]
        
        if fallback_used and polygon_rows == 0:
            notes_parts.append("no_data_from_yf_and_polygon")
        
        metadata = {
            "symbol": ticker,
            "asset_type": "stock", 
            "source": source,
            "date_range": f"{start_date} to {end_date}",
            "total_records": len(df),
            "missing_dates": missing_dates,
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": "; ".join(notes_parts)
        }
        
        return metadata
    
    def validate_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Validate data against schema contract requirements
        
        Args:
            df: DataFrame to validate
            ticker: Stock symbol for logging
            
        Returns:
            True if validation passes, False otherwise
        """
        self.logger.info(f"Validating data for {ticker}")
        
        try:
            # Check for required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for duplicate dates
            if df['date'].duplicated().any():
                raise ValueError("Duplicate timestamps found")
            
            # Check if dates are sorted ascending
            dates = pd.to_datetime(df['date'])
            if not dates.is_monotonic_increasing:
                raise ValueError("Dates are not sorted in ascending order")
            
            # Check Close is between Low and High
            invalid_close = (df['close'] < df['low']) | (df['close'] > df['high'])
            if invalid_close.any():
                invalid_count = invalid_close.sum()
                raise ValueError(f"Close price outside High/Low range in {invalid_count} records")
            
            # Check volume is non-negative
            if (df['volume'] < 0).any():
                raise ValueError("Negative volume values found")
            
            # Check for reasonable price values (no zeros or extreme values)
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    raise ValueError(f"Zero or negative prices found in {col}")
                if (df[col] > 1000000).any():  # Sanity check for extreme prices
                    self.logger.warning(f"Extremely high prices found in {col} for {ticker}")
            
            self.logger.info(f"Data validation passed for {ticker}")
            return True
            
        except ValueError as e:
            self.logger.error(f"Data validation failed for {ticker}: {str(e)}")
            return False
    
    def get_versioned_filename(self, base_path: Path) -> Path:
        """Generate versioned filename if file already exists"""
        if not base_path.exists():
            return base_path
        
        # Create versioned filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        name_parts = base_path.stem.split('_')
        versioned_name = f"{name_parts[0]}_stock_data_v{timestamp}.csv"
        versioned_path = base_path.parent / versioned_name
        
        self.logger.info(f"File exists, creating versioned file: {versioned_path}")
        return versioned_path
    
    def save_data(self, df: pd.DataFrame, metadata: Dict[str, Any], ticker: str):
        """
        Save DataFrame and metadata to files
        
        Args:
            df: Stock data DataFrame
            metadata: Metadata dictionary
            ticker: Stock symbol
        """
        # Generate file paths
        csv_path = self.output_dir / f"{ticker}_stock_data.csv"
        meta_path = self.output_dir / f"{ticker}_stock_data.meta.json"
        
        # Handle versioning if files exist
        csv_path = self.get_versioned_filename(csv_path)
        if csv_path.name != f"{ticker}_stock_data.csv":
            # Update metadata with version info
            version = csv_path.stem.split('_v')[-1]
            metadata["version"] = version
            meta_path = self.output_dir / f"{ticker}_stock_data_v{version}.meta.json"
        
        try:
            # Save CSV with exact column order
            df.to_csv(csv_path, index=False, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            
            # Save metadata JSON
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved {ticker} data to {csv_path}")
            self.logger.info(f"Saved {ticker} metadata to {meta_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data for {ticker}: {str(e)}")
            raise
    
    def collect_stocks(self, tickers: List[str], start_date: str, end_date: str):
        """
        Collect stock data for multiple tickers
        
        Args:
            tickers: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.logger.info(f"Starting stock collection for {len(tickers)} tickers")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Tickers: {', '.join(tickers)}")
        
        success_count = 0
        failed_tickers = []
        
        for ticker in tickers:
            try:
                # Fetch data
                df, metadata = self.fetch_stock_data(ticker, start_date, end_date)
                
                # Validate data
                if not self.validate_data(df, ticker):
                    failed_tickers.append(ticker)
                    continue
                
                # Save data
                self.save_data(df, metadata, ticker)
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to collect {ticker}: {str(e)}")
                failed_tickers.append(ticker)
                continue
        
        # Summary logging
        self.logger.info(f"Stock collection completed")
        self.logger.info(f"Successful: {success_count}/{len(tickers)}")
        if failed_tickers:
            self.logger.warning(f"Failed tickers: {', '.join(failed_tickers)}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Stock Price Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_prices.py stocks --tickers GME AMC BB --start 2020-12-01 --end 2023-12-31
  python collect_prices.py stocks --tickers KOSS BBBY --start 2021-01-01 --end 2023-06-30 --use-polygon-fallback true
  python collect_prices.py stocks --tickers BBBY --start 2020-12-01 --end 2023-12-31 --use-polygon-fallback false
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
        collector = StockPriceCollector(log_level=args.log_level, use_polygon_fallback=use_fallback)
        collector.collect_stocks(args.tickers, args.start, args.end)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()