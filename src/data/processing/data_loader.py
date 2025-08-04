"""
Data Loading Pipeline for Meme Stock Prediction Project
Day 1: Environment Setup & Data Infrastructure

This module provides comprehensive data loading functionality:
- Load Reddit WSB data
- Load stock price data from multiple sources
- Handle missing data and data quality issues
- Create unified dataset for modeling
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
import yfinance as yf
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Comprehensive data loader for meme stock prediction datasets
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader with data directory path
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.reddit_data = None
        self.stock_data = {}
        self.unified_data = None
        
    def load_reddit_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and preprocess Reddit WSB data
        
        Args:
            file_path: Optional custom path to Reddit data file
            
        Returns:
            Preprocessed Reddit DataFrame
        """
        logger.info("Loading Reddit WSB data...")
        
        if file_path is None:
            file_path = self.data_dir / "raw" / "reddit_wsb.csv"
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Reddit data file not found: {file_path}")
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            logger.info(f"Loaded Reddit data: {len(df):,} rows, {len(df.columns)} columns")
            
            # Basic preprocessing
            df = self._preprocess_reddit_data(df)
            
            self.reddit_data = df
            logger.info("Reddit data loading completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Reddit data: {str(e)}")
            raise
    
    def _preprocess_reddit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Reddit data for analysis
        
        Args:
            df: Raw Reddit DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing Reddit data...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Convert timestamp to datetime
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
            df['date'] = df['created_utc'].dt.date
            df['hour'] = df['created_utc'].dt.hour
            df['day_of_week'] = df['created_utc'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Handle missing values
        if 'title' in df.columns:
            df['title'] = df['title'].fillna('')
        
        if 'score' in df.columns:
            df['score'] = df['score'].fillna(0)
        
        if 'num_comments' in df.columns:
            df['num_comments'] = df['num_comments'].fillna(0)
        
        # Create engagement metrics
        df['total_engagement'] = df['score'] + df['num_comments']
        df['engagement_rate'] = df['total_engagement'] / (df['total_engagement'].max() + 1)
        
        # Basic text preprocessing
        if 'title' in df.columns:
            df['title_length'] = df['title'].str.len()
            df['word_count'] = df['title'].str.split().str.len()
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        logger.info(f"Reddit preprocessing completed. Final shape: {df.shape}")
        return df
    
    def load_stock_data(self, symbols: List[str] = None, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load stock price data for specified symbols
        
        Args:
            symbols: List of stock symbols to load (default: ['GME', 'AMC', 'BB'])
            start_date: Start date for data (format: 'YYYY-MM-DD')
            end_date: End date for data (format: 'YYYY-MM-DD')
            
        Returns:
            Dictionary of stock DataFrames
        """
        if symbols is None:
            symbols = ['GME', 'AMC', 'BB']
        
        logger.info(f"Loading stock data for symbols: {symbols}")
        
        stock_data = {}
        
        for symbol in tqdm(symbols, desc="Loading stock data"):
            try:
                # Try to load from local files first
                local_data = self._load_local_stock_data(symbol)
                
                if local_data is not None:
                    stock_data[symbol] = local_data
                    logger.info(f"Loaded {symbol} from local file: {len(local_data):,} rows")
                else:
                    # Fallback to Yahoo Finance API
                    logger.info(f"Loading {symbol} from Yahoo Finance API...")
                    api_data = self._load_stock_from_api(symbol, start_date, end_date)
                    if api_data is not None:
                        stock_data[symbol] = api_data
                        logger.info(f"Loaded {symbol} from API: {len(api_data):,} rows")
                    else:
                        logger.warning(f"Failed to load data for {symbol}")
                        
            except Exception as e:
                logger.error(f"Error loading {symbol}: {str(e)}")
        
        self.stock_data = stock_data
        logger.info(f"Stock data loading completed. Loaded {len(stock_data)} symbols")
        return stock_data
    
    def _load_local_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from local archive files
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stock DataFrame or None if not found
        """
        archive_dirs = [
            self.data_dir / "raw" / "archive-2",
            self.data_dir / "raw" / "archive-3"
        ]
        
        for archive_dir in archive_dirs:
            if archive_dir.exists():
                # Look for files containing the symbol
                for file in archive_dir.glob("*.csv"):
                    if symbol.upper() in file.name.upper():
                        try:
                            df = pd.read_csv(file)
                            
                            # Standardize column names
                            df = self._standardize_stock_columns(df)
                            
                            # Basic preprocessing
                            df = self._preprocess_stock_data(df)
                            
                            return df
                            
                        except Exception as e:
                            logger.warning(f"Error reading {file}: {str(e)}")
                            continue
        
        return None
    
    def _load_stock_from_api(self, symbol: str, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load stock data from Yahoo Finance API
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Stock DataFrame or None if failed
        """
        try:
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                # Default to last 2 years if no dates specified
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
                df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Standardize column names
            df = self._standardize_stock_columns(df)
            
            # Basic preprocessing
            df = self._preprocess_stock_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {symbol} from API: {str(e)}")
            return None
    
    def _standardize_stock_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize stock data column names
        
        Args:
            df: Raw stock DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Common column name mappings
        column_mappings = {
            'Date': 'date',
            'Time': 'date',
            'Datetime': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        # Rename columns
        df = df.rename(columns=column_mappings)
        
        # Ensure we have required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
        
        return df
    
    def _preprocess_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess stock data for analysis
        
        Args:
            df: Raw stock DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['date_only'] = df['date'].dt.date
        
        # Handle missing values
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                # Forward fill for missing values
                df[col] = df[col].fillna(method='ffill')
                # Backward fill for any remaining missing values at the beginning
                df[col] = df[col].fillna(method='bfill')
        
        # Calculate additional features
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Returns
            df['daily_return'] = df['close'].pct_change()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price ranges
            df['price_range'] = df['high'] - df['low']
            df['price_range_pct'] = df['price_range'] / df['close']
            
            # Volume features
            if 'volume' in df.columns:
                df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
                df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # Remove rows with missing data
        initial_rows = len(df)
        df = df.dropna()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} rows with missing data")
        
        return df
    
    def create_unified_dataset(self, 
                             reddit_data: Optional[pd.DataFrame] = None,
                             stock_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Create unified dataset combining Reddit and stock data
        
        Args:
            reddit_data: Reddit DataFrame (uses self.reddit_data if None)
            stock_data: Stock data dictionary (uses self.stock_data if None)
            
        Returns:
            Unified DataFrame
        """
        logger.info("Creating unified dataset...")
        
        if reddit_data is None:
            reddit_data = self.reddit_data
        if stock_data is None:
            stock_data = self.stock_data
        
        if reddit_data is None:
            raise ValueError("Reddit data not loaded. Call load_reddit_data() first.")
        
        if not stock_data:
            raise ValueError("Stock data not loaded. Call load_stock_data() first.")
        
        # Prepare Reddit data for merging
        reddit_daily = self._aggregate_reddit_daily(reddit_data)
        
        # Prepare stock data for merging
        stock_daily = self._aggregate_stock_daily(stock_data)
        
        # Merge datasets
        unified_data = self._merge_datasets(reddit_daily, stock_daily)
        
        self.unified_data = unified_data
        logger.info(f"Unified dataset created: {unified_data.shape}")
        return unified_data
    
    def _aggregate_reddit_daily(self, reddit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate Reddit data to daily level
        
        Args:
            reddit_data: Reddit DataFrame
            
        Returns:
            Daily aggregated Reddit DataFrame
        """
        logger.info("Aggregating Reddit data to daily level...")
        
        # Ensure we have date column
        if 'date' not in reddit_data.columns and 'created_utc' in reddit_data.columns:
            reddit_data['date'] = reddit_data['created_utc'].dt.date
        
        # Daily aggregation
        daily_agg = reddit_data.groupby('date').agg({
            'score': ['count', 'sum', 'mean', 'std'],
            'num_comments': ['sum', 'mean', 'std'],
            'total_engagement': ['sum', 'mean', 'std'],
            'title_length': ['mean', 'std'],
            'word_count': ['mean', 'std'],
            'is_weekend': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = ['date'] + [f'reddit_{col[0]}_{col[1]}' for col in daily_agg.columns[1:]]
        
        # Add additional features
        daily_agg['reddit_post_count'] = daily_agg['reddit_score_count']
        daily_agg['reddit_avg_score'] = daily_agg['reddit_score_mean']
        daily_agg['reddit_total_comments'] = daily_agg['reddit_num_comments_sum']
        daily_agg['reddit_avg_comments'] = daily_agg['reddit_num_comments_mean']
        daily_agg['reddit_total_engagement'] = daily_agg['reddit_total_engagement_sum']
        daily_agg['reddit_avg_engagement'] = daily_agg['reddit_total_engagement_mean']
        
        logger.info(f"Reddit daily aggregation completed: {daily_agg.shape}")
        return daily_agg
    
    def _aggregate_stock_daily(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate stock data to daily level and combine all symbols
        
        Args:
            stock_data: Dictionary of stock DataFrames
            
        Returns:
            Combined daily stock DataFrame
        """
        logger.info("Aggregating stock data to daily level...")
        
        combined_stocks = []
        
        for symbol, df in stock_data.items():
            if df is None or df.empty:
                continue
            
            # Ensure we have date column
            if 'date' in df.columns:
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                
                # Select relevant columns
                columns_to_keep = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                                 'daily_return', 'log_return', 'price_range', 'price_range_pct']
                available_columns = [col for col in columns_to_keep if col in df_copy.columns]
                
                df_subset = df_copy[available_columns].copy()
                
                # Add symbol prefix to columns (except date and symbol)
                for col in df_subset.columns:
                    if col not in ['date', 'symbol']:
                        df_subset[f'{symbol.lower()}_{col}'] = df_subset[col]
                        df_subset = df_subset.drop(columns=[col])
                
                combined_stocks.append(df_subset)
        
        if not combined_stocks:
            raise ValueError("No valid stock data found")
        
        # Combine all stocks
        combined_df = pd.concat(combined_stocks, ignore_index=True)
        
        # Pivot to get one row per date with all symbols
        pivot_columns = [col for col in combined_df.columns if col not in ['date', 'symbol']]
        
        if pivot_columns:
            combined_df = combined_df.pivot(index='date', columns='symbol', values=pivot_columns)
            combined_df = combined_df.reset_index()
            
            # Flatten column names
            combined_df.columns = ['date'] + [f'{col[1]}_{col[0]}' for col in combined_df.columns[1:]]
        
        logger.info(f"Stock daily aggregation completed: {combined_df.shape}")
        return combined_df
    
    def _merge_datasets(self, reddit_daily: pd.DataFrame, 
                       stock_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Reddit and stock daily data
        
        Args:
            reddit_daily: Daily Reddit DataFrame
            stock_daily: Daily stock DataFrame
            
        Returns:
            Merged unified DataFrame
        """
        logger.info("Merging Reddit and stock datasets...")
        
        # Ensure date columns are the same type
        reddit_daily['date'] = pd.to_datetime(reddit_daily['date'])
        stock_daily['date'] = pd.to_datetime(stock_daily['date'])
        
        # Merge on date
        merged_df = pd.merge(reddit_daily, stock_daily, on='date', how='outer')
        
        # Sort by date
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        # Handle missing values
        # Forward fill for stock data
        stock_columns = [col for col in merged_df.columns if any(symbol in col.lower() for symbol in ['gme', 'amc', 'bb'])]
        merged_df[stock_columns] = merged_df[stock_columns].fillna(method='ffill')
        
        # Fill Reddit data with 0 for missing days
        reddit_columns = [col for col in merged_df.columns if col.startswith('reddit_')]
        merged_df[reddit_columns] = merged_df[reddit_columns].fillna(0)
        
        # Add date features
        merged_df['year'] = merged_df['date'].dt.year
        merged_df['month'] = merged_df['date'].dt.month
        merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
        merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Dataset merging completed: {merged_df.shape}")
        return merged_df
    
    def save_unified_dataset(self, output_path: str = "data/processed/unified_dataset.csv"):
        """
        Save unified dataset to file
        
        Args:
            output_path: Path to save the unified dataset
        """
        if self.unified_data is None:
            raise ValueError("No unified dataset available. Call create_unified_dataset() first.")
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        self.unified_data.to_csv(output_file, index=False)
        logger.info(f"Unified dataset saved to {output_file}")
        
        # Also save a summary
        summary_file = output_file.parent / "unified_dataset_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== UNIFIED DATASET SUMMARY ===\n\n")
            f.write(f"Dataset Shape: {self.unified_data.shape}\n")
            f.write(f"Date Range: {self.unified_data['date'].min()} to {self.unified_data['date'].max()}\n")
            f.write(f"Total Days: {len(self.unified_data)}\n\n")
            
            f.write("COLUMN SUMMARY:\n")
            reddit_cols = [col for col in self.unified_data.columns if col.startswith('reddit_')]
            stock_cols = [col for col in self.unified_data.columns if any(symbol in col.lower() for symbol in ['gme', 'amc', 'bb'])]
            other_cols = [col for col in self.unified_data.columns if col not in reddit_cols + stock_cols + ['date']]
            
            f.write(f"Reddit Features: {len(reddit_cols)}\n")
            f.write(f"Stock Features: {len(stock_cols)}\n")
            f.write(f"Other Features: {len(other_cols)}\n")
            f.write(f"Total Features: {len(self.unified_data.columns) - 1}\n\n")
            
            f.write("MISSING DATA SUMMARY:\n")
            missing_data = self.unified_data.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                for col, count in missing_data.items():
                    percentage = (count / len(self.unified_data)) * 100
                    f.write(f"  {col}: {count} ({percentage:.1f}%)\n")
            else:
                f.write("  No missing data found\n")
        
        logger.info(f"Dataset summary saved to {summary_file}")
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of loaded data
        
        Returns:
            Dictionary containing data summary
        """
        summary = {
            "reddit_data": None,
            "stock_data": {},
            "unified_data": None
        }
        
        if self.reddit_data is not None:
            summary["reddit_data"] = {
                "shape": self.reddit_data.shape,
                "date_range": None,
                "columns": list(self.reddit_data.columns)
            }
            if 'created_utc' in self.reddit_data.columns:
                summary["reddit_data"]["date_range"] = {
                    "start": self.reddit_data['created_utc'].min().strftime('%Y-%m-%d'),
                    "end": self.reddit_data['created_utc'].max().strftime('%Y-%m-%d')
                }
        
        for symbol, data in self.stock_data.items():
            if data is not None:
                summary["stock_data"][symbol] = {
                    "shape": data.shape,
                    "date_range": None,
                    "columns": list(data.columns)
                }
                if 'date' in data.columns:
                    summary["stock_data"][symbol]["date_range"] = {
                        "start": data['date'].min().strftime('%Y-%m-%d'),
                        "end": data['date'].max().strftime('%Y-%m-%d')
                    }
        
        if self.unified_data is not None:
            summary["unified_data"] = {
                "shape": self.unified_data.shape,
                "date_range": {
                    "start": self.unified_data['date'].min().strftime('%Y-%m-%d'),
                    "end": self.unified_data['date'].max().strftime('%Y-%m-%d')
                },
                "columns": list(self.unified_data.columns)
            }
        
        return summary


def main():
    """
    Main function to run Day 1 data loading pipeline
    """
    logger.info("Starting Day 1 Data Loading Pipeline...")
    
    # Initialize data loader
    loader = DataLoader()
    
    try:
        # Load Reddit data
        reddit_data = loader.load_reddit_data()
        
        # Load stock data
        stock_data = loader.load_stock_data()
        
        # Create unified dataset
        unified_data = loader.create_unified_dataset()
        
        # Save unified dataset
        loader.save_unified_dataset()
        
        # Get and print summary
        summary = loader.get_data_summary()
        
        print("\n" + "="*50)
        print("DAY 1 DATA LOADING PIPELINE COMPLETE")
        print("="*50)
        print(f"Reddit Data: {summary['reddit_data']['shape'] if summary['reddit_data'] else 'Not loaded'}")
        print(f"Stock Data: {len(summary['stock_data'])} symbols loaded")
        print(f"Unified Dataset: {summary['unified_data']['shape'] if summary['unified_data'] else 'Not created'}")
        print("="*50)
        
        logger.info("✅ Day 1 data loading pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in data loading pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main() 