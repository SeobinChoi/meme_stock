"""
Day 2: Data Cleaning and Preprocessing Pipeline
Handles text preprocessing, financial data cleaning, and unified dataset creation
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing for Day 2
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.reddit_data = None
        self.stock_data = {}
        self.cleaned_data = {}
        self.cleaning_results = {}
        
    def load_datasets(self) -> Dict:
        """
        Load all datasets for cleaning
        """
        logger.info("Loading datasets for Day 2 cleaning...")
        
        # Load Reddit data
        try:
            reddit_file = f"{self.data_dir}/raw/reddit_wsb.csv"
            self.reddit_data = pd.read_csv(reddit_file)
            logger.info(f"✅ Loaded Reddit data: {len(self.reddit_data):,} rows")
        except Exception as e:
            logger.error(f"❌ Failed to load Reddit data: {e}")
            return {"status": "ERROR", "message": f"Reddit data loading failed: {e}"}
        
        # Load stock data
        stock_symbols = ["GME", "AMC", "BB"]
        for symbol in stock_symbols:
            try:
                stock_file = f"{self.data_dir}/raw/{symbol}_stock_data.csv"
                self.stock_data[symbol] = pd.read_csv(stock_file)
                logger.info(f"✅ Loaded {symbol} data: {len(self.stock_data[symbol]):,} rows")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {symbol} data: {e}")
        
        return {"status": "SUCCESS", "datasets_loaded": len(self.stock_data) + 1}
    
    def clean_reddit_text(self) -> pd.DataFrame:
        """
        Comprehensive Reddit text preprocessing
        """
        logger.info("🧹 Cleaning Reddit text data...")
        
        if self.reddit_data is None:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying original
        cleaned_reddit = self.reddit_data.copy()
        
        # 1. Content Standardization
        if 'title' in cleaned_reddit.columns:
            # Unicode normalization
            cleaned_reddit['title'] = cleaned_reddit['title'].astype(str).apply(
                lambda x: unicodedata.normalize('NFKC', x)
            )
            
            # Case handling (convert to lowercase for analysis)
            cleaned_reddit['title_lower'] = cleaned_reddit['title'].str.lower()
            
            # Remove special characters but keep important ones
            cleaned_reddit['title_clean'] = cleaned_reddit['title'].apply(
                lambda x: re.sub(r'[^\w\s\-\$\.\,\!\?]', '', str(x))
            )
        
        if 'body' in cleaned_reddit.columns:
            # Handle missing body content
            cleaned_reddit['body'] = cleaned_reddit['body'].fillna('')
            cleaned_reddit['body_clean'] = cleaned_reddit['body'].apply(
                lambda x: re.sub(r'[^\w\s\-\$\.\,\!\?]', '', str(x))
            )
        
        # 2. Spam Detection and Filtering
        # Remove posts with very low engagement (potential spam)
        if 'score' in cleaned_reddit.columns:
            score_threshold = cleaned_reddit['score'].quantile(0.1)  # Bottom 10%
            spam_mask = cleaned_reddit['score'] <= score_threshold
            logger.info(f"Identified {spam_mask.sum()} potential spam posts (score <= {score_threshold})")
        
        # Remove posts with suspicious patterns
        if 'title' in cleaned_reddit.columns:
            # Remove posts with excessive caps
            caps_ratio = cleaned_reddit['title'].str.count(r'[A-Z]') / cleaned_reddit['title'].str.len()
            excessive_caps = caps_ratio > 0.8
            logger.info(f"Identified {excessive_caps.sum()} posts with excessive caps")
            
            # Remove posts with suspicious keywords
            spam_keywords = ['buy now', 'click here', 'make money', 'get rich', 'investment opportunity']
            spam_keyword_mask = cleaned_reddit['title_lower'].str.contains('|'.join(spam_keywords), na=False)
            logger.info(f"Identified {spam_keyword_mask.sum()} posts with spam keywords")
        
        # 3. Content Quality Assessment
        if 'title' in cleaned_reddit.columns:
            # Calculate content quality metrics
            cleaned_reddit['title_length'] = cleaned_reddit['title'].str.len()
            cleaned_reddit['word_count'] = cleaned_reddit['title'].str.split().str.len()
            
            # Filter out very short or very long titles
            quality_mask = (cleaned_reddit['title_length'] >= 5) & (cleaned_reddit['title_length'] <= 300)
            logger.info(f"Filtered out {~quality_mask.sum()} posts with poor title quality")
        
        # 4. Language Filtering (basic English detection)
        if 'title' in cleaned_reddit.columns:
            # Simple English character ratio check
            english_chars = cleaned_reddit['title'].str.count(r'[a-zA-Z]')
            total_chars = cleaned_reddit['title'].str.len()
            english_ratio = english_chars / total_chars
            english_mask = english_ratio > 0.5  # At least 50% English characters
            logger.info(f"Identified {~english_mask.sum()} posts with low English content")
        
        # 5. Apply all filters
        final_mask = quality_mask & english_mask
        if 'score' in cleaned_reddit.columns:
            final_mask = final_mask & (cleaned_reddit['score'] > score_threshold)
        
        cleaned_reddit = cleaned_reddit[final_mask].reset_index(drop=True)
        logger.info(f"✅ Reddit data cleaned: {len(cleaned_reddit):,} posts remaining")
        
        return cleaned_reddit
    
    def clean_financial_data(self) -> Dict:
        """
        Comprehensive financial data cleaning
        """
        logger.info("📈 Cleaning financial data...")
        
        cleaned_stocks = {}
        
        for symbol, data in self.stock_data.items():
            logger.info(f"Cleaning {symbol} data...")
            
            # Create a copy
            cleaned_data = data.copy()
            
            # 1. Date handling
            if 'Date' in cleaned_data.columns:
                cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
                cleaned_data = cleaned_data.sort_values('Date').reset_index(drop=True)
            
            # 2. Price consistency checks and fixes
            if all(col in cleaned_data.columns for col in ['Open', 'High', 'Low', 'Close']):
                # Fix High < Low violations
                high_low_violations = cleaned_data['High'] < cleaned_data['Low']
                if high_low_violations.any():
                    logger.warning(f"Found {high_low_violations.sum()} High < Low violations in {symbol}")
                    # Swap High and Low where needed
                    mask = high_low_violations
                    cleaned_data.loc[mask, ['High', 'Low']] = cleaned_data.loc[mask, ['Low', 'High']].values
                
                # Fix High < Open violations
                high_open_violations = cleaned_data['High'] < cleaned_data['Open']
                if high_open_violations.any():
                    logger.warning(f"Found {high_open_violations.sum()} High < Open violations in {symbol}")
                    cleaned_data.loc[high_open_violations, 'High'] = cleaned_data.loc[high_open_violations, 'Open']
                
                # Fix High < Close violations
                high_close_violations = cleaned_data['High'] < cleaned_data['Close']
                if high_close_violations.any():
                    logger.warning(f"Found {high_close_violations.sum()} High < Close violations in {symbol}")
                    cleaned_data.loc[high_close_violations, 'High'] = cleaned_data.loc[high_close_violations, 'Close']
            
            # 3. Outlier detection and handling
            if 'Close' in cleaned_data.columns:
                # Calculate returns
                cleaned_data['returns'] = cleaned_data['Close'].pct_change()
                
                # Identify extreme returns (outliers)
                returns_std = cleaned_data['returns'].std()
                extreme_returns = abs(cleaned_data['returns']) > 3 * returns_std
                if extreme_returns.any():
                    logger.warning(f"Found {extreme_returns.sum()} extreme returns in {symbol}")
                    # Mark outliers for review (don't remove automatically)
                    cleaned_data['is_outlier'] = extreme_returns
            
            # 4. Volume data cleaning
            if 'Volume' in cleaned_data.columns:
                # Handle zero or negative volume
                invalid_volume = cleaned_data['Volume'] <= 0
                if invalid_volume.any():
                    logger.warning(f"Found {invalid_volume.sum()} invalid volume records in {symbol}")
                    # Replace with median volume
                    median_volume = cleaned_data['Volume'].median()
                    cleaned_data.loc[invalid_volume, 'Volume'] = median_volume
            
            # 5. Missing data handling
            missing_data = cleaned_data.isnull().sum()
            if missing_data.any():
                logger.warning(f"Found missing data in {symbol}: {missing_data[missing_data > 0].to_dict()}")
                
                # Forward fill for price data
                price_columns = ['Open', 'High', 'Low', 'Close']
                for col in price_columns:
                    if col in cleaned_data.columns:
                        cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
                
                # Forward fill for volume
                if 'Volume' in cleaned_data.columns:
                    cleaned_data['Volume'] = cleaned_data['Volume'].fillna(method='ffill')
            
            # 6. Remove any remaining rows with missing critical data
            critical_columns = ['Date', 'Close']
            if all(col in cleaned_data.columns for col in critical_columns):
                before_count = len(cleaned_data)
                cleaned_data = cleaned_data.dropna(subset=critical_columns)
                after_count = len(cleaned_data)
                if before_count != after_count:
                    logger.info(f"Removed {before_count - after_count} rows with missing critical data from {symbol}")
            
            cleaned_stocks[symbol] = cleaned_data
            logger.info(f"✅ {symbol} data cleaned: {len(cleaned_data):,} records")
        
        return cleaned_stocks
    
    def create_unified_dataset(self, cleaned_reddit: pd.DataFrame, cleaned_stocks: Dict) -> pd.DataFrame:
        """
        Create unified dataset with temporal alignment
        """
        logger.info("🔗 Creating unified dataset...")
        
        # 1. Prepare Reddit data for daily aggregation
        if cleaned_reddit.empty:
            logger.error("No cleaned Reddit data available")
            return pd.DataFrame()
        
        # Convert timestamp to date
        if 'created' in cleaned_reddit.columns:
            cleaned_reddit['date'] = pd.to_datetime(cleaned_reddit['created'], unit='s').dt.date
        else:
            logger.error("No 'created' column found in Reddit data")
            return pd.DataFrame()
        
        # 2. Daily Reddit aggregation
        agg_dict = {
            'title': 'count',  # Post count
            'score': ['sum', 'mean', 'std']  # Engagement metrics
        }
        
        if 'comms_num' in cleaned_reddit.columns:
            agg_dict['comms_num'] = ['sum', 'mean', 'std']
        
        daily_reddit = cleaned_reddit.groupby('date').agg(agg_dict).reset_index()
        
        # Flatten column names
        if 'comms_num' in cleaned_reddit.columns:
            daily_reddit.columns = ['date', 'post_count', 'total_score', 'avg_score', 'score_std', 'total_comments', 'avg_comments', 'comment_std']
        else:
            daily_reddit.columns = ['date', 'post_count', 'total_score', 'avg_score', 'score_std']
        
        # 3. Prepare stock data for merging
        stock_dfs = []
        for symbol, data in cleaned_stocks.items():
            if 'Date' in data.columns and 'Close' in data.columns:
                stock_daily = data[['Date', 'Close', 'Volume']].copy()
                # Convert Date to datetime if it's not already, handle tz-aware datetimes
                stock_daily['Date'] = pd.to_datetime(stock_daily['Date'], utc=True)
                stock_daily['date'] = stock_daily['Date'].dt.date
                stock_daily = stock_daily.rename(columns={
                    'Close': f'{symbol}_close',
                    'Volume': f'{symbol}_volume'
                })
                stock_daily = stock_daily[['date', f'{symbol}_close', f'{symbol}_volume']]
                stock_dfs.append(stock_daily)
        
        # 4. Merge all datasets
        unified_df = daily_reddit.copy()
        
        for stock_df in stock_dfs:
            unified_df = unified_df.merge(stock_df, on='date', how='left')
        
        # 5. Handle missing values in unified dataset
        # Forward fill for stock prices (weekends/holidays)
        stock_columns = [col for col in unified_df.columns if '_close' in col or '_volume' in col]
        for col in stock_columns:
            unified_df[col] = unified_df[col].fillna(method='ffill')
        
        # Fill remaining missing values with 0 for social metrics
        social_columns = ['post_count', 'total_score', 'avg_score', 'score_std']
        if 'total_comments' in unified_df.columns:
            social_columns.extend(['total_comments', 'avg_comments', 'comment_std'])
        
        for col in social_columns:
            if col in unified_df.columns:
                unified_df[col] = unified_df[col].fillna(0)
        
        # 6. Add derived features
        # Calculate returns for each stock
        for symbol in cleaned_stocks.keys():
            close_col = f'{symbol}_close'
            if close_col in unified_df.columns:
                unified_df[f'{symbol}_returns'] = unified_df[close_col].pct_change()
        
        # Calculate social engagement ratios
        if 'total_comments' in unified_df.columns:
            unified_df['engagement_ratio'] = unified_df['total_comments'] / unified_df['post_count'].replace(0, 1)
        
        # 7. Final cleaning
        # Remove rows with all missing stock data
        stock_data_mask = unified_df[stock_columns].notna().any(axis=1)
        unified_df = unified_df[stock_data_mask].reset_index(drop=True)
        
        # Sort by date
        unified_df = unified_df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✅ Unified dataset created: {len(unified_df):,} daily records")
        logger.info(f"Date range: {unified_df['date'].min()} to {unified_df['date'].max()}")
        logger.info(f"Features: {list(unified_df.columns)}")
        
        return unified_df
    
    def generate_cleaning_report(self) -> Dict:
        """
        Generate comprehensive cleaning report
        """
        logger.info("📊 Generating cleaning report...")
        
        # Load datasets first
        load_result = self.load_datasets()
        if load_result["status"] != "SUCCESS":
            logger.error(f"Failed to load datasets: {load_result}")
            return {"status": "ERROR", "message": "Dataset loading failed"}
        
        # Run cleaning pipeline
        cleaned_reddit = self.clean_reddit_text()
        cleaned_stocks = self.clean_financial_data()
        unified_dataset = self.create_unified_dataset(cleaned_reddit, cleaned_stocks)
        
        # Compile cleaning statistics
        report = {
            "cleaning_timestamp": datetime.now().isoformat(),
            "original_data": {
                "reddit_posts": len(self.reddit_data) if self.reddit_data is not None else 0,
                "stock_records": {symbol: len(data) for symbol, data in self.stock_data.items()}
            },
            "cleaned_data": {
                "reddit_posts": len(cleaned_reddit),
                "stock_records": {symbol: len(data) for symbol, data in cleaned_stocks.items()},
                "unified_records": len(unified_dataset)
            },
            "cleaning_metrics": {
                "reddit_retention_rate": round(len(cleaned_reddit) / len(self.reddit_data) * 100, 2) if self.reddit_data is not None else 0,
                "stock_retention_rate": round(np.mean([len(data) / len(self.stock_data[symbol]) for symbol, data in cleaned_stocks.items()]) * 100, 2) if self.stock_data else 0
            },
            "data_quality_improvements": self._assess_quality_improvements(cleaned_reddit, cleaned_stocks, unified_dataset)
        }
        
        # Save cleaned datasets
        self._save_cleaned_datasets(cleaned_reddit, cleaned_stocks, unified_dataset)
        
        self.cleaning_results = report
        return report
    
    def _assess_quality_improvements(self, cleaned_reddit: pd.DataFrame, cleaned_stocks: Dict, unified_dataset: pd.DataFrame) -> Dict:
        """
        Assess data quality improvements after cleaning
        """
        improvements = {
            "reddit_quality": {},
            "stock_quality": {},
            "unified_quality": {}
        }
        
        # Reddit quality improvements
        if self.reddit_data is not None and not cleaned_reddit.empty:
            original_missing = self.reddit_data.isnull().sum().sum()
            cleaned_missing = cleaned_reddit.isnull().sum().sum()
            improvements["reddit_quality"] = {
                "missing_data_reduction": original_missing - cleaned_missing,
                "data_completeness": round((1 - cleaned_missing / (len(cleaned_reddit) * len(cleaned_reddit.columns))) * 100, 2)
            }
        
        # Stock quality improvements
        stock_improvements = {}
        for symbol, original_data in self.stock_data.items():
            if symbol in cleaned_stocks:
                cleaned_data = cleaned_stocks[symbol]
                original_missing = original_data.isnull().sum().sum()
                cleaned_missing = cleaned_data.isnull().sum().sum()
                stock_improvements[symbol] = {
                    "missing_data_reduction": original_missing - cleaned_missing,
                    "data_completeness": round((1 - cleaned_missing / (len(cleaned_data) * len(cleaned_data.columns))) * 100, 2)
                }
        improvements["stock_quality"] = stock_improvements
        
        # Unified dataset quality
        if not unified_dataset.empty:
            improvements["unified_quality"] = {
                "total_records": len(unified_dataset),
                "date_coverage": (unified_dataset['date'].max() - unified_dataset['date'].min()).days,
                "feature_count": len(unified_dataset.columns),
                "missing_data_percentage": round(unified_dataset.isnull().sum().sum() / (len(unified_dataset) * len(unified_dataset.columns)) * 100, 2)
            }
        
        return improvements
    
    def _save_cleaned_datasets(self, cleaned_reddit: pd.DataFrame, cleaned_stocks: Dict, unified_dataset: pd.DataFrame):
        """
        Save cleaned datasets to processed directory
        """
        from pathlib import Path
        
        # Ensure processed directory exists
        processed_dir = Path(f"{self.data_dir}/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned Reddit data
        if not cleaned_reddit.empty:
            reddit_output = processed_dir / "cleaned_reddit_wsb.csv"
            cleaned_reddit.to_csv(reddit_output, index=False)
            logger.info(f"Saved cleaned Reddit data to {reddit_output}")
        
        # Save cleaned stock data
        for symbol, data in cleaned_stocks.items():
            stock_output = processed_dir / f"cleaned_{symbol}_stock_data.csv"
            data.to_csv(stock_output, index=False)
            logger.info(f"Saved cleaned {symbol} data to {stock_output}")
        
        # Save unified dataset
        if not unified_dataset.empty:
            unified_output = processed_dir / "unified_dataset.csv"
            unified_dataset.to_csv(unified_output, index=False)
            logger.info(f"Saved unified dataset to {unified_output}")
    
    def save_cleaning_report(self, report: Dict, output_path: str = "results/day2_cleaning_report.json"):
        """
        Save cleaning report to file
        """
        import json
        from pathlib import Path
        
        # Ensure results directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Cleaning report saved to {output_file}")
        
        # Save human-readable summary
        summary_file = output_file.parent / "day2_cleaning_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== DAY 2 DATA CLEANING SUMMARY ===\n\n")
            f.write(f"Cleaning Timestamp: {report['cleaning_timestamp']}\n\n")
            
            # Original data summary
            f.write("ORIGINAL DATA:\n")
            f.write(f"  Reddit Posts: {report['original_data']['reddit_posts']:,}\n")
            for symbol, count in report['original_data']['stock_records'].items():
                f.write(f"  {symbol} Records: {count:,}\n")
            f.write("\n")
            
            # Cleaned data summary
            f.write("CLEANED DATA:\n")
            f.write(f"  Reddit Posts: {report['cleaned_data']['reddit_posts']:,}\n")
            for symbol, count in report['cleaned_data']['stock_records'].items():
                f.write(f"  {symbol} Records: {count:,}\n")
            f.write(f"  Unified Records: {report['cleaned_data']['unified_records']:,}\n")
            f.write("\n")
            
            # Retention rates
            f.write("RETENTION RATES:\n")
            f.write(f"  Reddit: {report['cleaning_metrics']['reddit_retention_rate']}%\n")
            f.write(f"  Stock Data: {report['cleaning_metrics']['stock_retention_rate']}%\n")
            f.write("\n")
            
            # Quality improvements
            f.write("QUALITY IMPROVEMENTS:\n")
            if report['data_quality_improvements']['reddit_quality']:
                reddit_quality = report['data_quality_improvements']['reddit_quality']
                f.write(f"  Reddit Completeness: {reddit_quality['data_completeness']}%\n")
            
            for symbol, quality in report['data_quality_improvements']['stock_quality'].items():
                f.write(f"  {symbol} Completeness: {quality['data_completeness']}%\n")
            
            if report['data_quality_improvements']['unified_quality']:
                unified_quality = report['data_quality_improvements']['unified_quality']
                f.write(f"  Unified Dataset: {unified_quality['total_records']:,} records\n")
                f.write(f"  Date Coverage: {unified_quality['date_coverage']} days\n")
                f.write(f"  Missing Data: {unified_quality['missing_data_percentage']}%\n")
        
        logger.info(f"Cleaning summary saved to {summary_file}")


def main():
    """
    Main function to run Day 2 data cleaning
    """
    logger.info("Starting Day 2 Data Cleaning...")
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Load datasets
    load_result = cleaner.load_datasets()
    if load_result["status"] != "SUCCESS":
        logger.error(f"Failed to load datasets: {load_result}")
        return
    
    # Generate comprehensive cleaning report
    report = cleaner.generate_cleaning_report()
    
    # Save report
    cleaner.save_cleaning_report(report)
    
    # Print summary
    print("\n" + "="*50)
    print("DAY 2 DATA CLEANING COMPLETE")
    print("="*50)
    print(f"Reddit Retention: {report['cleaning_metrics']['reddit_retention_rate']}%")
    print(f"Stock Retention: {report['cleaning_metrics']['stock_retention_rate']}%")
    print(f"Unified Records: {report['cleaned_data']['unified_records']:,}")
    print("="*50)
    
    if report['cleaning_metrics']['reddit_retention_rate'] >= 80:
        logger.info("✅ Day 2 cleaning PASSED - Ready for feature engineering")
    else:
        logger.warning("⚠️ Day 2 cleaning has WARNINGS - Review data retention")


if __name__ == "__main__":
    main() 