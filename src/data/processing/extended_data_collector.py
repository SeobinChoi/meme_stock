"""
Extended Data Collector for Meme Stock Analysis
Collects comprehensive data from multiple sources and time periods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# Data collection imports
import yfinance as yf
import requests
import json
import time
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtendedDataCollector:
    """
    Extended data collector for comprehensive meme stock analysis
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.symbols = ["GME", "AMC", "BB", "TSLA", "AAPL", "SPY", "QQQ"]
        self.reddit_subreddits = [
            "wallstreetbets", "investing", "stocks", "pennystocks",
            "shortsqueeze", "superstonk", "amcstock", "bb_blackberry"
        ]
        
    def collect_extended_stock_data(self, start_year: int = 2019, end_year: int = 2024):
        """
        Collect extended stock data from 2019-2024
        """
        logger.info(f"📈 Collecting extended stock data from {start_year}-{end_year}...")
        
        # Create raw directory if it doesn't exist
        os.makedirs(self.data_dir / "raw", exist_ok=True)
        
        for symbol in self.symbols:
            try:
                logger.info(f"Downloading {symbol} extended data...")
                
                # Download extended historical data
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Reset index and clean data
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Add technical indicators
                df = self._add_technical_indicators(df)
                
                # Save extended data
                output_file = self.data_dir / "raw" / f"{symbol}_extended_stock_data.csv"
                df.to_csv(output_file, index=False)
                
                # Create description file
                self._create_extended_data_description(symbol, df, start_year, end_year)
                
                logger.info(f"✅ Downloaded {len(df):,} records for {symbol}")
                logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {symbol}: {e}")
    
    def collect_market_index_data(self):
        """
        Collect market index data for broader market context
        """
        logger.info("📊 Collecting market index data...")
        
        indices = ["^GSPC", "^DJI", "^IXIC", "^VIX", "^TNX"]  # S&P500, Dow, Nasdaq, VIX, 10Y Treasury
        
        for index in indices:
            try:
                logger.info(f"Downloading {index} index data...")
                
                ticker = yf.Ticker(index)
                df = ticker.history(start="2019-01-01", end="2024-12-31")
                
                if df.empty:
                    logger.warning(f"No data found for {index}")
                    continue
                
                # Reset index and clean data
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Save index data
                output_file = self.data_dir / "raw" / f"{index.replace('^', '')}_index_data.csv"
                df.to_csv(output_file, index=False)
                
                logger.info(f"✅ Downloaded {len(df):,} records for {index}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {index}: {e}")
    
    def collect_crypto_data(self):
        """
        Collect cryptocurrency data for broader market context
        """
        logger.info("🪙 Collecting cryptocurrency data...")
        
        cryptos = ["BTC-USD", "ETH-USD", "DOGE-USD"]  # Bitcoin, Ethereum, Dogecoin
        
        for crypto in cryptos:
            try:
                logger.info(f"Downloading {crypto} data...")
                
                ticker = yf.Ticker(crypto)
                df = ticker.history(start="2019-01-01", end="2024-12-31")
                
                if df.empty:
                    logger.warning(f"No data found for {crypto}")
                    continue
                
                # Reset index and clean data
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Save crypto data
                output_file = self.data_dir / "raw" / f"{crypto.replace('-USD', '')}_crypto_data.csv"
                df.to_csv(output_file, index=False)
                
                logger.info(f"✅ Downloaded {len(df):,} records for {crypto}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {crypto}: {e}")
    
    def collect_additional_reddit_data(self):
        """
        Collect additional Reddit data from multiple subreddits
        """
        logger.info("📱 Collecting additional Reddit data...")
        
        # This would require Reddit API authentication
        # For now, we'll create a framework for data collection
        logger.info("⚠️ Reddit API collection requires authentication")
        logger.info("   Consider using Pushshift API or Reddit API with proper auth")
        
        # Create sample structure for additional Reddit data
        sample_data = self._create_reddit_data_structure()
        
        # Save sample structure
        output_file = self.data_dir / "raw" / "additional_reddit_data_structure.csv"
        sample_data.to_csv(output_file, index=False)
        
        logger.info("✅ Created Reddit data collection structure")
    
    def collect_news_data(self):
        """
        Collect news data related to meme stocks
        """
        logger.info("📰 Collecting news data...")
        
        # This would require news API (e.g., NewsAPI, Alpha Vantage)
        # For now, create framework
        logger.info("⚠️ News collection requires API key")
        logger.info("   Consider: NewsAPI, Alpha Vantage, or web scraping")
        
        # Create sample structure
        sample_data = self._create_news_data_structure()
        
        # Save sample structure
        output_file = self.data_dir / "raw" / "news_data_structure.csv"
        sample_data.to_csv(output_file, index=False)
        
        logger.info("✅ Created news data collection structure")
    
    def collect_options_data(self):
        """
        Collect options trading data for meme stocks
        """
        logger.info("🔄 Collecting options data...")
        
        # This would require options data API
        # For now, create framework
        logger.info("⚠️ Options data requires specialized API")
        logger.info("   Consider: TD Ameritrade, Interactive Brokers, or CBOE")
        
        # Create sample structure
        sample_data = self._create_options_data_structure()
        
        # Save sample structure
        output_file = self.data_dir / "raw" / "options_data_structure.csv"
        sample_data.to_csv(output_file, index=False)
        
        logger.info("✅ Created options data collection structure")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to stock data
        """
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change indicators
        df['daily_return'] = df['close'].pct_change()
        df['volatility'] = df['daily_return'].rolling(window=20).std()
        
        return df
    
    def _create_extended_data_description(self, symbol: str, df: pd.DataFrame, start_year: int, end_year: int):
        """
        Create description file for extended data
        """
        description_file = self.data_dir / "raw" / f"{symbol}_extended_data_DESCRIPTION.txt"
        
        with open(description_file, 'w') as f:
            f.write(f"=== {symbol} EXTENDED STOCK DATA DESCRIPTION ===\n\n")
            f.write(f"Data Source: Yahoo Finance (yfinance)\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Period: {start_year}-01-01 to {end_year}-12-31\n")
            f.write(f"Total Records: {len(df):,}\n")
            f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n")
            f.write(f"Columns: {list(df.columns)}\n\n")
            f.write(f"Description:\n")
            f.write(f"This dataset contains extended historical stock price data for {symbol}.\n")
            f.write(f"The data covers {end_year - start_year + 1} years including the meme stock phenomenon period.\n")
            f.write(f"Technical indicators have been added for enhanced analysis.\n\n")
            f.write(f"Technical Indicators Added:\n")
            f.write(f"- Moving Averages (SMA 20, 50, EMA 12, 26)\n")
            f.write(f"- RSI (Relative Strength Index)\n")
            f.write(f"- MACD (Moving Average Convergence Divergence)\n")
            f.write(f"- Bollinger Bands\n")
            f.write(f"- Volume indicators\n")
            f.write(f"- Volatility measures\n\n")
            f.write(f"Data Quality:\n")
            f.write(f"- Missing values: {df.isnull().sum().sum()}\n")
            f.write(f"- Data completeness: {round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)}%\n")
            f.write(f"- Trading days: {len(df)}\n\n")
            f.write(f"Downloaded on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _create_reddit_data_structure(self) -> pd.DataFrame:
        """
        Create structure for additional Reddit data
        """
        dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
        
        sample_data = pd.DataFrame({
            'date': dates,
            'subreddit': 'wallstreetbets',
            'post_count': 0,
            'comment_count': 0,
            'total_score': 0,
            'avg_score': 0.0,
            'sentiment_score': 0.0,
            'meme_stock_mentions': 0,
            'gme_mentions': 0,
            'amc_mentions': 0,
            'bb_mentions': 0
        })
        
        return sample_data
    
    def _create_news_data_structure(self) -> pd.DataFrame:
        """
        Create structure for news data
        """
        dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
        
        sample_data = pd.DataFrame({
            'date': dates,
            'source': 'news_api',
            'title': '',
            'description': '',
            'content': '',
            'sentiment_score': 0.0,
            'relevance_score': 0.0,
            'meme_stock_related': False,
            'gme_mentioned': False,
            'amc_mentioned': False,
            'bb_mentioned': False
        })
        
        return sample_data
    
    def _create_options_data_structure(self) -> pd.DataFrame:
        """
        Create structure for options data
        """
        dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
        
        sample_data = pd.DataFrame({
            'date': dates,
            'symbol': 'GME',
            'expiration_date': '',
            'strike_price': 0.0,
            'option_type': 'call',
            'volume': 0,
            'open_interest': 0,
            'implied_volatility': 0.0,
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        })
        
        return sample_data
    
    def generate_data_collection_plan(self):
        """
        Generate comprehensive data collection plan
        """
        logger.info("📋 Generating data collection plan...")
        
        plan = {
            "current_data": {
                "stock_data": "2020-2021 (504 days per symbol)",
                "reddit_data": "2021 (365 days, WSB only)",
                "total_records": "~1,097 Reddit posts"
            },
            "planned_extensions": {
                "stock_data": "2019-2024 (6 years, ~1,500 days per symbol)",
                "reddit_data": "2019-2024 (6 years, multiple subreddits)",
                "news_data": "2019-2024 (6 years, multiple sources)",
                "options_data": "2019-2024 (6 years, options chain)"
            },
            "data_sources": {
                "stock_data": ["Yahoo Finance", "Alpha Vantage", "IEX Cloud"],
                "reddit_data": ["Reddit API", "Pushshift API", "Web scraping"],
                "news_data": ["NewsAPI", "Alpha Vantage", "Web scraping"],
                "options_data": ["TD Ameritrade", "Interactive Brokers", "CBOE"]
            },
            "estimated_total_records": {
                "stock_data": "~10,500 records (7 symbols × 1,500 days)",
                "reddit_data": "~50,000 records (8 subreddits × 2,190 days)",
                "news_data": "~20,000 records (daily news × 2,190 days)",
                "options_data": "~100,000 records (daily options × 2,190 days)"
            }
        }
        
        # Save plan
        plan_file = self.data_dir / "raw" / "extended_data_collection_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2, default=str)
        
        logger.info("✅ Generated data collection plan")
        return plan

def main():
    """
    Main function to run extended data collection
    """
    import os
    
    collector = ExtendedDataCollector()
    
    # Generate collection plan
    plan = collector.generate_data_collection_plan()
    
    # Collect extended stock data
    collector.collect_extended_stock_data(2019, 2024)
    
    # Collect market index data
    collector.collect_market_index_data()
    
    # Collect crypto data
    collector.collect_crypto_data()
    
    # Create structures for additional data
    collector.collect_additional_reddit_data()
    collector.collect_news_data()
    collector.collect_options_data()
    
    logger.info("✅ Extended data collection completed")

if __name__ == "__main__":
    main()
