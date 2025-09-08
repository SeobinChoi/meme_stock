#!/usr/bin/env python3
"""
Comprehensive feature engineering pipeline for meme stock contrarian effect prediction.

Combines text confidence, sentiment, Reddit dynamics, and technical features
as described in the manual.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import feature modules
from .text_confidence import calculate_confidence_features, aggregate_daily_confidence
from .sentiment import calculate_sentiment_features, aggregate_daily_sentiment
from .reddit_dynamics import calculate_reddit_dynamics_features, aggregate_daily_dynamics
from .technical_features import calculate_technical_features

logger = logging.getLogger(__name__)

class MemeStockFeatureEngine:
    """Main feature engineering engine for meme stock analysis."""
    
    def __init__(self,
                 target_tickers: List[str] = ['GME', 'AMC', 'BB'],
                 date_range: Tuple[str, str] = ('2021-01-01', '2023-12-31'),
                 use_finbert: bool = False):
        """
        Initialize meme stock feature engine.
        
        Args:
            target_tickers: List of tickers to analyze
            date_range: Date range for analysis
            use_finbert: Whether to use FinBERT for sentiment (requires transformers)
        """
        self.target_tickers = target_tickers
        self.date_range = date_range
        self.use_finbert = use_finbert
        
        logger.info(f"Initialized feature engine for tickers: {target_tickers}")
        logger.info(f"Date range: {date_range}")
    
    def load_and_prepare_data(self, 
                            reddit_data_path: str,
                            price_data_paths: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load and prepare Reddit and price data.
        
        Args:
            reddit_data_path: Path to Reddit data file
            price_data_paths: Dictionary mapping ticker to price data path
            
        Returns:
            Tuple of (reddit_df, price_data_dict)
        """
        logger.info("Loading and preparing data...")
        
        # Load Reddit data
        reddit_df = pd.read_csv(reddit_data_path)
        reddit_df['date'] = pd.to_datetime(reddit_df['date'])
        
        # Filter by date range
        reddit_df = reddit_df[
            (reddit_df['date'] >= self.date_range[0]) & 
            (reddit_df['date'] <= self.date_range[1])
        ]
        
        # Load price data
        price_data = {}
        for ticker, path in price_data_paths.items():
            if ticker in self.target_tickers:
                price_df = pd.read_csv(path)
                price_df['date'] = pd.to_datetime(price_df['date'])
                
                # Filter by date range
                price_df = price_df[
                    (price_df['date'] >= self.date_range[0]) & 
                    (price_df['date'] <= self.date_range[1])
                ]
                
                price_data[ticker] = price_df
                logger.info(f"Loaded {len(price_df)} price records for {ticker}")
        
        logger.info(f"Loaded {len(reddit_df)} Reddit records")
        
        return reddit_df, price_data
    
    def create_ticker_specific_reddit_data(self, reddit_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create ticker-specific Reddit datasets.
        
        Args:
            reddit_df: Combined Reddit dataframe
            
        Returns:
            Dictionary mapping ticker to Reddit dataframe
        """
        logger.info("Creating ticker-specific Reddit datasets...")
        
        ticker_data = {}
        
        for ticker in self.target_tickers:
            # Filter Reddit data for this ticker
            ticker_reddit = reddit_df[reddit_df['ticker'] == ticker].copy()
            
            if len(ticker_reddit) > 0:
                ticker_data[ticker] = ticker_reddit
                logger.info(f"Created dataset for {ticker}: {len(ticker_reddit)} records")
            else:
                logger.warning(f"No Reddit data found for {ticker}")
        
        return ticker_data
    
    def engineer_text_features(self, reddit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer text-based features (confidence and sentiment).
        
        Args:
            reddit_df: Reddit dataframe with text data
            
        Returns:
            DataFrame with text features
        """
        logger.info("Engineering text features...")
        
        # Calculate confidence features
        if 'text' in reddit_df.columns:
            reddit_df = calculate_confidence_features(reddit_df, text_col='text')
        elif 'title' in reddit_df.columns:
            reddit_df = calculate_confidence_features(reddit_df, text_col='title')
        else:
            logger.warning("No text column found for confidence analysis")
            return reddit_df
        
        # Calculate sentiment features
        text_col = 'text' if 'text' in reddit_df.columns else 'title'
        reddit_df = calculate_sentiment_features(reddit_df, text_col=text_col, use_finbert=self.use_finbert)
        
        logger.info("Text features engineered successfully")
        
        return reddit_df
    
    def engineer_reddit_dynamics_features(self, reddit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer Reddit dynamics features.
        
        Args:
            reddit_df: Reddit dataframe
            
        Returns:
            DataFrame with Reddit dynamics features
        """
        logger.info("Engineering Reddit dynamics features...")
        
        # Ensure we have mentions column
        if 'mentions' not in reddit_df.columns:
            if 'score' in reddit_df.columns:
                reddit_df['mentions'] = reddit_df['score']  # Use score as proxy
            else:
                logger.warning("No mentions or score column found for dynamics analysis")
                return reddit_df
        
        # Calculate Reddit dynamics features
        reddit_df = calculate_reddit_dynamics_features(reddit_df, mentions_col='mentions', group_col='ticker')
        
        logger.info("Reddit dynamics features engineered successfully")
        
        return reddit_df
    
    def engineer_technical_features(self, price_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Engineer technical features for a ticker.
        
        Args:
            price_df: Price dataframe for the ticker
            ticker: Ticker symbol
            
        Returns:
            DataFrame with technical features
        """
        logger.info(f"Engineering technical features for {ticker}...")
        
        # Add ticker column if not present
        if 'ticker' not in price_df.columns:
            price_df = price_df.copy()
            price_df['ticker'] = ticker
        
        # Calculate technical features
        price_df = calculate_technical_features(price_df, price_col='close', volume_col='volume', date_col='date', group_col='ticker')
        
        logger.info(f"Technical features engineered for {ticker}")
        
        return price_df
    
    def create_daily_features(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create daily aggregated features for each ticker.
        
        Args:
            ticker_data: Dictionary of ticker-specific dataframes
            
        Returns:
            Dictionary of daily feature dataframes
        """
        logger.info("Creating daily aggregated features...")
        
        daily_features = {}
        
        for ticker, df in ticker_data.items():
            logger.info(f"Creating daily features for {ticker}...")
            
            # Aggregate text features
            if any(col in df.columns for col in ['confidence_score', 'vader_compound']):
                text_agg = aggregate_daily_confidence(df, group_cols=['date', 'ticker'])
                sentiment_agg = aggregate_daily_sentiment(df, group_cols=['date', 'ticker'])
                
                # Merge text aggregations
                daily_df = pd.merge(text_agg, sentiment_agg, on=['date', 'ticker'], how='outer')
            else:
                daily_df = df.groupby(['date', 'ticker']).agg({
                    'mentions': ['sum', 'count', 'mean'],
                    'score': ['sum', 'mean'],
                    'comments': ['sum', 'mean']
                }).reset_index()
                daily_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in daily_df.columns]
                daily_df = daily_df.rename(columns={'date_': 'date', 'ticker_': 'ticker'})
            
            # Aggregate Reddit dynamics features
            if any(col in df.columns for col in ['reddit_surprise', 'ema_3', 'momentum_3']):
                dynamics_agg = aggregate_daily_dynamics(df, group_cols=['date', 'ticker'])
                daily_df = pd.merge(daily_df, dynamics_agg, on=['date', 'ticker'], how='outer')
            
            daily_features[ticker] = daily_df
            logger.info(f"Created daily features for {ticker}: {len(daily_df)} records")
        
        return daily_features
    
    def merge_price_and_reddit_features(self, 
                                     daily_features: Dict[str, pd.DataFrame],
                                     price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Merge price and Reddit features for each ticker.
        
        Args:
            daily_features: Dictionary of daily Reddit features
            price_data: Dictionary of price data
            
        Returns:
            Dictionary of merged feature dataframes
        """
        logger.info("Merging price and Reddit features...")
        
        merged_data = {}
        
        for ticker in self.target_tickers:
            if ticker in daily_features and ticker in price_data:
                logger.info(f"Merging features for {ticker}...")
                
                reddit_df = daily_features[ticker]
                price_df = price_data[ticker]
                
                # Merge on date
                merged_df = pd.merge(price_df, reddit_df, on='date', how='inner')
                
                # Calculate next-day return (target variable)
                merged_df['next_return'] = merged_df.groupby('ticker')['close'].pct_change().shift(-1)
                
                # Calculate current return
                merged_df['return'] = merged_df.groupby('ticker')['close'].pct_change()
                
                # Remove rows with missing next_return (last day of data)
                merged_df = merged_df.dropna(subset=['next_return'])
                
                merged_data[ticker] = merged_df
                logger.info(f"Merged features for {ticker}: {len(merged_df)} records")
            else:
                logger.warning(f"Missing data for {ticker}")
        
        return merged_data
    
    def create_combined_dataset(self, merged_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create combined dataset across all tickers.
        
        Args:
            merged_data: Dictionary of merged ticker data
            
        Returns:
            Combined dataframe
        """
        logger.info("Creating combined dataset...")
        
        if not merged_data:
            logger.error("No merged data available")
            return pd.DataFrame()
        
        # Combine all ticker data
        combined_df = pd.concat(list(merged_data.values()), ignore_index=True)
        
        # Sort by ticker and date
        combined_df = combined_df.sort_values(['ticker', 'date'])
        
        # Add market regime features
        combined_df = self._add_market_regime_features(combined_df)
        
        logger.info(f"Created combined dataset: {len(combined_df)} records across {len(merged_data)} tickers")
        
        return combined_df
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        logger.info("Adding market regime features...")
        
        # VIX proxy (using volatility)
        df['volatility_regime'] = pd.cut(
            df['volatility_20d'], 
            bins=[0, 0.2, 0.4, float('inf')], 
            labels=['low', 'medium', 'high']
        )
        
        # Volume regime
        if 'volume_ratio10' in df.columns:
            df['volume_regime'] = pd.cut(
                df['volume_ratio10'], 
                bins=[0, 1, 2, float('inf')], 
                labels=['low', 'normal', 'high']
            )
        
        # Meme stock frenzy indicator (high mentions + high volatility)
        if 'mentions_sum' in df.columns and 'volatility_20d' in df.columns:
            high_mentions = df['mentions_sum'] > df['mentions_sum'].quantile(0.8)
            high_vol = df['volatility_20d'] > df['volatility_20d'].quantile(0.8)
            df['meme_frenzy'] = (high_mentions & high_vol).astype(int)
        
        return df
    
    def engineer_all_features(self, 
                            reddit_data_path: str,
                            price_data_paths: Dict[str, str]) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            reddit_data_path: Path to Reddit data
            price_data_paths: Dictionary mapping ticker to price data path
            
        Returns:
            Complete feature dataframe
        """
        logger.info("Starting complete feature engineering pipeline...")
        
        # Load and prepare data
        reddit_df, price_data = self.load_and_prepare_data(reddit_data_path, price_data_paths)
        
        # Create ticker-specific Reddit data
        ticker_reddit_data = self.create_ticker_specific_reddit_data(reddit_df)
        
        # Engineer features for each ticker
        ticker_features = {}
        
        for ticker in self.target_tickers:
            if ticker in ticker_reddit_data:
                logger.info(f"Processing {ticker}...")
                
                # Engineer text features
                ticker_df = self.engineer_text_features(ticker_reddit_data[ticker])
                
                # Engineer Reddit dynamics features
                ticker_df = self.engineer_reddit_dynamics_features(ticker_df)
                
                ticker_features[ticker] = ticker_df
        
        # Create daily features
        daily_features = self.create_daily_features(ticker_features)
        
        # Engineer technical features for price data
        for ticker, price_df in price_data.items():
            price_data[ticker] = self.engineer_technical_features(price_df, ticker)
        
        # Merge price and Reddit features
        merged_data = self.merge_price_and_reddit_features(daily_features, price_data)
        
        # Create combined dataset
        combined_df = self.create_combined_dataset(merged_data)
        
        logger.info("Feature engineering pipeline completed successfully!")
        
        return combined_df

def create_meme_stock_features(reddit_data_path: str,
                             price_data_paths: Dict[str, str],
                             target_tickers: List[str] = ['GME', 'AMC', 'BB'],
                             date_range: Tuple[str, str] = ('2021-01-01', '2023-12-31'),
                             use_finbert: bool = False,
                             output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create comprehensive meme stock features.
    
    Args:
        reddit_data_path: Path to Reddit data
        price_data_paths: Dictionary mapping ticker to price data path
        target_tickers: List of tickers to analyze
        date_range: Date range for analysis
        use_finbert: Whether to use FinBERT
        output_path: Optional path to save results
        
    Returns:
        Complete feature dataframe
    """
    engine = MemeStockFeatureEngine(
        target_tickers=target_tickers,
        date_range=date_range,
        use_finbert=use_finbert
    )
    
    features_df = engine.engineer_all_features(reddit_data_path, price_data_paths)
    
    if output_path:
        features_df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
    
    return features_df

if __name__ == "__main__":
    # Example usage
    reddit_path = "data/processed/reddit/reddit_archive_daily_2021_2023_20250812_194025.csv"
    price_paths = {
        'GME': "data/raw/stocks/GME_extended_stock_data.csv",
        'AMC': "data/raw/stocks/AMC_extended_stock_data.csv", 
        'BB': "data/raw/stocks/BB_extended_stock_data.csv"
    }
    
    features_df = create_meme_stock_features(
        reddit_data_path=reddit_path,
        price_data_paths=price_paths,
        target_tickers=['GME', 'AMC', 'BB'],
        date_range=('2021-01-01', '2023-12-31'),
        use_finbert=False,
        output_path="data/features/meme_stock_features_complete.csv"
    )
    
    print(f"Created features dataset with {len(features_df)} records and {len(features_df.columns)} columns")
    print(f"Columns: {list(features_df.columns)}")

