"""
Cross-Modal Feature Engineering (10 features)
Creates interaction features between social and financial data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossModalFeatureEngineer:
    """
    Cross-modal feature engineering (10 features)
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_features(self, data: Dict) -> pd.DataFrame:
        """
        Generate cross-modal features (10 features)
        """
        logger.info("🔄 Generating cross-modal features...")
        
        # Get Reddit data
        reddit_data = data['reddit']
        stock_data = data['stocks']
        
        # Create daily Reddit aggregation
        daily_reddit = reddit_data.groupby('date').agg({
            'title': 'count',
            'score': ['sum', 'mean'],
            'comms_num': 'sum' if 'comms_num' in reddit_data.columns else 'count'
        }).reset_index()
        
        # Flatten column names
        daily_reddit.columns = ['date', 'reddit_posts', 'reddit_total_score', 'reddit_avg_score', 'reddit_comments']
        daily_reddit['date'] = pd.to_datetime(daily_reddit['date'])
        daily_reddit = daily_reddit.set_index('date')
        
        # Create base DataFrame
        features_df = daily_reddit.copy()
        
        # Generate cross-modal features
        features_df = self._add_sentiment_price_correlations(features_df, data)
        features_df = self._add_volume_mention_synchronization(features_df, data)
        features_df = self._add_prediction_lag_effects(features_df, data)
        features_df = self._add_feedback_effects(features_df, data)
        
        # Fill missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ Cross-modal features generated: {features_df.shape[1]} features")
        return features_df
    
    def _add_sentiment_price_correlations(self, features_df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """
        Add sentiment-price correlations (3 features)
        """
        logger.info("  Generating sentiment-price correlations...")
        
        reddit_data = data['reddit']
        stock_data = data['stocks']
        
        # Calculate sentiment for Reddit posts
        reddit_data['sentiment'] = reddit_data['title'].apply(self._calculate_sentiment)
        
        # Daily sentiment aggregation
        daily_sentiment = reddit_data.groupby('date')['sentiment'].mean()
        
        # Calculate correlations with stock returns
        for symbol in ['GME', 'AMC', 'BB']:
            if symbol in stock_data:
                stock_df = stock_data[symbol].copy()
                stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                stock_df['returns'] = stock_df['Close'].pct_change()
                stock_df = stock_df.set_index('Date')
                
                # Merge sentiment and returns
                merged = pd.DataFrame({
                    'sentiment': daily_sentiment,
                    'returns': stock_df['returns']
                }).dropna()
                
                if len(merged) > 10:  # Need sufficient data
                    # Rolling correlation
                    correlation = merged['sentiment'].rolling(7).corr(merged['returns'])
                    features_df[f'cross_modal_sentiment_{symbol}_corr_7d'] = correlation
                else:
                    features_df[f'cross_modal_sentiment_{symbol}_corr_7d'] = 0
        
        return features_df
    
    def _add_volume_mention_synchronization(self, features_df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """
        Add volume-mention synchronization (3 features)
        """
        logger.info("  Generating volume-mention synchronization...")
        
        reddit_data = data['reddit']
        stock_data = data['stocks']
        
        # Calculate mention counts for each stock
        for symbol in ['GME', 'AMC', 'BB']:
            # Count mentions in Reddit titles
            reddit_data[f'{symbol}_mentions'] = reddit_data['title'].str.contains(
                symbol, case=False, regex=False
            ).astype(int)
            
            # Daily mention aggregation
            daily_mentions = reddit_data.groupby('date')[f'{symbol}_mentions'].sum()
            
            if symbol in stock_data:
                stock_df = stock_data[symbol].copy()
                stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                stock_df = stock_df.set_index('Date')
                
                # Merge mentions and volume
                merged = pd.DataFrame({
                    'mentions': daily_mentions,
                    'volume': stock_df['Volume']
                }).dropna()
                
                if len(merged) > 10:
                    # Volume-mention correlation
                    correlation = merged['mentions'].rolling(5).corr(merged['volume'])
                    features_df[f'cross_modal_volume_mention_{symbol}_corr'] = correlation
                    
                    # Volume-mention ratio
                    features_df[f'cross_modal_volume_mention_{symbol}_ratio'] = (
                        merged['volume'] / (merged['mentions'] + 1)
                    )
                else:
                    features_df[f'cross_modal_volume_mention_{symbol}_corr'] = 0
                    features_df[f'cross_modal_volume_mention_{symbol}_ratio'] = 0
        
        return features_df
    
    def _add_prediction_lag_effects(self, features_df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """
        Add prediction lag effects (2 features)
        """
        logger.info("  Generating prediction lag effects...")
        
        reddit_data = data['reddit']
        stock_data = data['stocks']
        
        # Calculate sentiment momentum
        reddit_data['sentiment'] = reddit_data['title'].apply(self._calculate_sentiment)
        daily_sentiment = reddit_data.groupby('date')['sentiment'].mean()
        
        # Sentiment momentum (change in sentiment)
        sentiment_momentum = daily_sentiment.diff(1)
        
        # Predict future returns using sentiment
        for symbol in ['GME']:  # Focus on GME for lag effects
            if symbol in stock_data:
                stock_df = stock_data[symbol].copy()
                stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                stock_df['returns_1d_future'] = stock_df['Close'].pct_change(1).shift(-1)
                stock_df = stock_df.set_index('Date')
                
                # Merge sentiment momentum and future returns
                merged = pd.DataFrame({
                    'sentiment_momentum': sentiment_momentum,
                    'future_returns': stock_df['returns_1d_future']
                }).dropna()
                
                if len(merged) > 10:
                    # Correlation between sentiment momentum and future returns
                    correlation = merged['sentiment_momentum'].rolling(7).corr(merged['future_returns'])
                    features_df[f'cross_modal_sentiment_future_{symbol}_corr'] = correlation
                    
                    # Sentiment prediction power
                    features_df[f'cross_modal_sentiment_prediction_power'] = (
                        merged['sentiment_momentum'].rolling(7).std() * 
                        merged['future_returns'].rolling(7).std()
                    )
                else:
                    features_df[f'cross_modal_sentiment_future_{symbol}_corr'] = 0
                    features_df[f'cross_modal_sentiment_prediction_power'] = 0
        
        return features_df
    
    def _add_feedback_effects(self, features_df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """
        Add feedback effects (2 features)
        """
        logger.info("  Generating feedback effects...")
        
        reddit_data = data['reddit']
        stock_data = data['stocks']
        
        # Calculate price momentum
        for symbol in ['GME']:  # Focus on GME for feedback effects
            if symbol in stock_data:
                stock_df = stock_data[symbol].copy()
                stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                stock_df['price_momentum'] = stock_df['Close'].pct_change(1)
                stock_df = stock_df.set_index('Date')
                
                # Calculate sentiment response to price movements
                reddit_data['sentiment'] = reddit_data['title'].apply(self._calculate_sentiment)
                daily_sentiment = reddit_data.groupby('date')['sentiment'].mean()
                
                # Merge price momentum and sentiment
                merged = pd.DataFrame({
                    'price_momentum': stock_df['price_momentum'],
                    'sentiment': daily_sentiment
                }).dropna()
                
                if len(merged) > 10:
                    # Feedback correlation (price → sentiment)
                    feedback_corr = merged['price_momentum'].rolling(5).corr(merged['sentiment'])
                    features_df[f'cross_modal_price_sentiment_feedback'] = feedback_corr
                    
                    # Sentiment response to price volatility
                    price_volatility = merged['price_momentum'].rolling(7).std()
                    sentiment_response = merged['sentiment'].rolling(7).std()
                    features_df[f'cross_modal_volatility_sentiment_response'] = (
                        sentiment_response / (price_volatility + 1)
                    )
                else:
                    features_df[f'cross_modal_price_sentiment_feedback'] = 0
                    features_df[f'cross_modal_volatility_sentiment_response'] = 0
        
        return features_df
    
    def _calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score using TextBlob
        """
        try:
            if pd.isna(text) or text == '':
                return 0.0
            from textblob import TextBlob
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0


def main():
    """
    Test cross-modal feature engineering
    """
    logger.info("🚀 Testing Cross-Modal Feature Engineering...")
    
    # Load sample data
    from pathlib import Path
    data_dir = Path("data")
    
    # Load Reddit data
    reddit_file = data_dir / "raw" / "reddit_wsb.csv"
    if not reddit_file.exists():
        logger.error("❌ Reddit data not found for testing")
        return None
    
    reddit_data = pd.read_csv(reddit_file)
    reddit_data['created'] = pd.to_datetime(reddit_data['created'])
    reddit_data['date'] = reddit_data['created'].dt.date
    
    # Load stock data
    stock_symbols = ["GME", "AMC", "BB"]
    stock_data = {}
    
    for symbol in stock_symbols:
        stock_file = data_dir / "raw" / f"{symbol}_enhanced_stock_data.csv"
        if stock_file.exists():
            stock_df = pd.read_csv(stock_file)
            stock_data[symbol] = stock_df
        else:
            # Try original data
            original_file = data_dir / "raw" / f"{symbol}_stock_data.csv"
            if original_file.exists():
                stock_df = pd.read_csv(original_file)
                stock_data[symbol] = stock_df
    
    if stock_data:
        # Create sample data structure
        data = {'reddit': reddit_data, 'stocks': stock_data}
        
        # Initialize feature engineer
        engineer = CrossModalFeatureEngineer()
        
        # Generate features
        features = engineer.generate_features(data)
        
        logger.info(f"✅ Cross-modal features generated: {features.shape[1]} features")
        logger.info(f"✅ Feature shape: {features.shape}")
        logger.info(f"✅ Feature columns: {list(features.columns)}")
        
        return features
    else:
        logger.error("❌ No stock data found for testing")
        return None


if __name__ == "__main__":
    main() 