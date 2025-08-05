"""
Cross-Modal Feature Engineering (10 features)
Creates features that capture interactions between social and financial data
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
        
        # Get unified data
        unified_data = data['unified']
        stock_data = data['stocks']
        
        # Use existing Reddit aggregation from unified data
        features_df = unified_data[['post_count', 'total_score', 'avg_score', 'total_comments']].copy()
        features_df.columns = ['reddit_posts', 'reddit_total_score', 'reddit_avg_score', 'reddit_comments']
        
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
        
        unified_data = data['unified']
        stock_data = data['stocks']
        
        # Use engagement as sentiment proxy
        daily_sentiment = unified_data['avg_score']
        
        # Calculate correlations with stock returns
        for symbol in ['GME', 'AMC', 'BB']:
            # Get stock data
            if symbol in stock_data and stock_data[symbol] is not None:
                stock_df = stock_data[symbol].copy()
                stock_df['returns'] = stock_df['Close'].pct_change()
                
                # Merge sentiment and returns
                merged = pd.DataFrame({
                    'sentiment': daily_sentiment,
                    'returns': stock_df['returns']
                }, index=unified_data.index).dropna()
                
                if len(merged) > 10:  # Need sufficient data
                    # Rolling correlation
                    correlation = merged['sentiment'].rolling(7).corr(merged['returns'])
                    features_df[f'cross_modal_sentiment_{symbol}_corr_7d'] = correlation
                else:
                    features_df[f'cross_modal_sentiment_{symbol}_corr_7d'] = 0
            else:
                features_df[f'cross_modal_sentiment_{symbol}_corr_7d'] = 0
        
        return features_df
    
    def _add_volume_mention_synchronization(self, features_df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """
        Add volume-mention synchronization (3 features)
        """
        logger.info("  Generating volume-mention synchronization...")
        
        unified_data = data['unified']
        stock_data = data['stocks']
        
        # Use post count as mention proxy
        daily_mentions = unified_data['post_count']
        
        # Calculate correlations with stock volume
        for symbol in ['GME', 'AMC', 'BB']:
            # Get stock data
            if symbol in stock_data and stock_data[symbol] is not None:
                stock_df = stock_data[symbol].copy()
                stock_df['volume'] = stock_df['Volume']
                
                # Merge mentions and volume
                merged = pd.DataFrame({
                    'mentions': daily_mentions,
                    'volume': stock_df['volume']
                }, index=unified_data.index).dropna()
                
                if len(merged) > 10:
                    # Rolling correlation
                    correlation = merged['mentions'].rolling(7).corr(merged['volume'])
                    features_df[f'cross_modal_volume_mention_{symbol}_corr'] = correlation
                    
                    # Volume-mention ratio
                    features_df[f'cross_modal_volume_mention_{symbol}_ratio'] = (
                        merged['volume'] / (merged['mentions'] + 1)
                    )
                else:
                    features_df[f'cross_modal_volume_mention_{symbol}_corr'] = 0
                    features_df[f'cross_modal_volume_mention_{symbol}_ratio'] = 0
            else:
                features_df[f'cross_modal_volume_mention_{symbol}_corr'] = 0
                features_df[f'cross_modal_volume_mention_{symbol}_ratio'] = 0
        
        return features_df
    
    def _add_prediction_lag_effects(self, features_df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """
        Add prediction lag effects (2 features)
        """
        logger.info("  Generating prediction lag effects...")
        
        unified_data = data['unified']
        stock_data = data['stocks']
        
        # Use engagement momentum as sentiment momentum proxy
        sentiment_momentum = unified_data['avg_score'].diff(1)
        
        # Predict future returns using sentiment
        for symbol in ['GME']:  # Focus on GME for lag effects
            if symbol in stock_data and stock_data[symbol] is not None:
                stock_df = stock_data[symbol].copy()
                stock_df['returns_1d_future'] = stock_df['Close'].pct_change(1).shift(-1)
                
                # Merge sentiment momentum and future returns
                merged = pd.DataFrame({
                    'sentiment_momentum': sentiment_momentum,
                    'future_returns': stock_df['returns_1d_future']
                }, index=unified_data.index).dropna()
                
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
            else:
                features_df[f'cross_modal_sentiment_future_{symbol}_corr'] = 0
                features_df[f'cross_modal_sentiment_prediction_power'] = 0
        
        return features_df
    
    def _add_feedback_effects(self, features_df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """
        Add feedback effects (2 features)
        """
        logger.info("  Generating feedback effects...")
        
        unified_data = data['unified']
        stock_data = data['stocks']
        
        # Use price momentum as proxy for market sentiment
        for symbol in ['GME']:  # Focus on GME for feedback effects
            if symbol in stock_data and stock_data[symbol] is not None:
                stock_df = stock_data[symbol].copy()
                stock_df['price_momentum'] = stock_df['Close'].pct_change(1)
                
                # Calculate subsequent social sentiment (using engagement as proxy)
                subsequent_sentiment = unified_data['avg_score'].shift(-1)
                
                # Merge price momentum and subsequent sentiment
                merged = pd.DataFrame({
                    'price_momentum': stock_df['price_momentum'],
                    'subsequent_sentiment': subsequent_sentiment
                }, index=unified_data.index).dropna()
                
                if len(merged) > 10:
                    # Price-sentiment feedback correlation
                    correlation = merged['price_momentum'].rolling(7).corr(merged['subsequent_sentiment'])
                    features_df[f'cross_modal_price_sentiment_feedback'] = correlation
                    
                    # Volatility-sentiment response
                    price_volatility = merged['price_momentum'].rolling(7).std()
                    sentiment_response = merged['subsequent_sentiment'].rolling(7).std()
                    features_df[f'cross_modal_volatility_sentiment_response'] = (
                        price_volatility * sentiment_response
                    )
                else:
                    features_df[f'cross_modal_price_sentiment_feedback'] = 0
                    features_df[f'cross_modal_volatility_sentiment_response'] = 0
            else:
                features_df[f'cross_modal_price_sentiment_feedback'] = 0
                features_df[f'cross_modal_volatility_sentiment_response'] = 0
        
        return features_df

def main():
    """Test the cross-modal feature engineer"""
    # Create sample data
    dates = pd.date_range('2021-01-01', '2021-01-10', freq='D')
    sample_unified = pd.DataFrame({
        'post_count': np.random.randint(100, 1000, len(dates)),
        'total_score': np.random.randint(10000, 100000, len(dates)),
        'avg_score': np.random.randint(50, 500, len(dates)),
        'total_comments': np.random.randint(5000, 50000, len(dates))
    }, index=dates)
    
    sample_stock = pd.DataFrame({
        'Close': np.random.uniform(10, 100, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Test feature generation
    engineer = CrossModalFeatureEngineer()
    data = {
        'unified': sample_unified,
        'stocks': {'GME': sample_stock}
    }
    features = engineer.generate_features(data)
    
    print(f"Generated {features.shape[1]} cross-modal features")
    print(f"Feature columns: {list(features.columns)}")

if __name__ == "__main__":
    main() 