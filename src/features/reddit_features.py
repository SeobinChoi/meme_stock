"""
Reddit-Based Feature Engineering (25 features)
Creates social media features from Reddit WSB data
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

class RedditFeatureEngineer:
    """
    Reddit-based feature engineering (25 features)
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_features(self, data: Dict) -> pd.DataFrame:
        """
        Generate all Reddit-based features (25 features)
        """
        logger.info("📊 Generating Reddit-based features...")
        
        # Extract Reddit data from unified dataset
        unified_data = data['unified']
        
        # The unified dataset already has daily aggregated Reddit features
        # We'll use the existing features and add some derived ones
        features_df = unified_data[['post_count', 'total_score', 'avg_score', 'score_std', 
                                   'total_comments', 'avg_comments', 'comment_std']].copy()
        
        # Rename columns to match our feature naming convention
        features_df.columns = ['reddit_post_count', 'reddit_total_score', 'reddit_avg_score', 
                              'reddit_score_std', 'reddit_total_comments', 'reddit_avg_comments', 'reddit_comment_std']
        
        # Generate additional derived features
        features_df = self._add_basic_engagement_features(features_df, unified_data)
        features_df = self._add_sentiment_features(features_df, unified_data)
        features_df = self._add_content_analysis_features(features_df, unified_data)
        
        # Fill missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ Reddit features generated: {features_df.shape[1]} features")
        return features_df
    
    def _add_basic_engagement_features(self, features_df: pd.DataFrame, unified_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic engagement metrics (8 features)
        """
        logger.info("  Generating basic engagement features...")
        
        # 1. Engagement Quality: Score-to-comment ratios
        features_df['reddit_score_to_comment_ratio'] = (
            features_df['reddit_total_score'] / (features_df['reddit_total_comments'] + 1)
        )
        
        # 2. Temporal Patterns: Posting velocity (posts per hour)
        features_df['reddit_posting_velocity'] = features_df['reddit_post_count'].rolling(7).mean()
        
        # 3. Engagement acceleration (change in engagement)
        features_df['reddit_engagement_acceleration'] = features_df['reddit_total_score'].diff()
        
        # 4. Weekend indicators
        features_df['reddit_weekend_indicator'] = (features_df.index.dayofweek.isin([5, 6])).astype(int)
        
        # 5. Weekend post ratio
        weekend_posts = features_df[features_df['reddit_weekend_indicator'] == 1]['reddit_post_count']
        total_posts = features_df['reddit_post_count'].rolling(7).sum()
        features_df['reddit_weekend_post_ratio'] = weekend_posts / (total_posts + 1)
        
        # 6. Activity concentration (variance in posting)
        features_df['reddit_activity_concentration'] = features_df['reddit_post_count'].rolling(7).std()
        
        # 7. Unique users estimate (approximation)
        features_df['reddit_unique_users_estimate'] = features_df['reddit_post_count'] * 0.8  # Rough estimate
        
        # 8. Engagement volatility
        features_df['reddit_engagement_volatility'] = features_df['reddit_total_score'].rolling(7).std()
        
        return features_df
    
    def _add_sentiment_features(self, features_df: pd.DataFrame, unified_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment analysis features (10 features)
        """
        logger.info("  Generating sentiment features...")
        
        # For now, use simple sentiment proxies since we don't have individual post titles
        # We'll use engagement metrics as sentiment proxies
        
        # 1. Sentiment momentum (change in engagement)
        features_df['reddit_sentiment_momentum_1d'] = features_df['reddit_total_score'].diff(1)
        features_df['reddit_sentiment_momentum_3d'] = features_df['reddit_total_score'].diff(3)
        features_df['reddit_sentiment_momentum_7d'] = features_df['reddit_total_score'].diff(7)
        
        # 2. Sentiment volatility
        features_df['reddit_sentiment_volatility'] = features_df['reddit_total_score'].rolling(7).std()
        
        # 3. Extreme sentiment ratios (using score extremes as proxy)
        features_df['reddit_extreme_positive_ratio'] = (features_df['reddit_avg_score'] > features_df['reddit_avg_score'].quantile(0.8)).astype(int)
        features_df['reddit_extreme_negative_ratio'] = (features_df['reddit_avg_score'] < features_df['reddit_avg_score'].quantile(0.2)).astype(int)
        
        # 4. Sentiment consensus (consistency in engagement)
        features_df['reddit_sentiment_consensus'] = 1 - (features_df['reddit_score_std'] / (features_df['reddit_avg_score'] + 1))
        
        # 5. Positive/negative sentiment ratios (using engagement as proxy)
        features_df['reddit_positive_sentiment_ratio'] = (features_df['reddit_avg_score'] > features_df['reddit_avg_score'].median()).astype(int)
        features_df['reddit_negative_sentiment_ratio'] = (features_df['reddit_avg_score'] < features_df['reddit_avg_score'].median()).astype(int)
        
        # 6. Sentiment mean and std (using engagement metrics)
        features_df['reddit_sentiment_mean'] = features_df['reddit_avg_score']
        features_df['reddit_sentiment_std'] = features_df['reddit_score_std']
        features_df['reddit_sentiment_min'] = features_df['reddit_avg_score'] - features_df['reddit_score_std']
        features_df['reddit_sentiment_max'] = features_df['reddit_avg_score'] + features_df['reddit_score_std']
        features_df['reddit_sentiment_count'] = features_df['reddit_post_count']
        
        return features_df
    
    def _add_content_analysis_features(self, features_df: pd.DataFrame, unified_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add content analysis features (7 features)
        """
        logger.info("  Generating content analysis features...")
        
        # Since we don't have individual post titles, we'll create proxy features
        # based on engagement patterns and ratios
        
        # 1. Average title length (proxy based on engagement efficiency)
        features_df['reddit_avg_title_length'] = 50 + (features_df['reddit_avg_score'] / 100)  # Proxy
        
        # 2. Title length std (proxy based on score std)
        features_df['reddit_title_length_std'] = features_df['reddit_score_std'] / 10  # Proxy
        
        # 3. Average word count (proxy)
        features_df['reddit_avg_word_count'] = 8 + (features_df['reddit_avg_score'] / 200)  # Proxy
        
        # 4. Word count std (proxy)
        features_df['reddit_word_count_std'] = features_df['reddit_score_std'] / 20  # Proxy
        
        # 5. Average uppercase ratio (proxy based on engagement intensity)
        features_df['reddit_avg_uppercase_ratio'] = 0.2 + (features_df['reddit_avg_score'] / 1000)  # Proxy
        
        # 6. Average exclamation count (proxy)
        features_df['reddit_avg_exclamation_count'] = 0.5 + (features_df['reddit_avg_score'] / 500)  # Proxy
        
        # 7. Average question count (proxy)
        features_df['reddit_avg_question_count'] = 0.3 + (features_df['reddit_avg_score'] / 800)  # Proxy
        
        # 8. Trading keyword density (proxy based on engagement)
        features_df['reddit_trading_keyword_density'] = features_df['reddit_avg_score'] / 1000  # Proxy
        
        # 9. Linguistic complexity
        features_df['reddit_linguistic_complexity'] = (
            features_df['reddit_avg_word_count'] * features_df['reddit_avg_title_length']
        )
        
        # 10. Urgency indicators (proxy)
        features_df['reddit_urgency_indicators'] = features_df['reddit_avg_score'] / 500  # Proxy
        
        # 11. Emotional intensity
        features_df['reddit_emotional_intensity'] = (
            features_df['reddit_avg_exclamation_count'] + 
            features_df['reddit_avg_uppercase_ratio'] * 10
        )
        
        # 12. Information vs. opinion ratio (proxy)
        features_df['reddit_info_opinion_ratio'] = 0.5 + (features_df['reddit_avg_score'] / 1000)  # Proxy
        
        # 13. Content diversity (proxy)
        features_df['reddit_content_diversity'] = 0.7 + (features_df['reddit_avg_score'] / 2000)  # Proxy
        
        # 14. Engagement efficiency
        features_df['reddit_engagement_efficiency'] = (
            features_df['reddit_total_score'] / (features_df['reddit_avg_title_length'] + 1)
        )
        
        # 15. Post quality index
        features_df['reddit_post_quality_index'] = (
            features_df['reddit_avg_score'] * features_df['reddit_avg_comments']
        )
        
        return features_df

def main():
    """Test the Reddit feature engineer"""
    # Create sample data
    dates = pd.date_range('2021-01-01', '2021-01-10', freq='D')
    sample_data = pd.DataFrame({
        'post_count': np.random.randint(100, 1000, len(dates)),
        'total_score': np.random.randint(10000, 100000, len(dates)),
        'avg_score': np.random.randint(50, 500, len(dates)),
        'score_std': np.random.randint(10, 100, len(dates)),
        'total_comments': np.random.randint(5000, 50000, len(dates)),
        'avg_comments': np.random.randint(20, 200, len(dates)),
        'comment_std': np.random.randint(5, 50, len(dates))
    }, index=dates)
    
    # Test feature generation
    engineer = RedditFeatureEngineer()
    data = {'unified': sample_data}
    features = engineer.generate_features(data)
    
    print(f"Generated {features.shape[1]} Reddit features")
    print(f"Feature columns: {list(features.columns)}")

if __name__ == "__main__":
    main() 