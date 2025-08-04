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

# NLP imports for sentiment analysis
from textblob import TextBlob
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditFeatureEngineer:
    """
    Reddit-based feature engineering (25 features)
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_features(self, reddit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all Reddit-based features (25 features)
        """
        logger.info("📊 Generating Reddit-based features...")
        
        # Group by date for daily aggregation
        agg_dict = {
            'title': 'count',  # Post count
            'score': ['sum', 'mean', 'std']  # Engagement metrics
        }
        
        if 'comms_num' in reddit_data.columns:
            agg_dict['comms_num'] = ['sum', 'mean', 'std']
        
        daily_data = reddit_data.groupby('date').agg(agg_dict).reset_index()
        
        # Flatten column names
        if 'comms_num' in reddit_data.columns:
            daily_data.columns = ['date', 'reddit_post_count', 'reddit_total_score', 'reddit_avg_score', 'reddit_score_std', 'reddit_total_comments', 'reddit_avg_comments', 'reddit_comment_std']
        else:
            daily_data.columns = ['date', 'reddit_post_count', 'reddit_total_score', 'reddit_avg_score', 'reddit_score_std']
        
        # Set date as index
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data = daily_data.set_index('date').sort_index()
        
        # Generate all feature categories
        features_df = daily_data.copy()
        
        # 1. Basic Engagement Metrics (8 features)
        features_df = self._add_basic_engagement_features(features_df, reddit_data)
        
        # 2. Sentiment Analysis Features (10 features)
        features_df = self._add_sentiment_features(features_df, reddit_data)
        
        # 3. Content Analysis Features (7 features)
        features_df = self._add_content_analysis_features(features_df, reddit_data)
        
        # Fill missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ Reddit features generated: {features_df.shape[1]} features")
        return features_df
    
    def _add_basic_engagement_features(self, features_df: pd.DataFrame, reddit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic engagement metrics (8 features)
        """
        logger.info("  Generating basic engagement features...")
        
        # Calculate daily metrics
        daily_metrics = reddit_data.groupby('date').agg({
            'title': 'count',  # Already exists
            'score': ['sum', 'mean', 'std'],  # Already exists
            'comms_num': ['sum', 'mean', 'std'] if 'comms_num' in reddit_data.columns else 'sum'
        })
        
        # 1. Engagement Quality: Score-to-comment ratios
        if 'comms_num' in reddit_data.columns:
            features_df['reddit_score_to_comment_ratio'] = (
                features_df['reddit_total_score'] / (features_df['reddit_total_comments'] + 1)
            )
        else:
            features_df['reddit_score_to_comment_ratio'] = features_df['reddit_total_score']
        
        # 2. Temporal Patterns: Posting velocity (posts per hour)
        features_df['reddit_posting_velocity'] = features_df['reddit_post_count'].rolling(7).mean()
        
        # 3. Engagement acceleration (change in engagement)
        features_df['reddit_engagement_acceleration'] = features_df['reddit_total_score'].diff()
        
        # 4. Weekend effects
        features_df['reddit_weekend_indicator'] = features_df.index.dayofweek.isin([5, 6]).astype(int)
        features_df['reddit_weekend_post_ratio'] = (
            features_df['reddit_post_count'] * features_df['reddit_weekend_indicator']
        )
        
        # 5. Activity concentration (Gini coefficient approximation)
        features_df['reddit_activity_concentration'] = self._calculate_activity_concentration(reddit_data)
        
        # 6. Unique user count (approximation)
        features_df['reddit_unique_users_estimate'] = features_df['reddit_post_count'] * 0.8  # Approximation
        
        # 7. Engagement volatility
        features_df['reddit_engagement_volatility'] = features_df['reddit_total_score'].rolling(7).std()
        
        # 8. Post quality index
        features_df['reddit_post_quality_index'] = (
            features_df['reddit_avg_score'] * features_df['reddit_avg_comments']
        )
        
        return features_df
    
    def _add_sentiment_features(self, features_df: pd.DataFrame, reddit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment analysis features (10 features)
        """
        logger.info("  Generating sentiment analysis features...")
        
        # Calculate sentiment for each post
        reddit_data['sentiment'] = reddit_data['title'].apply(self._calculate_sentiment)
        
        # Daily sentiment aggregation
        daily_sentiment = reddit_data.groupby('date')['sentiment'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).rename(columns={
            'mean': 'reddit_sentiment_mean',
            'std': 'reddit_sentiment_std',
            'min': 'reddit_sentiment_min',
            'max': 'reddit_sentiment_max',
            'count': 'reddit_sentiment_count'
        })
        
        # Merge with features
        features_df = features_df.merge(daily_sentiment, left_index=True, right_index=True, how='left')
        
        # 1. Polarity metrics
        features_df['reddit_positive_sentiment_ratio'] = (
            reddit_data.groupby('date')['sentiment'].apply(lambda x: (x > 0.1).mean())
        )
        features_df['reddit_negative_sentiment_ratio'] = (
            reddit_data.groupby('date')['sentiment'].apply(lambda x: (x < -0.1).mean())
        )
        
        # 2. Sentiment momentum (rate of change)
        features_df['reddit_sentiment_momentum_1d'] = features_df['reddit_sentiment_mean'].diff(1)
        features_df['reddit_sentiment_momentum_3d'] = features_df['reddit_sentiment_mean'].diff(3)
        features_df['reddit_sentiment_momentum_7d'] = features_df['reddit_sentiment_mean'].diff(7)
        
        # 3. Sentiment volatility
        features_df['reddit_sentiment_volatility'] = features_df['reddit_sentiment_std'].rolling(7).mean()
        
        # 4. Extreme sentiment
        features_df['reddit_extreme_positive_ratio'] = (
            reddit_data.groupby('date')['sentiment'].apply(lambda x: (x > 0.5).mean())
        )
        features_df['reddit_extreme_negative_ratio'] = (
            reddit_data.groupby('date')['sentiment'].apply(lambda x: (x < -0.5).mean())
        )
        
        # 5. Sentiment consensus (inverse of standard deviation)
        features_df['reddit_sentiment_consensus'] = 1 / (features_df['reddit_sentiment_std'] + 1)
        
        return features_df
    
    def _add_content_analysis_features(self, features_df: pd.DataFrame, reddit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add content analysis features (7 features)
        """
        logger.info("  Generating content analysis features...")
        
        # Calculate text-based features
        reddit_data['title_length'] = reddit_data['title'].str.len()
        reddit_data['word_count'] = reddit_data['title'].str.split().str.len()
        reddit_data['uppercase_ratio'] = reddit_data['title'].str.count(r'[A-Z]') / reddit_data['title'].str.len()
        reddit_data['exclamation_count'] = reddit_data['title'].str.count(r'!')
        reddit_data['question_count'] = reddit_data['title'].str.count(r'\?')
        
        # Daily aggregation
        daily_content = reddit_data.groupby('date').agg({
            'title_length': ['mean', 'std'],
            'word_count': ['mean', 'std'],
            'uppercase_ratio': 'mean',
            'exclamation_count': 'mean',
            'question_count': 'mean'
        })
        
        # Flatten column names
        daily_content.columns = [
            'reddit_avg_title_length', 'reddit_title_length_std',
            'reddit_avg_word_count', 'reddit_word_count_std',
            'reddit_avg_uppercase_ratio', 'reddit_avg_exclamation_count', 'reddit_avg_question_count'
        ]
        
        # Merge with features
        features_df = features_df.merge(daily_content, left_index=True, right_index=True, how='left')
        
        # 1. Keyword density (trading-related terms)
        trading_keywords = ['buy', 'sell', 'hold', 'short', 'long', 'diamond', 'hands', 'moon', 'rocket', 'tendies']
        features_df['reddit_trading_keyword_density'] = self._calculate_keyword_density(reddit_data, trading_keywords)
        
        # 2. Linguistic complexity
        features_df['reddit_linguistic_complexity'] = (
            features_df['reddit_avg_word_count'] * features_df['reddit_avg_title_length']
        )
        
        # 3. Urgency indicators
        urgency_keywords = ['now', 'today', 'urgent', 'quick', 'fast', 'immediate', 'deadline']
        features_df['reddit_urgency_indicators'] = self._calculate_keyword_density(reddit_data, urgency_keywords)
        
        # 4. Emotional intensity
        features_df['reddit_emotional_intensity'] = (
            features_df['reddit_avg_exclamation_count'] + 
            features_df['reddit_avg_uppercase_ratio'] * 10
        )
        
        # 5. Information vs. opinion ratio (approximation)
        info_keywords = ['earnings', 'revenue', 'profit', 'loss', 'financial', 'report', 'data']
        opinion_keywords = ['think', 'believe', 'feel', 'hope', 'wish', 'maybe', 'probably']
        
        features_df['reddit_info_opinion_ratio'] = (
            self._calculate_keyword_density(reddit_data, info_keywords) /
            (self._calculate_keyword_density(reddit_data, opinion_keywords) + 1)
        )
        
        # 6. Content diversity (unique words ratio)
        features_df['reddit_content_diversity'] = self._calculate_content_diversity(reddit_data)
        
        # 7. Engagement efficiency
        features_df['reddit_engagement_efficiency'] = (
            features_df['reddit_total_score'] / (features_df['reddit_avg_title_length'] + 1)
        )
        
        return features_df
    
    def _calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score using TextBlob
        """
        try:
            if pd.isna(text) or text == '':
                return 0.0
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
    
    def _calculate_activity_concentration(self, reddit_data: pd.DataFrame) -> pd.Series:
        """
        Calculate activity concentration (Gini coefficient approximation)
        """
        # Simplified Gini calculation based on score distribution
        daily_concentration = reddit_data.groupby('date')['score'].apply(
            lambda x: 1 - (x.var() / (x.mean() ** 2 + 1)) if x.mean() > 0 else 0
        )
        return daily_concentration
    
    def _calculate_keyword_density(self, reddit_data: pd.DataFrame, keywords: List[str]) -> pd.Series:
        """
        Calculate keyword density for given keywords
        """
        def count_keywords(text):
            if pd.isna(text):
                return 0
            text_lower = str(text).lower()
            return sum(1 for keyword in keywords if keyword in text_lower)
        
        reddit_data['keyword_count'] = reddit_data['title'].apply(count_keywords)
        daily_density = reddit_data.groupby('date')['keyword_count'].mean()
        return daily_density
    
    def _calculate_content_diversity(self, reddit_data: pd.DataFrame) -> pd.Series:
        """
        Calculate content diversity (unique words ratio)
        """
        def unique_words_ratio(text):
            if pd.isna(text):
                return 0
            words = str(text).lower().split()
            if len(words) == 0:
                return 0
            return len(set(words)) / len(words)
        
        reddit_data['diversity_ratio'] = reddit_data['title'].apply(unique_words_ratio)
        daily_diversity = reddit_data.groupby('date')['diversity_ratio'].mean()
        return daily_diversity


def main():
    """
    Test Reddit feature engineering
    """
    logger.info("🚀 Testing Reddit Feature Engineering...")
    
    # Load sample data
    from pathlib import Path
    data_dir = Path("data")
    reddit_file = data_dir / "raw" / "reddit_wsb.csv"
    
    if reddit_file.exists():
        reddit_data = pd.read_csv(reddit_file)
        reddit_data['created'] = pd.to_datetime(reddit_data['created'])
        reddit_data['date'] = reddit_data['created'].dt.date
        
        # Initialize feature engineer
        engineer = RedditFeatureEngineer()
        
        # Generate features
        features = engineer.generate_features(reddit_data)
        
        logger.info(f"✅ Reddit features generated: {features.shape[1]} features")
        logger.info(f"✅ Feature shape: {features.shape}")
        logger.info(f"✅ Feature columns: {list(features.columns)}")
        
        return features
    else:
        logger.error("❌ Reddit data not found for testing")
        return None


if __name__ == "__main__":
    main() 