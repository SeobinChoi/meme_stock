#!/usr/bin/env python3
"""
Reddit dynamics feature engineering for meme stock analysis.

Implements Reddit Surprise (RS), EMAs, momentum, and other dynamics features
as described in the manual for contrarian effect prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class RedditDynamicsEngine:
    """Engine for calculating Reddit dynamics features."""
    
    def __init__(self, 
                 ema_windows: List[int] = [3, 5, 10],
                 momentum_windows: List[int] = [3, 7],
                 volatility_windows: List[int] = [5, 10, 20],
                 surprise_window: int = 7):
        """
        Initialize Reddit dynamics engine.
        
        Args:
            ema_windows: Windows for EMA calculations
            momentum_windows: Windows for momentum calculations  
            volatility_windows: Windows for volatility calculations
            surprise_window: Window for Reddit Surprise calculation
        """
        self.ema_windows = ema_windows
        self.momentum_windows = momentum_windows
        self.volatility_windows = volatility_windows
        self.surprise_window = surprise_window
    
    def calculate_ema(self, series: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """Calculate Exponential Moving Average."""
        if alpha is None:
            alpha = 2.0 / (window + 1)
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    def calculate_reddit_surprise(self, mentions: pd.Series, window: int = None) -> pd.Series:
        """
        Calculate Reddit Surprise (RS) - deviation from expected level.
        
        RS_t = (Mentions_t - MA_t) / MA_t
        
        Args:
            mentions: Series of mention counts
            window: Window for moving average (default: self.surprise_window)
            
        Returns:
            Series of Reddit Surprise values
        """
        if window is None:
            window = self.surprise_window
        
        ma = mentions.rolling(window=window, min_periods=1).mean()
        
        # Avoid division by zero
        rs = (mentions - ma) / ma.replace(0, np.nan)
        rs = rs.fillna(0)
        
        return rs
    
    def calculate_momentum(self, mentions: pd.Series, window: int) -> pd.Series:
        """
        Calculate momentum as difference from lag.
        
        Momentum_t = Mentions_t - Mentions_{t-window}
        """
        return mentions - mentions.shift(window)
    
    def calculate_velocity(self, mentions: pd.Series) -> pd.Series:
        """Calculate mention velocity (first derivative)."""
        return mentions.diff()
    
    def calculate_acceleration(self, mentions: pd.Series) -> pd.Series:
        """Calculate mention acceleration (second derivative)."""
        velocity = self.calculate_velocity(mentions)
        return velocity.diff()
    
    def calculate_volatility(self, mentions: pd.Series, window: int) -> pd.Series:
        """Calculate rolling volatility of mentions."""
        return mentions.rolling(window=window, min_periods=1).std()
    
    def calculate_skewness(self, mentions: pd.Series, window: int) -> pd.Series:
        """Calculate rolling skewness of mentions."""
        return mentions.rolling(window=window, min_periods=3).skew()
    
    def calculate_kurtosis(self, mentions: pd.Series, window: int) -> pd.Series:
        """Calculate rolling kurtosis of mentions."""
        return mentions.rolling(window=window, min_periods=4).kurt()
    
    def calculate_percentile_rank(self, mentions: pd.Series, window: int) -> pd.Series:
        """Calculate percentile rank within rolling window."""
        return mentions.rolling(window=window, min_periods=1).rank(pct=True)
    
    def calculate_z_score(self, mentions: pd.Series, window: int) -> pd.Series:
        """Calculate z-score within rolling window."""
        rolling_mean = mentions.rolling(window=window, min_periods=1).mean()
        rolling_std = mentions.rolling(window=window, min_periods=1).std()
        
        z_score = (mentions - rolling_mean) / rolling_std.replace(0, np.nan)
        return z_score.fillna(0)
    
    def calculate_spike_indicator(self, mentions: pd.Series, threshold: float = 2.0) -> pd.Series:
        """
        Calculate spike indicator based on z-score threshold.
        
        Args:
            mentions: Series of mention counts
            threshold: Z-score threshold for spike detection
            
        Returns:
            Binary series indicating spikes
        """
        z_scores = self.calculate_z_score(mentions, window=20)
        return (z_scores > threshold).astype(int)
    
    def calculate_engagement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate engagement-based metrics.
        
        Args:
            df: DataFrame with columns ['mentions', 'score', 'comments', 'authors']
            
        Returns:
            DataFrame with additional engagement metrics
        """
        result_df = df.copy()
        
        # Engagement ratios
        if 'score' in df.columns and 'mentions' in df.columns:
            result_df['score_per_mention'] = df['score'] / df['mentions'].replace(0, np.nan)
            result_df['score_per_mention'] = result_df['score_per_mention'].fillna(0)
        
        if 'comments' in df.columns and 'mentions' in df.columns:
            result_df['comments_per_mention'] = df['comments'] / df['mentions'].replace(0, np.nan)
            result_df['comments_per_mention'] = result_df['comments_per_mention'].fillna(0)
        
        if 'authors' in df.columns and 'mentions' in df.columns:
            result_df['authors_per_mention'] = df['authors'] / df['mentions'].replace(0, np.nan)
            result_df['authors_per_mention'] = result_df['authors_per_mention'].fillna(0)
        
        # Engagement diversity
        if 'authors' in df.columns and 'mentions' in df.columns:
            result_df['engagement_diversity'] = df['authors'] / df['mentions'].replace(0, np.nan)
            result_df['engagement_diversity'] = result_df['engagement_diversity'].fillna(0)
        
        return result_df
    
    def calculate_all_dynamics(self, df: pd.DataFrame, 
                             mentions_col: str = 'mentions',
                             group_col: str = 'ticker') -> pd.DataFrame:
        """
        Calculate all Reddit dynamics features.
        
        Args:
            df: DataFrame with mention data
            mentions_col: Column name for mention counts
            group_col: Column to group by (e.g., ticker)
            
        Returns:
            DataFrame with all dynamics features
        """
        logger.info(f"Calculating Reddit dynamics features for {len(df)} records")
        
        result_df = df.copy()
        
        # Sort by group and date
        if 'date' in df.columns:
            result_df = result_df.sort_values([group_col, 'date'])
        
        # Group by ticker and calculate features
        features_list = []
        
        for ticker, group in result_df.groupby(group_col):
            ticker_features = group.copy()
            
            mentions = group[mentions_col]
            
            # Log volume
            ticker_features['log_mentions'] = np.log1p(mentions)
            
            # EMAs
            for window in self.ema_windows:
                ticker_features[f'ema_{window}'] = self.calculate_ema(mentions, window)
                ticker_features[f'ema_{window}_ratio'] = mentions / ticker_features[f'ema_{window}'].replace(0, np.nan)
                ticker_features[f'ema_{window}_ratio'] = ticker_features[f'ema_{window}_ratio'].fillna(1)
            
            # Momentum
            for window in self.momentum_windows:
                ticker_features[f'momentum_{window}'] = self.calculate_momentum(mentions, window)
                ticker_features[f'momentum_{window}_pct'] = ticker_features[f'momentum_{window}'] / mentions.replace(0, np.nan)
                ticker_features[f'momentum_{window}_pct'] = ticker_features[f'momentum_{window}_pct'].fillna(0)
            
            # Reddit Surprise
            ticker_features['reddit_surprise'] = self.calculate_reddit_surprise(mentions)
            ticker_features['reddit_surprise_abs'] = np.abs(ticker_features['reddit_surprise'])
            
            # Velocity and acceleration
            ticker_features['velocity'] = self.calculate_velocity(mentions)
            ticker_features['acceleration'] = self.calculate_acceleration(mentions)
            
            # Volatility
            for window in self.volatility_windows:
                ticker_features[f'volatility_{window}'] = self.calculate_volatility(mentions, window)
                ticker_features[f'volatility_{window}_pct'] = ticker_features[f'volatility_{window}'] / mentions.replace(0, np.nan)
                ticker_features[f'volatility_{window}_pct'] = ticker_features[f'volatility_{window}_pct'].fillna(0)
            
            # Statistical features
            ticker_features['z_score_20'] = self.calculate_z_score(mentions, 20)
            ticker_features['percentile_rank_20'] = self.calculate_percentile_rank(mentions, 20)
            ticker_features['skewness_20'] = self.calculate_skewness(mentions, 20)
            ticker_features['kurtosis_20'] = self.calculate_kurtosis(mentions, 20)
            
            # Spike indicators
            ticker_features['is_spike'] = self.calculate_spike_indicator(mentions, threshold=2.0)
            ticker_features['is_mega_spike'] = self.calculate_spike_indicator(mentions, threshold=3.0)
            
            # Trend features
            ticker_features['trend_5'] = mentions.rolling(5).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0)
            ticker_features['trend_10'] = mentions.rolling(10).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0)
            
            features_list.append(ticker_features)
        
        # Combine all ticker features
        result_df = pd.concat(features_list, ignore_index=True)
        
        # Calculate engagement metrics if available
        engagement_cols = ['score', 'comments', 'authors']
        if any(col in df.columns for col in engagement_cols):
            result_df = self.calculate_engagement_metrics(result_df)
        
        logger.info(f"Reddit dynamics features calculated. Features added: {len(result_df.columns) - len(df.columns)}")
        
        return result_df

def calculate_reddit_dynamics_features(df: pd.DataFrame,
                                     mentions_col: str = 'mentions',
                                     group_col: str = 'ticker',
                                     **kwargs) -> pd.DataFrame:
    """
    Calculate Reddit dynamics features for a dataframe.
    
    Args:
        df: DataFrame with mention data
        mentions_col: Column name for mention counts
        group_col: Column to group by
        **kwargs: Additional arguments for RedditDynamicsEngine
        
    Returns:
        DataFrame with additional dynamics features
    """
    engine = RedditDynamicsEngine(**kwargs)
    return engine.calculate_all_dynamics(df, mentions_col, group_col)

def aggregate_daily_dynamics(df: pd.DataFrame,
                            group_cols: List[str] = ['date', 'ticker']) -> pd.DataFrame:
    """
    Aggregate dynamics features by date and ticker.
    
    Args:
        df: DataFrame with dynamics features
        group_cols: Columns to group by
        
    Returns:
        Aggregated DataFrame with daily dynamics metrics
    """
    logger.info(f"Aggregating dynamics features by {group_cols}")
    
    # Identify dynamics columns
    dynamics_cols = [col for col in df.columns if any(x in col for x in 
                   ['ema', 'momentum', 'reddit_surprise', 'velocity', 'acceleration',
                    'volatility', 'z_score', 'percentile', 'skewness', 'kurtosis',
                    'trend', 'log_mentions'])]
    
    agg_dict = {}
    for col in dynamics_cols:
        if col in df.columns:
            agg_dict[col] = ['mean', 'std', 'min', 'max']
    
    # Special aggregation for binary features
    binary_cols = [col for col in df.columns if col.startswith('is_')]
    for col in binary_cols:
        if col in df.columns:
            agg_dict[col] = ['sum', 'mean']
    
    # Add count
    agg_dict['mentions'] = ['sum', 'count']
    
    # Aggregate
    agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns]
    
    # Rename key columns
    rename_dict = {
        f"{group_cols[0]}_": group_cols[0],
        f"{group_cols[1]}_": group_cols[1] if len(group_cols) > 1 else None
    }
    agg_df = agg_df.rename(columns={k: v for k, v in rename_dict.items() if v})
    
    logger.info(f"Aggregated to {len(agg_df)} daily records")
    
    return agg_df

if __name__ == "__main__":
    # Test Reddit dynamics calculation
    np.random.seed(42)
    
    # Create sample data
    dates = pd.date_range('2021-01-01', periods=100, freq='D')
    tickers = ['GME', 'AMC', 'BB']
    
    data = []
    for ticker in tickers:
        # Simulate mention patterns with spikes
        base_mentions = np.random.poisson(100, 100)
        spikes = np.random.choice(range(100), size=5, replace=False)
        base_mentions[spikes] *= 10
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'ticker': ticker,
                'mentions': base_mentions[i],
                'score': np.random.poisson(base_mentions[i] * 0.5),
                'comments': np.random.poisson(base_mentions[i] * 0.3),
                'authors': np.random.poisson(base_mentions[i] * 0.1)
            })
    
    df = pd.DataFrame(data)
    
    # Calculate dynamics features
    engine = RedditDynamicsEngine()
    result_df = engine.calculate_all_dynamics(df)
    
    print("Sample Reddit dynamics features:")
    print(result_df[['date', 'ticker', 'mentions', 'reddit_surprise', 'is_spike', 'z_score_20']].head(10))

