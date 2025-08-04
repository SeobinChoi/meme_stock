"""
Temporal Feature Engineering (9 features)
Creates time-based features for market analysis
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

class TemporalFeatureEngineer:
    """
    Temporal feature engineering (9 features)
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_features(self, data: Dict) -> pd.DataFrame:
        """
        Generate temporal features (9 features)
        """
        logger.info("⏰ Generating temporal features...")
        
        # Get date range from Reddit data
        reddit_data = data['reddit']
        date_range = pd.date_range(
            start=reddit_data['date'].min(),
            end=reddit_data['date'].max(),
            freq='D'
        )
        
        # Create base DataFrame with all dates
        features_df = pd.DataFrame(index=date_range)
        features_df.index.name = 'date'
        
        # Generate temporal features
        features_df = self._add_calendar_features(features_df)
        features_df = self._add_market_session_features(features_df)
        features_df = self._add_seasonal_features(features_df)
        features_df = self._add_event_window_features(features_df)
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"✅ Temporal features generated: {features_df.shape[1]} features")
        return features_df
    
    def _add_calendar_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar effects (3 features)
        """
        logger.info("  Generating calendar features...")
        
        # 1. Day of week effects
        features_df['temporal_day_of_week'] = features_df.index.dayofweek
        
        # 2. Month effects
        features_df['temporal_month'] = features_df.index.month
        
        # 3. Holiday proximity effects (simplified)
        # Major US holidays in 2020-2021
        holidays_2020 = [
            '2020-01-01', '2020-01-20', '2020-02-17', '2020-04-10', '2020-05-25',
            '2020-07-03', '2020-09-07', '2020-10-12', '2020-11-11', '2020-11-26',
            '2020-12-25'
        ]
        holidays_2021 = [
            '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31',
            '2021-07-05', '2021-09-06', '2021-10-11', '2021-11-11', '2021-11-25',
            '2021-12-24'
        ]
        
        all_holidays = [pd.to_datetime(h) for h in holidays_2020 + holidays_2021]
        
        # Calculate days to nearest holiday
        def days_to_holiday(date):
            min_days = float('inf')
            for holiday in all_holidays:
                days = abs((date - holiday).days)
                min_days = min(min_days, days)
            return min_days
        
        features_df['temporal_days_to_holiday'] = features_df.index.map(days_to_holiday)
        
        return features_df
    
    def _add_market_session_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market session indicators (2 features)
        """
        logger.info("  Generating market session features...")
        
        # 1. Regular market hours indicator (simplified)
        # Assume regular market hours are Monday-Friday
        features_df['temporal_market_session'] = (
            features_df.index.dayofweek.isin([0, 1, 2, 3, 4]).astype(int)
        )
        
        # 2. Weekend indicator
        features_df['temporal_weekend_indicator'] = (
            features_df.index.dayofweek.isin([5, 6]).astype(int)
        )
        
        return features_df
    
    def _add_seasonal_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal patterns (2 features)
        """
        logger.info("  Generating seasonal features...")
        
        # 1. Quarterly earnings seasons
        # Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
        features_df['temporal_quarter'] = features_df.index.quarter
        
        # 2. Options expiration cycles (simplified)
        # Third Friday of each month
        def is_options_expiration(date):
            return date.day >= 15 and date.day <= 21 and date.dayofweek == 4  # Friday
        
        features_df['temporal_options_expiration'] = features_df.index.map(is_options_expiration).astype(int)
        
        return features_df
    
    def _add_event_window_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add event window features (2 features)
        """
        logger.info("  Generating event window features...")
        
        # 1. Month-end effects
        features_df['temporal_month_end'] = (
            features_df.index.day >= 25
        ).astype(int)
        
        # 2. Year-end effects
        features_df['temporal_year_end'] = (
            (features_df.index.month == 12) & (features_df.index.day >= 20)
        ).astype(int)
        
        return features_df


def main():
    """
    Test temporal feature engineering
    """
    logger.info("🚀 Testing Temporal Feature Engineering...")
    
    # Load sample data
    from pathlib import Path
    data_dir = Path("data")
    reddit_file = data_dir / "raw" / "reddit_wsb.csv"
    
    if reddit_file.exists():
        reddit_data = pd.read_csv(reddit_file)
        reddit_data['created'] = pd.to_datetime(reddit_data['created'])
        reddit_data['date'] = reddit_data['created'].dt.date
        
        # Create sample data structure
        data = {'reddit': reddit_data}
        
        # Initialize feature engineer
        engineer = TemporalFeatureEngineer()
        
        # Generate features
        features = engineer.generate_features(data)
        
        logger.info(f"✅ Temporal features generated: {features.shape[1]} features")
        logger.info(f"✅ Feature shape: {features.shape}")
        logger.info(f"✅ Feature columns: {list(features.columns)}")
        
        return features
    else:
        logger.error("❌ Reddit data not found for testing")
        return None


if __name__ == "__main__":
    main() 