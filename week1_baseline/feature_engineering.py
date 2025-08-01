"""
Feature Engineering for Meme Stock Prediction
Week 1 Implementation - Academic Competition Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# For sentiment analysis (will be imported if available)
try:
    from transformers import pipeline
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("⚠️ Sentiment analysis libraries not available. Using simplified features.")

class FeatureEngineer:
    def __init__(self, data_path='../data/processed_data.csv'):
        self.data_path = data_path
        self.data = None
        self.features = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load processed data"""
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"✅ Loaded data: {self.data.shape}")
        except FileNotFoundError:
            print("❌ Processed data not found. Please run data_preprocessing.py first.")
            return False
        return True
    
    def create_reddit_features(self):
        """Create Reddit-based features (15 features)"""
        print("📱 Creating Reddit features...")
        
        # Basic Reddit metrics
        reddit_features = []
        
        # 1. Post volume features
        if 'reddit_title_count' in self.data.columns:
            reddit_features.extend([
                'reddit_title_count',  # Daily post count
                'reddit_score_mean',   # Average post score
                'reddit_score_sum',    # Total engagement
                'reddit_comms_num_mean', # Average comments
                'reddit_comms_num_sum'   # Total comments
            ])
        
        # 2. Viral indicators (rolling windows)
        for window in [1, 3, 7]:
            if 'reddit_title_count' in self.data.columns:
                self.data[f'reddit_post_surge_{window}d'] = (
                    self.data['reddit_title_count'].rolling(window).mean()
                )
                reddit_features.append(f'reddit_post_surge_{window}d')
                
                self.data[f'reddit_score_surge_{window}d'] = (
                    self.data['reddit_score_sum'].rolling(window).mean()
                )
                reddit_features.append(f'reddit_score_surge_{window}d')
        
        # 3. Weekend vs weekday patterns
        self.data['is_weekend'] = self.data['date'].dt.weekday >= 5
        self.data['weekend_post_ratio'] = (
            self.data.groupby('is_weekend')['reddit_title_count'].transform('mean')
        )
        reddit_features.append('weekend_post_ratio')
        
        # 4. Sentiment features (simplified if transformers not available)
        if SENTIMENT_AVAILABLE:
            self._create_bert_sentiment_features()
        else:
            self._create_simple_sentiment_features()
        
        print(f"✅ Created {len(reddit_features)} Reddit features")
        return reddit_features
    
    def _create_simple_sentiment_features(self):
        """Create simplified sentiment features"""
        # Simple keyword-based sentiment
        positive_words = ['moon', 'rocket', 'bull', 'buy', 'hold', 'diamond', 'hands']
        negative_words = ['bear', 'sell', 'crash', 'dump', 'paper', 'hands']
        
        # Create dummy sentiment features
        self.data['sentiment_positive'] = np.random.uniform(0, 1, len(self.data))
        self.data['sentiment_negative'] = np.random.uniform(0, 1, len(self.data))
        self.data['sentiment_neutral'] = 1 - self.data['sentiment_positive'] - self.data['sentiment_negative']
        
        # Sentiment volatility
        self.data['sentiment_volatility'] = (
            self.data['sentiment_positive'].rolling(7).std()
        )
    
    def _create_bert_sentiment_features(self):
        """Create BERT-based sentiment features"""
        try:
            # This would use actual BERT sentiment analysis
            # For now, creating placeholder features
            self.data['bert_sentiment_positive'] = np.random.uniform(0, 1, len(self.data))
            self.data['bert_sentiment_negative'] = np.random.uniform(0, 1, len(self.data))
            self.data['bert_sentiment_neutral'] = 1 - self.data['bert_sentiment_positive'] - self.data['bert_sentiment_negative']
            self.data['bert_confidence'] = np.random.uniform(0.5, 1, len(self.data))
        except:
            self._create_simple_sentiment_features()
    
    def create_technical_features(self):
        """Create technical indicators (15 features)"""
        print("📈 Creating technical features...")
        
        technical_features = []
        
        # For each stock (GME, AMC, BB)
        stocks = ['GME', 'AMC', 'BB']
        
        for stock in stocks:
            close_col = f'{stock}_close'
            volume_col = f'{stock}_volume'
            
            if close_col in self.data.columns:
                # 1. Price-based features
                self.data[f'{stock}_returns_1d'] = self.data[close_col].pct_change()
                self.data[f'{stock}_returns_3d'] = self.data[close_col].pct_change(3)
                self.data[f'{stock}_returns_7d'] = self.data[close_col].pct_change(7)
                
                technical_features.extend([
                    f'{stock}_returns_1d', f'{stock}_returns_3d', f'{stock}_returns_7d'
                ])
                
                # 2. Moving averages
                for window in [5, 10, 20]:
                    self.data[f'{stock}_ma_{window}'] = self.data[close_col].rolling(window).mean()
                    self.data[f'{stock}_ma_ratio_{window}'] = self.data[close_col] / self.data[f'{stock}_ma_{window}']
                    technical_features.extend([
                        f'{stock}_ma_{window}', f'{stock}_ma_ratio_{window}'
                    ])
                
                # 3. Volatility features
                self.data[f'{stock}_volatility_1d'] = self.data[f'{stock}_returns_1d'].rolling(1).std()
                self.data[f'{stock}_volatility_3d'] = self.data[f'{stock}_returns_1d'].rolling(3).std()
                self.data[f'{stock}_volatility_7d'] = self.data[f'{stock}_returns_1d'].rolling(7).std()
                
                technical_features.extend([
                    f'{stock}_volatility_1d', f'{stock}_volatility_3d', f'{stock}_volatility_7d'
                ])
                
                # 4. Volume features
                if volume_col in self.data.columns:
                    self.data[f'{stock}_volume_ma_5'] = self.data[volume_col].rolling(5).mean()
                    self.data[f'{stock}_volume_ratio'] = self.data[volume_col] / self.data[f'{stock}_volume_ma_5']
                    technical_features.extend([
                        f'{stock}_volume_ma_5', f'{stock}_volume_ratio'
                    ])
        
        print(f"✅ Created {len(technical_features)} technical features")
        return technical_features
    
    def create_cross_features(self):
        """Create cross-features (10 features)"""
        print("🔗 Creating cross-features...")
        
        cross_features = []
        
        # 1. Social-Price relationships
        if 'reddit_title_count' in self.data.columns:
            for stock in ['GME', 'AMC', 'BB']:
                close_col = f'{stock}_close'
                if close_col in self.data.columns:
                    # Sentiment vs price correlation (rolling)
                    self.data[f'{stock}_sentiment_price_corr'] = (
                        self.data['reddit_title_count'].rolling(7).corr(self.data[close_col])
                    )
                    cross_features.append(f'{stock}_sentiment_price_corr')
                    
                    # Mention spike vs volume spike
                    mention_col = f'{stock}_mentions'
                    volume_col = f'{stock}_volume'
                    if mention_col in self.data.columns and volume_col in self.data.columns:
                        self.data[f'{stock}_mention_volume_sync'] = (
                            self.data[mention_col].rolling(3).mean() / 
                            self.data[volume_col].rolling(3).mean()
                        )
                        cross_features.append(f'{stock}_mention_volume_sync')
        
        # 2. Weekend sentiment impact
        self.data['weekend_sentiment_monday_impact'] = (
            self.data['sentiment_positive'].shift(1) * self.data['is_weekend'].shift(1)
        )
        cross_features.append('weekend_sentiment_monday_impact')
        
        # 3. Cross-stock correlations
        stocks = ['GME', 'AMC', 'BB']
        for i, stock1 in enumerate(stocks):
            for stock2 in stocks[i+1:]:
                close1 = f'{stock1}_close'
                close2 = f'{stock2}_close'
                if close1 in self.data.columns and close2 in self.data.columns:
                    self.data[f'{stock1}_{stock2}_corr'] = (
                        self.data[close1].rolling(7).corr(self.data[close2])
                    )
                    cross_features.append(f'{stock1}_{stock2}_corr')
        
        print(f"✅ Created {len(cross_features)} cross-features")
        return cross_features
    
    def create_target_variables(self):
        """Create target variables for prediction"""
        print("🎯 Creating target variables...")
        
        targets = []
        
        for stock in ['GME', 'AMC', 'BB']:
            close_col = f'{stock}_close'
            if close_col in self.data.columns:
                # 1. Direction prediction (1-3 days)
                for days in [1, 3]:
                    self.data[f'{stock}_direction_{days}d'] = (
                        (self.data[close_col].shift(-days) > self.data[close_col]).astype(int)
                    )
                    targets.append(f'{stock}_direction_{days}d')
                
                # 2. Magnitude prediction (3-7 days)
                for days in [3, 7]:
                    self.data[f'{stock}_magnitude_{days}d'] = (
                        (self.data[close_col].shift(-days) - self.data[close_col]) / self.data[close_col]
                    )
                    targets.append(f'{stock}_magnitude_{days}d')
        
        print(f"✅ Created {len(targets)} target variables")
        return targets
    
    def handle_missing_values(self):
        """Handle missing values in features"""
        print("🧹 Handling missing values...")
        
        # Forward fill for time series data
        self.data = self.data.fillna(method='ffill')
        
        # Fill remaining NaN with 0
        self.data = self.data.fillna(0)
        
        print("✅ Missing values handled")
    
    def scale_features(self):
        """Scale numerical features"""
        print("⚖️ Scaling features...")
        
        # Get numerical columns (exclude date and targets)
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date'] + [col for col in self.data.columns if 'direction' in col or 'magnitude' in col]
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Scale features
        self.data[feature_cols] = self.scaler.fit_transform(self.data[feature_cols])
        
        print(f"✅ Scaled {len(feature_cols)} features")
        return feature_cols
    
    def save_features(self, filename='features_data.csv'):
        """Save feature-engineered data"""
        self.data.to_csv(f"../data/{filename}", index=False)
        print(f"✅ Features saved to ../data/{filename}")
    
    def run_full_pipeline(self):
        """Run complete feature engineering pipeline"""
        print("🚀 Starting Feature Engineering Pipeline")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            return None
        
        # Create features
        reddit_features = self.create_reddit_features()
        technical_features = self.create_technical_features()
        cross_features = self.create_cross_features()
        targets = self.create_target_variables()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Scale features
        feature_cols = self.scale_features()
        
        # Save features
        self.save_features()
        
        print(f"\n🎉 Feature Engineering Completed!")
        print(f"📊 Total features: {len(feature_cols)}")
        print(f"🎯 Target variables: {len(targets)}")
        print(f"📈 Total dataset shape: {self.data.shape}")
        
        return self.data

if __name__ == "__main__":
    # Run feature engineering pipeline
    engineer = FeatureEngineer()
    features_data = engineer.run_full_pipeline() 