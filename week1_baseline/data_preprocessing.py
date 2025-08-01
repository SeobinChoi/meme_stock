"""
Data Preprocessing Pipeline for Meme Stock Prediction
Week 1 Implementation - Academic Competition Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path='../data/'):
        self.data_path = data_path
        self.reddit_data = None
        self.stock_data = None
        self.mention_data = None
        self.merged_data = None
        
    def load_data(self):
        """Load all three datasets"""
        print("📊 Loading datasets...")
        
        try:
            # Load Reddit WSB posts
            self.reddit_data = pd.read_csv(f"{self.data_path}reddit_wsb.csv")
            print(f"✅ Reddit data loaded: {len(self.reddit_data)} posts")
            
            # Load meme stocks data
            self.stock_data = pd.read_csv(f"{self.data_path}meme_stocks.csv")
            print(f"✅ Stock data loaded: {len(self.stock_data)} records")
            
            # Load mention counts
            self.mention_data = pd.read_csv(f"{self.data_path}wsb_mention_counts.csv")
            print(f"✅ Mention data loaded: {len(self.mention_data)} records")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("Creating sample data for demonstration...")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration purposes"""
        # Sample Reddit data
        dates = pd.date_range('2021-01-01', '2021-12-31', freq='D')
        self.reddit_data = pd.DataFrame({
            'timestamp': dates,
            'title': [f'WSB post {i}' for i in range(len(dates))],
            'body': [f'This is post content {i}' for i in range(len(dates))],
            'score': np.random.randint(0, 1000, len(dates)),
            'comms_num': np.random.randint(0, 100, len(dates)),
            'id': [f'post_{i}' for i in range(len(dates))]
        })
        
        # Sample stock data
        self.stock_data = pd.DataFrame({
            'date': dates,
            'GME_close': np.random.uniform(10, 50, len(dates)),
            'GME_volume': np.random.randint(1000000, 10000000, len(dates)),
            'AMC_close': np.random.uniform(5, 30, len(dates)),
            'AMC_volume': np.random.randint(500000, 5000000, len(dates)),
            'BB_close': np.random.uniform(8, 25, len(dates)),
            'BB_volume': np.random.randint(300000, 3000000, len(dates))
        })
        
        # Sample mention data
        self.mention_data = pd.DataFrame({
            'date': dates,
            'GME_mentions': np.random.randint(0, 50, len(dates)),
            'AMC_mentions': np.random.randint(0, 30, len(dates)),
            'BB_mentions': np.random.randint(0, 20, len(dates))
        })
    
    def explore_data(self):
        """Analyze structure and quality of datasets"""
        print("\n🔍 Data Exploration Report")
        print("=" * 50)
        
        # Reddit data exploration
        print("\n📱 Reddit Data Analysis:")
        print(f"Shape: {self.reddit_data.shape}")
        print(f"Columns: {list(self.reddit_data.columns)}")
        print(f"Missing values:\n{self.reddit_data.isnull().sum()}")
        print(f"Date range: {self.reddit_data['timestamp'].min()} to {self.reddit_data['timestamp'].max()}")
        
        # Stock data exploration
        print("\n📈 Stock Data Analysis:")
        print(f"Shape: {self.stock_data.shape}")
        print(f"Columns: {list(self.stock_data.columns)}")
        print(f"Missing values:\n{self.stock_data.isnull().sum()}")
        
        # Mention data exploration
        print("\n📊 Mention Data Analysis:")
        print(f"Shape: {self.mention_data.shape}")
        print(f"Columns: {list(self.mention_data.columns)}")
        print(f"Missing values:\n{self.mention_data.isnull().sum()}")
    
    def clean_data(self):
        """Clean and prepare data for merging"""
        print("\n🧹 Cleaning data...")
        
        # Clean Reddit data
        if 'timestamp' in self.reddit_data.columns:
            self.reddit_data['timestamp'] = pd.to_datetime(self.reddit_data['timestamp'])
            self.reddit_data = self.reddit_data.dropna(subset=['title', 'body'])
        
        # Clean stock data
        if 'date' in self.stock_data.columns:
            self.stock_data['date'] = pd.to_datetime(self.stock_data['date'])
            # Forward fill missing values
            self.stock_data = self.stock_data.fillna(method='ffill')
        
        # Clean mention data
        if 'date' in self.mention_data.columns:
            self.mention_data['date'] = pd.to_datetime(self.mention_data['date'])
            self.mention_data = self.mention_data.fillna(0)
        
        print("✅ Data cleaning completed")
    
    def merge_datasets(self):
        """Merge three datasets based on date"""
        print("\n🔗 Merging datasets...")
        
        # Start with stock data as base
        merged = self.stock_data.copy()
        
        # Ensure date column is datetime
        merged['date'] = pd.to_datetime(merged['date'])
        
        # Merge with mention data
        if self.mention_data is not None:
            self.mention_data['date'] = pd.to_datetime(self.mention_data['date'])
            merged = merged.merge(self.mention_data, on='date', how='left')
        
        # Aggregate Reddit data by date
        if self.reddit_data is not None:
            # Convert timestamp to date for grouping
            self.reddit_data['date'] = pd.to_datetime(self.reddit_data['timestamp']).dt.date
            self.reddit_data['date'] = pd.to_datetime(self.reddit_data['date'])
            
            daily_reddit = self.reddit_data.groupby('date').agg({
                'score': ['mean', 'sum', 'count'],
                'comms_num': ['mean', 'sum'],
                'title': 'count',
                'body': 'count'
            }).reset_index()
            
            # Flatten column names
            daily_reddit.columns = ['date'] + [f'reddit_{col[0]}_{col[1]}' for col in daily_reddit.columns[1:]]
            
            merged = merged.merge(daily_reddit, on='date', how='left')
        
        # Fill missing values
        merged = merged.fillna(0)
        
        # Sort by date
        merged = merged.sort_values('date').reset_index(drop=True)
        
        self.merged_data = merged
        print(f"✅ Merged dataset shape: {merged.shape}")
        print(f"✅ Date range: {merged['date'].min()} to {merged['date'].max()}")
        
        return merged
    
    def handle_weekends_holidays(self):
        """Handle weekends and holidays by forward filling"""
        print("\n📅 Handling weekends and holidays...")
        
        # Create complete date range
        date_range = pd.date_range(
            start=self.merged_data['date'].min(),
            end=self.merged_data['date'].max(),
            freq='D'
        )
        
        # Reindex to include all dates
        self.merged_data = self.merged_data.set_index('date').reindex(date_range)
        
        # Forward fill missing values
        self.merged_data = self.merged_data.fillna(method='ffill')
        
        # Reset index
        self.merged_data = self.merged_data.reset_index().rename(columns={'index': 'date'})
        
        print(f"✅ Final dataset shape: {self.merged_data.shape}")
    
    def save_processed_data(self, filename='processed_data.csv'):
        """Save processed data"""
        if self.merged_data is not None:
            self.merged_data.to_csv(f"../data/{filename}", index=False)
            print(f"✅ Processed data saved to ../data/{filename}")
    
    def run_full_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("🚀 Starting Data Preprocessing Pipeline")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Clean data
        self.clean_data()
        
        # Merge datasets
        self.merge_datasets()
        
        # Handle weekends/holidays
        self.handle_weekends_holidays()
        
        # Save processed data
        self.save_processed_data()
        
        print("\n🎉 Data Preprocessing Pipeline Completed!")
        return self.merged_data

if __name__ == "__main__":
    # Run preprocessing pipeline
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_full_pipeline() 