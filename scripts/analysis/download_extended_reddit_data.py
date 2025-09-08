#!/usr/bin/env python3
"""
Extended Reddit Data Downloader for Meme Stock Analysis
Downloads WSB posts from 2020-2022 to expand the dataset beyond just 2021
"""

import praw
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtendedRedditDownloader:
    def __init__(self, config_file: str = "config/reddit_config.json"):
        """Initialize Reddit API client"""
        self.config = self.load_config(config_file)
        self.reddit = self.initialize_reddit_client()
        self.subreddit = self.reddit.subreddit("wallstreetbets")
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load Reddit API configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_file} not found. Please create it with your Reddit API credentials.")
            raise
            
    def initialize_reddit_client(self) -> praw.Reddit:
        """Initialize Reddit API client"""
        return praw.Reddit(
            client_id=self.config['client_id'],
            client_secret=self.config['client_secret'],
            user_agent=self.config['user_agent'],
            username=self.config['username'],
            password=self.config['password']
        )
    
    def get_posts_by_date_range(self, start_date: datetime, end_date: datetime, 
                               keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Get posts within a date range, optionally filtered by keywords"""
        posts = []
        
        # Convert dates to timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        logger.info(f"Searching posts from {start_date} to {end_date}")
        
        try:
            # Search for posts in the date range
            search_query = " OR ".join(keywords) if keywords else "GME OR AMC OR BB OR meme OR stock"
            
            for submission in self.subreddit.search(search_query, sort='new', time_filter='year'):
                # Check if post is within our date range
                if start_timestamp <= submission.created_utc <= end_timestamp:
                    post_data = {
                        'title': submission.title,
                        'score': submission.score,
                        'id': submission.id,
                        'url': submission.url,
                        'comms_num': submission.num_comments,
                        'created': submission.created_utc,
                        'body': submission.selftext if submission.selftext else '',
                        'timestamp': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'upvote_ratio': submission.upvote_ratio,
                        'author': str(submission.author) if submission.author else '[deleted]',
                        'subreddit': submission.subreddit.display_name,
                        'permalink': submission.permalink
                    }
                    posts.append(post_data)
                    
                    if len(posts) % 100 == 0:
                        logger.info(f"Collected {len(posts)} posts so far...")
                        
                # Stop if we've gone past our end date
                if submission.created_utc < start_timestamp:
                    break
                    
        except Exception as e:
            logger.error(f"Error collecting posts: {e}")
            
        return posts
    
    def download_extended_data(self, output_dir: str = "data/raw/reddit/extended") -> None:
        """Download extended Reddit data from 2020-2022"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Define date ranges for each year
        date_ranges = [
            (datetime(2020, 1, 1), datetime(2020, 12, 31), "2020"),
            (datetime(2021, 1, 1), datetime(2021, 12, 31), "2021"),
            (datetime(2022, 1, 1), datetime(2022, 12, 31), "2022")
        ]
        
        # Keywords to focus on meme stocks
        keywords = [
            "GME", "GameStop", "AMC", "BlackBerry", "BB", "meme stock",
            "short squeeze", "diamond hands", "🚀", "💎", "🙌", "WSB"
        ]
        
        all_posts = []
        
        for start_date, end_date, year in date_ranges:
            logger.info(f"Downloading data for {year}...")
            
            # Get posts for this year
            year_posts = self.get_posts_by_date_range(start_date, end_date, keywords)
            
            # Save year-specific data
            if year_posts:
                year_df = pd.DataFrame(year_posts)
                year_file = os.path.join(output_dir, f"wsb_{year}_extended.csv")
                year_df.to_csv(year_file, index=False)
                logger.info(f"Saved {len(year_posts)} posts for {year} to {year_file}")
                
                all_posts.extend(year_posts)
            
            # Rate limiting
            time.sleep(2)
        
        # Save combined dataset
        if all_posts:
            combined_df = pd.DataFrame(all_posts)
            combined_file = os.path.join(output_dir, "wsb_2020_2022_combined.csv")
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"Saved combined dataset with {len(all_posts)} posts to {combined_file}")
            
            # Save summary statistics
            self.save_summary_stats(all_posts, output_dir)
    
    def save_summary_stats(self, posts: List[Dict[str, Any]], output_dir: str) -> None:
        """Save summary statistics of the downloaded data"""
        df = pd.DataFrame(posts)
        
        # Convert timestamp to datetime for analysis
        df['date'] = pd.to_datetime(df['timestamp'])
        
        # Daily post counts
        daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='post_count')
        daily_counts.to_csv(os.path.join(output_dir, "daily_post_counts.csv"), index=False)
        
        # Monthly post counts
        monthly_counts = df.groupby([df['date'].dt.year, df['date'].dt.month]).size().reset_index(name='post_count')
        monthly_counts.columns = ['year', 'month', 'post_count']
        monthly_counts.to_csv(os.path.join(output_dir, "monthly_post_counts.csv"), index=False)
        
        # Top posts by score
        top_posts = df.nlargest(100, 'score')[['title', 'score', 'comms_num', 'timestamp']]
        top_posts.to_csv(os.path.join(output_dir, "top_100_posts.csv"), index=False)
        
        logger.info("Summary statistics saved")

def main():
    """Main function to run the extended Reddit data download"""
    try:
        downloader = ExtendedRedditDownloader()
        downloader.download_extended_data()
        logger.info("Extended Reddit data download completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to download extended Reddit data: {e}")
        raise

if __name__ == "__main__":
    main()
