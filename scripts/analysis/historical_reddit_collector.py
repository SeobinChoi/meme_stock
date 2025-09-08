#!/usr/bin/env python3
"""
Historical Reddit Data Collector for WSB (2020-2022)
Uses Reddit API with strategic approach to collect historical meme stock data
"""

import praw
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalRedditCollector:
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
    
    def get_top_posts_by_year(self, year: int, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get top posts from a specific year using different sorting methods"""
        posts = []
        
        # Try different sorting methods to get more historical data
        sort_methods = ['top', 'hot', 'new']
        
        for sort_method in sort_methods:
            logger.info(f"Collecting {sort_method} posts for {year}...")
            
            try:
                if sort_method == 'top':
                    # Get top posts with different time filters
                    time_filters = ['year', 'month', 'week', 'all']
                    for time_filter in time_filters:
                        try:
                            submissions = self.subreddit.top(time_filter=time_filter, limit=limit//len(time_filters))
                            for submission in submissions:
                                post_date = datetime.fromtimestamp(submission.created_utc)
                                if post_date.year == year:
                                    post_data = self.extract_post_data(submission)
                                    if post_data not in posts:
                                        posts.append(post_data)
                                        
                                if len(posts) >= limit:
                                    break
                                    
                        except Exception as e:
                            logger.warning(f"Error with {time_filter} filter: {e}")
                            continue
                            
                elif sort_method == 'hot':
                    # Get hot posts (might include some historical)
                    submissions = self.subreddit.hot(limit=limit//len(sort_methods))
                    for submission in submissions:
                        post_date = datetime.fromtimestamp(submission.created_utc)
                        if post_date.year == year:
                            post_data = self.extract_post_data(submission)
                            if post_data not in posts:
                                posts.append(post_data)
                                
                        if len(posts) >= limit:
                            break
                            
                elif sort_method == 'new':
                    # Get newer posts and work backwards
                    submissions = self.subreddit.new(limit=limit//len(sort_methods))
                    for submission in submissions:
                        post_date = datetime.fromtimestamp(submission.created_utc)
                        if post_date.year == year:
                            post_data = self.extract_post_data(submission)
                            if post_data not in posts:
                                posts.append(post_data)
                                
                        if len(posts) >= limit:
                            break
                            
            except Exception as e:
                logger.error(f"Error with {sort_method} method: {e}")
                continue
                
            # Rate limiting
            time.sleep(1)
            
        return posts
    
    def search_keyword_posts(self, keywords: List[str], year: int, limit: int = 500) -> List[Dict[str, Any]]:
        """Search for posts with specific keywords from a specific year"""
        posts = []
        
        for keyword in keywords:
            logger.info(f"Searching for '{keyword}' posts from {year}...")
            
            try:
                # Search with different time filters
                time_filters = ['year', 'month', 'week']
                
                for time_filter in time_filters:
                    try:
                        search_query = f"{keyword} year:{year}"
                        submissions = self.subreddit.search(search_query, sort='relevance', time_filter=time_filter, limit=limit//len(keywords))
                        
                        for submission in submissions:
                            post_date = datetime.fromtimestamp(submission.created_utc)
                            if post_date.year == year:
                                post_data = self.extract_post_data(submission)
                                if post_data not in posts:
                                    posts.append(post_data)
                                    
                            if len(posts) >= limit:
                                break
                                
                    except Exception as e:
                        logger.warning(f"Error with {time_filter} filter for {keyword}: {e}")
                        continue
                        
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error searching for {keyword}: {e}")
                continue
                
        return posts
    
    def extract_post_data(self, submission) -> Dict[str, Any]:
        """Extract relevant data from a Reddit submission"""
        return {
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
            'permalink': submission.permalink,
            'is_original_content': submission.is_original_content,
            'over_18': submission.over_18,
            'spoiler': submission.spoiler,
            'locked': submission.locked,
            'stickied': submission.stickied
        }
    
    def collect_historical_data(self, output_dir: str = "data/raw/reddit/extended") -> None:
        """Collect historical Reddit data from 2020-2022"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Keywords to focus on meme stocks
        keywords = [
            "GME", "GameStop", "AMC", "BlackBerry", "BB", "meme stock",
            "short squeeze", "diamond hands", "🚀", "💎", "🙌", "WSB",
            "roaring kitty", "DFV", "Keith Gill", "Robinhood", "TD Ameritrade"
        ]
        
        all_posts = []
        
        # Collect data for each year
        for year in [2020, 2021, 2022]:
            logger.info(f"Collecting data for {year}...")
            
            # Method 1: Get top posts by year
            year_posts = self.get_top_posts_by_year(year, limit=1000)
            logger.info(f"Collected {len(year_posts)} top posts for {year}")
            
            # Method 2: Search by keywords
            keyword_posts = self.search_keyword_posts(keywords, year, limit=500)
            logger.info(f"Collected {len(keyword_posts)} keyword posts for {year}")
            
            # Combine and remove duplicates
            combined_posts = year_posts + keyword_posts
            unique_posts = []
            seen_ids = set()
            
            for post in combined_posts:
                if post['id'] not in seen_ids:
                    unique_posts.append(post)
                    seen_ids.add(post['id'])
            
            logger.info(f"Total unique posts for {year}: {len(unique_posts)}")
            
            # Save year-specific data
            if unique_posts:
                year_df = pd.DataFrame(unique_posts)
                year_file = os.path.join(output_dir, f"wsb_{year}_historical.csv")
                year_df.to_csv(year_file, index=False)
                logger.info(f"Saved {len(unique_posts)} posts for {year} to {year_file}")
                
                all_posts.extend(unique_posts)
            
            # Rate limiting between years
            time.sleep(5)
        
        # Save combined dataset
        if all_posts:
            combined_df = pd.DataFrame(all_posts)
            combined_file = os.path.join(output_dir, "wsb_2020_2022_historical_combined.csv")
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"Saved combined dataset with {len(all_posts)} posts to {combined_file}")
            
            # Save summary statistics
            self.save_summary_stats(all_posts, output_dir)
    
    def save_summary_stats(self, posts: List[Dict[str, Any]], output_dir: str) -> None:
        """Save summary statistics of the collected data"""
        df = pd.DataFrame(posts)
        
        # Convert timestamp to datetime for analysis
        df['date'] = pd.to_datetime(df['timestamp'])
        
        # Daily post counts
        daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='post_count')
        daily_counts.to_csv(os.path.join(output_dir, "daily_post_counts_historical.csv"), index=False)
        
        # Monthly post counts
        monthly_counts = df.groupby([df['date'].dt.year, df['date'].dt.month]).size()
        monthly_counts.index.names = ['year', 'month']
        monthly_counts = monthly_counts.reset_index(name='post_count')
        monthly_counts.to_csv(os.path.join(output_dir, "monthly_post_counts_historical.csv"), index=False)
        
        # Top posts by score
        top_posts = df.nlargest(100, 'score')[['title', 'score', 'comms_num', 'timestamp']]
        top_posts.to_csv(os.path.join(output_dir, "top_100_posts_historical.csv"), index=False)
        
        # Keyword frequency analysis
        keyword_counts = {}
        for keyword in ["GME", "AMC", "BB", "meme", "short squeeze"]:
            keyword_counts[keyword] = df['title'].str.contains(keyword, case=False, na=False).sum()
        
        keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['keyword', 'count'])
        keyword_df.to_csv(os.path.join(output_dir, "keyword_frequency_historical.csv"), index=False)
        
        logger.info("Summary statistics saved")

def main():
    """Main function to run the historical Reddit data collection"""
    try:
        collector = HistoricalRedditCollector()
        collector.collect_historical_data()
        logger.info("Historical Reddit data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to collect historical Reddit data: {e}")
        raise

if __name__ == "__main__":
    main()