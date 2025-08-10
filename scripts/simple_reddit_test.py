#!/usr/bin/env python3
"""
Simple Reddit Test - Verify we can download actual posts
"""

import praw
import json
import pandas as pd
from datetime import datetime

def simple_reddit_test():
    """Simple test to download some Reddit posts"""
    
    # Load config
    with open("config/reddit_config.json", 'r') as f:
        config = json.load(f)
    
    # Initialize Reddit client
    reddit = praw.Reddit(
        client_id=config['client_id'],
        client_secret=config['client_secret'],
        user_agent=config['user_agent'],
        username=config['username'],
        password=config['password']
    )
    
    print("🔗 Reddit client initialized successfully")
    
    # Get WSB subreddit
    subreddit = reddit.subreddit("wallstreetbets")
    print(f"✅ Accessing r/{subreddit.display_name}")
    
    # Try different methods to get posts
    posts = []
    
    print("\n📊 Testing different post collection methods...")
    
    # Method 1: Hot posts
    print("1. Getting hot posts...")
    try:
        hot_posts = list(subreddit.hot(limit=10))
        print(f"   ✅ Got {len(hot_posts)} hot posts")
        
        for post in hot_posts[:3]:  # First 3 posts
            post_data = {
                'title': post.title,
                'score': post.score,
                'id': post.id,
                'created': post.created_utc,
                'timestamp': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'comms_num': post.num_comments,
                'upvote_ratio': post.upvote_ratio
            }
            posts.append(post_data)
            print(f"   📝 {post.title[:50]}... (Score: {post.score})")
            
    except Exception as e:
        print(f"   ❌ Error getting hot posts: {e}")
    
    # Method 2: New posts
    print("\n2. Getting new posts...")
    try:
        new_posts = list(subreddit.new(limit=10))
        print(f"   ✅ Got {len(new_posts)} new posts")
        
        for post in new_posts[:3]:  # First 3 posts
            post_data = {
                'title': post.title,
                'score': post.score,
                'id': post.id,
                'created': post.created_utc,
                'timestamp': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'comms_num': post.num_comments,
                'upvote_ratio': post.upvote_ratio
            }
            posts.append(post_data)
            print(f"   📝 {post.title[:50]}... (Score: {post.score})")
            
    except Exception as e:
        print(f"   ❌ Error getting new posts: {e}")
    
    # Method 3: Search for GME
    print("\n3. Searching for GME posts...")
    try:
        search_posts = list(subreddit.search("GME", limit=10))
        print(f"   ✅ Got {len(search_posts)} GME search results")
        
        for post in search_posts[:3]:  # First 3 posts
            post_data = {
                'title': post.title,
                'score': post.score,
                'id': post.id,
                'created': post.created_utc,
                'timestamp': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'comms_num': post.num_comments,
                'upvote_ratio': post.upvote_ratio
            }
            posts.append(post_data)
            print(f"   📝 {post.title[:50]}... (Score: {post.score})")
            
    except Exception as e:
        print(f"   ❌ Error searching GME posts: {e}")
    
    # Save results
    if posts:
        print(f"\n💾 Saving {len(posts)} posts to test_output.csv")
        df = pd.DataFrame(posts)
        df.to_csv("test_output.csv", index=False)
        print("✅ Test completed successfully!")
        
        # Show date range
        if 'created' in df.columns:
            dates = [datetime.fromtimestamp(ts) for ts in df['created']]
            print(f"📅 Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
    else:
        print("❌ No posts collected")

if __name__ == "__main__":
    simple_reddit_test()
