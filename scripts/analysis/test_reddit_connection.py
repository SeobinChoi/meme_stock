#!/usr/bin/env python3
"""
Test Reddit API Connection
Simple script to test if Reddit API credentials are working correctly
"""

import praw
import json
import sys
import os

def test_reddit_connection(config_file: str = "config/reddit_config.json"):
    """Test Reddit API connection with provided credentials"""
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Config file {config_file} not found!")
        print("Please create the config file with your Reddit API credentials.")
        return False
    
    try:
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = ['client_id', 'client_secret', 'user_agent', 'username', 'password']
        missing_fields = [field for field in required_fields if field not in config or config[field].startswith('YOUR_')]
        
        if missing_fields:
            print(f"❌ Missing or placeholder values for: {', '.join(missing_fields)}")
            print("Please update the config file with your actual Reddit API credentials.")
            return False
        
        # Test connection
        print("🔗 Testing Reddit API connection...")
        
        reddit = praw.Reddit(
            client_id=config['client_id'],
            client_secret=config['client_secret'],
            user_agent=config['user_agent'],
            username=config['username'],
            password=config['password']
        )
        
        # Test basic API calls
        print("✅ Reddit client created successfully")
        
        # Test subreddit access
        subreddit = reddit.subreddit("wallstreetbets")
        print(f"✅ Successfully accessed r/{subreddit.display_name}")
        
        # Test getting a few posts
        posts = list(subreddit.hot(limit=3))
        print(f"✅ Successfully retrieved {len(posts)} posts")
        
        # Test user info
        user = reddit.user.me()
        print(f"✅ Successfully authenticated as u/{user.name}")
        
        print("\n🎉 Reddit API connection test passed!")
        print("You can now run the extended data downloader.")
        return True
        
    except Exception as e:
        print(f"❌ Reddit API connection test failed: {e}")
        print("\nCommon issues:")
        print("1. Check your Reddit API credentials")
        print("2. Ensure your Reddit account is not suspended")
        print("3. Verify the app type is set to 'script'")
        print("4. Check if you need to verify your email")
        return False

def main():
    """Main function"""
    print("Reddit API Connection Test")
    print("=" * 40)
    
    success = test_reddit_connection()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
