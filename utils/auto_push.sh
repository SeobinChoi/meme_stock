#!/bin/bash

# Auto push script for meme_stock project
echo "🤖 Auto-pushing to GitHub..."

# Add all changes
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to commit"
else
    # Get current timestamp for commit message
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Commit with timestamp
    git commit -m "Auto-push: $timestamp - Updates from development"
    
    # Push to GitHub
    git push
    
    echo "✅ Successfully pushed to GitHub at $timestamp"
fi 