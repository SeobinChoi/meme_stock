# Reddit Data Analysis Summary

## Overview
This document provides a comprehensive analysis of all Reddit data available in the meme stock project, including data quantity, content structure, and key findings.

## Data Sources Summary

### 1. Main Reddit Data Files

#### `data/raw/reddit/raw_reddit_wsb.csv`
- **Total Records**: 399,637 posts
- **File Size**: ~43.7 MB
- **Content**: Raw WallStreetBets (WSB) Reddit posts
- **Columns**: title, score, id, url, comms_num, created, body, timestamp
- **Time Period**: 2021 (based on timestamp 1611862661.0 = Jan 28, 2021)

#### `data/raw/reddit/reddit_wsb.csv`
- **Total Records**: 29,793 posts
- **File Size**: ~29.8 KB
- **Content**: Processed/filtered WSB posts
- **Note**: This appears to be a cleaned subset of the raw data

### 2. Archive Data Structure

#### Archive-2 (Historical Stock Data)
- **AMC.csv**: 2,111 records - Stock price data (2013-2021)
- **BlackBerry.csv**: 5,853 records - Stock price data (2013-2021)  
- **GameStock.csv**: 5,094 records - Stock price data (2002-2021)
- **Wish.csv**: 350 records - Stock price data
- **Total**: 13,408 stock price records

#### Archive-3 (Reddit Data by Year)
- **2021/wallstreetbets_2021.csv**: 101 records
- **2022/wallstreetbets_2022.csv**: 101 records  
- **2023/wallstreetbets_2023.csv**: 101 records
- **Note**: These appear to be sample/example files, not full datasets

### 3. Additional Data Structure Files
- **additional_reddit_data_structure.csv**: 105 KB - Reddit data schema
- **news_data_structure.csv**: 120 KB - News data schema
- **options_data_structure.csv**: 107 KB - Options trading data schema

## Content Analysis

### Reddit Posts Content
The raw Reddit data contains:
- **Post titles**: Meme-style titles with emojis and references to stocks
- **Scores**: Upvote counts (e.g., 55, 110)
- **Comments**: Number of comments per post
- **Timestamps**: Unix timestamps converted to readable dates
- **URLs**: Links to Reddit posts and media content

### Sample Content Examples
1. "It's not about the money, it's about sending a message. 🚀💎🙌" (Score: 55)
2. "Math Professor Scott Steiner says the numbers spell DISASTER for Gamestop shorts" (Score: 110)

### Stock Data Content
The archive stock data contains standard OHLCV (Open, High, Low, Close, Volume) data:
- **AMC**: 2013-2021 daily stock data
- **BlackBerry**: 2013-2021 daily stock data  
- **GameStop**: 2002-2021 daily stock data
- **Wish**: Limited stock data

## Key Findings

### Data Volume
- **Total Reddit Posts**: ~430,000+ posts (raw + processed)
- **Total Stock Records**: ~13,400 daily price records
- **Main Dataset**: 399,637 raw WSB posts from 2021

### Data Quality
- Raw Reddit data appears complete with 399K+ posts
- Processed data is significantly smaller (29K posts) - likely filtered/cleaned
- Archive data shows consistent structure across different stocks
- Timestamps are properly formatted and cover relevant meme stock period

### Content Characteristics
- Posts focus on meme stocks (GME, AMC, BB, etc.)
- High engagement posts with scores and comment counts
- Rich metadata including URLs and timestamps
- Stock data covers both meme stock period and historical context

## Recommendations

1. **Primary Dataset**: Use `raw_reddit_wsb.csv` (399K posts) as main Reddit dataset
2. **Data Processing**: The processed `reddit_wsb.csv` may contain important filtering logic
3. **Stock Integration**: Combine Reddit sentiment with corresponding stock price data
4. **Time Alignment**: Focus on 2021 data where Reddit and stock data overlap

## Next Steps
- Analyze sentiment patterns in the 399K Reddit posts
- Correlate Reddit activity with stock price movements
- Extract key features for meme stock prediction models
- Validate data quality and completeness for modeling purposes
