# Meme Stock Analysis Project

## Overview
This project analyzes meme stock movements using Reddit sentiment data, news data, and stock price data to predict market trends.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Reddit API Setup
1. Get Reddit API credentials from [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Update `config/reddit_config.json` with your credentials
3. Test connection: `python scripts/test_reddit_connection.py`

### 3. Download Extended Reddit Data
```bash
python scripts/download_extended_reddit_data.py
```

## 📊 Data Sources

### Current Data
- **Reddit WSB Posts**: 399,637 posts from 2021
- **Stock Price Data**: AMC, BlackBerry, GameStop, Wish (2013-2021)
- **Processed Data**: 29,793 filtered posts

### Extended Data (After Download)
- **2020-2022 WSB Posts**: Complete 3-year dataset
- **Enhanced Metadata**: Upvote ratios, author info, permalinks
- **Keyword Filtering**: Focus on meme stock related posts
- **Statistical Analysis**: Daily/monthly post counts, top posts

## 📁 Project Structure

```
meme_stock/
├── data/
│   ├── raw/
│   │   ├── reddit/           # Reddit data files
│   │   └── archive/          # Historical stock data
│   └── processed/            # Cleaned and processed data
├── scripts/
│   ├── download_extended_reddit_data.py  # Main downloader
│   └── test_reddit_connection.py         # Connection test
├── config/
│   └── reddit_config.json    # API credentials
├── docs/
│   ├── reddit_api_setup_guide.md         # Setup instructions
│   └── reddit_data_analysis_summary.md   # Data overview
└── requirements.txt           # Python dependencies
```

## 🔧 Configuration

### Reddit API Credentials
Create `config/reddit_config.json`:
```json
{
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "user_agent": "MemeStockAnalysis/1.0 (by /u/YOUR_USERNAME)",
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD"
}
```

## 📈 Features

### Data Collection
- **Multi-year Coverage**: 2020-2022 Reddit data
- **Smart Filtering**: Meme stock keyword detection
- **Rate Limiting**: Respects Reddit API limits
- **Error Handling**: Robust error recovery

### Data Analysis
- **Sentiment Analysis**: Post sentiment scoring
- **Engagement Metrics**: Upvotes, comments, ratios
- **Temporal Analysis**: Daily/monthly trends
- **Top Content**: Highest scoring posts

## 🚨 Important Notes

- **API Limits**: Reddit has rate limiting - be patient
- **Data Size**: Extended dataset will be several GB
- **Authentication**: Requires Reddit account and app
- **Storage**: Ensure sufficient disk space

## 📚 Documentation

- [Reddit API Setup Guide](docs/reddit_api_setup_guide.md)
- [Data Analysis Summary](docs/reddit_data_analysis_summary.md)
- [Project Plan](docs/plan_meme_stock_formatted_v3.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. 